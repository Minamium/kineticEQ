# kineticEQ/src/kineticEQ/CNN/BGK1D1V/eval_warmstart.py
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# --- make repo importable when executed as a script ---
# file: .../kineticEQ/src/kineticEQ/CNN/BGK1D1V/eval_warmstart.py
# add:  .../kineticEQ/src  to sys.path
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.CNN.BGK1D1V.models import MomentCNN1D


def _load_ckpt_state(ckpt_path: str) -> dict:
    """
    kineticEQ/CNN/BGK1D1V/train.py が保存する形式に対応:
      ckpt = {"epoch":..., "model": state_dict, "opt":..., ...}

    さらに以下も吸収:
      - {"model_state": state_dict}
      - {"state_dict": state_dict}
      - raw state_dict
      - DataParallel/DDP の module. prefix
    """
    obj = torch.load(ckpt_path, map_location="cpu")

    # dict checkpoint
    if isinstance(obj, dict):
        # kineticEQ train.py
        for k in ["model", "model_state", "state_dict", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # raw state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj

    # nn.Module保存など
    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
        if isinstance(sd, dict):
            return sd

    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def _strip_module_prefix(state: dict) -> dict:
    """Remove 'module.' prefix if saved from DataParallel/DistributedDataParallel."""
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def infer_arch_from_state(state: dict) -> tuple[int, int, int, int]:
    """
    Infer (in_ch, hidden, kernel, n_blocks) for kineticEQ.CNN.BGK1D1V.models.MomentCNN1D

    Expected keys:
      - stem.0.weight : (hidden, in_ch, kernel)
      - blocks.{i}.conv1.weight : used to infer n_blocks
    """
    state = _strip_module_prefix(state)

    stem_key = "stem.0.weight"
    if stem_key not in state:
        # Provide a more helpful error
        keys = sorted([k for k in state.keys() if "stem" in k or "blocks" in k])[:50]
        raise KeyError(f"Cannot find '{stem_key}' in checkpoint. Nearby keys (up to 50): {keys}")

    w = state[stem_key]
    if w.ndim != 3:
        raise ValueError(f"Unexpected '{stem_key}' shape: {tuple(w.shape)} (expected 3D: hidden,in_ch,kernel)")

    hidden = int(w.shape[0])
    in_ch = int(w.shape[1])
    kernel = int(w.shape[2])

    # Infer n_blocks by scanning blocks.*.conv1.weight
    idx = set()
    for k in state.keys():
        if k.startswith("blocks.") and k.endswith(".conv1.weight"):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                idx.add(int(parts[1]))

    if not idx:
        # Fallback: some variants might name blocks differently; try blocks.{i}.*
        idx2 = set()
        for k in state.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx2.add(int(parts[1]))
        if idx2:
            idx = idx2
        else:
            raise ValueError("Failed to infer n_blocks from checkpoint keys starting with 'blocks.{i}.'")

    n_blocks = max(idx) + 1
    if n_blocks <= 0:
        raise ValueError(f"Inferred n_blocks invalid: {n_blocks}")

    return in_ch, hidden, kernel, n_blocks


def load_model(ckpt_path: str, device: torch.device) -> tuple[MomentCNN1D, dict]:
    state = _load_ckpt_state(ckpt_path)
    state = _strip_module_prefix(state)

    in_ch, hidden, kernel, n_blocks = infer_arch_from_state(state)
    model = MomentCNN1D(in_ch=in_ch, hidden=hidden, out_ch=3, kernel=kernel, n_blocks=n_blocks)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    arch = {"in_ch": in_ch, "out_ch": 3, "hidden": hidden, "kernel": kernel, "n_blocks": n_blocks}
    return model, arch


@torch.no_grad()
def predict_next_moments_delta(model: MomentCNN1D,
                               n0: torch.Tensor, u0: torch.Tensor, T0: torch.Tensor,
                               logdt: float, logtau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Model outputs Δn, Δu, ΔT (trained with dataloader_npz.py).
    Return predicted absolute next moments:
      n1 = n0 + Δn, u1 = u0 + Δu, T1 = T0 + ΔT.

    Inputs are device tensors of shape (nx,).
    Model runs in float32; outputs are cast back to n0.dtype (usually float64).
    """
    nx = n0.numel()
    x = torch.empty((1, 5, nx), device=n0.device, dtype=torch.float32)
    x[0, 0] = n0.to(torch.float32)
    x[0, 1] = u0.to(torch.float32)
    x[0, 2] = T0.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))

    dy = model(x)[0]  # (3, nx) float32 = (Δn, Δu, ΔT)

    dn = dy[0].to(n0.dtype)
    du = dy[1].to(u0.dtype)
    dT = dy[2].to(T0.dtype)

    n1 = n0 + dn
    u1 = u0 + du
    T1 = T0 + dT
    return n1, u1, T1


@torch.no_grad()
def _maxwellian_from_nuT(state, n: torch.Tensor, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Non-destructive Maxwellian builder.
    Requires state.v_col (nx, nv) and state.inv_sqrt_2pi (scalar tensor) to exist.
    Returns (nx, nv) in state dtype/device.
    """
    # coeff = n / sqrt(2*pi*T)
    coeff = (n * state.inv_sqrt_2pi) / torch.sqrt(T)          # (nx,)
    invT = 0.5 / T                                            # (nx,)
    diff = state.v_col - u[:, None]                           # (nx, nv)
    expo = diff.mul(diff)                                     # (nx, nv)
    expo.mul_(-invT[:, None])
    torch.exp(expo, out=expo)
    expo.mul_(coeff[:, None])
    return expo


@torch.no_grad()
def build_fz_from_moments(state,
                          n1: torch.Tensor, u1: torch.Tensor, T1: torch.Tensor,
                          n_floor: float = 1e-12, T_floor: float = 1e-12) -> torch.Tensor:
    """
    Build fz Maxwellian from predicted moments without mutating state.n/u/T.

    Safety:
      - NaN/Inf -> fallback to state.f clone
      - clamp n,T to positive floors
      - boundary is copied from current distribution state.f
    """
    if (not torch.isfinite(n1).all()) or (not torch.isfinite(u1).all()) or (not torch.isfinite(T1).all()):
        return state.f.clone()

    n1 = torch.clamp(n1, min=n_floor)
    T1 = torch.clamp(T1, min=T_floor)

    fz = _maxwellian_from_nuT(state, n1, u1, T1)  # (nx, nv)

    # keep boundary from current distribution (important for implicit boundary Maxwell)
    fz[0, :].copy_(state.f[0, :])
    fz[-1, :].copy_(state.f[-1, :])
    return fz


def build_cfg(tau: float, dt: float, T_total: float,
              nx: int = 512, nv: int = 256, Lx: float = 1.0, v_max: float = 10.0,
              picard_iter: int = 1000, picard_tol: float = 1e-6, abs_tol: float = 1e-13) -> Config:
    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=nx, nv=nv, Lx=Lx, v_max=v_max),
        time=BGK1D.TimeConfig(dt=dt, T_total=T_total),
        params=BGK1D.BGK1D1VParams(tau_tilde=tau),
        scheme_params=BGK1D.implicit.Params(picard_iter=picard_iter, picard_tol=picard_tol, abs_tol=abs_tol),
        initial=BGK1D.InitialCondition1D(initial_regions=(
            {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
            {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
        )),
    )
    cfg = Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cuda_kernel",
        model_cfg=model_cfg,
        log_level="err",
        use_tqdm=False,
    )
    return cfg


@torch.no_grad()
def run_case_baseline(cfg: Config, n_steps: int, device: torch.device) -> dict:
    # Ensure torch uses the requested device for any implicit allocations
    torch.set_default_device(device)

    eng = Engine(cfg)
    it_hist = np.empty((n_steps,), dtype=np.int32)
    resid_hist = np.empty((n_steps,), dtype=np.float32)

    t0 = time.perf_counter()
    for s in range(n_steps):
        eng.stepper(s)
        bench = getattr(eng.stepper, "benchlog", None) or {}
        it_hist[s] = int(bench.get("picard_iter", -1))
        resid_hist[s] = float(bench.get("std_picard_residual", np.nan))
    t1 = time.perf_counter()

    # final moments for sanity
    n, u, T = calculate_moments(eng.state, eng.state.f)
    out = {
        "walltime_sec": float(t1 - t0),
        "picard_iter_sum": int(it_hist[it_hist > 0].sum()),
        "picard_iter_mean": float(np.mean(it_hist[it_hist > 0])) if np.any(it_hist > 0) else float("nan"),
        "picard_iter_hist": it_hist.tolist(),
        "std_resid_hist": resid_hist.tolist(),
        "final_moments": {
            "n": n.detach().cpu().double().numpy().tolist(),
            "u": u.detach().cpu().double().numpy().tolist(),
            "T": T.detach().cpu().double().numpy().tolist(),
        }
    }
    return out


@torch.no_grad()
def run_case_warmstart(cfg: Config,
                      model: MomentCNN1D,
                      n_steps: int,
                      device: torch.device,
                      mix_alpha: float = 1.0) -> dict:
    """
    mix_alpha:
      1.0 -> pure warmstart Maxwellian
      0.0 -> no warmstart (equivalent to baseline init fz=state.f)
      0<alpha<1 -> convex mix: fz = alpha*fz_pred + (1-alpha)*state.f
    """
    torch.set_default_device(device)

    eng = Engine(cfg)
    ws = getattr(eng.stepper, "ws", None)
    if ws is None:
        raise RuntimeError("eng.stepper.ws is missing. Ensure implicit build_stepper sets _stepper.ws = ws.")

    dt = float(cfg.model_cfg.time.dt)
    tau = float(cfg.model_cfg.params.tau_tilde)
    logdt = float(np.log10(dt))
    logtau = float(np.log10(tau))

    it_hist = np.empty((n_steps,), dtype=np.int32)
    resid_hist = np.empty((n_steps,), dtype=np.float32)

    t0 = time.perf_counter()
    for s in range(n_steps):
        # moments at current step
        n0, u0, T0 = calculate_moments(eng.state, eng.state.f)

        # predict next moments (Δ model)
        n1, u1, T1 = predict_next_moments_delta(model, n0, u0, T0, logdt=logdt, logtau=logtau)

        # build fz init from predicted moments and inject via workspace hook
        fz_init = build_fz_from_moments(eng.state, n1, u1, T1)

        # optional stabilization via mixing
        if mix_alpha < 1.0:
            a = float(mix_alpha)
            fz_init = a * fz_init + (1.0 - a) * eng.state.f

        ws._init_fz = fz_init

        # step
        eng.stepper(s)
        bench = getattr(eng.stepper, "benchlog", None) or {}
        it_hist[s] = int(bench.get("picard_iter", -1))
        resid_hist[s] = float(bench.get("std_picard_residual", np.nan))
    t1 = time.perf_counter()

    # final moments for sanity
    n, u, T = calculate_moments(eng.state, eng.state.f)
    out = {
        "walltime_sec": float(t1 - t0),
        "picard_iter_sum": int(it_hist[it_hist > 0].sum()),
        "picard_iter_mean": float(np.mean(it_hist[it_hist > 0])) if np.any(it_hist > 0) else float("nan"),
        "picard_iter_hist": it_hist.tolist(),
        "std_resid_hist": resid_hist.tolist(),
        "final_moments": {
            "n": n.detach().cpu().double().numpy().tolist(),
            "u": u.detach().cpu().double().numpy().tolist(),
            "T": T.detach().cpu().double().numpy().tolist(),
        }
    }
    return out


def moments_error(m0: dict, m1: dict) -> dict:
    # m is dict with "n/u/T" list
    def _arr(k, src):
        return np.array(src["final_moments"][k], dtype=np.float64)

    err = {}
    for k in ["n", "u", "T"]:
        a = _arr(k, m0)
        b = _arr(k, m1)
        d = b - a
        err[k] = {
            "l2": float(np.linalg.norm(d) / (np.linalg.norm(a) + 1e-30)),
            "linf": float(np.max(np.abs(d)) / (np.max(np.abs(a)) + 1e-30)),
        }
    return err


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt / last.pt")
    p.add_argument("--tau", type=float, nargs="+", required=True, help="tau_tilde (one or more)")
    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--T_total", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=-1, help="override steps; if <0 uses round(T_total/dt)")
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--nv", type=int, default=256)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--v_max", type=float, default=10.0)

    p.add_argument("--picard_iter", type=int, default=1000)
    p.add_argument("--picard_tol", type=float, default=1e-6)
    p.add_argument("--abs_tol", type=float, default=1e-13)

    p.add_argument("--mix_alpha", type=float, default=1.0, help="0..1 mixing for warmstart fz")

    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, arch = load_model(args.ckpt, device=device)

    results = {
        "meta": {
            "ckpt": str(args.ckpt),
            "dt": float(args.dt),
            "T_total": float(args.T_total),
            "nx": int(args.nx),
            "nv": int(args.nv),
            "Lx": float(args.Lx),
            "v_max": float(args.v_max),
            "picard_iter": int(args.picard_iter),
            "picard_tol": float(args.picard_tol),
            "abs_tol": float(args.abs_tol),
            "mix_alpha": float(args.mix_alpha),
            "model_arch": arch,
        },
        "cases": [],
    }

    for tau in args.tau:
        cfg = build_cfg(
            tau=float(tau), dt=float(args.dt), T_total=float(args.T_total),
            nx=int(args.nx), nv=int(args.nv), Lx=float(args.Lx), v_max=float(args.v_max),
            picard_iter=int(args.picard_iter), picard_tol=float(args.picard_tol), abs_tol=float(args.abs_tol),
        )
        if args.n_steps > 0:
            n_steps = int(args.n_steps)
        else:
            n_steps = int(round(cfg.model_cfg.time.T_total / cfg.model_cfg.time.dt))

        base = run_case_baseline(cfg, n_steps=n_steps, device=device)
        warm = run_case_warmstart(cfg, model=model, n_steps=n_steps, device=device, mix_alpha=float(args.mix_alpha))

        speed = (base["picard_iter_sum"] / max(warm["picard_iter_sum"], 1))
        wall_speed = (base["walltime_sec"] / max(warm["walltime_sec"], 1e-30))

        case_out = {
            "tau_tilde": float(tau),
            "n_steps": int(n_steps),
            "baseline": base,
            "warmstart": warm,
            "speedup_picard_sum": float(speed),
            "speedup_walltime": float(wall_speed),
            "final_moment_error": moments_error(base, warm),
        }
        results["cases"].append(case_out)

        print(
            f"[tau={tau:.3e}] picard_sum: base={base['picard_iter_sum']} warm={warm['picard_iter_sum']} "
            f"(x{speed:.2f}), wall: base={base['walltime_sec']:.2f}s warm={warm['walltime_sec']:.2f}s (x{wall_speed:.2f})",
            flush=True
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
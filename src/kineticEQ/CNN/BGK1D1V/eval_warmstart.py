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
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_maxwellian import maxwellian


class MomentCNN1D(nn.Module):
    """
    Simple Conv1d network:
      input:  (B, 5, nx)  [n,u,T,logdt,logtau]
      output: (B, 3, nx)  [n_next,u_next,T_next]

    Note:
    - This is a baseline. Later you can add residual blocks / normalization.
    """
    def __init__(self, ch_in: int = 5, ch_hidden: int = 64, ch_out: int = 3, kernel: int = 5) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch_in, ch_hidden, kernel_size=kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(ch_hidden, ch_hidden, kernel_size=kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(ch_hidden, ch_out, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_ckpt_state(ckpt_path: str) -> dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # could be raw state_dict
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)} keys={list(obj.keys())[:5] if isinstance(obj, dict) else ''}")


def infer_arch_from_state(state: dict) -> tuple[int, int, int]:
    """
    Infer (in_ch, hidden, kernel) from first conv weight: net.0.weight
      weight shape = (hidden, in_ch, kernel)
    """
    key = None
    for cand in ["net.0.weight", "module.net.0.weight"]:
        if cand in state:
            key = cand
            break
    if key is None:
        raise KeyError("Cannot find first conv weight (net.0.weight) in checkpoint.")

    w = state[key]
    if w.ndim != 3:
        raise ValueError(f"Unexpected net.0.weight dim: {w.shape}")
    hidden = int(w.shape[0])
    in_ch = int(w.shape[1])
    kernel = int(w.shape[2])
    return in_ch, hidden, kernel


def load_model(ckpt_path: str, device: torch.device) -> MomentCNN1D:
    state = _load_ckpt_state(ckpt_path)
    in_ch, hidden, kernel = infer_arch_from_state(state)
    model = MomentCNN1D(in_ch=in_ch, out_ch=3, hidden=hidden, kernel=kernel)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_next_moments(model: MomentCNN1D,
                         n: torch.Tensor, u: torch.Tensor, T: torch.Tensor,
                         logdt: float, logtau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inputs are device tensors of shape (nx,).
    Model runs in float32; outputs are cast back to n.dtype (usually float64).
    """
    nx = n.numel()
    x = torch.empty((1, 5, nx), device=n.device, dtype=torch.float32)
    x[0, 0] = n.to(torch.float32)
    x[0, 1] = u.to(torch.float32)
    x[0, 2] = T.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))

    y = model(x)[0]  # (3, nx) float32
    n1 = y[0].to(n.dtype)
    u1 = y[1].to(n.dtype)
    T1 = y[2].to(n.dtype)
    return n1, u1, T1


@torch.no_grad()
def build_fz_from_moments(state, n1: torch.Tensor, u1: torch.Tensor, T1: torch.Tensor,
                          n_floor: float = 1e-12, T_floor: float = 1e-12) -> torch.Tensor:
    """
    Use existing maxwellian(state) implementation.
    Boundary (0, -1) is overwritten by current state.f boundary to avoid injecting nonsense.
    """
    # enforce positivity
    n1 = torch.clamp(n1, min=n_floor)
    T1 = torch.clamp(T1, min=T_floor)

    # write into state moment buffers (maxwellian reads state.n/u/T)
    state.n.copy_(n1)
    state.u.copy_(u1)
    state.T.copy_(T1)

    fz = maxwellian(state)  # (nx, nv), dtype follows state

    # keep boundary from current distribution
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
def run_case_warmstart(cfg: Config, model: MomentCNN1D, n_steps: int, device: torch.device) -> dict:
    eng = Engine(cfg)
    ws = getattr(eng.stepper, "ws", None)
    if ws is None:
        raise RuntimeError("eng.stepper.ws is missing. Ensure build_stepper sets _stepper.ws = ws for implicit scheme.")

    dt = float(cfg.model_cfg.time.dt)
    tau = float(cfg.model_cfg.params.tau_tilde)
    logdt = np.log10(dt)
    logtau = np.log10(tau)

    it_hist = np.empty((n_steps,), dtype=np.int32)
    resid_hist = np.empty((n_steps,), dtype=np.float32)

    t0 = time.perf_counter()
    for s in range(n_steps):
        # moments at current step
        n0, u0, T0 = calculate_moments(eng.state, eng.state.f)

        # predict next moments
        n1, u1, T1 = predict_next_moments(model, n0, u0, T0, logdt=logdt, logtau=logtau)

        # build fz init from predicted moments and inject via workspace hook
        fz_init = build_fz_from_moments(eng.state, n1, u1, T1)
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

    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.ckpt, device=device)

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
            "model_arch": {
                "in_ch": 5,
                "out_ch": 3,
                "hidden": int(next(iter(model.net[0].weight.shape))),
                "kernel": int(model.net[0].weight.shape[2]),
            },
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
        warm = run_case_warmstart(cfg, model=model, n_steps=n_steps, device=device)

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

        print(f"[tau={tau:.3e}] picard_sum: base={base['picard_iter_sum']} warm={warm['picard_iter_sum']} "
              f"(x{speed:.2f}), wall: base={base['walltime_sec']:.2f}s warm={warm['walltime_sec']:.2f}s (x{wall_speed:.2f})",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

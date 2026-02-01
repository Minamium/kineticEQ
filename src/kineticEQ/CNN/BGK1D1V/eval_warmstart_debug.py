# kineticEQ/CNN/BGK1D1V/eval_warmstart_debug.py

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# --- make repo importable when executed as a script ---
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

    if isinstance(obj, dict):
        for k in ["model", "model_state", "state_dict", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj

    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
        if isinstance(sd, dict):
            return sd

    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def _strip_module_prefix(state: dict) -> dict:
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def infer_arch_from_state(state: dict) -> tuple[int, int, int, int]:
    state = _strip_module_prefix(state)

    stem_key = "stem.0.weight"
    if stem_key not in state:
        keys = sorted([k for k in state.keys() if "stem" in k or "blocks" in k])[:80]
        raise KeyError(f"Cannot find '{stem_key}' in checkpoint. Nearby keys (up to 80): {keys}")

    w = state[stem_key]
    if w.ndim != 3:
        raise ValueError(f"Unexpected '{stem_key}' shape: {tuple(w.shape)}")
    hidden = int(w.shape[0])
    in_ch = int(w.shape[1])
    kernel = int(w.shape[2])

    idx = set()
    for k in state.keys():
        if k.startswith("blocks.") and k.endswith(".conv1.weight"):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                idx.add(int(parts[1]))
    if not idx:
        idx2 = set()
        for k in state.keys():
            if k.startswith("blocks."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx2.add(int(parts[1]))
        if idx2:
            idx = idx2
        else:
            raise ValueError("Failed to infer n_blocks from checkpoint keys 'blocks.{i}.*'")

    n_blocks = max(idx) + 1
    return in_ch, hidden, kernel, n_blocks


def load_model(ckpt_path: str, device: torch.device) -> tuple[MomentCNN1D, dict]:
    state = _load_ckpt_state(ckpt_path)
    state = _strip_module_prefix(state)

    in_ch, hidden, kernel, n_blocks = infer_arch_from_state(state)
    model = MomentCNN1D(in_ch=in_ch, hidden=hidden, out_ch=3, kernel=kernel, n_blocks=n_blocks)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    arch = {"in_ch": in_ch, "out_ch": 3, "hidden": hidden, "kernel": kernel, "n_blocks": n_blocks}
    return model, arch


@torch.no_grad()
def predict_next_moments_delta(
    model: MomentCNN1D,
    n0: torch.Tensor, u0: torch.Tensor, T0: torch.Tensor,
    logdt: float, logtau: float,
    n_floor: float = 1e-12,
    delta_type: str = "dnu"
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    dnu 学習用:
      model outputs dy = [Δn, Δ(nu), ΔT]
      
    dw 学習用:
      model outputs dy = [Δn, Δu, ΔT]

    Returns:(dnu 学習用)
      (n1, u1, T1, dn, dm, dT)

    Returns:(dw 学習用)
      (n1, u1, T1, dn, du, dT)

    Note:
      u1 is reconstructed from momentum:
        m0 = n0*u0, m1 = m0 + dm, n1 = n0 + dn, u1 = m1 / max(n1, n_floor)
    """
    nx = n0.numel()
    x = torch.empty((1, 5, nx), device=n0.device, dtype=torch.float32)
    x[0, 0] = n0.to(torch.float32)
    x[0, 1] = u0.to(torch.float32)
    x[0, 2] = T0.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))

    dy = model(x)[0]  # (3, nx) float32
    dn = dy[0].to(n0.dtype)
    dT = dy[2].to(T0.dtype)

    n1 = n0 + dn
    if delta_type == "dnu":
        dm = dy[1].to(n0.dtype)   # momentum delta uses n dtype
        n1_safe = torch.clamp(n1, min=float(n_floor))
        m0 = n0 * u0
        m1 = m0 + dm
        u1 = m1 / n1_safe
    elif delta_type == "dw":
        du = dy[1].to(n0.dtype)
        u1 = u0 + du
        dm = du
    else:
        raise ValueError(f"unknown delta_type={delta_type}")
    
    T1 = T0 + dT
    return n1[1:-1], u1[1:-1], T1[1:-1], dn, dm, dT


@torch.no_grad()
def _maxwellian_from_nuT(state, n: torch.Tensor, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Non-destructive Maxwellian builder.
    Requires state.v_col (nx, nv) and state.inv_sqrt_2pi (scalar tensor).
    """
    coeff = (n * state.inv_sqrt_2pi) / torch.sqrt(T)          # (nx,)
    invT = 0.5 / T                                            # (nx,)
    diff = state.v_col - u[:, None]                           # (nx, nv)
    expo = diff.mul(diff)
    expo.mul_(-invT[:, None])
    torch.exp(expo, out=expo)
    expo.mul_(coeff[:, None])
    return expo


@torch.no_grad()
def build_fz_from_moments(
    state,
    n1: torch.Tensor, u1: torch.Tensor, T1: torch.Tensor,
    n_floor: float = 1e-12, T_floor: float = 1e-12
) -> torch.Tensor:
    """
    Build Maxwellian fz from moments (with boundary overwrite).
    Note: boundary overwrite intentionally breaks exact moment reproduction (full-domain).
    """
    if (not torch.isfinite(n1).all()) or (not torch.isfinite(u1).all()) or (not torch.isfinite(T1).all()):
        return state.f.clone()

    n1 = torch.clamp(n1, min=n_floor)
    T1 = torch.clamp(T1, min=T_floor)

    fz = _maxwellian_from_nuT(state, n1, u1, T1)

    # keep boundary from current distribution
    fz[0, :].copy_(state.f[0, :])
    fz[-1, :].copy_(state.f[-1, :])
    return fz


def build_cfg(
    tau: float, dt: float, T_total: float,
    nx: int = 512, nv: int = 256, Lx: float = 1.0, v_max: float = 10.0,
    picard_iter: int = 1000, picard_tol: float = 1e-6, abs_tol: float = 1e-13
) -> Config:
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
        device="cuda",
        model_cfg=model_cfg,
        log_level="err",
        use_tqdm=False,
    )
    return cfg


def _rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30) -> dict:
    d = (b - a)
    l2 = torch.linalg.norm(d) / (torch.linalg.norm(a) + eps)
    linf = torch.max(torch.abs(d)) / (torch.max(torch.abs(a)) + eps)
    return {"l2": float(l2.detach().cpu()), "linf": float(linf.detach().cpu())}


def _abs_stats(x: torch.Tensor) -> dict:
    ax = torch.abs(x)
    return {
        "abs_max": float(torch.max(ax).detach().cpu()),
        "abs_mean": float(torch.mean(ax).detach().cpu()),
    }


@torch.no_grad()
def run_case_debug(
    cfg: Config,
    model: MomentCNN1D,
    n_steps: int,
    device: torch.device,
    mix_alpha: float,
    debug_steps: int,
    n_floor: float,
    T_floor: float,
    delta_type: str,
) -> dict:
    """
    baseline と warmstart を並走し、stepごとの誤差を観測。
    debug_steps: 先頭何stepだけ詳細ログを残すか

    dnu 版:
      - model output: [Δn, Δ(nu), ΔT]
      - u は (n,u) から復元して評価
    """

    eng_base = Engine(cfg)
    eng_warm = Engine(cfg)

    ws = getattr(eng_warm.stepper, "ws", None)
    if ws is None:
        raise RuntimeError("eng.stepper.ws is missing. Ensure implicit build_stepper sets _stepper.ws = ws.")

    dt = float(cfg.model_cfg.time.dt)
    tau = float(cfg.model_cfg.params.tau_tilde)
    logdt = float(np.log10(dt))
    logtau = float(np.log10(tau))

    it_hist_base = np.empty((n_steps,), dtype=np.int32)
    it_hist_warm = np.empty((n_steps,), dtype=np.int32)
    resid_hist_base = np.empty((n_steps,), dtype=np.float32)
    resid_hist_warm = np.empty((n_steps,), dtype=np.float32)

    debug_log = []

    t0 = time.perf_counter()
    for s in range(n_steps):
        # baseline current moments
        n0_b, u0_b, T0_b = calculate_moments(eng_base.state, eng_base.state.f)

        # predict using baseline inputs (fair)
        n1p = n0_b.clone()
        u1p = u0_b.clone()
        T1p = T0_b.clone()
        
        n1p[1:-1], u1p[1:-1], T1p[1:-1], dn, dm, dT = predict_next_moments_delta(
            model, n0_b, u0_b, T0_b, logdt, logtau, n_floor=float(n_floor), delta_type=delta_type
        )

        # build warmstart fz
        fz_pure = build_fz_from_moments(eng_warm.state, n1p, u1p, T1p, n_floor=n_floor, T_floor=T_floor)

        fz_used = fz_pure
        if mix_alpha < 1.0:
            a = float(mix_alpha)
            fz_used = a * fz_pure + (1.0 - a) * eng_warm.state.f

        ws._init_fz = fz_used

        # step both
        eng_base.stepper(s)
        eng_warm.stepper(s)

        # bench
        bench_b = getattr(eng_base.stepper, "benchlog", None) or {}
        bench_w = getattr(eng_warm.stepper, "benchlog", None) or {}
        it_hist_base[s] = int(bench_b.get("picard_iter", -1))
        it_hist_warm[s] = int(bench_w.get("picard_iter", -1))
        resid_hist_base[s] = float(bench_b.get("std_picard_residual", np.nan))
        resid_hist_warm[s] = float(bench_w.get("std_picard_residual", np.nan))

        if s < debug_steps:
            # baseline true next moments
            n1_b, u1_b, T1_b = calculate_moments(eng_base.state, eng_base.state.f)

            # warm next moments
            n1_w, u1_w, T1_w = calculate_moments(eng_warm.state, eng_warm.state.f)

            # (i) prediction vs baseline true next
            pred_err = {
                "n": _rel_err(n1_b, n1p),
                "u": _rel_err(u1_b, u1p),
                "T": _rel_err(T1_b, T1p),
            }

            # (ii-a) moments of PURE fz vs (n1p,u1p,T1p)
            n_fz, u_fz, T_fz = calculate_moments(eng_warm.state, fz_pure)
            fz_moment_err_full = {
                "n": _rel_err(n1p, n_fz),
                "u": _rel_err(u1p, u_fz),
                "T": _rel_err(T1p, T_fz),
            }
            fz_moment_err_interior = {
                "n": _rel_err(n1p[1:-1], n_fz[1:-1]),
                "u": _rel_err(u1p[1:-1], u_fz[1:-1]),
                "T": _rel_err(T1p[1:-1], T_fz[1:-1]),
            }

            # (ii-c) USED fz moments stats
            n_used, u_used, T_used = calculate_moments(eng_warm.state, fz_used)
            used_fz_moments_stats = {
                "n": _abs_stats(n_used),
                "u": _abs_stats(u_used),
                "T": _abs_stats(T_used),
            }

            # (iii) baseline vs warm divergence
            sol_err = {
                "n": _rel_err(n1_b, n1_w),
                "u": _rel_err(u1_b, u1_w),
                "T": _rel_err(T1_b, T1_w),
            }

            # (iv-a) predicted step-change metrics
            # dn is dn
            # dm corresponds to momentum change; to compare with du_true, compute du_pred = u1p-u0_b
            du_pred = u1p - u0_b
            pred_step_change = {
                "dn_over_n0_linf": float(torch.max(torch.abs(dn) / (torch.abs(n0_b) + 1e-30)).detach().cpu()),
                "dT_over_T0_linf": float(torch.max(torch.abs(dT) / (torch.abs(T0_b) + 1e-30)).detach().cpu()),
                "du_abs_linf": float(torch.max(torch.abs(du_pred)).detach().cpu()),
                "dm_abs_linf": float(torch.max(torch.abs(dm)).detach().cpu()),
            }

            # (iv-b) true step-change metrics
            dn_true = n1_b - n0_b
            du_true = u1_b - u0_b
            dT_true = T1_b - T0_b
            m0_true = n0_b * u0_b
            m1_true = n1_b * u1_b
            dm_true = m1_true - m0_true
            true_step_change = {
                "dn_over_n0_linf": float(torch.max(torch.abs(dn_true) / (torch.abs(n0_b) + 1e-30)).detach().cpu()),
                "dT_over_T0_linf": float(torch.max(torch.abs(dT_true) / (torch.abs(T0_b) + 1e-30)).detach().cpu()),
                "du_abs_linf": float(torch.max(torch.abs(du_true)).detach().cpu()),
                "dm_abs_linf": float(torch.max(torch.abs(dm_true)).detach().cpu()),
            }

            # (v) scale stats
            u_scale = {
                "u0_true": _abs_stats(u0_b),
                "u1_true": _abs_stats(u1_b),
                "u1_pred": _abs_stats(u1p),
                "du_pred": _abs_stats(du_pred),
                "du_true": _abs_stats(du_true),
                "dm_pred": _abs_stats(dm),
                "dm_true": _abs_stats(dm_true),
            }

            debug_log.append({
                "step": int(s),
                "picard_iter_base": int(it_hist_base[s]),
                "picard_iter_warm": int(it_hist_warm[s]),
                "std_resid_base": float(resid_hist_base[s]),
                "std_resid_warm": float(resid_hist_warm[s]),
                "pred_err_vs_baseline_next": pred_err,
                "fz_moment_reproduction_err_full": fz_moment_err_full,
                "fz_moment_reproduction_err_interior": fz_moment_err_interior,
                "used_fz_moments_stats": used_fz_moments_stats,
                "baseline_vs_warm_next_err": sol_err,
                "pred_step_change_metrics": pred_step_change,
                "true_step_change_metrics": true_step_change,
                "u_scale_stats": u_scale,
            })

    t1 = time.perf_counter()

    nb, ub, Tb = calculate_moments(eng_base.state, eng_base.state.f)
    nw, uw, Tw = calculate_moments(eng_warm.state, eng_warm.state.f)

    out = {
        "walltime_sec": float(t1 - t0),
        "picard_iter_sum_base": int(it_hist_base[it_hist_base > 0].sum()),
        "picard_iter_sum_warm": int(it_hist_warm[it_hist_warm > 0].sum()),
        "picard_iter_mean_base": float(np.mean(it_hist_base[it_hist_base > 0])) if np.any(it_hist_base > 0) else float("nan"),
        "picard_iter_mean_warm": float(np.mean(it_hist_warm[it_hist_warm > 0])) if np.any(it_hist_warm > 0) else float("nan"),
        "picard_iter_hist_base": it_hist_base.tolist(),
        "picard_iter_hist_warm": it_hist_warm.tolist(),
        "std_resid_hist_base": resid_hist_base.tolist(),
        "std_resid_hist_warm": resid_hist_warm.tolist(),
        "final_moments_base": {
            "n": nb.detach().cpu().double().numpy().tolist(),
            "u": ub.detach().cpu().double().numpy().tolist(),
            "T": Tb.detach().cpu().double().numpy().tolist(),
        },
        "final_moments_warm": {
            "n": nw.detach().cpu().double().numpy().tolist(),
            "u": uw.detach().cpu().double().numpy().tolist(),
            "T": Tw.detach().cpu().double().numpy().tolist(),
        },
        "debug_steps": int(debug_steps),
        "debug_log": debug_log,
    }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tau", type=float, nargs="+", required=True)
    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--T_total", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=-1)
    p.add_argument("--delta_type", type=str, default="dw", choices=["dnu", "dw"])

    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--nv", type=int, default=256)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--v_max", type=float, default=10.0)

    p.add_argument("--picard_iter", type=int, default=1000)
    p.add_argument("--picard_tol", type=float, default=1e-3)
    p.add_argument("--abs_tol", type=float, default=1e-13)

    p.add_argument("--mix_alpha", type=float, default=1.0)

    p.add_argument("--debug_steps", type=int, default=10)
    p.add_argument("--n_floor", type=float, default=1e-12)
    p.add_argument("--T_floor", type=float, default=1e-12)

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
            "debug_steps": int(args.debug_steps),
            "n_floor": float(args.n_floor),
            "T_floor": float(args.T_floor),
            "model_arch": arch,
            "target": args.delta_type,  # important metadata
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

        out = run_case_debug(
            cfg=cfg,
            model=model,
            n_steps=n_steps,
            device=device,
            mix_alpha=float(args.mix_alpha),
            debug_steps=int(args.debug_steps),
            n_floor=float(args.n_floor),
            T_floor=float(args.T_floor),
            delta_type=args.delta_type,
        )

        base_sum = out["picard_iter_sum_base"]
        warm_sum = out["picard_iter_sum_warm"]
        speed = (base_sum / max(warm_sum, 1))
        results["cases"].append({
            "tau_tilde": float(tau),
            "n_steps": int(n_steps),
            "picard_sum_base": int(base_sum),
            "picard_sum_warm": int(warm_sum),
            "speedup_picard_sum": float(speed),
            "walltime_sec_total": float(out["walltime_sec"]),
            "detail": out,
        })

        print(
            f"[tau={tau:.3e}] picard_sum base={base_sum} warm={warm_sum} (x{speed:.2f}) "
            f"debug_steps={args.debug_steps} mix_alpha={args.mix_alpha}",
            flush=True
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
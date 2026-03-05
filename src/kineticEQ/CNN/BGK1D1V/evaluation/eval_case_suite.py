from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[4]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kineticEQ import BGK1D, Config
from kineticEQ.CNN.BGK1D1V.evaluation.eval_warmstart_debug import (
    build_cfg,
    run_case_baseline_input,
    run_case_pair,
)


def _shock_basic() -> tuple[dict, ...]:
    return (
        {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.1, "u": 0.0, "T": 0.4},
    )


def _shock_strong() -> tuple[dict, ...]:
    return (
        {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 4.0, "u": 0.0, "T": 0.25},
    )


def _isobaric_contact() -> tuple[dict, ...]:
    p0 = 1.0
    TL, TR = 1.0, 0.25
    nL, nR = p0 / TL, p0 / TR
    return (
        {"x_range": (0.0, 0.5), "n": float(nL), "u": 0.0, "T": float(TL)},
        {"x_range": (0.5, 1.0), "n": float(nR), "u": 0.0, "T": float(TR)},
    )


def _double_hotspot_isobaric() -> tuple[dict, ...]:
    p0 = 0.2
    Th, Tc = 1.0, 0.2
    return (
        {"x_range": (0.0, 0.15), "n": p0 / Tc, "u": 0.0, "T": Tc},
        {"x_range": (0.15, 0.25), "n": p0 / Th, "u": 0.0, "T": Th},
        {"x_range": (0.25, 0.75), "n": p0 / Tc, "u": 0.0, "T": Tc},
        {"x_range": (0.75, 0.85), "n": p0 / Th, "u": 0.0, "T": Th},
        {"x_range": (0.85, 1.0), "n": p0 / Tc, "u": 0.0, "T": Tc},
    )


def _velocity_ramp() -> tuple[dict, ...]:
    regions: list[dict] = []
    U = 0.5
    N = 16
    for k in range(N):
        x0 = k / N
        x1 = (k + 1) / N
        xc = 0.5 * (x0 + x1)
        u = -U + 2.0 * U * xc
        regions.append({"x_range": (x0, x1), "n": 1.0, "u": float(u), "T": 1.0})
    return tuple(regions)


def _thermal_velocity_mixed() -> tuple[dict, ...]:
    U = 0.5
    TL, TR = 1.0, 0.2
    p0 = 0.6
    TM = 0.5 * (TL + TR)
    return (
        {"x_range": (0.0, 0.02), "n": p0 / TL, "u": -U, "T": TL},
        {"x_range": (0.02, 0.98), "n": p0 / TM, "u": 0.0, "T": TM},
        {"x_range": (0.98, 1.0), "n": p0 / TR, "u": +U, "T": TR},
    )


CASE_BUILDERS = {
    "shock_basic": _shock_basic,
    "shock_strong": _shock_strong,
    "isobaric_contact": _isobaric_contact,
    "double_hotspot_isobaric": _double_hotspot_isobaric,
    "velocity_ramp": _velocity_ramp,
    "thermal_velocity_mixed": _thermal_velocity_mixed,
}


def _parse_warm_enable(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v == "auto":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"invalid warm_enable={value!r}")


def _apply_initial(cfg: Config, initial_regions: tuple[dict, ...]) -> Config:
    new_initial = BGK1D.InitialCondition1D(initial_regions=initial_regions)
    new_model_cfg = replace(cfg.model_cfg, initial=new_initial)
    return replace(cfg, model_cfg=new_model_cfg)


def _calc_final_drift(detail: dict) -> dict[str, float]:
    nb = np.asarray(detail["final_moments_base"]["n"], dtype=np.float64)
    ub = np.asarray(detail["final_moments_base"]["u"], dtype=np.float64)
    Tb = np.asarray(detail["final_moments_base"]["T"], dtype=np.float64)

    nw = np.asarray(detail["final_moments_warm"]["n"], dtype=np.float64)
    uw = np.asarray(detail["final_moments_warm"]["u"], dtype=np.float64)
    Tw = np.asarray(detail["final_moments_warm"]["T"], dtype=np.float64)

    def _mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    def _linf(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    return {
        "n_mae": _mae(nb, nw),
        "u_mae": _mae(ub, uw),
        "T_mae": _mae(Tb, Tw),
        "n_linf": _linf(nb, nw),
        "u_linf": _linf(ub, uw),
        "T_linf": _linf(Tb, Tw),
    }


def _summarize(results: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for rec in results:
        cname = str(rec["case_name"])
        out.setdefault(cname, {"speedup": [], "drift_n": [], "drift_u": [], "drift_T": []})
        out[cname]["speedup"].append(float(rec["speedup_picard_sum"]))
        out[cname]["drift_n"].append(float(rec["final_drift"]["n_mae"]))
        out[cname]["drift_u"].append(float(rec["final_drift"]["u_mae"]))
        out[cname]["drift_T"].append(float(rec["final_drift"]["T_mae"]))

    summary: dict[str, dict] = {}
    for cname, vals in out.items():
        summary[cname] = {
            "speedup_picard_sum_mean": float(np.mean(vals["speedup"])),
            "speedup_picard_sum_min": float(np.min(vals["speedup"])),
            "speedup_picard_sum_max": float(np.max(vals["speedup"])),
            "final_drift_n_mae_mean": float(np.mean(vals["drift_n"])),
            "final_drift_u_mae_mean": float(np.mean(vals["drift_u"])),
            "final_drift_T_mae_mean": float(np.mean(vals["drift_T"])),
            "num_runs": int(len(vals["speedup"])),
        }

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--eval_type", type=str, choices=["rollout", "baseline_input"], default="rollout")
    p.add_argument("--cases", type=str, default="all", help="comma-separated case names or 'all'")

    p.add_argument("--tau", type=float, nargs="+", required=True)
    p.add_argument("--dt", type=float, default=5e-4)
    p.add_argument("--T_total", type=float, default=0.05)
    p.add_argument("--n_steps", type=int, default=-1)

    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--nv", type=int, default=256)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--v_max", type=float, default=10.0)

    p.add_argument("--picard_iter", type=int, default=100000)
    p.add_argument("--picard_tol", type=float, default=1e-6)
    p.add_argument("--abs_tol", type=float, default=1e-8)
    p.add_argument("--conv_type", type=str, choices=["f", "w"], default="w")

    p.add_argument("--aa_enable", action="store_true")
    p.add_argument("--aa_m", type=int, default=6)
    p.add_argument("--aa_beta", type=float, default=1.0)
    p.add_argument("--aa_stride", type=int, default=1)
    p.add_argument("--aa_start_iter", type=int, default=2)
    p.add_argument("--aa_reg", type=float, default=1e-10)
    p.add_argument("--aa_alpha_max", type=float, default=50.0)

    p.add_argument("--warm_enable", type=str, choices=["auto", "true", "false"], default="auto")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warm_enable_cfg = _parse_warm_enable(args.warm_enable)
    device = torch.device(args.device)

    if args.cases == "all":
        case_names = list(CASE_BUILDERS.keys())
    else:
        case_names = [s.strip() for s in str(args.cases).split(",") if s.strip()]
        unknown = [c for c in case_names if c not in CASE_BUILDERS]
        if unknown:
            raise ValueError(f"unknown cases: {unknown}")

    results: list[dict] = []

    aa_kw = dict(
        aa_enable=bool(args.aa_enable),
        aa_m=int(args.aa_m),
        aa_beta=float(args.aa_beta),
        aa_stride=int(args.aa_stride),
        aa_start_iter=int(args.aa_start_iter),
        aa_reg=float(args.aa_reg),
        aa_alpha_max=float(args.aa_alpha_max),
    )

    for tau in args.tau:
        for case_name in case_names:
            regions = CASE_BUILDERS[case_name]()

            cfg_base = build_cfg(
                tau=float(tau),
                dt=float(args.dt),
                T_total=float(args.T_total),
                nx=int(args.nx),
                nv=int(args.nv),
                Lx=float(args.Lx),
                v_max=float(args.v_max),
                picard_iter=int(args.picard_iter),
                picard_tol=float(args.picard_tol),
                abs_tol=float(args.abs_tol),
                conv_type=str(args.conv_type),
                warm_enable=None,
                moments_cnn_modelpath=None,
                **aa_kw,
            )
            cfg_base = _apply_initial(cfg_base, regions)

            if args.n_steps > 0:
                n_steps = int(args.n_steps)
            else:
                n_steps = int(cfg_base.model_cfg.time.n_steps)

            if args.eval_type == "rollout":
                cfg_warm = build_cfg(
                    tau=float(tau),
                    dt=float(args.dt),
                    T_total=float(args.T_total),
                    nx=int(args.nx),
                    nv=int(args.nv),
                    Lx=float(args.Lx),
                    v_max=float(args.v_max),
                    picard_iter=int(args.picard_iter),
                    picard_tol=float(args.picard_tol),
                    abs_tol=float(args.abs_tol),
                    conv_type=str(args.conv_type),
                    warm_enable=warm_enable_cfg,
                    moments_cnn_modelpath=str(args.ckpt),
                    **aa_kw,
                )
                cfg_warm = _apply_initial(cfg_warm, regions)
                out = run_case_pair(cfg_base=cfg_base, cfg_warm=cfg_warm, n_steps=n_steps, device=device)
            else:
                out = run_case_baseline_input(
                    cfg_base=cfg_base,
                    ckpt_path=str(args.ckpt),
                    n_steps=n_steps,
                    device=device,
                )

            pic = out["picard"]
            base_sum = int(pic.get("picard_iter_sum_base", 0))
            warm_sum = int(pic.get("picard_iter_sum_warm", 0))
            speed = float(base_sum) / float(max(warm_sum, 1))
            drift = _calc_final_drift(out)

            rec = {
                "tau_tilde": float(tau),
                "case_name": str(case_name),
                "initial_regions": [dict(r) for r in regions],
                "n_steps": int(n_steps),
                "speedup_picard_sum": float(speed),
                "final_drift": drift,
                "detail": out,
            }
            results.append(rec)

            print(
                f"[case={case_name}, tau={tau:.3e}] "
                f"picard_sum base={base_sum} warm={warm_sum} (x{speed:.2f}) "
                f"drift(n/u/T)={drift['n_mae']:.2e}/{drift['u_mae']:.2e}/{drift['T_mae']:.2e}",
                flush=True,
            )

    payload = {
        "meta": {
            "ckpt": str(args.ckpt),
            "eval_type": str(args.eval_type),
            "device": str(args.device),
            "warm_enable": warm_enable_cfg,
            "dt": float(args.dt),
            "T_total": float(args.T_total),
            "nx": int(args.nx),
            "nv": int(args.nv),
            "Lx": float(args.Lx),
            "v_max": float(args.v_max),
            "picard_iter": int(args.picard_iter),
            "picard_tol": float(args.picard_tol),
            "abs_tol": float(args.abs_tol),
            "conv_type": str(args.conv_type),
            "aa": aa_kw,
            "tau_list": [float(t) for t in args.tau],
            "case_names": case_names,
        },
        "summary": _summarize(results),
        "cases": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()

# kineticEQ/CNN/BGK1D1V/eval_warmstart_debug.py

# usage:
# python -m kineticEQ.CNN.BGK1D1V.eval_warmstart_debug \
#   --ckpt cnn_models/bgk1d1v/best_speed.pt \
#   --tau 5e-7 \
#   --dt 5e-5 --T_total 0.05 \
#   --picard_iter 1000 --picard_tol 1e-3 --abs_tol 1e-13 \
#   --out eval_runs/debug_job123.json \
#   --device cuda
#
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import math
import numpy as np
import torch
from dataclasses import fields, replace

# --- make repo importable when executed as a script ---
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_momentCNN_util import (
    load_moments_cnn_model, predict_next_moments_delta
)
from kineticEQ.plotting.bgk1d.plot_state import plot_state


def build_cfg(
    *,
    tau: float,
    dt: float,
    T_total: float,
    nx: int = 512,
    nv: int = 256,
    Lx: float = 1.0,
    v_max: float = 10.0,
    picard_iter: int = 1000,
    picard_tol: float = 1e-3,
    abs_tol: float = 1e-13,
    aa_enable: bool = False,
    aa_m: int = 6,
    aa_beta: float = 1.0,
    aa_stride: int = 1,
    aa_start_iter: int = 2,
    aa_reg: float = 1e-10,
    aa_alpha_max: float = 50.0,
    moments_cnn_modelpath: str | None = None,
) -> Config:
    scheme_params = BGK1D.implicit.Params(
        picard_iter=int(picard_iter),
        picard_tol=float(picard_tol),
        abs_tol=float(abs_tol),
    )

    # frozen + optional field 対応：存在する時だけ replace で埋める
    fnames = {f.name for f in fields(scheme_params)}
    if "aa_enable" in fnames:
        scheme_params = replace(scheme_params, aa_enable=bool(aa_enable))
    if "aa_m" in fnames:
        scheme_params = replace(scheme_params, aa_m=int(aa_m))
    if "aa_beta" in fnames:
        scheme_params = replace(scheme_params, aa_beta=float(aa_beta))
    if "aa_stride" in fnames:
        scheme_params = replace(scheme_params, aa_stride=int(aa_stride))
    if "aa_start_iter" in fnames:
        scheme_params = replace(scheme_params, aa_start_iter=int(aa_start_iter))
    if "aa_reg" in fnames:
        scheme_params = replace(scheme_params, aa_reg=float(aa_reg))
    if "aa_alpha_max" in fnames:
        scheme_params = replace(scheme_params, aa_alpha_max=float(aa_alpha_max))

    if moments_cnn_modelpath is not None and str(moments_cnn_modelpath) != "":
        if "moments_cnn_modelpath" in fnames:
            scheme_params = replace(scheme_params, moments_cnn_modelpath=str(moments_cnn_modelpath))
        else:
            raise AttributeError("BGK1D.implicit.Params has no field 'moments_cnn_modelpath'")

    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=int(nx), nv=int(nv), Lx=float(Lx), v_max=float(v_max)),
        time=BGK1D.TimeConfig(dt=float(dt), T_total=float(T_total)),
        params=BGK1D.BGK1D1VParams(tau_tilde=float(tau)),
        scheme_params=scheme_params,
        initial=BGK1D.InitialCondition1D(initial_regions=(
            {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
            {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
        )),
    )

    return Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cuda_kernel",
        device="cuda",
        model_cfg=model_cfg,
        log_level="err",
        use_tqdm=False,
    )


def _now_sync(device: torch.device) -> float:
    # CUDAカーネルの非同期を潰して walltime をまともに測る
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def run_case_pair(
    *,
    cfg_base: Config,
    cfg_warm: Config,
    n_steps: int,
    device: torch.device,
) -> dict:
    print(f"[log] creating eng_base (JIT compile may take minutes)...", flush=True)
    eng_base = Engine(cfg_base)
    print(f"[log] eng_base created", flush=True)
    eng_warm = Engine(cfg_warm)
    print(f"[log] eng_warm created", flush=True)

    # stepper が benchlog を持つ前提（無い場合も壊れないようにする）
    # warmstart が実際に有効かの sanity:
    # - eng_warm.stepper 内部でモデルがロードされるはず
    # - ただし実装次第なので、ここでは明示チェックしない

    it_base = np.empty((n_steps,), dtype=np.int32)
    it_warm = np.empty((n_steps,), dtype=np.int32)

    resid_base = np.empty((n_steps,), dtype=np.float32)
    resid_warm = np.empty((n_steps,), dtype=np.float32)

    tstep_base = np.empty((n_steps,), dtype=np.float64)
    tstep_warm = np.empty((n_steps,), dtype=np.float64)

    # 推論時間推定用（stepごと推定と、全体推定の両方を保存）
    t_inf_est_step = np.empty((n_steps,), dtype=np.float64)

    # --- main loop ---
    t0_all = _now_sync(device)

    for s in range(n_steps):
        if s % 100 == 0:
            print(f"[log] rollout step {s}/{n_steps}", flush=True)
        # baseline step timing
        t0 = _now_sync(device)
        eng_base.stepper(s)
        t1 = _now_sync(device)
        tstep_base[s] = float(t1 - t0)

        bench_b = getattr(eng_base.stepper, "benchlog", None) or {}
        it_base[s] = int(bench_b.get("picard_iter", -1))
        resid_base[s] = float(bench_b.get("std_picard_residual", np.nan))

        # warm step timing (includes inference + picard)
        t0 = _now_sync(device)
        eng_warm.stepper(s)
        t1 = _now_sync(device)
        tstep_warm[s] = float(t1 - t0)

        bench_w = getattr(eng_warm.stepper, "benchlog", None) or {}
        it_warm[s] = int(bench_w.get("picard_iter", -1))
        resid_warm[s] = float(bench_w.get("std_picard_residual", np.nan))

        # inference time estimate (per-step):
        # baseline 1-iter time ~ tstep_base / it_base  (it_base<=0 のときはfallback)
        ib = int(it_base[s])
        iw = int(it_warm[s])
        if ib > 0 and iw >= 0:
            t_iter = float(tstep_base[s]) / float(ib)
            t_inf = float(tstep_warm[s]) - float(iw) * t_iter
            t_inf_est_step[s] = float(max(t_inf, 0.0))
        else:
            t_inf_est_step[s] = float("nan")

    t1_all = _now_sync(device)
    wall_all = float(t1_all - t0_all)

    # final moments
    nb, ub, Tb = calculate_moments(eng_base.state, eng_base.state.f)
    nw, uw, Tw = calculate_moments(eng_warm.state, eng_warm.state.f)

    # global inference estimate (more stable than per-step):
    # baseline per-iter time ~ sum(t_base)/sum(it_base)
    itb_pos = it_base[it_base > 0]
    iw_pos = it_warm[it_warm > 0]

    sum_t_base = float(np.sum(tstep_base))
    sum_t_warm = float(np.sum(tstep_warm))
    sum_it_base = int(np.sum(itb_pos)) if itb_pos.size else 0
    sum_it_warm = int(np.sum(iw_pos)) if iw_pos.size else 0

    if sum_it_base > 0:
        t_iter_global = sum_t_base / float(sum_it_base)
        t_inf_global = sum_t_warm - float(sum_it_warm) * t_iter_global
        t_inf_global = float(max(t_inf_global, 0.0))
    else:
        t_iter_global = float("nan")
        t_inf_global = float("nan")

    out = {
        "walltime_sec_total": wall_all,
        "n_steps": int(n_steps),
        "timing": {
            "step_walltime_base_sec": tstep_base.tolist(),
            "step_walltime_warm_sec": tstep_warm.tolist(),
            "infer_time_est_step_sec": t_inf_est_step.tolist(),
            "infer_time_est_global_sec": t_inf_global,
            "baseline_time_per_iter_global_sec": t_iter_global,
            "sum_walltime_base_sec": sum_t_base,
            "sum_walltime_warm_sec": sum_t_warm,
        },
        "picard": {
            "picard_iter_hist_base": it_base.tolist(),
            "picard_iter_hist_warm": it_warm.tolist(),
            "std_resid_hist_base": resid_base.tolist(),
            "std_resid_hist_warm": resid_warm.tolist(),
            "picard_iter_sum_base": int(np.sum(itb_pos)) if itb_pos.size else 0,
            "picard_iter_sum_warm": int(np.sum(iw_pos)) if iw_pos.size else 0,
            "picard_iter_mean_base": float(np.mean(itb_pos)) if itb_pos.size else float("nan"),
            "picard_iter_mean_warm": float(np.mean(iw_pos)) if iw_pos.size else float("nan"),
        },
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
    }

    return out


@torch.no_grad()
def run_case_baseline_input(
    *,
    cfg_base: Config,
    ckpt_path: str,
    n_steps: int,
    device: torch.device,
) -> dict:
    """
    Teacher-forcing evaluation (true 1-step TF via state sync):
      - eng_base: baseline (no CNN), provides TRUE state f_t
      - eng_tf:   baseline config (no CNN in stepper), but each step:
          (1) sync eng_tf.state to eng_base.state (so both start from same f_t)
          (2) build W=(n,nu,T) from CNN prediction using eng_base's TRUE moments
          (3) inject W and run one step
    This isolates the CNN's single-step quality from rollout accumulation.

    Output schema is identical to run_case_pair (warm = tf).
    """
    print(f"[log] creating eng_base (JIT compile may take minutes)...", flush=True)
    eng_base = Engine(cfg_base)
    print(f"[log] eng_base created", flush=True)
    eng_tf   = Engine(cfg_base)   # CNN disabled inside stepper
    print(f"[log] eng_tf created", flush=True)

    # load model on eval side only
    model_tf, model_meta_tf = load_moments_cnn_model(ckpt_path, device)
    dtp = model_meta_tf.get("delta_type", "dnu")
    print(f"[log] model loaded, delta_type={dtp}", flush=True)

    dt_val  = float(cfg_base.model_cfg.time.dt)
    tau_val = float(cfg_base.model_cfg.params.tau_tilde)

    it_base  = np.empty((n_steps,), dtype=np.int32)
    it_warm  = np.empty((n_steps,), dtype=np.int32)

    resid_base = np.empty((n_steps,), dtype=np.float32)
    resid_warm = np.empty((n_steps,), dtype=np.float32)

    tstep_base = np.empty((n_steps,), dtype=np.float64)
    tstep_warm = np.empty((n_steps,), dtype=np.float64)

    t_inf_est_step = np.empty((n_steps,), dtype=np.float64)

    t0_all = _now_sync(device)

    for s in range(n_steps):

        # 10% interval logging
        if s % (n_steps // 10) == 0:
            print(f"[log] tf step {s}/{n_steps}", flush=True)
        # ============================================================
        # (A) true moments from BASE at time t=s  (BEFORE stepping base)
        # ============================================================
        n0, u0, T0 = calculate_moments(eng_base.state, eng_base.state.f)

        # ============================================================
        # (B) sync TF engine state to BASE state so TF starts from same f_t
        # ============================================================
        eng_tf.state.f.copy_(eng_base.state.f)
        # safest: also sync f_tmp if exists (implementation-dependent)
        if hasattr(eng_tf.state, "f_tmp") and hasattr(eng_base.state, "f_tmp"):
            eng_tf.state.f_tmp.copy_(eng_base.state.f_tmp)

        # ============================================================
        # (C) CNN prediction -> build W=(n,nu,T)
        # ============================================================
        n1p = n0.clone()
        u1p = u0.clone()
        T1p = T0.clone()

        n1p[1:-1], u1p[1:-1], T1p[1:-1], _, _, _ = predict_next_moments_delta(
            model_tf, n0, u0, T0,
            math.log10(dt_val),
            math.log10(tau_val),
            delta_type=dtp,
        )

        # inject W for THIS step (t=s -> s+1)
        # _init_W is consumed once inside implicit stepper
        eng_tf.stepper.ws._init_W = (n1p, n1p * u1p, T1p)

        # ============================================================
        # (D) baseline step (advance base to t=s+1)
        # ============================================================
        t0 = _now_sync(device)
        eng_base.stepper(s)
        t1 = _now_sync(device)
        tstep_base[s] = float(t1 - t0)

        bench_b = getattr(eng_base.stepper, "benchlog", None) or {}
        it_base[s]    = int(bench_b.get("picard_iter", -1))
        resid_base[s] = float(bench_b.get("std_picard_residual", np.nan))

        # ============================================================
        # (E) teacher-forcing step (solve from same f_t with CNN-init W)
        # ============================================================
        t0 = _now_sync(device)
        eng_tf.stepper(s)
        t1 = _now_sync(device)
        tstep_warm[s] = float(t1 - t0)

        bench_w = getattr(eng_tf.stepper, "benchlog", None) or {}
        it_warm[s]    = int(bench_w.get("picard_iter", -1))
        resid_warm[s] = float(bench_w.get("std_picard_residual", np.nan))

        # ============================================================
        # (F) inference time estimate
        # ============================================================
        ib = int(it_base[s])
        iw = int(it_warm[s])
        if ib > 0 and iw >= 0:
            t_iter = float(tstep_base[s]) / float(ib)
            t_inf  = float(tstep_warm[s]) - float(iw) * t_iter
            t_inf_est_step[s] = float(max(t_inf, 0.0))
        else:
            t_inf_est_step[s] = float("nan")

    t1_all = _now_sync(device)
    wall_all = float(t1_all - t0_all)

    # final moments (NOTE: eng_tf is NOT a physical rollout; only last-step state is meaningful)
    nb, ub, Tb = calculate_moments(eng_base.state, eng_base.state.f)
    nw, uw, Tw = calculate_moments(eng_tf.state, eng_tf.state.f)

    itb_pos = it_base[it_base > 0]
    iw_pos  = it_warm[it_warm > 0]

    sum_t_base = float(np.sum(tstep_base))
    sum_t_warm = float(np.sum(tstep_warm))
    sum_it_base = int(np.sum(itb_pos)) if itb_pos.size else 0
    sum_it_warm = int(np.sum(iw_pos))  if iw_pos.size  else 0

    if sum_it_base > 0:
        t_iter_global = sum_t_base / float(sum_it_base)
        t_inf_global  = sum_t_warm - float(sum_it_warm) * t_iter_global
        t_inf_global  = float(max(t_inf_global, 0.0))
    else:
        t_iter_global = float("nan")
        t_inf_global  = float("nan")

    out = {
        "walltime_sec_total": wall_all,
        "n_steps": int(n_steps),
        "timing": {
            "step_walltime_base_sec": tstep_base.tolist(),
            "step_walltime_warm_sec": tstep_warm.tolist(),
            "infer_time_est_step_sec": t_inf_est_step.tolist(),
            "infer_time_est_global_sec": t_inf_global,
            "baseline_time_per_iter_global_sec": t_iter_global,
            "sum_walltime_base_sec": sum_t_base,
            "sum_walltime_warm_sec": sum_t_warm,
        },
        "picard": {
            "picard_iter_hist_base": it_base.tolist(),
            "picard_iter_hist_warm": it_warm.tolist(),
            "std_resid_hist_base": resid_base.tolist(),
            "std_resid_hist_warm": resid_warm.tolist(),
            "picard_iter_sum_base": int(np.sum(itb_pos)) if itb_pos.size else 0,
            "picard_iter_sum_warm": int(np.sum(iw_pos))  if iw_pos.size  else 0,
            "picard_iter_mean_base": float(np.mean(itb_pos)) if itb_pos.size else float("nan"),
            "picard_iter_mean_warm": float(np.mean(iw_pos))  if iw_pos.size  else float("nan"),
        },
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
    }

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="moments_cnn_modelpath for warmstart (baseline is None)")
    p.add_argument("--eval_type", type=str, choices=["baseline_input", "rollout"], default="rollout", help="eval type")
    p.add_argument("--tau", type=float, nargs="+", required=True)

    p.add_argument("--dt", type=float, default=5e-5)
    p.add_argument("--T_total", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=-1)

    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--nv", type=int, default=256)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--v_max", type=float, default=10.0)

    p.add_argument("--picard_iter", type=int, default=10_000)
    p.add_argument("--picard_tol", type=float, default=1e-3)
    p.add_argument("--abs_tol", type=float, default=1e-13)

    p.add_argument("--aa_enable", action="store_true", help="enable Anderson Acceleration in implicit Picard")
    p.add_argument("--aa_m", type=int, default=6)
    p.add_argument("--aa_beta", type=float, default=1.0)
    p.add_argument("--aa_stride", type=int, default=1, help="apply AA every k Picard iterations")
    p.add_argument("--aa_start_iter", type=int, default=2, help="first Picard iteration index to allow AA")
    p.add_argument("--aa_reg", type=float, default=1e-10)
    p.add_argument("--aa_alpha_max", type=float, default=50.0)

    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"args parsed: eval_type={args.eval_type}", flush=True)
    device = torch.device(args.device)
    print(f"device={device}", flush=True)

    aa_enable_effective = bool(args.aa_enable)

    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(device.index or 0)
        except Exception:
            gpu_name = torch.cuda.get_device_name(0)

    results = {
        "meta": {
            "warm_ckpt": str(args.ckpt),
            "dt": float(args.dt),
            "T_total": float(args.T_total),
            "nx": int(args.nx),
            "nv": int(args.nv),
            "Lx": float(args.Lx),
            "v_max": float(args.v_max),
            "picard_iter": int(args.picard_iter),
            "picard_tol": float(args.picard_tol),
            "abs_tol": float(args.abs_tol),
            "aa_enable": bool(args.aa_enable),
            "aa_m": int(args.aa_m),
            "aa_beta": float(args.aa_beta),
            "aa_stride": int(args.aa_stride),
            "aa_start_iter": int(args.aa_start_iter),
            "aa_reg": float(args.aa_reg),
            "aa_alpha_max": float(args.aa_alpha_max),
            "device": str(args.device),
            "gpu_name": gpu_name,
            "eval_type": str(args.eval_type),
        },
        "cases": [],
    }

    eval_type = str(args.eval_type)

    for tau in args.tau:
        print(f"[log] tau={tau:.3e}, eval_type={eval_type}, building cfg...", flush=True)
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
            aa_enable=bool(aa_enable_effective),
            aa_m=int(args.aa_m),
            aa_beta=float(args.aa_beta),
            aa_stride=int(args.aa_stride),
            aa_start_iter=int(args.aa_start_iter),
            aa_reg=float(args.aa_reg),
            aa_alpha_max=float(args.aa_alpha_max),
            moments_cnn_modelpath=None,
        )

        if args.n_steps > 0:
            n_steps = int(args.n_steps)
        else:
            try:
                n_steps = int(cfg_base.model_cfg.time.n_steps)
            except Exception:
                n_steps = int(round(float(cfg_base.model_cfg.time.T_total) / float(cfg_base.model_cfg.time.dt)))

        if eval_type == "rollout":
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
                aa_enable=bool(aa_enable_effective),
                aa_m=int(args.aa_m),
                aa_beta=float(args.aa_beta),
                aa_stride=int(args.aa_stride),
                aa_start_iter=int(args.aa_start_iter),
                aa_reg=float(args.aa_reg),
                aa_alpha_max=float(args.aa_alpha_max),
                moments_cnn_modelpath=str(args.ckpt),
            )
            out = run_case_pair(
                cfg_base=cfg_base,
                cfg_warm=cfg_warm,
                n_steps=n_steps,
                device=device,
            )
        elif eval_type == "baseline_input":
            out = run_case_baseline_input(
                cfg_base=cfg_base,
                ckpt_path=str(args.ckpt),
                n_steps=n_steps,
                device=device,
            )
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")

        pic = out["picard"]
        base_sum = int(pic.get("picard_iter_sum_base", 0))
        warm_sum = int(pic.get("picard_iter_sum_warm", 0))
        speed = float(base_sum) / float(max(warm_sum, 1))

        results["cases"].append({
            "tau_tilde": float(tau),
            "n_steps": int(n_steps),
            "speedup_picard_sum": float(speed),
            "detail": out,
        })

        print(
            f"[tau={tau:.3e}] "
            f"picard_sum base={base_sum} warm={warm_sum} (x{speed:.2f}) "
            f"t_base_sum={out['timing']['sum_walltime_base_sec']:.3f}s "
            f"t_warm_sum={out['timing']['sum_walltime_warm_sec']:.3f}s "
            f"t_inf_est_global={out['timing']['infer_time_est_global_sec']:.3f}s",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
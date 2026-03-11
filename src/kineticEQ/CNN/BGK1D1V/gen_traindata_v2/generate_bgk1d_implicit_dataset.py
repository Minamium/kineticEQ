# kineticEQ/CNN/BGK1D1V/gen_traindata_v2/generate_bgk1d_implicit_dataset.py

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from kineticEQ import BGK1D, Config, Engine
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments


LOG_LEVEL_PRIORITY = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}


def normalize_log_level(log_level: str) -> str:
    s = str(log_level).strip().lower()
    aliases = {
        "warn": "warning",
        "err": "error",
    }
    s = aliases.get(s, s)
    if s not in LOG_LEVEL_PRIORITY:
        raise ValueError(f"unsupported log_level: {log_level}")
    return s


def should_log(current_level: str, message_level: str) -> bool:
    return LOG_LEVEL_PRIORITY[current_level] <= LOG_LEVEL_PRIORITY[message_level]


def emit_log(log_level: str, message_level: str, message: str, rank: int) -> None:
    if should_log(log_level, message_level):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} [{message_level.upper()}] rank={rank} {message}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="dataset_pt_v2")
    ap.add_argument("--cases", type=int, default=1000)
    ap.add_argument("--cases_per_shard", type=int, default=100)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--nv", type=int, default=256)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--v_max", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--T_total", type=float, default=5e-2)
    ap.add_argument("--picard_iter", type=int, default=100_000)
    ap.add_argument("--picard_max_iter", type=int, default=None)
    ap.add_argument("--picard_tol", type=float, default=1e-8)
    ap.add_argument("--abs_tol", type=float, default=1e-10)
    ap.add_argument("--conv_type", type=str, default="w", choices=["w", "f"])
    ap.add_argument("--n_floor", type=float, default=1e-3)
    ap.add_argument("--w_min", type=float, default=0.10)
    ap.add_argument(
        "--tau_tilde_list",
        type=float,
        nargs="+",
        default=[1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning", "warn", "error", "err"])
    args = ap.parse_args()
    if args.picard_max_iter is not None:
        args.picard_iter = int(args.picard_max_iter)
    args.cases = int(args.cases)
    args.cases_per_shard = int(args.cases_per_shard)
    args.picard_iter = int(args.picard_iter)
    args.nx = int(args.nx)
    args.nv = int(args.nv)
    args.log_level = normalize_log_level(args.log_level)
    if args.cases < 1:
        raise ValueError("--cases must be >= 1")
    if args.cases_per_shard < 1:
        raise ValueError("--cases_per_shard must be >= 1")
    if args.w_min <= 0.0 or args.w_min >= 0.25:
        raise ValueError("--w_min must satisfy 0 < w_min < 0.25")
    if len(args.tau_tilde_list) < 1:
        raise ValueError("--tau_tilde_list must be non-empty")
    return args


def setup_dist(device_arg: str) -> tuple[bool, int, int, int, str]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if str(device_arg).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda was requested but CUDA is not available")
            return False, 0, 0, 1, str(device_arg)
        return False, 0, 0, 1, "cpu"

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = str(device_arg).startswith("cuda")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed CUDA generation requested but CUDA is not available")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world, device


def balanced_local_plan(
    total_cases: int,
    tau_values: list[float],
    rank: int,
    world_size: int,
) -> list[tuple[int, float]]:
    tau_values = [float(v) for v in tau_values]
    n_tau = len(tau_values)
    block_size = world_size * n_tau
    plan: list[tuple[int, float]] = []
    for global_case_id in range(total_cases):
        slot = global_case_id % block_size
        tau_pos = slot // world_size
        slot_rank = slot % world_size
        tau_value = tau_values[(tau_pos + slot_rank) % n_tau]
        if slot_rank == rank:
            plan.append((global_case_id, float(tau_value)))
    return plan


def planned_shard_offset(
    total_cases: int,
    tau_values: list[float],
    world_size: int,
    rank: int,
    cases_per_shard: int,
) -> int:
    offset = 0
    for r in range(rank):
        local_cases = len(balanced_local_plan(total_cases, tau_values, r, world_size))
        offset += math.ceil(local_cases / cases_per_shard)
    return offset


def make_case_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def rand_uniform(g: torch.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * torch.rand((), generator=g).item())


def sample_initial_regions(g: torch.Generator, n_floor: float, w_min: float) -> list[dict[str, Any]]:
    rem = 1.0 - 4.0 * float(w_min)
    widths = torch.rand((4,), generator=g)
    widths = widths / widths.sum()
    widths = float(w_min) + rem * widths
    x0 = float(widths[0].item())
    x1 = float((widths[0] + widths[1]).item())
    x2 = float((widths[0] + widths[1] + widths[2]).item())

    n_vals = [rand_uniform(g, n_floor, 1.0) for _ in range(4)]
    T_vals = [0.5 + torch.rand((), generator=g).item() for _ in range(4)]
    u_vals = [0.0, rand_uniform(g, -0.2, 0.2), rand_uniform(g, -0.2, 0.2), 0.0]

    return [
        {"x_range": [0.0, x0], "n": n_vals[0], "u": u_vals[0], "T": T_vals[0]},
        {"x_range": [x0, x1], "n": n_vals[1], "u": u_vals[1], "T": T_vals[1]},
        {"x_range": [x1, x2], "n": n_vals[2], "u": u_vals[2], "T": T_vals[2]},
        {"x_range": [x2, 1.0], "n": n_vals[3], "u": u_vals[3], "T": T_vals[3]},
    ]


def region_stats(initial_regions: list[dict[str, Any]]) -> dict[str, float]:
    n_vals = [float(r["n"]) for r in initial_regions]
    T_vals = [float(r["T"]) for r in initial_regions]
    u_vals = [float(r["u"]) for r in initial_regions]
    n_min = min(n_vals)
    n_max = max(n_vals)
    T_min = min(T_vals)
    T_max = max(T_vals)
    return {
        "n_min": float(n_min),
        "n_max": float(n_max),
        "n_ratio": float(n_max / max(n_min, 1e-30)),
        "T_min": float(T_min),
        "T_max": float(T_max),
        "T_ratio": float(T_max / max(T_min, 1e-30)),
        "u_min": float(min(u_vals)),
        "u_max": float(max(u_vals)),
        "u_rms": float(math.sqrt(sum(v * v for v in u_vals) / max(len(u_vals), 1))),
    }


def make_model_cfg(args, tau_tilde: float, initial_regions: list[dict[str, Any]]):
    return BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=args.nx, nv=args.nv, Lx=args.Lx, v_max=args.v_max),
        time=BGK1D.TimeConfig(dt=args.dt, T_total=args.T_total),
        params=BGK1D.BGK1D1VParams(tau_tilde=float(tau_tilde)),
        scheme_params=BGK1D.implicit.Params(
            picard_iter=int(args.picard_iter),
            picard_tol=float(args.picard_tol),
            abs_tol=float(args.abs_tol),
            conv_type=str(args.conv_type),
        ),
        initial=BGK1D.InitialCondition1D(
            initial_regions=tuple(
                {
                    "x_range": (float(r["x_range"][0]), float(r["x_range"][1])),
                    "n": float(r["n"]),
                    "u": float(r["u"]),
                    "T": float(r["T"]),
                }
                for r in initial_regions
            )
        ),
    )


def run_case(
    case_id: int,
    tau_tilde: float,
    args,
    device: str,
    rank: int,
    local_idx: int,
    local_total: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    case_seed = int(args.seed) + int(case_id)
    g = make_case_generator(case_seed)
    initial_regions = sample_initial_regions(g, float(args.n_floor), float(args.w_min))
    stats = region_stats(initial_regions)
    model_cfg = make_model_cfg(args, tau_tilde=float(tau_tilde), initial_regions=initial_regions)
    emit_log(
        args.log_level,
        "debug",
        (
            f"case_start local={local_idx + 1}/{local_total} case_id={case_id} seed={case_seed} "
            f"tau={float(tau_tilde):.3e} dt={float(args.dt):.3e} nx={int(args.nx)} nv={int(args.nv)}"
        ),
        rank,
    )
    emit_log(
        args.log_level,
        "debug",
        "case_props "
        + json.dumps(
            {
                "case_id": int(case_id),
                "seed": int(case_seed),
                "tau_tilde": float(tau_tilde),
                "dt": float(args.dt),
                "T_total": float(args.T_total),
                "n_steps": int(model_cfg.time.n_steps),
                "initial_regions": initial_regions,
                "region_stats": stats,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        rank,
    )
    engine = Engine(
        Config(
            model="BGK1D1V",
            scheme="implicit",
            backend="cuda_kernel",
            model_cfg=model_cfg,
            device=device,
            dtype="float64",
            log_level="err",
            use_tqdm="false",
        )
    )

    n_steps = int(model_cfg.time.n_steps)
    n_frames = n_steps + 1
    nx = int(model_cfg.grid.nx)
    W = torch.empty((n_frames, 3, nx), dtype=torch.float32)
    picard_iter = torch.empty((n_frames,), dtype=torch.int32)
    std_picard_residual = torch.empty((n_frames,), dtype=torch.float32)

    t0 = time.time()
    next_progress_pct = 5
    with torch.no_grad():
        n0, u0, T0 = calculate_moments(engine.state, engine.state.f)
    W[0, 0].copy_(n0.detach().cpu().to(torch.float32))
    W[0, 1].copy_(u0.detach().cpu().to(torch.float32))
    W[0, 2].copy_(T0.detach().cpu().to(torch.float32))
    picard_iter[0] = 0
    std_picard_residual[0] = 0.0

    for step in range(n_steps):
        engine.stepper(step)
        bench = getattr(engine.stepper, "benchlog", None) or {}
        with torch.no_grad():
            n1, u1, T1 = calculate_moments(engine.state, engine.state.f)
        W[step + 1, 0].copy_(n1.detach().cpu().to(torch.float32))
        W[step + 1, 1].copy_(u1.detach().cpu().to(torch.float32))
        W[step + 1, 2].copy_(T1.detach().cpu().to(torch.float32))
        picard_iter[step + 1] = int(bench.get("picard_iter", -1))
        std_picard_residual[step + 1] = float(bench.get("std_picard_residual", float("nan")))
        progress_pct = int((100 * (step + 1)) // max(n_steps, 1))
        while next_progress_pct <= 100 and progress_pct >= next_progress_pct:
            emit_log(
                args.log_level,
                "debug",
                (
                    f"case_progress case_id={case_id} progress={next_progress_pct}% "
                    f"step={step + 1}/{n_steps} elapsed={time.time() - t0:.3f}s "
                    f"picard_iter={int(picard_iter[step + 1].item())} "
                    f"std_picard_residual={float(std_picard_residual[step + 1].item()):.3e}"
                ),
                rank,
            )
            next_progress_pct += 5

    rec = {
        "case_id": int(case_id),
        "seed": int(case_seed),
        "frame_start": -1,
        "n_frames": int(n_frames),
        "n_steps": int(n_steps),
        "nx": int(model_cfg.grid.nx),
        "nv": int(model_cfg.grid.nv),
        "Lx": float(model_cfg.grid.Lx),
        "v_max": float(model_cfg.grid.v_max),
        "dt": float(model_cfg.time.dt),
        "T_total": float(model_cfg.time.T_total),
        "tau_tilde": float(model_cfg.params.tau_tilde),
        "log10_dt": float(math.log10(float(model_cfg.time.dt))),
        "log10_tau": float(math.log10(float(model_cfg.params.tau_tilde))),
        "log10_dt_over_tau": float(math.log10(float(model_cfg.time.dt) / float(model_cfg.params.tau_tilde))),
        "picard_iter_limit": int(args.picard_iter),
        "picard_tol": float(args.picard_tol),
        "abs_tol": float(args.abs_tol),
        "conv_type": str(args.conv_type),
        "initial_regions": initial_regions,
        "elapsed_sec": float(time.time() - t0),
        "failed": False,
        "error_message": "",
        "shard_path": "",
    }
    rec.update(stats)
    payload = {
        "W": W,
        "picard_iter": picard_iter,
        "std_picard_residual": std_picard_residual,
    }
    return rec, payload


def flush_shard(
    out_dir: Path,
    shard_global_id: int,
    shard_cases: list[dict[str, Any]],
    shard_payloads: list[dict[str, torch.Tensor]],
) -> None:
    if not shard_cases:
        return
    total_frames = sum(int(rec["n_frames"]) for rec in shard_cases)
    nx = int(shard_payloads[0]["W"].shape[-1])
    W = torch.empty((total_frames, 3, nx), dtype=torch.float32)
    picard_iter = torch.empty((total_frames,), dtype=torch.int32)
    std_picard_residual = torch.empty((total_frames,), dtype=torch.float32)
    cursor = 0
    shard_rel_path = f"shards/shard_{shard_global_id:05d}.pt"
    for rec, payload in zip(shard_cases, shard_payloads):
        n_frames = int(rec["n_frames"])
        W[cursor:cursor + n_frames].copy_(payload["W"])
        picard_iter[cursor:cursor + n_frames].copy_(payload["picard_iter"])
        std_picard_residual[cursor:cursor + n_frames].copy_(payload["std_picard_residual"])
        rec["frame_start"] = int(cursor)
        rec["shard_path"] = shard_rel_path
        cursor += n_frames
    torch.save(
        {
            "format": "kineticEQ_BGK1D1V_pt_v2",
            "W": W,
            "picard_iter": picard_iter,
            "std_picard_residual": std_picard_residual,
            "case_ids": [int(rec["case_id"]) for rec in shard_cases],
        },
        out_dir / shard_rel_path,
    )


def assign_split_iid(records: list[dict[str, Any]], seed: int) -> None:
    ok = [rec for rec in records if not bool(rec.get("failed", False))]
    ok_sorted = sorted(ok, key=lambda x: int(x["case_id"]))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 2026)
    perm = torch.randperm(len(ok_sorted), generator=g).tolist()
    n = len(ok_sorted)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    train_cut = n_train
    val_cut = min(n, n_train + n_val)
    label_by_case: dict[int, str] = {}
    for pos, order_idx in enumerate(perm):
        case_id = int(ok_sorted[order_idx]["case_id"])
        if pos < train_cut:
            label_by_case[case_id] = "train"
        elif pos < val_cut:
            label_by_case[case_id] = "val"
        else:
            label_by_case[case_id] = "test"
    for rec in records:
        rec["split_iid"] = label_by_case.get(int(rec["case_id"]), "test")


def assign_split_ood_tau(records: list[dict[str, Any]]) -> None:
    ok = [rec for rec in records if not bool(rec.get("failed", False))]
    tau_values = sorted({float(rec["tau_tilde"]) for rec in ok})
    if len(tau_values) == 0:
        tau_to_label = {}
    elif len(tau_values) == 1:
        tau_to_label = {tau_values[0]: "train"}
    elif len(tau_values) == 2:
        tau_to_label = {tau_values[0]: "train", tau_values[1]: "test"}
    else:
        tau_to_label = {tau: "train" for tau in tau_values}
        tau_to_label[tau_values[0]] = "test"
        tau_to_label[tau_values[-1]] = "val"
    for rec in records:
        rec["split_ood_tau"] = tau_to_label.get(float(rec["tau_tilde"]), "train")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_generation_config(args, world_size: int) -> dict[str, Any]:
    return {
        "format": "kineticEQ_BGK1D1V_pt_v2_partial_config",
        "version": 2,
        "num_cases_requested": int(args.cases),
        "cases_per_shard": int(args.cases_per_shard),
        "seed": int(args.seed),
        "world_size": int(world_size),
        "grid": {
            "nx": int(args.nx),
            "nv": int(args.nv),
            "Lx": float(args.Lx),
            "v_max": float(args.v_max),
        },
        "time": {
            "dt": float(args.dt),
            "T_total": float(args.T_total),
        },
        "scheme": {
            "solver": "implicit",
            "backend": "cuda_kernel",
            "picard_iter": int(args.picard_iter),
            "picard_tol": float(args.picard_tol),
            "abs_tol": float(args.abs_tol),
            "conv_type": str(args.conv_type),
        },
        "sampling": {
            "n_floor": float(args.n_floor),
            "w_min": float(args.w_min),
            "T_rule": "0.5 + U(0,1)",
            "u_rule": "u0=u3=0, u1=u2=(2U-1)*0.2",
            "tau_tilde_list": [float(v) for v in args.tau_tilde_list],
        },
        "paths": {
            "case_manifest": "case_manifest.jsonl",
            "shard_dir": "shards",
            "partial_dir": "partials",
        },
    }


def write_partial_generation_config(out_dir: Path, rank: int, args, world_size: int) -> None:
    cfg = build_generation_config(args=args, world_size=world_size)
    cfg["rank"] = int(rank)
    write_json(out_dir / "partials" / f"generation_config.rank{rank:05d}.json", cfg)


def main():
    args = parse_args()
    is_dist, rank, local_rank, world_size, device = setup_dist(args.device)
    if not str(device).startswith("cuda"):
        raise ValueError("gen_traindata_v2 requires CUDA because the implicit generator uses backend='cuda_kernel'")
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "partials").mkdir(parents=True, exist_ok=True)
    write_partial_generation_config(out_dir=out_dir, rank=rank, args=args, world_size=world_size)

    local_plan = balanced_local_plan(
        total_cases=int(args.cases),
        tau_values=[float(v) for v in args.tau_tilde_list],
        rank=rank,
        world_size=world_size,
    )
    emit_log(
        args.log_level,
        "debug",
        (
            f"generator_start world_size={world_size} local_cases={len(local_plan)} "
            f"cases={int(args.cases)} cases_per_shard={int(args.cases_per_shard)} "
            f"tau_count={len(args.tau_tilde_list)} out_dir={str(out_dir)} device={device}"
        ),
        rank,
    )
    shard_id_base = planned_shard_offset(
        total_cases=int(args.cases),
        tau_values=[float(v) for v in args.tau_tilde_list],
        world_size=world_size,
        rank=rank,
        cases_per_shard=int(args.cases_per_shard),
    )

    local_records: list[dict[str, Any]] = []
    shard_cases: list[dict[str, Any]] = []
    shard_payloads: list[dict[str, torch.Tensor]] = []
    local_shard_count = 0

    for local_idx, (case_id, tau_tilde) in enumerate(local_plan):
        try:
            rec, payload = run_case(
                case_id=int(case_id),
                tau_tilde=float(tau_tilde),
                args=args,
                device=device,
                rank=rank,
                local_idx=local_idx,
                local_total=len(local_plan),
            )
            shard_cases.append(rec)
            shard_payloads.append(payload)
            local_records.append(rec)
            if len(shard_cases) >= int(args.cases_per_shard):
                flush_shard(
                    out_dir=out_dir,
                    shard_global_id=shard_id_base + local_shard_count,
                    shard_cases=shard_cases,
                    shard_payloads=shard_payloads,
                )
                local_shard_count += 1
                shard_cases = []
                shard_payloads = []
            emit_log(
                args.log_level,
                "info",
                f"local={local_idx + 1}/{len(local_plan)} case_id={case_id} tau={tau_tilde:.3e} elapsed={rec['elapsed_sec']:.3f}s",
                rank,
            )
        except Exception as e:
            local_records.append(
                {
                    "case_id": int(case_id),
                    "seed": int(args.seed) + int(case_id),
                    "frame_start": -1,
                    "n_frames": 0,
                    "n_steps": 0,
                    "nx": int(args.nx),
                    "nv": int(args.nv),
                    "Lx": float(args.Lx),
                    "v_max": float(args.v_max),
                    "dt": float(args.dt),
                    "T_total": float(args.T_total),
                    "tau_tilde": float(tau_tilde),
                    "log10_dt": float(math.log10(float(args.dt))),
                    "log10_tau": float(math.log10(float(tau_tilde))),
                    "log10_dt_over_tau": float(math.log10(float(args.dt) / float(tau_tilde))),
                    "picard_iter_limit": int(args.picard_iter),
                    "picard_tol": float(args.picard_tol),
                    "abs_tol": float(args.abs_tol),
                    "conv_type": str(args.conv_type),
                    "initial_regions": [],
                    "n_min": None,
                    "n_max": None,
                    "n_ratio": None,
                    "T_min": None,
                    "T_max": None,
                    "T_ratio": None,
                    "u_min": None,
                    "u_max": None,
                    "u_rms": None,
                    "elapsed_sec": 0.0,
                    "failed": True,
                    "error_message": str(e),
                    "shard_path": "",
                }
            )
            emit_log(
                args.log_level,
                "error",
                f"case_id={case_id} tau={tau_tilde:.3e} failed={type(e).__name__}: {e}",
                rank,
            )

    if shard_cases:
        flush_shard(
            out_dir=out_dir,
            shard_global_id=shard_id_base + local_shard_count,
            shard_cases=shard_cases,
            shard_payloads=shard_payloads,
        )

    write_jsonl(out_dir / "partials" / f"case_manifest.rank{rank:05d}.jsonl", local_records)
    emit_log(
        args.log_level,
        "info",
        (
            f"local_generation_done local_cases={len(local_records)} "
            f"partial_manifest=partials/case_manifest.rank{rank:05d}.jsonl "
            f"partial_config=partials/generation_config.rank{rank:05d}.json"
        ),
        rank,
    )


if __name__ == "__main__":
    main()

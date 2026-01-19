# kineticEQ/CNN/BGK1D1V/generate_bgk1d_implicit_dataset.py
"""
BGK1D1V implicit (cuda_kernel) dataset generator for NN warm-start.

- torchrun / NCCL を想定（ケース並列）
- Engine を生成し engine.stepper(step_idx) を叩いて time evolution
- 各 step 後に moments を計算して保存（implicit stepper は state.n/u/T を更新しないため）
- 出力：Zarr（1 case = 1 store）

例:
torchrun --standalone --nproc_per_node=8 src/CNN/GBK1D1V/generate_bgk1d_implicit_dataset.py \
  --out-dir /path/to/dataset \
  --nx 4096 --nv 256 --lx 1.0 --vmax 10.0 \
  --t-total 0.05 \
  --dt-logspace -6 -4 9 \
  --tau-logspace -6 -2 9 \
  --dtype float64 \
  --flush-interval 64
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# kineticEQ
from kineticEQ import Config, Engine, BGK1D
from kineticEQ.api.config import Scheme, Backend
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments

# CUDA extension warmup (avoid torchrun JIT races)
from kineticEQ.cuda_kernel.compile import load_implicit_fused, load_gtsv


# -------------------------
# utilities
# -------------------------
def dist_setup() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size). Initialize process group if launched by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # single process
        return 0, 0, 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def log_rank0(msg: str):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)


def safe_float_str(x: float) -> str:
    # filesystem friendly
    return f"{x:.3e}".replace("+", "").replace(".", "p")


def make_default_inits() -> list[dict[str, Any]]:
    """
    初期条件のテンプレ（必要なら増やす）
    set_initial_condition() が読む形式に合わせて initial_regions を dict list で持つ。
    """
    return [
        # init_id = 0 : Riemann (Sod-like)
        {
            "name": "riemann_sod",
            "initial_regions": [
                {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
                {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
            ],
        },
        # init_id = 1 : smooth bump
        {
            "name": "smooth_bump",
            "initial_regions": [
                {"x_range": (0.0, 1.0), "n": 1.0, "u": 0.0, "T": 1.0},
            ],
            # 追加で smooth を作りたい場合は、現状の set_initial_condition では piecewise のみなので、
            # ここは「後で state.n/u/T を上書きして maxwellian を作る」などの拡張が必要。
        },
    ]


def build_case_grid(dt_values: list[float], tau_values: list[float], init_ids: list[int]) -> list[tuple[int, float, float, int]]:
    cases = []
    case_idx = 0
    for init_id in init_ids:
        for dt in dt_values:
            for tau in tau_values:
                cases.append((case_idx, dt, tau, init_id))
                case_idx += 1
    return cases


# -------------------------
# zarr writer
# -------------------------
def require_zarr():
    try:
        import zarr  # noqa: F401
        import numcodecs  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Zarr/numcodecs が見つかりません。データ生成ノードで以下を入れてください:\n"
            "  pip install zarr numcodecs\n"
            "HPC では v2 系をまず推奨（例: zarr<3）です。"
        ) from e


def open_case_zarr(case_dir: Path, nx: int, chunk_t: int, compressor_level: int = 3):
    import zarr
    from numcodecs import Blosc

    case_dir.mkdir(parents=True, exist_ok=True)

    # 1 case = 1 store (directory)
    root = zarr.open_group(str(case_dir), mode="a")

    compressor = Blosc(cname="zstd", clevel=int(compressor_level), shuffle=Blosc.BITSHUFFLE)

    # W: (T, 3, Nx) float32
    if "W" not in root:
        root.create_dataset(
            "W",
            shape=(0, 3, nx),
            chunks=(chunk_t, 3, nx),
            dtype="f4",
            compressor=compressor,
            overwrite=False,
        )

    # iters: (T,) int16
    if "picard_iter" not in root:
        root.create_dataset(
            "picard_iter",
            shape=(0,),
            chunks=(chunk_t,),
            dtype="i2",
            compressor=compressor,
            overwrite=False,
        )

    # residuals: (T,) float32
    if "picard_residual" not in root:
        root.create_dataset(
            "picard_residual",
            shape=(0,),
            chunks=(chunk_t,),
            dtype="f4",
            compressor=compressor,
            overwrite=False,
        )

    return root


def zarr_append(root, W_block: torch.Tensor, iters_block: torch.Tensor, resid_block: torch.Tensor):
    """
    W_block: (B, 3, Nx) float32 on CPU
    iters_block: (B,) int
    resid_block: (B,) float32
    """
    import numpy as np

    W = root["W"]
    iters = root["picard_iter"]
    resid = root["picard_residual"]

    bsz = W_block.shape[0]
    t0 = W.shape[0]
    t1 = t0 + bsz

    W.resize(t1, axis=0)
    iters.resize(t1, axis=0)
    resid.resize(t1, axis=0)

    W[t0:t1, :, :] = W_block.numpy(force=True) if hasattr(W_block, "numpy") else np.asarray(W_block)
    iters[t0:t1] = iters_block.cpu().numpy()
    resid[t0:t1] = resid_block.cpu().numpy()


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--nv", type=int, default=128)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument("--t-total", type=float, default=0.05)

    # dt / tau sampling
    parser.add_argument("--dt-logspace", type=float, nargs=3, metavar=("LOG10_MIN", "LOG10_MAX", "N"),
                        default=[-6, -4, 9],
                        help="dt を 10^a..10^b の logspace で N 点")
    parser.add_argument("--tau-logspace", type=float, nargs=3, metavar=("LOG10_MIN", "LOG10_MAX", "N"),
                        default=[-6, -2, 9],
                        help="tau_tilde を 10^a..10^b の logspace で N 点")

    parser.add_argument("--init-ids", type=int, nargs="*", default=[0], help="使う初期条件 ID")
    parser.add_argument("--init-json", type=str, default="", help="初期条件定義を JSON で与える（任意）")

    # compute / logging
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    parser.add_argument("--flush-interval", type=int, default=64, help="RAM バッファを何 step で flush するか")
    parser.add_argument("--chunk-t", type=int, default=64, help="Zarr chunk の time 次元")
    parser.add_argument("--compressor-level", type=int, default=3)

    # safety
    parser.add_argument("--max-cases", type=int, default=-1, help="デバッグ用：ケース数上限")
    args = parser.parse_args()

    rank, local_rank, world_size = dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Zarr dependency check
    require_zarr()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # initial condition definitions
    init_defs = make_default_inits()
    if args.init_json:
        p = Path(args.init_json)
        init_defs = json.loads(p.read_text())

    # dt/tau grids
    dt_log_min, dt_log_max, dt_n = args.dt_logspace
    tau_log_min, tau_log_max, tau_n = args.tau_logspace
    dt_values = torch.logspace(dt_log_min, dt_log_max, int(dt_n), base=10.0).tolist()
    tau_values = torch.logspace(tau_log_min, tau_log_max, int(tau_n), base=10.0).tolist()

    cases = build_case_grid(dt_values, tau_values, args.init_ids)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    # Warmup compile to avoid torchrun JIT races
    if world_size > 1:
        if rank == 0:
            log_rank0("[rank0] warmup JIT compile: implicit_fused, gtsv_batch ...")
            _ = load_implicit_fused()
            _ = load_gtsv()
            log_rank0("[rank0] warmup done.")
        barrier()

    log_rank0(f"Total cases: {len(cases)} | world_size={world_size}")

    # Main loop: rank handles its subset
    for (case_idx, dt, tau, init_id) in cases:
        if (case_idx % world_size) != rank:
            continue

        init_def = init_defs[init_id]
        init_name = init_def.get("name", f"init{init_id}")
        initial_regions = init_def["initial_regions"]

        # Build model cfg
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=args.nx, nv=args.nv, Lx=args.lx, v_max=args.vmax),
            time=BGK1D.TimeConfig(dt=float(dt), T_total=float(args.t_total)),
            params=BGK1D.BGK1D1VParams(tau_tilde=float(tau)),
            initial=BGK1D.InitialCondition1D(initial_regions=tuple(initial_regions)),
        )

        cfg = Config(
            model="BGK1D1V",
            scheme=Scheme.IMPLICIT.value,
            backend=Backend.CUDA_KERNEL.value,
            device="cuda",
            dtype=args.dtype,
            use_tqdm="false",
            log_level="error",
            model_cfg=model_cfg,
        )

        # Engine (disable logging config for speed)
        engine = Engine(cfg, apply_logging_flag=False)

        n_steps = engine.config.model_cfg.time.n_steps
        nx = engine.state.f.shape[0]

        # output path per case
        case_name = (
            f"case_{case_idx:06d}"
            f"_init-{init_name}"
            f"_dt-{safe_float_str(float(dt))}"
            f"_tau-{safe_float_str(float(tau))}"
            f"_nx-{nx}_nv-{engine.state.f.shape[1]}"
        )
        case_dir = out_dir / case_name

        root = open_case_zarr(case_dir, nx=nx, chunk_t=int(args.chunk_t), compressor_level=int(args.compressor_level))

        # store metadata as attrs
        root.attrs["case_idx"] = int(case_idx)
        root.attrs["init_id"] = int(init_id)
        root.attrs["init_name"] = init_name
        root.attrs["dt"] = float(dt)
        root.attrs["tau_tilde"] = float(tau)
        root.attrs["log10_dt"] = float(math.log10(dt))
        root.attrs["log10_tau_tilde"] = float(math.log10(tau))
        root.attrs["nx"] = int(nx)
        root.attrs["nv"] = int(engine.state.f.shape[1])
        root.attrs["lx"] = float(args.lx)
        root.attrs["vmax"] = float(args.vmax)
        root.attrs["t_total"] = float(args.t_total)
        root.attrs["scheme"] = "implicit"
        root.attrs["backend"] = "cuda_kernel"
        root.attrs["dtype_sim"] = str(args.dtype)
        root.attrs["created_unix"] = int(time.time())

        # buffers for flush
        W_buf = []
        it_buf = []
        res_buf = []

        # record t=0 (initial)
        with torch.no_grad():
            n0, u0, T0 = calculate_moments(engine.state, engine.state.f)
            nu0 = n0 * u0
            U0 = 0.5 * n0 * (u0 * u0 + T0)  # energy density proxy
            W0 = torch.stack([n0, nu0, U0], dim=0)  # (3, Nx)
            W_buf.append(W0.float().cpu())
            it_buf.append(torch.tensor(0, dtype=torch.int16))
            res_buf.append(torch.tensor(0.0, dtype=torch.float32))

        # time evolution
        for step in range(n_steps):
            engine.stepper(step)
            bench = getattr(engine.stepper, "benchlog", None) or {}

            with torch.no_grad():
                n, u, T = calculate_moments(engine.state, engine.state.f)
                nu = n * u
                U = 0.5 * n * (u * u + T)
                W = torch.stack([n, nu, U], dim=0)  # (3, Nx)

            W_buf.append(W.float().cpu())
            it_buf.append(torch.tensor(int(bench.get("picard_iter", -1)), dtype=torch.int16))
            res_buf.append(torch.tensor(float(bench.get("std_picard_residual", float("nan"))), dtype=torch.float32))

            # flush
            if len(W_buf) >= int(args.flush_interval):
                W_block = torch.stack(W_buf, dim=0)        # (B, 3, Nx)
                it_block = torch.stack(it_buf, dim=0)      # (B,)
                rs_block = torch.stack(res_buf, dim=0)     # (B,)
                zarr_append(root, W_block, it_block, rs_block)
                W_buf.clear(); it_buf.clear(); res_buf.clear()

        # final flush
        if W_buf:
            W_block = torch.stack(W_buf, dim=0)
            it_block = torch.stack(it_buf, dim=0)
            rs_block = torch.stack(res_buf, dim=0)
            zarr_append(root, W_block, it_block, rs_block)
            W_buf.clear(); it_buf.clear(); res_buf.clear()

        log_rank0(f"[rank {rank}] finished {case_name}")

    barrier()
    log_rank0("All done.")


if __name__ == "__main__":
    main()
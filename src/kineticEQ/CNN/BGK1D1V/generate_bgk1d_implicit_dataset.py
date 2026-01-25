# kineticEQ/CNN/BGK1D1V/generate_bgk1d_implicit_dataset.py

import os
import json
import torch
import torch.distributed as dist
from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
import numpy as np
import time
import argparse

def setup_dist():
    # torchrun起動の環境変数をキャッチ
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, local_rank, world, device
    else:
        # 単体実行（デバッグ用）
        return False, 0, 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="mgpu_output")
    args = ap.parse_args()

    is_dist, rank, local_rank, world_size, device = setup_dist()

    # 出力ディレクトリ作成
    out_dir = f"{args.out_dir}/shard_rank{rank:02d}"
    os.makedirs(out_dir, exist_ok=True)

    # 計算負荷の分散
    g = torch.Generator()
    g.manual_seed(0)  # 全rankで同じ
    all_cases = torch.randperm(240, generator=g).tolist()
    my_cases = all_cases[rank::world_size]

    for case_id in my_cases:
        # ケースごとのパラメータ設定
        # 乱数生成
        g_case = torch.Generator(device="cpu")
        g_case.manual_seed(1234 + int(case_id))

        # --- tau: logscale sweep, 5e-7周辺を重点的に (中心70%) ---
        # 70%: tau = 5e-7
        # 30%: tau = 5e-7 * 10^{U[-0.3, +0.3]}  ≈ 0.5x .. 2x
        if torch.rand((), generator=g_case).item() < 0.70:
            tau = 5e-7
        else:
            log10_factor = (torch.rand((), generator=g_case).item() * 0.6) - 0.3
            tau = 5e-7 * (10.0 ** log10_factor)

        # 安全柵（極端値が入らないように任意でクランプ）
        # ここは必要なら調整：中心近傍sweepなので基本不要だが、保険として入れておくのはアリ
        tau = float(max(min(tau, 2.0e-6), 5.0e-8))

        # --- dt: logscale sweep, 5e-5周辺を重点的に (中心70%) ---
        # 70%: dt = 5e-5
        # 30%: dt = 5e-5 * 10^{U[-0.2, +0.2]}  ≈ 0.63x .. 1.58x
        if torch.rand((), generator=g_case).item() < 0.70:
            dt = 5e-5
        else:
            log10_factor = (torch.rand((), generator=g_case).item() * 0.4) - 0.2
            dt = 5e-5 * (10.0 ** log10_factor)

        # 安全柵（任意）
        dt = float(max(min(dt, 2.0e-4), 1.0e-5))


        n_1 = 1.0 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        n_2 = 0.5 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        n_3 = 1.0 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        n_4 = 0.5 + (torch.rand((), generator=g_case) - 0.5) * 0.4

        u_2 = 0.0 + (torch.rand((), generator=g_case) - 0.5) * 0.04
        u_3 = 0.0 + (torch.rand((), generator=g_case) - 0.5) * 0.04

        T_1 = 1.0 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        T_2 = 0.8 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        T_3 = 1.0 + (torch.rand((), generator=g_case) - 0.5) * 0.4
        T_4 = 0.8 + (torch.rand((), generator=g_case) - 0.5) * 0.4

        # モデル設定
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=512, nv=256, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=dt, T_total=0.05),
            params=BGK1D.BGK1D1VParams(tau_tilde=tau),
            scheme_params=BGK1D.implicit.Params(picard_iter=1_000, picard_tol=1e-7, abs_tol=1e-13),
            initial=BGK1D.InitialCondition1D(initial_regions=(
                {"x_range": (0.0, 0.2), "n": float(n_1), "u": 0.0, "T": float(T_1)},
                {"x_range": (0.2, 0.4), "n": float(n_2), "u": u_2, "T": float(T_2)},
                {"x_range": (0.4, 0.7), "n": float(n_3), "u": u_3, "T": float(T_3)},
                {"x_range": (0.7, 1.0), "n": float(n_4), "u": 0.0, "T": float(T_4)},
                )
            )
        )
        maker = Engine(Config(model="BGK1D1V", 
                              scheme="implicit",
                              backend="cuda_kernel",
                              model_cfg=model_cfg,
                              device=device,
                              log_level="err",
                              use_tqdm=False))

        n_steps = model_cfg.time.n_steps
        nx = model_cfg.grid.nx

        # 事前確保： (n_steps+1, nx)
        n_hist = np.empty((n_steps + 1, nx), dtype=np.float32)
        u_hist = np.empty((n_steps + 1, nx), dtype=np.float32)
        T_hist = np.empty((n_steps + 1, nx), dtype=np.float32)

        picard_iter_hist = np.empty((n_steps + 1,), dtype=np.int16)
        std_resid_hist   = np.empty((n_steps + 1,), dtype=np.float32)

        # t=0 記録
        with torch.no_grad():
            n, u, T = calculate_moments(maker.state, maker.state.f)
        n_hist[0] = n.detach().cpu().float().numpy()
        u_hist[0] = u.detach().cpu().float().numpy()
        T_hist[0] = T.detach().cpu().float().numpy()
        picard_iter_hist[0] = 0
        std_resid_hist[0] = 0.0

        # evolve
        start_time = time.time()
        for step in range(n_steps):
            maker.stepper(step)
            bench = getattr(maker.stepper, "benchlog", None) or {}

            with torch.no_grad():
                n, u, T = calculate_moments(maker.state, maker.state.f)

            n_hist[step + 1] = n.detach().cpu().float().numpy()
            u_hist[step + 1] = u.detach().cpu().float().numpy()
            T_hist[step + 1] = T.detach().cpu().float().numpy()
            picard_iter_hist[step + 1] = int(bench.get("picard_iter", -1))
            std_resid_hist[step + 1]   = float(bench.get("std_picard_residual", np.nan))

        meta = dict(
            nx=nx, nv=model_cfg.grid.nv, Lx=model_cfg.grid.Lx, v_max=model_cfg.grid.v_max,
            dt=float(model_cfg.time.dt), T_total=float(model_cfg.time.T_total),
            tau_tilde=float(model_cfg.params.tau_tilde),
            scheme="implicit", backend="cuda_kernel",
            case_id=int(case_id), rank=int(rank),
        )

        np.savez(
            os.path.join(out_dir, f"case_{case_id:05d}.npz"),
            meta=np.bytes_(json.dumps(meta).encode("utf-8")),
            n=n_hist, u=u_hist, T=T_hist,
            picard_iter=picard_iter_hist,
            std_picard_residual=std_resid_hist,
        )

        print(f"Rank {rank}: saved case {case_id} tau={tau:.3e} elapsed={time.time() - start_time:.3f}sec", flush=True)

    # 同期
    if is_dist:
        pass
        #dist.barrier(device_ids=[local_rank])
        #dist.destroy_process_group()

if __name__ == "__main__":
    main()

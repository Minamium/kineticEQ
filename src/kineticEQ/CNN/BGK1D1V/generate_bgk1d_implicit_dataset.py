# kineticEQ/CNN/BGK1D1V/generate_bgk1d_implicit_dataset.py

import os
import torch
import torch.distributed as dist
from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.plotting.bgk1d import plot_state

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
    is_dist, rank, local_rank, world_size, device = setup_dist()

    # 出力ディレクトリ作成
    out_dir = f"mgpu_output/shard_rank{rank:02d}"
    os.makedirs(out_dir, exist_ok=True)

    # 計算負荷の分散
    g = torch.Generator()
    g.manual_seed(0)  # 全rankで同じ
    all_cases = torch.randperm(100, generator=g).tolist()
    my_cases = all_cases[rank::world_size]

    for case_id in my_cases:
        # ケースごとのパラメータ設定
        print(f"Rank {rank}: Processing case {case_id}")
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=512, nv=256, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-5, T_total=0.05),
            params=BGK1D.BGK1D1VParams(tau_tilde=(5e-8*(case_id+1)*10)),
            scheme_params=BGK1D.implicit.Params(picard_iter=1_000, picard_tol=1e-6, abs_tol=1e-13),
            initial=BGK1D.InitialCondition1D(initial_regions=(
                {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
                {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
                )
            
            )
        )
        maker = Engine(Config(model="BGK1D1V", 
                              scheme="implicit",
                              backend="cuda_kernel",
                              model_cfg=model_cfg,
                              log_level="err",
                              use_tqdm=False))
        
        # ステッパーを叩いて学習データを得る
        for steps in range(model_cfg.time.n_steps):
            maker.stepper(steps)

            # calculate moments
            moments = calculate_moments(maker.state.f)

    # 同期
    if is_dist:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

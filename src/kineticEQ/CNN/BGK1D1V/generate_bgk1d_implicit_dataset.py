# kineticEQ/CNN/BGK1D1V/generate_bgk1d_implicit_dataset.py

import os
import torch
import torch.distributed as dist
from kineticEQ import Engine, Config, BGK1D
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

    # 例：処理したいケース一覧（dt,tau,init など）
    all_cases = list(range(8))

    # rank で仕事分割（ストライド）
    my_cases = all_cases[rank::world_size]

    # 出力は rank ごとに分ける（衝突防止）
    out_dir = f"mgpu_output/shard_rank{rank:03d}"
    os.makedirs(out_dir, exist_ok=True)

    # 乱数 seed（rank ごとにずらす：重複回避）
    base_seed = 1234
    torch.manual_seed(base_seed + rank)
    torch.cuda.manual_seed_all(base_seed + rank)

    for case_id in my_cases:
        # --- ここに kineticEQ の 1ケース生成を入れる ---
        # engine = Engine(cfg_for(case_id), ...)
        # for step in range(n_steps): engine.stepper(step)
        # W = compute_moments(...) など
        # save(out_dir, case_id, data)
        print(f"Rank {rank}: Processing case {case_id}")
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=512, nv=256, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-5, T_total=5e-3),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-3 * (case_id + 1)),
            initial=BGK1D.InitialCondition1D(initial_regions=(
                {"x_range": (0.0, 0.5), "n": 0.1 , "u": 0.0, "T":  0.5},
                {"x_range": (0.5, 1.0), "n": 0.01, "u": 0.0, "T":  0.1},
                )
            )
        )
        maker = Engine(Config(model="BGK1D1V", 
                              scheme="explicit",
                              backend="cuda_kernel",
                              model_cfg=model_cfg))
        maker.run()
        plot_state(state=maker.state, output_dir=out_dir, 
                   filename=f"case_{model_cfg.params.tau_tilde:.6f}.png")

    # 必要なら同期（任意）
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

import argparse
from kineticEQ import BGK1DPlot
import pickle

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='Acc_bench', help='Output filename prefix')
parser.add_argument('--benc_type', type=str, default='spatial', help='Benchmark type')
parser.add_argument('--solver', type=str, default='implicit', help='Solver type')
parser.add_argument('--explicit_solver', type=str, default='backend', help='Explicit solver type')
parser.add_argument('--implicit_solver', type=str, default='holo', help='Implicit solver type')
args = parser.parse_args()

config = {
    "solver": args.solver,
    "explicit_solver": args.explicit_solver,
    "implicit_solver": args.implicit_solver,
    "tau_tilde": 5e-6,
    "v_max": 10.0,
    "dt": 5e-6,
    "nv": 200,
    "initial_regions": [
        {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    ],
    "n_left": 1.0,   "u_left": 0.0, "T_left": 1.0,
    "n_right": 0.125,"u_right": 0.0,"T_right": 0.8,
    "T_total": 0.01,
    "dtype": "float64",
    "use_tqdm": False,
    "device": "cuda",
}

sim = BGK1DPlot(**config)

# 1. ベンチ実行
bench_results = sim.run_benchmark(
    benc_type=args.benc_type,
    grid_list=[65, 129, 257, 513, 1025, 2049, 4129],
)

# 2. 誤差計算（run_benchmark の戻り値をそのまま渡す）
error_dict = sim.compute_error(bench_results)

# 3. pkl に保存（今の save_benchmark_results の仕様に合わせる）
sim.save_benchmark_results(
    filename=f"{args.output}.pkl",
    bench_results=bench_results,
    error_dict=error_dict,
)

# 4. 図をプロット
sim.plot_benchmark_results(
    bench_results=bench_results,
    error_dict=error_dict,
    fname_moment=f"{args.output}_moments.png",
    fname_error=f"{args.output}_error.png",
    logscale=True,
    show_plots=False,
)

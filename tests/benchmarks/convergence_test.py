import argparse
from kineticEQ import BGK1DPlot
import pickle

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='Conv_bench', help='Output filename prefix')
parser.add_argument('--dt', type=float, default=5e-4, help='Time step')
parser.add_argument('--nv', type=int, default=200, help='Number of velocity points')
parser.add_argument('--nx', type=int, default=1000, help='Number of space points')
parser.add_argument('--picard_iter', type=int, default=4096, help='Number of picard iterations')
parser.add_argument('--picard_tol', type=float, default=1e-6, help='Tolerance for picard iterations')
parser.add_argument('--ho_iter', type=int, default=4096, help='Number of ho iterations')
parser.add_argument('--lo_iter', type=int, default=4096, help='Number of lo iterations')
parser.add_argument('--ho_tol', type=float, default=1e-6, help='Tolerance for ho iterations')
parser.add_argument('--lo_tol', type=float, default=1e-6, help='Tolerance for lo iterations')
parser.add_argument('--use_tqdm', type=bool, default=True, help='Use tqdm')
args = parser.parse_args()

config = {
    "v_max": 10.0,
    "dt": args.dt,
    "nv": args.nv,
    "nx": args.nx,

    # 陰解法パラメータ
    "picard_iter": args.picard_iter,
    "picard_tol": args.picard_tol,

    # HOLOパラメータ
    "ho_iter": args.ho_iter,
    "lo_iter": args.lo_iter,
    "ho_tol": args.ho_tol,
    "lo_tol": args.lo_tol,

    "initial_regions": [
        {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    ],
    "n_left": 1.0,   "u_left": 0.0, "T_left": 1.0,
    "n_right": 0.125,"u_right": 0.0,"T_right": 0.8,
    "T_total": 0.05,
    "dtype": "float64",
    "use_tqdm": args.use_tqdm,
    "device": "cuda",
}

sim = BGK1DPlot(**config)
conv_result = sim.run_convergence_test()
sim.save_benchmark_results(
    filename=f"{args.output}.pkl",
    bench_results=conv_result,
)

sim.plot_convergence_results(conv_result, filename=f"{args.output}.png")

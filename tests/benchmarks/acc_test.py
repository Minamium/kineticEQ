import argparse
from kineticEQ import BGK1DPlot

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='Acc_bench', help='Output filename prefix')
parser.add_argument('--solver', type=str, default='implicit', help='Solver type')
parser.add_argument('--explicit_solver', type=str, default='backend', help='Explicit solver type')
parser.add_argument('--implicit_solver', type=str, default='holo', help='Implicit solver type')
args = parser.parse_args()

config = {
        # ソルバ選択
        "solver": args.solver,

        # 陽解法ソルバー
        "explicit_solver": args.explicit_solver,

        # 陰解法ソルバー
        "implicit_solver": args.implicit_solver,

        # ハイパーパラメータ
        "tau_tilde": 5e-6,

        # 数値計算パラメータ
        "v_max": 10.0,
        "dt": 5e-6,
        "nv": 200,

        "initial_regions": [
        {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},    
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8}
    ],

        # 固定モーメント境界条件
        "n_left": 1.0,
        "u_left": 0.0,
        "T_left": 1.0,
        "n_right": 0.125,
        "u_right": 0.0,
        "T_right": 0.8,

        # シミュレーション時間
        "T_total":0.01,

        # 精度
        "dtype": "float64",

        "use_tqdm" : False,

        "device" : "cuda"

    }

sim = BGK1DPlot(**config)
bench_result = sim.run_benchmark(benc_type="spatial", grid_list=[65, 129, 257, 513, 1025, 2049, 4129])
sim.save_benchmark_results(filename=f"{args.output}.pkl")
error_dict = sim.compute_error(filename=f"{args.output}.pkl", kind='nearest')
result = sim.plot_benchmark_results(filename=f"{args.output}.pkl", error_dict=error_dict, show_plots=True)
print(f"収束次数: {result['convergence_orders']}")
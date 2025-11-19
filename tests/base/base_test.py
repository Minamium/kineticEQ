import argparse
from kineticEQ import BGK1DPlot

# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='base_bench', help='Output filename prefix')
parser.add_argument('--solver', type=str, default='implicit', help='Solver type')
parser.add_argument('--implicit_solver', type=str, default='backend', help='Implicit solver type')
parser.add_argument('--picard_iter', type=int, default=64, help='Picard iteration number')
parser.add_argument('--picard_tol', type=float, default=1e-6, help='Picard tolerance')
parser.add_argument('--ho_iter', type=int, default=10, help='HOLO iteration number')
parser.add_argument('--ho_tol', type=float, default=1e-4, help='HOLO tolerance')
parser.add_argument('--lo_iter', type=int, default=10, help='LOLO iteration number')
parser.add_argument('--lo_tol', type=float, default=1e-4, help='LOLO tolerance')
parser.add_argument('--tau_tilde', type=float, default=5e-6, help='Hyperparameter')
parser.add_argument('--dt', type=float, default=5e-6, help='Time step')
parser.add_argument('--T_total', type=float, default=0.01, help='Total simulation time')
parser.add_argument('--device', type=str, default='cuda', help='Device')
parser.add_argument('--use_tqdm', type=bool, default=True, help='Use tqdm')
args = parser.parse_args()

config = {
        # ソルバ選択
        "solver": args.solver,

        # 陰解法ソルバー
        "implicit_solver": args.implicit_solver,

        # 陰解法パラメータ
        "picard_iter": args.picard_iter,
        "picard_tol": args.picard_tol,

        # HOLOパラメータ
        "ho_iter": args.ho_iter,
        "lo_iter": args.lo_iter,
        "ho_tol": args.ho_tol,
        "lo_tol": args.lo_tol,

        # ハイパーパラメータ
        "tau_tilde": args.tau_tilde,

        # 数値計算パラメータ
        "v_max": 10.0,
        "dt": args.dt,

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
        "T_total":args.T_total,

        # 精度
        "dtype": "float64",

        "use_tqdm" : args.use_tqdm,

        "device" : args.device

    }

sim = BGK1DPlot(**config)
sim.run_simulation()
sim.plot_state(filename=f"{args.output}.png")
sim.create_gif(filename=f"{args.output}.gif")


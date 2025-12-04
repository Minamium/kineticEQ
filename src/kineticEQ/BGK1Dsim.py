import torch
import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Any, Union
import time

from .progress_bar import get_progress_bar, progress_write

@torch.jit.script
def _tdma_vec_impl(a: torch.Tensor,
                   b: torch.Tensor,
                   c: torch.Tensor,
                   d: torch.Tensor) -> torch.Tensor:
    n = b.shape[1]

    cp = torch.empty_like(c)
    dp = torch.empty_like(d)

    beta    = b[:, 0]
    cp[:,0] = c[:,0] / beta
    dp[:,0] = d[:,0] / beta

    for k in range(1, n):
        beta    = b[:,k] - a[:,k] * cp[:,k-1]
        cp[:,k] = c[:,k] / beta
        dp[:,k] = (d[:,k] - a[:,k] * dp[:,k-1]) / beta

    x = torch.empty_like(d)
    x[:,-1] = dp[:,-1]
    for k in range(n-2, -1, -1):
        x[:,k] = dp[:,k] - cp[:,k] * x[:,k+1]
    return x

# 数値計算クラス
class BGK1D:
    """数値計算・データ生成関数群"""
    def __init__(self, 
                 # 解法選択
                 solver=None,

                 # 陽解法ソルバー
                 explicit_solver=None,

                 # 陰解法ソルバー
                 implicit_solver=None,

                 # 陰解法パラメータ
                 picard_iter=10,
                 picard_tol=1e-4,

                 # HOLOパラメータ
                 ho_iter=10,
                 lo_iter=10,
                 ho_tol=1e-4,
                 lo_tol=1e-4,
                 Con_Terms_do=False,
                 flux_consistency_do=False,
                 SVdown=False,

                 # ハイパーパラメータ
                 tau_tilde=1.0,

                 # 数値計算パラメータ
                 dt=0.01,
                 Lx=1.0,
                 T_total=1.0,
                 nx=64,
                 nv=32,
                 v_max=5.0,

                 # 初期状態設定
                 ic_fn=None,
                 initial_regions=[{"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},    
                                  {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8}],

                 # 固定モーメント境界条件
                 n_left=1.0, u_left=0.0, T_left=1.0,
                 n_right=1.0, u_right=0.0, T_right=1.0,

                 # 精度
                 dtype='float64',

                 # GPU設定
                 device='cuda',

                 # tqdm設定
                 use_tqdm=True,
                 
                 # GIF用のrecord_stateのフラグ
                 record_state=False,

                 # インスタンス作成時に自動コンパイル
                 auto_compile=True
                 ):
        # flag保存
        self.flag_record_state = record_state

        # パラメータ保存
        self.solver = solver
        self.implicit_solver = implicit_solver
        self.explicit_solver = explicit_solver
        self.picard_iter = picard_iter
        self.picard_tol = picard_tol
        self.ho_iter = ho_iter
        self.lo_iter = lo_iter
        self.ho_tol = ho_tol
        self.lo_tol = lo_tol
        self.tau_tilde = tau_tilde
        self.dt = dt
        self.Lx = Lx
        self.T_total = T_total
        self.nx = nx
        self.nv = nv
        self.v_max = v_max

        # 初期状態保存
        self.ic_fn = ic_fn
        self.initial_regions = initial_regions

        # 境界条件保存
        self.n_left, self.u_left, self.T_left = n_left, u_left, T_left
        self.n_right, self.u_right, self.T_right = n_right, u_right, T_right

        # 精度設定（文字列からTorch型に変換）
        if dtype == 'float32':
            self.dtype = torch.float32
        elif dtype == 'float64':
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32' or 'float64'")
        self.dtype_str = dtype  # 表示用
        torch.set_default_dtype(self.dtype)

        # デバイス設定
        self.device = torch.device(device)

        # tqdm設定
        self.use_tqdm = use_tqdm

        # HOLO補正項オプション
        self.Con_Terms_do = Con_Terms_do
        self.flux_consistency_do = flux_consistency_do
        self.SVdown = SVdown

        # 派生パラメータ計算
        self.dx = self.Lx / (self.nx - 1)
        self.dv = 2 * self.v_max / (self.nv - 1)
        self.nt = int(self.T_total / self.dt) + 1

        self.animation_data = []  # 状態記録用

        # コンパイル
        if auto_compile:
            print("--- auto compile ---")
            self.compile()
            print("--- auto compile complete ---")

        # 初期化完了通知
        print(f"initiaze complete:")
        print(f"  solver: {self.solver}")
        if self.solver == 'implicit':
            print(f"  implicit solver: {self.implicit_solver}")
            if self.implicit_solver == 'holo':
                print(f"  ho_iter: {self.ho_iter}, ho_tol: {self.ho_tol}")
                print(f"  lo_iter: {self.lo_iter}, lo_tol: {self.lo_tol}")
                if self.Con_Terms_do:
                    print("  collision terms: included")
                else:
                    print("  collision terms: excluded")
                if self.flux_consistency_do:
                    print("  flux consistency: enabled")
                else:
                    print("  flux consistency: disabled")

        print(" ---- hyperparameter ----")
        print(f"  hyperparameter: tau_tilde={self.tau_tilde}")
        print(" ---- space ----")
        print(f"  space: nx={self.nx}, dx={self.dx:.4f}, Lx={self.Lx}")
        print(" ---- velocity ----")
        print(f"  velocity: nv={self.nv}, dv={self.dv:.4f}, v_max={self.v_max}")
        print(" ---- time ----")
        print(f"  time: nt={self.nt}, dt={self.dt:.4f}, T_total={self.T_total}")
        print(f"  dtype: {self.dtype}")
        print(f"  device: {self.device}, GPU name: {torch.cuda.get_device_name(0)}")

    #シミュレーションメソッド
    def run_simulation(self):
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        print("--- run simulation ---")

        if self.solver == "explicit":
            self.explicit_scheme()
        elif self.solver == "implicit":
            self.implicit_scheme()
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        print("--- run simulation complete, Result is saved in self.f ---")

    # コンパイルメソッド
    def compile(self, cuSOLVER=False, 
                fused_explicit=False,
                fused_implicit=False,
                lo_block=False):
        # cuSOLVERコンパイル
        if cuSOLVER or (self.solver == 'implicit' and (self.implicit_solver == 'backend' or self.implicit_solver == 'holo')):
            print("--- compile cuSOLVER ---")
            from torch.utils.cpp_extension import load
            import os, sysconfig
            from pathlib import Path
            os.makedirs('build', exist_ok=True)
            src_dir = Path(__file__).resolve().parent / "backends" / "gtsv"
            self._cusolver = load(
                name='gtsv_batch',
                sources=[str(src_dir/'gtsv_binding.cpp'),
                         str(src_dir/'gtsv_batch.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                extra_include_paths=[sysconfig.get_paths()['include']],
                extra_ldflags=['-lcusparse'],
                build_directory='build',
                verbose=True
            )

        # CUDA fused explicitコンパイル
        self._explicit_cuda = None
        if fused_explicit or (self.solver == 'explicit' and self.explicit_solver == 'backend'):
            print("--- compile CUDA fused explicit backend ---")
            from torch.utils.cpp_extension import load
            import traceback, os, sysconfig
            from pathlib import Path
            os.makedirs('build', exist_ok=True)
            src_dir = Path(__file__).resolve().parent / "backends" / "explicit_fused"
            # A100/3070 両対応にするなら環境変数でアーキを指定:
            # os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6")
            self._explicit_cuda = load(
                name='explicit_fused',
                sources=[str(src_dir/'explicit_binding.cpp'),
                         str(src_dir/'explicit_kernel.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],  # FP64なので -use_fast_math は付けない
                extra_include_paths=[sysconfig.get_paths()['include']],
                build_directory='build',
                verbose=True
            )
            traceback.print_exc()
            print('--- fused CUDA backend loaded ---')

        # __init__ 内でのビルド（explicit_fused と同様に分離コンパイル）
        if fused_implicit or (self.solver == "implicit" and self.implicit_solver == "backend"):
            print("--- compile CUDA fused implicit backend ---")
            from torch.utils.cpp_extension import load
            import traceback, os, sysconfig
            from pathlib import Path
            src_dir = Path(__file__).resolve().parent / "backends" / "implicit_fused"
            os.makedirs('build', exist_ok=True)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6")  # 必要に応じて
            self._implicit_cuda = load(
                name='implicit_fused',
                sources=[str(src_dir/'implicit_binding.cpp'),
                         str(src_dir/'implicit_kernels.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                extra_include_paths=[sysconfig.get_paths()['include']],
                extra_ldflags=['-lcusparse'],
                build_directory='build',
                verbose=True
            )
            traceback.print_exc()
            print('--- fused CUDA backend loaded ---')

        # HOLO 用ブロック三重対角ソルバ
        if lo_block or (self.solver == "implicit" and self.implicit_solver == "holo"):
            print("--- compile LO block-tridiag backend ---")
            from torch.utils.cpp_extension import load
            import traceback, os, sysconfig
            from pathlib import Path

            src_dir = Path(__file__).resolve().parent / "backends" / "lo_blocktridiag"
            os.makedirs('build', exist_ok=True)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6")

            self._lo_block = load(
                name='lo_blocktridiag',
                sources=[str(src_dir / 'block_tridiag_binding.cpp'),
                         str(src_dir / 'block_tridiag_kernel.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                extra_include_paths=[sysconfig.get_paths()['include']],
                build_directory='build',
                verbose=True
            )
            traceback.print_exc()
            print('--- LO block-tridiag backend loaded ---')
        
    # ベンチマークメソッド
    def run_benchmark(self, 
                      benc_type="spatial", 
                      grid_list=[16, 32, 64, 128, 256, 512, 1024],
                      nv_list=[64, 128, 256]):
        print(f"--- Benchmark Start, benc_type: {benc_type} ---")

        # 結果保存用辞書
        self.benchmark_results = {
            'bench_type' : benc_type,
        }

        if benc_type == "spatial":
            self._run_benchmark_space(grid_list)
        elif benc_type == "velocity":
            self._run_benchmark_velocity(grid_list)
        elif benc_type == "time":
            self._run_benchmark_time(grid_list, nv_list)
        else:
            raise ValueError(f"Unknown benchmark type: {benc_type}")

        #self.plot_benchmark_comparison(benc_type)

        print("--- Benchmark Completed ---")
        return self.benchmark_results

    # 収束性テスト(HOLO反復 vs picard反復)
    def run_convergence_test(self,
                             tau_tilde_list=[5e-4, 5e-5, 5e-6, 5e-7]):
        # 必要なバイナリをコンパイル
        self.compile(cuSOLVER=True,
                     fused_explicit=False,
                     fused_implicit=True,
                     lo_block=True)
        print(f"--- Convergence Test Start ---")
        
        # 結果保存用コンテナ (meta + records)
        self.convergence_results = {
            "meta": {
                # タイトルに使っている実行条件をここにまとめておく
                "dt": float(self.dt),
                "nx": int(self.nx),
                "nv": int(self.nv),
                "Lx": float(self.Lx),
                "T_total": float(self.T_total),
                "v_max": float(self.v_max),

                "picard_tol": float(self.picard_tol),
                "ho_tol": float(self.ho_tol),
                "lo_tol": float(self.lo_tol),
                "picard_iter": int(self.picard_iter),
                "ho_iter": int(self.ho_iter),
                "lo_iter": int(self.lo_iter),

                # どの tau_tilde を回したかも残しておく
                "tau_tilde_list": [float(t) for t in tau_tilde_list],

                # あると便利そうなもの（必要に応じて削ってよい）
                "dtype": str(self.dtype),
                "device": (
                    self.device.type
                    if isinstance(self.device, torch.device)
                    else str(self.device)
                ),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if isinstance(self.device, torch.device)
                    and self.device.type == "cuda"
                    else None
                ),
            },
            "records": [],  # 実際の1%サンプリング結果はここに append していく
        }

        # tau_tilde_listに従って収束性テストを実行
        for tau in tau_tilde_list:
            # tau_tildeを設定
            print(f"--- tau_tilde: {tau} ---")
            self.tau_tilde = tau
            
            # 収束性テストメソッド
            self._run_holo_test()
            self._run_picard_test()

        print("--- Convergence Test Completed ---")
        return self.convergence_results
    
    # スキーム別比較テスト
    def run_scheme_comparison_test(self, 
                                    scheme_list = ["explicit", "holo", "implicit"],
                                    scheme_delta_t_list = [5e-7, 5e-4, 5e-7],
                                    ):
        """
        指定されたスキームリストで比較テストを実行する
        scheme_list: スキームリスト
        scheme_delta_t_list: 各スキームに対応する時間刻み幅のリスト
        """
        # 必要なバイナリをコンパイル
        self.compile(cuSOLVER=True,
                     fused_explicit=True,
                     fused_implicit=True,
                     lo_block=True)
        print(f"Running scheme comparison test with schemes: {scheme_list}")
        print(f"Time step sizes: {scheme_delta_t_list}")

        # 結果保存用コンテナ (meta + records)
        self.cross_scheme_test_results = {
            "meta": {
                # テストメタデータ
                "scheme_list": scheme_list,
                "scheme_delta_t_list": [float(dt) for dt in scheme_delta_t_list],

                # 実行条件（dt はスキームごとに変えるのでここは初期値）
                "dt": float(self.dt),
                "nx": int(self.nx),
                "nv": int(self.nv),
                "Lx": float(self.Lx),
                "T_total": float(self.T_total),
                "v_max": float(self.v_max),
                "tau_tilde": float(self.tau_tilde),

                "picard_tol": float(self.picard_tol),
                "ho_tol": float(self.ho_tol),
                "lo_tol": float(self.lo_tol),
                "picard_iter": int(self.picard_iter),
                "ho_iter": int(self.ho_iter),
                "lo_iter": int(self.lo_iter),

                # cudaデバイス名
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if isinstance(self.device, torch.device)
                    and self.device.type == "cuda"
                    else None
                ),
            },
            "records": [],  # 各スキームの実行結果はここに append していく
        }
        
        # スキームごとにテストを実行
        for scheme, dt_scheme in zip(scheme_list, scheme_delta_t_list):
            dt_scheme = float(dt_scheme)

            if scheme == "explicit":
                print(f"  Running explicit scheme (dt={dt_scheme})")
                self._cross_scheme_test_explicit(dt_scheme)
                n, u, T = self.calculate_moments()
                self.cross_scheme_test_results["records"].append({
                    "scheme": "explicit",
                    "dt": dt_scheme,
                    "f": self.f.cpu().numpy().copy(),
                    "n": n.cpu().numpy().copy(),
                    "u": u.cpu().numpy().copy(),
                    "T": T.cpu().numpy().copy(),
                })

            elif scheme == "implicit":
                print(f"  Running implicit scheme (dt={dt_scheme})")
                self._cross_scheme_test_implicit(dt_scheme)
                n, u, T = self.calculate_moments()
                self.cross_scheme_test_results["records"].append({
                    "scheme": "implicit",
                    "dt": dt_scheme,
                    "f": self.f.cpu().numpy().copy(),
                    "n": n.cpu().numpy().copy(),
                    "u": u.cpu().numpy().copy(),
                    "T": T.cpu().numpy().copy(),
                })

            elif scheme == "holo":
                print(f"  Running HOLO scheme (dt={dt_scheme})")
                self._cross_scheme_test_holo(dt_scheme)
                n, u, T = self.calculate_moments()
                self.cross_scheme_test_results["records"].append({
                    "scheme": "holo",
                    "dt": dt_scheme,
                    "f": self.f.cpu().numpy().copy(),
                    "n": n.cpu().numpy().copy(),
                    "u": u.cpu().numpy().copy(),
                    "T": T.cpu().numpy().copy(),
                })

            else:
                raise ValueError(f"Unknown scheme: {scheme}")
        return self.cross_scheme_test_results

    # クロステスト(explicit scheme)
    def _cross_scheme_test_explicit(self, delta_t):
        print("--- explicit ---")

        # dt設定 + nt更新
        self.dt = float(delta_t)
        self.nt = int(self.T_total / self.dt) + 1

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # 1% 間隔
        progress_interval = max(1, self.nt // 100)

        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress(explicit)", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # 更新実行
                self._explicit_update_cuda_backend()

                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # クロステスト(implicit scheme: Picard)
    def _cross_scheme_test_implicit(self, delta_t):
        print("--- implicit ---")

        self.dt = float(delta_t)
        self.nt = int(self.T_total / self.dt) + 1

        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        progress_interval = max(1, self.nt // 100)

        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress(implicit)", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                self._implicit_update_cuda_backend()
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # クロステスト(HOLO scheme)
    def _cross_scheme_test_holo(self, delta_t):
        print("--- HOLO ---")

        self.dt = float(delta_t)
        self.nt = int(self.T_total / self.dt) + 1

        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        progress_interval = max(1, self.nt // 100)

        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress(HOLO)", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                self._implicit_update_holo()
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # HOLO収束性テスト
    def _run_holo_test(self):
        print("--- HOLO ---")

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # 1% 間隔
        progress_interval = max(1, self.nt // 100)

        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # ここから計測開始
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                #更新実行
                ho_iter, ho_residual, lo_iter_list, lo_residual = self._implicit_update_holo()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                step_walltime = time.perf_counter() - t0
                # ここまで 1 step の walltime 計測

                # 進捗1%ごとに収集
                if step % progress_interval == 0 or (step == self.nt - 1):
                    self.convergence_results["records"].append({
                        "scheme": "holo",
                        "tau_tilde": self.tau_tilde,
                        "step": step,
                        "time": step * self.dt,
                        "ho_iter": ho_iter,
                        "ho_residual": ho_residual,
                        "lo_iter_list": lo_iter_list,
                        "lo_residual": lo_residual,
                        "walltime": step_walltime,
                    })

                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # Implicit + picard収束性テスト
    def _run_picard_test(self):
        print("--- PICARD ---")

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # 1% 間隔
        progress_interval = max(1, self.nt // 100)

        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # ここから計測開始
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                #更新実行
                picard_iter, picard_residual = self._implicit_update_cuda_backend()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                step_walltime = time.perf_counter() - t0
                # ここまで 1 step の walltime 計測

                # 進捗1%ごとに収集
                if step % progress_interval == 0 or (step == self.nt - 1):
                    self.convergence_results["records"].append({
                        "scheme": "implicit_picard",
                        "tau_tilde": self.tau_tilde,
                        "step": step,
                        "time": step * self.dt,
                        "picard_iter": picard_iter,
                        "picard_residual": picard_residual,
                        "walltime": step_walltime,
                    })

                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    #空間差分ベンチマーク
    def _run_benchmark_space(self, grid_list):

        # grid_listに従って空間差分ベンチマークを実行
        for bench_iter in range(len(grid_list)):
            self._runbench_spatial(grid_list[bench_iter])
            n, u, T = self.calculate_moments()
            self.benchmark_results[grid_list[bench_iter]] = {
                'x': self.x.cpu().numpy().copy(),   
                'v': self.v.cpu().numpy().copy(),    
                'f': self.f.cpu().numpy().copy(), 
                'n': n.cpu().numpy().copy(),
                'u': u.cpu().numpy().copy(),
                'T': T.cpu().numpy().copy(),
                'dx': self.dx,
                'dv': self.dv
            }

    #速度差分ベンチマーク
    def _run_benchmark_velocity(self, grid_list):

        # grid_listに従って速度差分ベンチマークを実行
        for bench_iter in range(len(grid_list)):
            self._runbench_velocity(grid_list[bench_iter])
            n, u, T = self.calculate_moments()
            self.benchmark_results[grid_list[bench_iter]] = {
                'x': self.x.cpu().numpy().copy(),
                'v': self.v.cpu().numpy().copy(),    
                'f': self.f.cpu().numpy().copy(), 
                'n': n.cpu().numpy().copy(),
                'u': u.cpu().numpy().copy(),
                'T': T.cpu().numpy().copy(),
                'dv': self.dv,
                'dx': self.dx  
            }

    # 実行時間ベンチマーク
    def _run_benchmark_time(self, nx_list, nv_list):
        # 実行中の計算デバイス名の取得
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else str(self.device)
        self.benchmark_results['device_name'] = device_name
        
        # CPU名を取得
        import platform
        cpu_name = None
        
        # 方法1: /proc/cpuinfo から取得（Linux）
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        break
        except:
            pass
        
        # 方法2: cpuinfo ライブラリを使用
        if not cpu_name:
            try:
                import cpuinfo
                cpu_name = cpuinfo.get_cpu_info()['brand_raw']
            except:
                pass
        
        # 方法3: platform.processor()
        if not cpu_name:
            cpu_name = platform.processor()
        
        # フォールバック: アーキテクチャ名
        if not cpu_name:
            cpu_name = f"{platform.machine()} CPU"
        
        self.benchmark_results['cpu_name'] = cpu_name
        
        # 結果保存用辞書を初期化
        self.benchmark_results['timing_results'] = {}

        # grid_listに従って実行時間ベンチマークのグリッドテストを実行
        for nx in nx_list:
            for nv in nv_list:
                grid_key = f"{nx}x{nv}"
                print(f"--- Test Grid: {grid_key} ---")

                # グリッドテストベンチマークの実行と結果保存
                timing_result = self._runbench_time(nx, nv)
                self.benchmark_results['timing_results'][grid_key] = timing_result

        return self.benchmark_results

    # 空間分割数ベンチマーク用シミュレーションメソッド
    def _runbench_spatial(self, nx_num):
        print(f"--- Run Benchmark Simulation (nx = {nx_num}) ---")

        # パラメータ設定
        self.nx = nx_num
        self.dx = self.Lx / (self.nx - 1)
        self.nt = int(self.T_total / self.dt)

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                if self.solver == "explicit":
                    if self.explicit_solver == 'backend':
                        self._explicit_update_cuda_backend()
                    else:
                        self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == 'backend':
                        self._implicit_update_cuda_backend()
                    elif self.implicit_solver == 'holo':
                        self._implicit_update_holo()
                    else:
                        self._implicit_cusolver_update()
                    
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # 速度分割数ベンチマーク用シミュレーションメソッド
    def _runbench_velocity(self, nv_num):
        print(f"--- Run Benchmark Simulation (nv = {nv_num}) ---")

        # パラメータ設定
        self.nv = nv_num
        self.dv = 2 * self.v_max / (self.nv - 1)
        self.nt = int(self.T_total / self.dt)

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                if self.solver == "explicit":
                    if self.explicit_solver == 'backend':
                        self._explicit_update_cuda_backend()
                    else:
                        self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == 'backend':
                        self._implicit_update_cuda_backend()
                    elif self.implicit_solver == 'holo':
                        self._implicit_update_holo()
                    else:
                        self._implicit_cusolver_update()
                    
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # 実行時間ベンチマーク用シミュレーションメソッド
    def _runbench_time(self, nx_num, nv_num):
        import time
        
        self.nx = nx_num
        self.nv = nv_num
        self.dx = self.Lx / (self.nx - 1)
        self.dv = 2 * self.v_max / (self.nv - 1)

        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # CUDA時間測定の準備
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up: 初回JITコンパイルとメモリ確保のオーバーヘッドを除去
        warmup_steps = min(5, self.nt // 4)  # 最大5ステップ、全体の1/4まで
        for step in range(warmup_steps):
            if self.solver == "explicit":
                # self._explicit_update()
                if self.explicit_solver == 'backend':
                    self._explicit_update_cuda_backend()
                else:
                    self._explicit_update()
            elif self.solver == "implicit":
                if self.implicit_solver == 'backend':
                    self._implicit_update_cuda_backend()
                elif self.implicit_solver == 'holo':
                    self._implicit_update_holo()
                else:
                    self._implicit_cusolver_update()
                
            # 配列交換
            self.f, self.f_new = self.f_new, self.f
        
        # 本計測開始
        cpu_start_time = time.perf_counter()
        
        # GPU時間測定開始
        if self.device.type == 'cuda':
            start_event.record()

        # ベンチマーク用ループ（プログレスバー無し）
        for step in range(self.nt):
            if self.solver == "explicit":
                # self._explicit_update()
                if self.explicit_solver == 'backend':
                    self._explicit_update_cuda_backend()
                else:
                    self._explicit_update()
            elif self.solver == "implicit":
                if self.implicit_solver == 'backend':
                    self._implicit_update_cuda_backend()
                elif self.implicit_solver == 'holo':
                    self._implicit_update_holo()
                else:
                    self._implicit_cusolver_update()

            # 配列交換
            self.f, self.f_new = self.f_new, self.f

        # GPU完了を待ってから時間測定終了
        if self.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()  # GPU完了を保証
        
        cpu_total_time = time.perf_counter() - cpu_start_time
        
        # 結果をまとめる
        timing_result = {
            'nx': nx_num,
            'nv': nv_num,
            'total_grid_points': nx_num * nv_num,
            'device': str(self.device),
            'total_steps': self.nt,
            'warmup_steps': warmup_steps,
            'cpu_total_time_sec': cpu_total_time,  # GPU計算を包含した壁時計時間
        }
        
        # CUDA時間も記録（純粋なGPU計算時間）
        if self.device.type == 'cuda':
            gpu_total_time_ms = start_event.elapsed_time(end_event)
            timing_result.update({
                'gpu_total_time_ms': gpu_total_time_ms,
                'gpu_total_time_sec': gpu_total_time_ms / 1000,
            })
        
        return timing_result

    #配列確保メソッド
    def Array_allocation(self):
        #print("--- prepare for allocation ---")
        #print("Now, using device is: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

        # 配列確保
        #print("Start allocating arrays...")

        # 分布関数配列確保
        self.f = torch.zeros((self.nx, self.nv), dtype=self.dtype, device=self.device)
        self.f_new = torch.zeros((self.nx, self.nv), dtype=self.dtype, device=self.device)

        # 座標系配列確保
        self.x = torch.linspace(0, self.Lx, self.nx, dtype=self.dtype, device=self.device)
        self.v = torch.linspace(-self.v_max, self.v_max, self.nv, dtype=self.dtype, device=self.device)
        
        # ---- キャッシュ: 変化しないテンソルを保存 (FP64 前提) ----
        # 列ベクトル化した速度グリッドと 1/sqrt(2π)
        self._v_col = self.v[None, :]                       # shape (1, nv)
        self._inv_sqrt_2pi = torch.tensor(
            1.0 / math.sqrt(2.0 * math.pi),
            dtype=self.dtype,
            device=self.device,
        )
         
        # ---- 移流項計算用キャッシュ ----
        self._pos_mask = (self.v > 0)                       # (nv,)
        self._neg_mask = ~self._pos_mask                    # (nv,)
        # -v/dx を前計算しておくと stream = coeff * df だけで済む
        self._v_coeff = -self.v / self.dx                   # (nv,)

        # ---- ここから追加：陽解法の融合版で使うワーク（事前確保・再利用）----
        # (nv,3) 連続: [1, v, v^2] —— モーメントを1パスで取るため
        self._vv3 = torch.stack((
            torch.ones_like(self.v),
            self.v,
            self.v * self.v
        ), dim=1).contiguous()

        # 差分とフラックスの作業領域（毎回割り当てをしない）
        self._df   = torch.empty((self.nx - 1, self.nv), dtype=self.dtype, device=self.device)
        self._flux = torch.empty_like(self._df)

        self._s    = torch.empty((self.nx, 3), dtype=self.dtype, device=self.device)     
        self._work = torch.empty((self.nx, self.nv), dtype=self.dtype, device=self.device)  

        # v>=0 と v<0 を分ける閾インデックス（ブールマスクを避ける）
        self._k0 = int(torch.searchsorted(
            self.v, torch.tensor(0.0, dtype=self.dtype, device=self.device)
        ))

        # ===== Picard で再利用する内部ワーク領域 =====
        # (n, u, T)
        self._n  = torch.empty(self.nx, dtype=self.dtype, device=self.device)
        self._u  = torch.empty(self.nx, dtype=self.dtype, device=self.device)
        self._T  = torch.empty(self.nx, dtype=self.dtype, device=self.device)

        # 境界マクスウェル（RHS寄与用。セル値はここでは触らない）
        self._fL = torch.empty(self.nv, dtype=self.dtype, device=self.device)
        self._fR = torch.empty(self.nv, dtype=self.dtype, device=self.device)

        # 3重対角＋右辺（内部セル i=1..nx-2 用）
        n_inner = max(self.nx - 2, 0)
        if n_inner > 0:
            self._dl = torch.empty((self.nv, n_inner), dtype=self.dtype, device=self.device)
            self._dd = torch.empty((self.nv, n_inner), dtype=self.dtype, device=self.device)
            self._du = torch.empty((self.nv, n_inner), dtype=self.dtype, device=self.device)
            self._B  = torch.empty((self.nv, n_inner), dtype=self.dtype, device=self.device)
        else:
            self._dl = self._dd = self._du = self._B = None

        # Picard反復用の候補バッファ（境界は最後に前状態で固定）
        self._fz     = torch.empty_like(self.f)
        self._fn_tmp = torch.empty_like(self.f)

        # 残差用スカラ（GPU）
        self._res_buf = torch.zeros(1, dtype=self.dtype, device=self.device)

         
        #print("Allocated successfully")

    # モーメント計算メソッド
    @torch.no_grad()
    def calculate_moments(self, f=None):
        # 引数がなければ、現在の分布関数を使用
        if f is None:
            f = self.f

        # それぞれの空間座標における粒子数密度、平均速度、温度を計算して
        # 物理空間分割点をインデックスとしたモーメント配列を作成

        # 愚直な実装, sumを三回呼ぶためメモリ帯域浪費が激しい
        #n = torch.sum(f, dim=1) * self.dv
        #u = torch.sum(f * self.v[None, : ], dim=1) * self.dv / n
        #T = torch.sum(f * (self.v[None, : ] - u[:, None])**2, dim=1) * self.dv / n

        # torch.einsumを使用した高速化 
        dv = self.dv
        ones = torch.ones_like(self.v)
        vv = torch.stack((ones, self.v * self.v))  # shape (2, nv)

        # s0, s2 を同時に
        s02 = f @ vv.T
        s02.mul_(dv)

        n  = s02[:, 0]
        s2 = s02[:, 1]

        s1 = (f @ self.v) * dv
        u  = s1 / n
        T  = s2 / n - u * u

        return n, u, T

    # マックスウェル分布関数計算メソッド
    @torch.no_grad()
    def Maxwellian(self, n: torch.Tensor, u: torch.Tensor, T: torch.Tensor):
        """マックスウェル分布 f_M を高速計算 (FP64 前提)

        Parameters
        ----------
        n, u, T : (nx,) torch.Tensor
            密度, 流速, 温度 (いずれも self.dtype, self.device)

        Returns
        -------
        f_M : (nx, nv) torch.Tensor
            Maxwellian 分布
        """

        # マックスウェル分布関数を計算(old実装)
        # メモリ帯域効率が悪い
        #coeff = n / torch.sqrt(2 * torch.pi * T)
        #exponet = -(self.v[None, :] - u[:, None])**2 / (2 * T[:, None])
        #return coeff[:, None] * torch.exp(exponet)

        # 係数部  n / sqrt(2π T)
        coeff = (n * self._inv_sqrt_2pi) / torch.sqrt(T)      # (nx,)

        # 指数部  exp( -(v-u)^2 / (2T) )
        invT  = 0.5 / T                                       # (nx,)
        diff  = self._v_col - u[:, None]                      # (nx, nv), view+broadcast

        exponent = diff.mul(diff)                             # (nx, nv): (v-u)^2
        exponent.mul_(-invT[:, None])                         # -(v-u)^2 / (2T)
        torch.exp(exponent, out=exponent)                     # exp(·) in-place

        exponent.mul_(coeff[:, None])                         # f = coeff * exp
        return exponent

    # 初期条件設定メソッド
    def set_initial_condition(self):
        """config設定による初期条件設定"""
        x = self.x  # (nx,) 物理座標

        # デフォルト値で初期化
        n_init = torch.ones(self.nx, dtype=self.dtype, device=self.device)
        u_init = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
        T_init = torch.ones(self.nx, dtype=self.dtype, device=self.device)

        # ic_fn 優先
        if self.ic_fn is not None:
            n_i, u_i, T_i = self.ic_fn(x)

            # 型・shape チェック
            if n_i.shape != x.shape or u_i.shape != x.shape or T_i.shape != x.shape:
                raise ValueError("ic_fn must return (n, u, T) with same shape as x")

            n_init = n_i.to(device=self.device, dtype=self.dtype)
            u_init = u_i.to(device=self.device, dtype=self.dtype)
            T_init = T_i.to(device=self.device, dtype=self.dtype)

        # 後方互換: initial_regions があればそれを使う
        elif hasattr(self, 'initial_regions') and self.initial_regions is not None:
            for region in self.initial_regions:
                x_start, x_end = region['x_range']
                # 物理座標でマスクを切る（nxをユーザに意識させない）
                mask = (x >= x_start) & (x < x_end)

                if 'n' in region:
                    n_init[mask] = region['n']
                if 'u' in region:
                    u_init[mask] = region['u']
                if 'T' in region:
                    T_init[mask] = region['T']

        # マクスウェル分布で初期化
        self.f = self.Maxwellian(n_init, u_init, T_init)

    #境界条件設定メソッド
    def apply_boundary_condition(self):
        #print("--- apply boundary condition ---")
        # 左境界（スカラー値を直接使用）
        coeff_left = self.n_left / math.sqrt(2 * math.pi * self.T_left)
        exp_left = -(self.v - self.u_left)**2 / (2 * self.T_left)
        self.f[0, :] = coeff_left * torch.exp(exp_left)

        # 右境界（スカラー値を直接使用）
        coeff_right = self.n_right / math.sqrt(2 * math.pi * self.T_right)
        exp_right = -(self.v - self.u_right)**2 / (2 * self.T_right)
        self.f[-1, :] = coeff_right * torch.exp(exp_right)
        #print("Boundary condition applied")

    # torchによる陽解法更新メソッド
    def _explicit_update(self):
        """BGK explicit scheme"""
        # 現在のモーメント計算
        n, u, T = self.calculate_moments()

        # 緩和時間計算
        tau = self.tau_tilde / (n * torch.sqrt(T))

        # マクスウェル分布計算
        f_maxwell = self.Maxwellian(n, u, T)

        # 移流項計算（Upwind）
        streaming = self._compute_streaming()

        # 衝突項計算
        collision = (f_maxwell - self.f) / tau[:, None]

        # 時間発展
        self.f_new[1:-1, :] = self.f[1:-1, :] + self.dt * (streaming[1:-1, :] + collision[1:-1, :])

        # 境界固定
        self.f_new[0, :] = self.f[0, :]   
        self.f_new[-1, :] = self.f[-1, :]

    #陰解法更新メソッド(cuSOLVER)
    def _implicit_cusolver_update(self):
        """BGK implicit scheme"""
        f_z = self.f.clone()
        f_z_new = self.f.clone()

        # Picard反復
        for z in range(self.picard_iter):
            # 前回反復より得たf(k+1)の候補であるf_oldを使ってモーメント計算
            n, u, T = self.calculate_moments(f_z)

            # 緩和時間計算
            tau = self.tau_tilde / (n * torch.sqrt(T))

            # マクスウェル分布計算
            f_maxwell = self.Maxwellian(n, u, T)

            # 係数行列を構築
            # 要素a
            a_coeff = -self.dt / self.dx * torch.clamp(self.v, min=0)
            a_batch = torch.zeros(self.nv, self.nx -2, dtype=self.dtype, device=self.device)
            a_batch[:,:] = a_coeff[:, None]
            a_batch[:, 0] = 0.0

            # 要素c
            c_coeff = -self.dt / self.dx * torch.clamp(-self.v, min=0)
            c_batch = torch.zeros(self.nv, self.nx -2, dtype=self.dtype, device=self.device)
            c_batch[:,:] = c_coeff[:, None]
            c_batch[:, -1] = 0.0 

            # 要素b
            b_batch = torch.zeros(self.nv, self.nx - 2, dtype=self.dtype, device=self.device)
            b_batch[:, :] = 1.0 + (-a_coeff)[:, None] + (-c_coeff)[:, None] + self.dt / tau[1:-1][None, :]

            # ソース項行列を構築
            B_batch = (self.f[1:-1, :] + (self.dt / tau[1:-1][:, None]) * f_maxwell[1:-1, :]).T

            #境界からの移流
            B_batch[:, 0] += self.dt / self.dx * torch.clamp(self.v, min=0) *f_maxwell[0, :]
            B_batch[:, -1] += self.dt / self.dx * torch.clamp(-self.v, min=0) *f_maxwell[-1, :]

            # 線形方程式を構築, 計算
            solution = self._cusolver.gtsv_strided(
                    a_batch.contiguous(),    # 下対角
                    b_batch.contiguous(),    # 主対角
                    c_batch.contiguous(),    # 上対角
                    B_batch.contiguous()     # 右辺 → 解に上書き
                )
            f_z_new[1:-1, :] = solution.T


            # 収束判定
            residual = torch.max(torch.abs(f_z_new - f_z))
            if residual < self.picard_tol:
                break
            f_z = f_z_new.clone()

        self.f_new = f_z_new.clone()
        return z + 1, residual

    #移流項計算メソッド
    @torch.no_grad()
    def _compute_streaming(self, f: torch.Tensor | None = None):
        """Upwind streaming term ∂f/∂x を高速計算.

        前後セル差分を `torch.diff` で取得し、事前計算した係数
        `self._v_coeff = -v/dx` を掛けて一度でフラックスを作成する。
        その後、正負の速度に応じてフラックスをセルに割り当てる。
        メモリアロケーションを最小化し、ブロードキャストを回避.
        """

        # 引数がなければ、現在の分布関数を使用
        # メモリ効率が悪い
        #if f is None:
        #    f = self.f

        # v の符号マスク
        #pos = self.v > 0   # (nv,)
        #neg = ~pos
        # 前向き・後向き差分
        #df_forward  = f[1:, :] - f[:-1, :]
        #streaming   = torch.zeros_like(f, dtype=self.dtype, device=self.device)
        #streaming[1:,  pos] = - self.v[pos] * df_forward[:,  pos] / self.dx
        #streaming[:-1, neg] = - self.v[neg] * df_forward[:,  neg] / self.dx
        #return streaming

        if f is None:
            f = self.f  # (nx, nv)

        # forward difference along x (size nx-1, nv)
        df = torch.diff(f, dim=0)

        # flux = (-v/dx) * df   (nx-1, nv)
        flux = df * self._v_coeff  # broadcast over rows

        # 出力テンソルを用意（境界は 0）
        streaming = torch.zeros_like(f)

        # 正速度: 対象セルは i>=1
        streaming[1:,  self._pos_mask] = flux[:, self._pos_mask]

        # 負速度: 対象セルは i<=nx-2 (=-1 シフト)
        streaming[:-1, self._neg_mask] = flux[:, self._neg_mask]

        return streaming

    #安定性条件チェックメソッド(CFL条件)
    def check_cfl_condition(self):
        """CFL安定性条件確認"""
        cfl = self.dt * torch.max(torch.abs(self.v)) / self.dx
        print(f"CFL number: {cfl:.4f}")
        if cfl > 1.0:
            print("WARNING: CFL > 1, unstable!")
        return cfl

    #陽解法時間発展メソッド
    def explicit_scheme(self):
        """陽解法時間発展"""
        print("--- Starting explicit time evolution ---")
        self.check_cfl_condition()

        progress_interval = max(1, self.nt // 50)

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Explicit Evolution", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # 時間発展ステップ
                if self.explicit_solver == 'backend':
                    self._explicit_update_cuda_backend()
                else:
                    self._explicit_update()

                # 配列交換
                self.f, self.f_new = self.f_new, self.f

                # 進捗表示
                if self.flag_record_state:
                    if step % progress_interval == 0:
                        # 状態記録
                        self._record_state(step * self.dt)
                pbar.update(1)

        #self.create_gif()
        print("Time evolution completed!")

    #陰解法時間発展メソッド
    def implicit_scheme(self):
        """陰解法時間発展"""
        print("--- Starting implicit time evolution ---")
        self.check_cfl_condition()

        progress_interval = max(1, self.nt // 10)

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Implicit Evolution", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # 時間発展ステップ
                if self.implicit_solver == 'backend':
                    Picard_iter, residual = self._implicit_update_cuda_backend()
                elif self.implicit_solver == 'holo':
                    Picard_iter, residual, lo_iter_list, lo_residual_list = self._implicit_update_holo()
                else:
                    Picard_iter, residual = self._implicit_cusolver_update()
                
                # 配列交換
                self.f, self.f_new = self.f_new, self.f

                # 進捗表示
                if step % progress_interval == 0:
                    # 状態記録
                    self._record_state(step * self.dt)
                    current_time = step * self.dt
                    progress_write(f"Step {step:5d}/{self.nt - 1} (t={current_time:.3f})")
                    progress_write(f"Picard iteration: {Picard_iter:5d}, residual: {residual:.6e}")
                    if self.implicit_solver == 'holo':
                        progress_write(f"LO iteration: {lo_iter_list}, residual: {lo_residual_list}")

                pbar.update(1)

        #self.create_gif()
        print("Time evolution completed!")

    #状態記録メソッド
    def _record_state(self, time):
        # CPUに転送してnumpy配列として保存
        f_cpu = self.f.cpu().numpy()
        n, u, T = self.calculate_moments()

        state_data = {
            'time': time,
            'f': f_cpu,
            'n': n.cpu().numpy(),
            'u': u.cpu().numpy(),
            'T': T.cpu().numpy()
        }
        self.animation_data.append(state_data)

    # エラー計算メソッド
    def compute_error(self, result, kind='nearest'):
        """f, n, u, TのL1, L2, L∞を計算"""
        # エラー計算結果を保存する辞書
        error_dict = {}

        # 引数辞書型をコピー
        result_dict = result.copy()

        # ベンチタイプを取得
        bench_type = result_dict['bench_type']
        result_dict.pop('bench_type')

        # 参照解を取得
        ref_result = result_dict.pop(max(result_dict.keys()))

        # エラー計算
        for grid_num in result_dict:
            if bench_type == 'spatial':
                # 空間ベンチマーク：x方向のみ補間（v方向は同じ格子）
                current_result = result_dict[grid_num]

                # 分布関数f(x,v)：各v点でx方向に1次元補間
                ref_f_on_coarse = np.zeros_like(current_result['f'])
                for v_idx in range(len(current_result['v'])):
                    f_interp_1d = interp1d(ref_result['x'], ref_result['f'][:, v_idx], kind=kind)
                    ref_f_on_coarse[:, v_idx] = f_interp_1d(current_result['x'])

                # モーメントn,u,T(x)：x方向に1次元補間
                ref_n_interp = interp1d(ref_result['x'], ref_result['n'], kind=kind)
                ref_u_interp = interp1d(ref_result['x'], ref_result['u'], kind=kind)
                ref_T_interp = interp1d(ref_result['x'], ref_result['T'], kind=kind)

                ref_n_on_coarse = ref_n_interp(current_result['x'])
                ref_u_on_coarse = ref_u_interp(current_result['x'])
                ref_T_on_coarse = ref_T_interp(current_result['x'])

            elif bench_type == 'velocity':
                # 速度ベンチマーク：v方向のみ補間（x方向は同じ格子）

                current_result = result_dict[grid_num]

                # 分布関数f(x,v)：各x点でv方向に1次元補間
                ref_f_on_coarse = np.zeros_like(current_result['f'])
                for x_idx in range(len(current_result['x'])):
                    f_interp_1d = interp1d(ref_result['v'], ref_result['f'][x_idx, :], kind=kind)
                    ref_f_on_coarse[x_idx, :] = f_interp_1d(current_result['v'])

                # モーメントは補間不要（同じx格子）
                ref_n_on_coarse = ref_result['n']
                ref_u_on_coarse = ref_result['u'] 
                ref_T_on_coarse = ref_result['T']

            # 誤差計算
            f_error = current_result['f'] - ref_f_on_coarse
            n_error = current_result['n'] - ref_n_on_coarse
            u_error = current_result['u'] - ref_u_on_coarse
            T_error = current_result['T'] - ref_T_on_coarse

            # 重み計算
            dx = current_result['dx']
            dv = current_result['dv']
            weights_2d = dx * dv  # 分布関数用
            weights_1d = dx       # モーメント用

            # L1ノルム計算
            L1_f = np.sum(np.abs(f_error)) * weights_2d
            L1_n = np.sum(np.abs(n_error)) * weights_1d
            L1_u = np.sum(np.abs(u_error)) * weights_1d
            L1_T = np.sum(np.abs(T_error)) * weights_1d

            L1_error_dict = {'f': L1_f, 'n': L1_n, 'u': L1_u, 'T': L1_T}

            # L2ノルム計算
            L2_f = np.sqrt(np.sum(f_error**2) * weights_2d)
            L2_n = np.sqrt(np.sum(n_error**2) * weights_1d)
            L2_u = np.sqrt(np.sum(u_error**2) * weights_1d)
            L2_T = np.sqrt(np.sum(T_error**2) * weights_1d)

            L2_error_dict = {'f': L2_f, 'n': L2_n, 'u': L2_u, 'T': L2_T}

            # L∞ノルム計算
            Linf_f = np.max(np.abs(f_error))
            Linf_n = np.max(np.abs(n_error))
            Linf_u = np.max(np.abs(u_error))
            Linf_T = np.max(np.abs(T_error))

            Linf_error_dict = {'f': Linf_f, 'n': Linf_n, 'u': Linf_u, 'T': Linf_T}

            error_dict[grid_num] = {
                'L1': L1_error_dict,
                'L2': L2_error_dict,
                'Linf': Linf_error_dict
            }

        return error_dict

    # 陽解法のバックエンド呼び出し版
    def _explicit_update_cuda_backend(self):
        # カーネル呼び出し（境界は後で上書き）
        self._explicit_cuda.explicit_step(
            self.f, self.f_new, self.v,
            float(self.dv), float(self.dt), float(self.dx),
            float(self.tau_tilde), float(self._inv_sqrt_2pi.item()), int(self._k0)
        )
        # 境界固定（極小オーバーヘッド）
        self.f_new[0, :].copy_(self.f[0, :])
        self.f_new[-1, :].copy_(self.f[-1, :])

    # 陰解法の行列構成のバックエンド実装版
    def _implicit_update_cuda_backend(self):
        if self._implicit_cuda is None:
            raise RuntimeError("implicit_fused backend is not loaded. Set implicit_solver='backend'.")

        # 初期候補：前ステップ
        self._fz.copy_(self.f)
        swapped_last = False
        residual_val = float('inf')

        for z in range(self.picard_iter):
            # (a,b,c,B) を一括構築（Maxwellの境界寄与も旧実装と同等）
            self._implicit_cuda.build_system_fused(
                self.f, self._fz, self.v,
                float(self.dv), float(self.dt), float(self.dx),
                float(self.tau_tilde), float(self._inv_sqrt_2pi.item()),
                self._dl, self._dd, self._du, self._B
            )

            # 既存 cuSOLVER バインダで一括解法（戻り値 shape: (nv, nx-2)）
            solution = self._cusolver.gtsv_strided(
                self._dl.contiguous(),
                self._dd.contiguous(),
                self._du.contiguous(),
                self._B.contiguous()
            )

            # 内部セルのみ書き戻し。境界は前状態を維持
            self._fn_tmp.copy_(self._fz)
            self._fn_tmp[1:-1, :].copy_(solution.T)

            # 残差
            residual = torch.max(torch.abs(self._fn_tmp - self._fz))
            residual_val = float(residual)

            if residual <= self.picard_tol:
                swapped_last = False
                break

            # 次反復へ
            self._fz, self._fn_tmp = self._fn_tmp, self._fz
            swapped_last = True

        # 直近で swap したかで最新候補の位置が変わる
        latest = self._fz if swapped_last else self._fn_tmp
        self.f_new.copy_(latest)
        # 念のため境界は前状態を維持（latest の境界は _fz と同じだが、方針の明確化）
        self.f_new[0, :].copy_(self.f[0, :])
        self.f_new[-1, :].copy_(self.f[-1, :])

        return (z + 1), residual_val

    # 陰解法のHOLOアルゴリズムによる加速更新関数
    def _implicit_update_holo(self):
        """BGK HOLO scheme"""
        self._fz.copy_(self.f)
        self._fn_tmp.copy_(self.f)
        ho_residual = float('inf')

        # HOLO反復前の既知のモーメント(W^HO, k | S_1 ~ 3)
        n_HO, u_HO, T_HO = self.calculate_moments(self.f)
        S_1_HO, S_2_HO, S_3_HO = self._HO_calculate_moments(self.f)
        swapped_last = False

        # θSchemeの値を計算(暫定処置の強制クランクニコルソン)
        theta = 0.5

        # Explicit_termの計算
        Explicit_term = self._compute_Explicit_term(theta)

        # LO反復回数を保存するリスト
        lo_iter_list = []
        lo_residual_list = []
        
        # HOLOアルゴリズム
        for z in range(self.ho_iter):
            # z-1ステップのfk+1を用いて熱流束を計算
            Q_HO = self._HO_calculate_fluxes(self._fz)

            # y_i整合項を計算
            if self.Con_Terms_do:
                Y_I_terms = self._compute_Y_I_terms(self._fz, n_HO, u_HO, T_HO, Q_HO, theta)
            else:
                Y_I_terms = torch.zeros((self.nx, 3), dtype=self.dtype, device=self.device)

            # LO_calculate_momentsによる次状態のモーメントの近似(線形化反復を含む)
            n_lo, u_lo, T_lo, tau_lo, lo_residual, lo_iter = self._LO_calculate_moments(n_HO, u_HO, T_HO, Q_HO,
                                                                                        S_1_HO, S_2_HO, S_3_HO, Y_I_terms)

            # LO反復回数を保存
            lo_iter_list.append(lo_iter)
            lo_residual_list.append(lo_residual)

            # 近似したモーメントを用いて分布関数を更新
            # 係数行列の構築
            # 上対角
            beta = (self.dt / (2 * self.dx)) * torch.clamp(-self.v, min=0)
            _du = torch.zeros(self.nv, self.nx -2, dtype=self.dtype, device=self.device)
            _du[:,:] = -beta[:, None]
            _du[:, -1] = 0.0

            # 下対角
            alpha = (self.dt / (2 * self.dx)) * torch.clamp(self.v, min=0)
            _dl = torch.zeros(self.nv, self.nx -2, dtype=self.dtype, device=self.device)
            _dl[:,:] = -alpha[:, None]
            _dl[:, 0] = 0.0 

            # 主対角
            _dd = torch.zeros(self.nv, self.nx - 2, dtype=self.dtype, device=self.device)
            _dd[:, :] = 1.0 + (alpha)[:, None] + (beta)[:, None] + theta * (self.dt / tau_lo[1:-1][None, :])

            ###########
            # ソース項. #
            ###########
            B_batch = torch.zeros(self.nv, self.nx - 2, dtype=self.dtype, device=self.device)
            
            # LO系から得たモーメントでMaxwell分布を計算
            f_M_LO = self.Maxwellian(n_lo, u_lo, T_lo)

            # ソース項: d[i,j] = EXP[i,j] + θ*(dt/τ^LO)*f_M^LO[i,j]
            B_batch = (Explicit_term[1:-1, :] + 
                       theta * (self.dt / tau_lo[1:-1, None]) * f_M_LO[1:-1, :]).T

            # 境界合わせ
            B_batch[:, 0] += alpha * self.f[0, :]
            B_batch[:, -1] += beta * self.f[-1, :]
            self._fn_tmp.copy_(self._fz)

            # cuSOLVERによる線形方程式のバッチ解法
            solution = self._cusolver.gtsv_strided(
                _dl.contiguous(),
                _dd.contiguous(),
                _du.contiguous(),
                B_batch.contiguous()
            )

            # 解を内部セルのみ書き戻し
            self._fn_tmp[1:-1, :].copy_(solution.T)

            # 残差計算と収束判定
            ho_residual = torch.max(torch.abs(self._fn_tmp - self._fz))
            if ho_residual <= self.ho_tol:
                swapped_last = False
                break

            # 次反復へ
            self._fz, self._fn_tmp = self._fn_tmp, self._fz
            swapped_last = True

        # 境界固定（極小オーバーヘッド）
        latest = self._fz if swapped_last else self._fn_tmp
        self.f_new.copy_(latest)
        self.f_new[0, :].copy_(self.f[0, :])
        self.f_new[-1, :].copy_(self.f[-1, :])

        return (z + 1), ho_residual, lo_iter_list, lo_residual_list

    # LO系のモーメント方程式をPicard反復により線形化して解く
    def _LO_calculate_moments(self, n_HO, u_HO, T_HO, Q_HO, S_1_HO, S_2_HO, S_3_HO, Y_I_terms):
        """
        LO系のモーメント方程式をPicard反復により線形化して解く（GPU版）。
        3×3ブロック三重対角系は CUDA 拡張 lo_blocktridiag を用いて解く。

        入力はすべて (nx,) or (nx-1,) の torch.Tensor (self.device, self.dtype) を想定。
        Y_I_terms: 右辺整合項ベクトル
        戻り値:
            n_lo, u_lo, T_lo, tau_lo : torch.Tensor (nx,)
            lo_residual              : float
            lo_iter                  : int
        """
        device = self.device
        dtype = self.dtype

        # 念のため型・デバイスを揃える
        n_HO   = n_HO.to(device=device, dtype=dtype)
        u_HO   = u_HO.to(device=device, dtype=dtype)
        T_HO   = T_HO.to(device=device, dtype=dtype)
        Q_HO   = Q_HO.to(device=device, dtype=dtype)
        S_1_HO = S_1_HO.to(device=device, dtype=dtype)
        S_2_HO = S_2_HO.to(device=device, dtype=dtype)
        S_3_HO = S_3_HO.to(device=device, dtype=dtype)
        Y_I_terms = Y_I_terms.to(device=device, dtype=dtype)

        nx        = self.nx
        dt        = self.dt
        dx        = self.dx
        tau_tilde = self.tau_tilde

        dt_over_4dx = dt / (4.0 * dx)
        dt_over_2dx = dt / (2.0 * dx)

        # ── HO 系モーメントベクトル W^{HO} ─────────────────────
        # W_HO[i] = [ n_i, (nu)_i, U_i ]
        W_HO = torch.empty((nx, 3), dtype=dtype, device=device)
        W_HO[:, 0] = n_HO
        W_HO[:, 1] = n_HO * u_HO                           # nu
        W_HO[:, 2] = 0.5 * n_HO * (u_HO * u_HO + T_HO)     # U

        # Picard 初期値: LO^{k+1,m=0} = HO^k
        W_m = W_HO.clone()  # (nx,3)

        # ── HO 側フラックス F^{HO}_{i+1/2} を前計算 ─────────────
        # Q_half[i+1/2] = 0.5 (Q_i + Q_{i+1})
        Q_half = 0.5 * (Q_HO[:-1] + Q_HO[1:])  # (nx-1,)

        # F_{i+1/2}^{HO} = [S_1, S_2, S_3 + 2 Q_half]
        F_HO_half = torch.empty((nx - 1, 3), dtype=dtype, device=device)
        F_HO_half[:, 0] = S_1_HO
        F_HO_half[:, 1] = S_2_HO
        F_HO_half[:, 2] = S_3_HO + 2.0 * Q_half

        n_inner = nx - 2
        I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1,3,3)

        lo_residual = float("inf")

        # ── Picard 反復 ────────────────────────────────────────
        for m in range(self.lo_iter):
            # 前反復 W_m から非線形項を計算（凍結）
            n_m  = W_m[:, 0]    # (nx,)
            nu_m = W_m[:, 1]
            U_m  = W_m[:, 2]

            denom = n_m + 1e-30
            u_star = nu_m / denom
            T_star = 2.0 * (U_m / denom - 0.5 * u_star * u_star)
            T_star = torch.clamp(T_star, min=1e-30)
            P_star = n_m * T_star

            # 界面値 i+1/2 の u*, P* （中心平均）
            u_half_p = 0.5 * (u_star[:-1] + u_star[1:])  # (nx-1,)
            P_half_p = 0.5 * (P_star[:-1] + P_star[1:])  # (nx-1,)

            # ── 界面行列 A_{i+1/2}, b_{i+1/2} を一括構築 ────────
            # A_int: (nx-1,3,3), b_int: (nx-1,3)
            A_int = torch.zeros((nx - 1, 3, 3), dtype=dtype, device=device)
            b_int = torch.zeros((nx - 1, 3),    dtype=dtype, device=device)

            # A の非ゼロ成分
            A_int[:, 0, 1] = 1.0                      # (0,1) = 1
            A_int[:, 1, 0] = u_half_p * u_half_p      # (1,0) = u*^2
            A_int[:, 2, 2] = u_half_p                 # (2,2) = u*

            # b の非ゼロ成分
            b_int[:, 1] = P_half_p                    # P*
            b_int[:, 2] = u_half_p * P_half_p         # u* P*

            # 内部セル i=1..nx-2 への対応
            A_L = A_int[:-1]    # (n_inner,3,3)
            A_R = A_int[1:]     # (n_inner,3,3)
            b_L = b_int[:-1]    # (n_inner,3)
            b_R = b_int[1:]     # (n_inner,3)

            F_L = F_HO_half[:-1]  # (n_inner,3)
            F_R = F_HO_half[1:]   # (n_inner,3)
            F_diff = F_R - F_L
            b_diff = b_R - b_L

            coef = dt_over_4dx

            # 3×3 ブロック三重対角係数
            if n_inner <= 0:
                # 内部セル無し（nx<=2）の退化ケース
                W_full = W_HO.clone()
            else:
                AA = torch.zeros((n_inner, 3, 3), dtype=dtype, device=device)
                CC = torch.zeros((n_inner, 3, 3), dtype=dtype, device=device)

                # 下対角: i=2..nx-2 (→ k=1..n_inner-1)
                AA[1:] = -coef * A_L[1:]
                # 上対角: i=1..nx-3 (→ k=0..n_inner-2)
                CC[:-1] = coef * A_R[:-1]

                # 主対角
                BB = I3 + coef * (A_R - A_L)  # (n_inner,3,3)

                # 右辺
                DD = (
                    W_HO[1:-1] 
                    - dt_over_2dx * F_diff
                    - dt_over_2dx * b_diff
                    + Y_I_terms[1:-1]   # 右辺整合項ベクトル
                )  # (n_inner,3)

                # ── CUDA ブロック三重対角ソルバ呼び出し ───────
                # lo_blocktridiag.block_tridiag_solve は
                #   A:(n,3,3), B:(n,3,3), C:(n,3,3), D:(n,3) を受け取り
                #   X:(n,3) を返すように実装してある前提
                X_inner = self._lo_block.block_tridiag_solve(
                    AA.contiguous(),
                    BB.contiguous(),
                    CC.contiguous(),
                    DD.contiguous()
                )  # (n_inner,3)

                # 境界セルは HO と同じ
                W_full = W_HO.clone()
                W_full[1:-1, :] = X_inner

            # 収束判定
            diff = torch.abs(W_full - W_m)
            lo_residual = float(torch.max(diff).item())
            W_m = W_full

            if lo_residual < self.lo_tol:
                break

        # ── 結果を (n,u,T,τ) に戻す ─────────────────────────
        n_lo  = W_m[:, 0]
        nu_lo = W_m[:, 1]
        U_lo  = W_m[:, 2]

        denom = n_lo + 1e-30
        u_lo = nu_lo / denom
        T_lo = 2.0 * (U_lo / denom - 0.5 * u_lo * u_lo)
        T_lo = torch.clamp(T_lo, min=1e-30)

        tau_lo = tau_tilde / (n_lo * torch.sqrt(T_lo))

        return n_lo, u_lo, T_lo, tau_lo, lo_residual, m + 1

    # 高次モーメント S_1, S_2, S_3 を分布関数から計算
    @torch.no_grad()
    def _HO_calculate_moments(self, f_z: torch.Tensor):
        """
        LO 系で使う高次モーメント S_1, S_2, S_3 を、
        分布 f_z と速度 v による **風上 upwind スキーム** で
        「界面 i+1/2 のフラックス」として計算する。

        戻り値:
            S_1_HO, S_2_HO, S_3_HO : いずれも shape (nx-1,)
                i+1/2 の界面フラックスを表す。
        """
        # f_z : (nx, nv)
        nx, nv = f_z.shape
        assert nv == self.nv

        dv = self.dv
        v = self.v  # (nv,)

        # 左・右セル
        fL = f_z[:-1, :]  # i
        fR = f_z[ 1:, :]  # i+1

        # upwind 分布: v>0 なら左セル、v<0 なら右セル
        f_up = torch.empty((nx - 1, nv), dtype=self.dtype, device=self.device)
        f_up[:, self._pos_mask] = fL[:, self._pos_mask]
        f_up[:, self._neg_mask] = fR[:, self._neg_mask]

        # 界面フラックス
        if self.SVdown:
            w1 = torch.ones_like(v)
            w2 = v
            w3 = 0.5 * v * v
        else:
            w1 = v
            w2 = v * v
            w3 = 0.5 * v * v * v
            
        S_1_HO = torch.sum(f_up * w1[None, :], dim=1) * dv  # (nx-1,)
        S_2_HO = torch.sum(f_up * w2[None, :], dim=1) * dv  # (nx-1,)
        S_3_HO = torch.sum(f_up * w3[None, :], dim=1) * dv  # (nx-1,)

        return S_1_HO, S_2_HO, S_3_HO

    # 熱流束を分布関数より計算
    @torch.no_grad()
    def _HO_calculate_fluxes(self, f_z: torch.Tensor):
        """
        HO 系で使う熱流束 Q^HO を計算する。
        論文の式 (4.18) に対応する形を意識して、
          Q_i ≈ 1/2 ∫ (v - u_i)^3 f_i(v) dv
        を離散化して求める。
        """
        # f_z : (nx, nv)
        dv = self.dv
        # HO モーメント（n, u, T）を f_z から取得
        _, u_HO, _ = self.calculate_moments(f_z)

        # (v - u)^3 の計算（ブロードキャストで一括）
        diff = self.v[None, :] - u_HO[:, None]   # (nx, nv)
        diff3 = diff * diff * diff               # (nx, nv)

        # Q_i = 0.5 * Σ_v (v - u_i)^3 f_i(v) dv
        Q_HO = 0.5 * torch.sum(diff3 * f_z, dim=1) * dv  # (nx,)

        return Q_HO

    # HOLOスキームにおけるHO差分式のソース項のExplicit_termの計算
    @torch.no_grad()
    def _compute_Explicit_term(self, theta: float = 0.5):
        """
        HOLO スキームにおける HO 差分式の Explicit 項を計算する.
        現在ステップ f^k における移流 + BGK 衝突を評価し,
        θ スキームの (1-θ) 部分をまとめた既知項を返す.

        戻り値: (nx, nv) Tensor
        """
        # 現在のモーメントと緩和時間
        n, u, T = self.calculate_moments(self.f)
        tau = self.tau_tilde / (n * torch.sqrt(T))  # (nx,)

        # Maxwell 分布 f_M^k
        f_M = self.Maxwellian(n, u, T)              # (nx, nv)

        # 移流項と衝突項（R(f^k) = streaming + collision）
        streaming = self._compute_streaming(self.f)             # (nx, nv)
        collision = (f_M - self.f) / tau[:, None]              # (nx, nv)

        rhs = streaming + collision  # R(f^k)

        # f^{k+1} - θΔt R^{k+1} = f^k + (1-θ)Δt R^k の右辺を Explicit_term として返す
        Explicit_term = self.f + (1.0 - theta) * self.dt * rhs  # (nx, nv)
        return Explicit_term

    # 右辺整合項ベクトルY_I_termsの計算
    def _compute_Y_I_terms(self,
                           f_z: torch.Tensor,
                           n_HO: torch.Tensor,
                           u_HO: torch.Tensor,
                           T_HO: torch.Tensor,
                           Q_HO: torch.Tensor,
                           theta: float):
        """
        右辺整合項 Y_I_terms (= Δt * (y_i + I_i)) を計算する。

        Parameters
        ----------
        f_z : (nx, nv)
            HO 系の現在の反復解 f^{HO, k+1, z-1} に対応する分布
            （ここでは「新しい」HO 状態として扱う）
        n_HO, u_HO, T_HO : (nx,)
            旧時刻 k の HO モーメント (n^{HO,k}, u^{HO,k}, T^{HO,k})
        Q_HO : (nx,)
            HO 系の熱流束 Q_i^{HO}（_HO_calculate_fluxes で計算されたもの）
        theta : float
            BGK θ-scheme パラメータ（Crank–Nicolson なら 0.5）

        Returns
        -------
        Y_I_terms : (nx, 3)
            各セル i に対する [Y_I^n, Y_I^{nu}, Y_I^U]
            （離散式上は Δt * (y_i + I_i) に対応）
        """
        device = self.device
        dtype  = self.dtype

        nx = self.nx
        dv = self.dv
        dx = self.dx
        dt = self.dt

        Y_I_terms = torch.zeros((nx, 3), dtype=dtype, device=device)

        # ============================================================
        # 1. フラックス整合 y_i : γ を通した定義
        #    γ_face = -S_face + F_LO_face
        #    y_i   ≈ (γ_{i+1/2} - γ_{i-1/2}) / Δx
        # ============================================================

        # --- (1-1) HO 数値流束の離散モーメント S_1, S_2, S_3 (v, v^2, 0.5 v^3) ---
        #     f_z に対する upwind flux のモーメント
        #     S_l(i+1/2) = Σ Δv w_l(v_j) f_hat_{i+1/2,j}
        S_1_face, S_2_face, S_3_face = self._HO_calculate_moments(f_z)  # (nx-1,)

        # --- (1-2) f_z から "新しい" HO モーメントを計算 ---
        n_new, u_new, T_new = self.calculate_moments(f_z)
        P_new = n_new * T_new
        nu_new = n_new * u_new
        U_new  = 0.5 * n_new * (u_new * u_new + T_new)

        # --- (1-3) これらから LO 形式の数値流束 F_LO^{HO} を界面で構成 ---
        # 質量流束 F1 = nu
        nu_half = 0.5 * (nu_new[:-1] + nu_new[1:])  # (nx-1,)

        # 運動量流束 F2 = nu^2/n + P
        mom_flux_center = nu_new * nu_new / (n_new + 1e-30) + P_new  # (nx,)
        mom_flux_half = 0.5 * (mom_flux_center[:-1] + mom_flux_center[1:])  # (nx-1,)

        # エネルギー流束 F3 = (U + P) u + Q
        energy_flux_center = (U_new + P_new) * u_new + Q_HO           # (nx,)
        energy_flux_half   = 0.5 * (energy_flux_center[:-1] +
                                    energy_flux_center[1:])          # (nx-1,)

        # F_LO_half = [F1, F2, F3]
        F_LO_half = torch.empty((nx - 1, 3), dtype=dtype, device=device)
        F_LO_half[:, 0] = nu_half
        F_LO_half[:, 1] = mom_flux_half
        F_LO_half[:, 2] = energy_flux_half

        # --- (1-4) HO 数値流束の離散モーメントベクトル S_face ---
        S_face = torch.stack((S_1_face, S_2_face, S_3_face), dim=1)  # (nx-1,3)

        # --- (1-5) γ_face = -S_face + F_LO_half  ---  (4.19)–(4.21) の離散化
        gamma_face = -S_face + F_LO_half  # (nx-1,3)

        # --- (1-6) セル中心の y_i = (γ_{i+1/2} - γ_{i-1/2}) / Δx ---
        if nx > 2:
            y_internal = (gamma_face[1:, :] - gamma_face[:-1, :]) / dx  # (nx-2,3)
            if self.flux_consistency_do:
                Y_I_terms[1:-1, :] += dt * y_internal  # Δt * y_i を右辺に加える
            else:
                pass  # flux_consistency_do が False の場合は項を加えない

        # ============================================================
        # 2. 衝突整合 I_i : BGK 衝突項の離散モーメント残差
        #    I_i ≈ - ∑_j Δv ψ(v_j) C_mix,i,j
        # ============================================================

        # --- (2-1) 新しい HO 解 f_z に対する BGK 衝突項 C_new ---
        tau_new = self.tau_tilde / (n_new * torch.sqrt(T_new))
        fM_new  = self.Maxwellian(n_new, u_new, T_new)
        C_new   = (fM_new - f_z) / tau_new[:, None]  # (nx,nv)

        # --- (2-2) 旧時刻 HO 解 self.f (= f^k) に対する BGK 衝突項 C_old ---
        n_old, u_old, T_old = n_HO, u_HO, T_HO
        tau_old = self.tau_tilde / (n_old * torch.sqrt(T_old))
        fM_old  = self.Maxwellian(n_old, u_old, T_old)
        C_old   = (fM_old - self.f) / tau_old[:, None]  # (nx,nv)

        # --- (2-3) θ-平均した time-centred 衝突項 ---
        C_mix = theta * C_new + (1.0 - theta) * C_old  # (nx,nv)

        # 衝突不変量 ψ = (1, v, 0.5 v^2)
        v  = self.v
        w1 = torch.ones_like(v)           # ψ_1 = 1
        w2 = v                            # ψ_2 = v
        w3 = 0.5 * v * v                  # ψ_3 = 0.5 v^2

        # 離散モーメント M_i = ∑ Δv ψ C_mix
        M_n  = torch.sum(C_mix * w1[None, :], dim=1) * dv  # (nx,)
        M_nu = torch.sum(C_mix * w2[None, :], dim=1) * dv
        M_U  = torch.sum(C_mix * w3[None, :], dim=1) * dv

        # I_i を「BGK モーメント残差のマイナス」として LO 系に与える
        # （連続極限では M_i → 0, I_i → 0 になる）
        Y_I_terms[:, 0] += -dt * M_n
        Y_I_terms[:, 1] += -dt * M_nu
        Y_I_terms[:, 2] += -dt * M_U

        return Y_I_terms



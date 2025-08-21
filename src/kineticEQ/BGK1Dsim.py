import torch
import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Any, Union

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
                 solver='explicit',

                 # 陽解法ソルバー
                 explicit_solver='backend',

                 # 陰解法ソルバー
                 implicit_solver='backend',

                 # 陰解法パラメータ
                 picard_iter=10,
                 picard_tol=1e-4,

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
                 use_tqdm=True):

        # パラメータ保存
        self.solver = solver
        self.implicit_solver = implicit_solver
        self.explicit_solver = explicit_solver
        self.picard_iter = picard_iter
        self.picard_tol = picard_tol
        self.tau_tilde = tau_tilde
        self.dt = dt
        self.Lx = Lx
        self.T_total = T_total
        self.nx = nx
        self.nv = nv
        self.v_max = v_max

        # 初期状態保存
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

        # 派生パラメータ計算
        self.dx = self.Lx / (self.nx - 1)
        self.dv = 2 * self.v_max / (self.nv - 1)
        self.nt = int(self.T_total / self.dt) + 1

        self.animation_data = []  # 状態記録用

        # cuSOLVERコンパイル
        if self.implicit_solver == 'cuSOLVER' and self.solver == 'implicit':
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
                extra_cuda_cflags=['-O3', '-lcusparse'],
                extra_include_paths=[sysconfig.get_paths()['include']],
                build_directory='build',
                verbose=True
            )

        # CUDA fused explicitコンパイル
        self._explicit_cuda = None
        if self.solver == 'explicit' and self.explicit_solver == 'backend':
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
        if self.solver == "implicit" and self.implicit_solver == "backend":
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
                         str(src_dir/'implicit_kernels.cu'),
                         str(src_dir/'gtsv_batch.cu')],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
                extra_include_paths=[sysconfig.get_paths()['include']],
                build_directory='build',
                verbose=True
            )
            traceback.print_exc()
            print('--- fused CUDA backend loaded ---')


        # 初期化完了通知
        print(f"initiaze complete:")
        print(f"  solver: {self.solver}")
        if self.solver == 'implicit':
            print(f"  implicit solver: {self.implicit_solver}")

        print(f"  space: nx={self.nx}, dx={self.dx:.4f}, Lx={self.Lx}")
        print(f"  velocity: nv={self.nv}, dv={self.dv:.4f}, v_max={self.v_max}")
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
                    self._implicit_cusolver_update()
                    
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # 実行時間ベンチマーク用シミュレーションメソッド
    def _runbench_time(self, nx_num, nv_num):
        import time
        
        self.nx = nx_num
        self.nv = nv_num

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
                #self._implicit_cusolver_update()
                self._implicit_update_cuda_backend()
                
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
        #print("--- set initial condition ---")

        # デフォルト値で初期化
        n_init = torch.ones(self.nx, dtype=self.dtype, device=self.device)
        u_init = torch.zeros(self.nx, dtype=self.dtype, device=self.device)
        T_init = torch.ones(self.nx, dtype=self.dtype, device=self.device)

        # config設定があれば適用
        if hasattr(self, 'initial_regions'):
            for region in self.initial_regions:
                # 座標範囲をインデックスに変換
                x_start, x_end = region['x_range']
                i_start = int(x_start / self.Lx * (self.nx - 1))
                i_end = int(x_end / self.Lx * (self.nx - 1)) + 1
                i_start = max(0, i_start)
                i_end = min(self.nx, i_end)

                # モーメント値設定
                n_init[i_start:i_end] = region.get('n', 1.0)
                u_init[i_start:i_end] = region.get('u', 0.0)
                T_init[i_start:i_end] = region.get('T', 1.0)

                #print(f"  Region x=[{x_start:.2f}, {x_end:.2f}]: n={region.get('n', 1.0)}, u={region.get('u', 0.0)}, T={region.get('T', 1.0)}")

        # マクスウェル分布で初期化
        self.f = self.Maxwellian(n_init, u_init, T_init)
        #print("Initial condition set")

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
                #Picard_iter, residual = self._implicit_cusolver_update()
                Picard_iter, residual = self._implicit_update_cuda_backend()

                # 配列交換
                self.f, self.f_new = self.f_new, self.f

                # 進捗表示
                if step % progress_interval == 0:
                    # 状態記録
                    self._record_state(step * self.dt)
                    current_time = step * self.dt
                    progress_write(f"Step {step:5d}/{self.nt - 1} (t={current_time:.3f})")
                    progress_write(f"Picard iteration: {Picard_iter:5d}, residual: {residual:.6e}")

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

    # 陽解法のバックエンド
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

    # 陰解法のバックエンド
    def _implicit_update_cuda_backend(self):
        # fn に結果を書き込みつつ、戻り値で (iters, residual) を受け取る
        iters, residual = self._implicit_cuda.implicit_step(
            self.f, self.f_new, self.v,
            float(self.dv), float(self.dt), float(self.dx),
            float(self.tau_tilde), float(self._inv_sqrt_2pi.item()),
            int(self._k0),                      # 未使用（API整合用）
            int(self.picard_iter), float(self.picard_tol)
        )
        # 境界セルは触らない方針なので、最後に前状態で固定しておく
        self.f_new[0, :].copy_(self.f[0, :])
        self.f_new[-1, :].copy_(self.f[-1, :])
        return int(iters), float(residual)


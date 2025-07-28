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
class BGK1DBase:
    """数値計算・データ生成関数群"""
    def __init__(self, 
                 # 解法選択
                 solver='explicit',

                 # 三重対角行列ソルバー
                 implicit_solver='cuSOLVER',

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

        # 初期化完了通知
        print(f"initiaze complete:")
        print(f"  solver: {self.solver}")
        if self.solver == 'implicit':
            print(f"  implicit solver: {self.implicit_solver}")

        print(f"  space: nx={self.nx}, dx={self.dx:.4f}, Lx={self.Lx}")
        print(f"  velocity: nv={self.nv}, dv={self.dv:.4f}, v_max={self.v_max}")
        print(f"  time: nt={self.nt}, dt={self.dt:.4f}, T_total={self.T_total}")
        print(f"  dtype: {self.dtype}")
        print(f"  device: {self.device}")

    #シミュレーションメソッド
    def run_simulation(self):
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        print("--- Plot Initial State ---")
        self.plot_state()

        if self.solver == "explicit":
            self.explicit_scheme()
        elif self.solver == "implicit":
            self.implicit_scheme()
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        print("--- Plot Final State ---")
        self.plot_state()

    # ベンチマークメソッド
    def run_benchmark(self, 
                      benc_type="spatial", 
                      grid_list=[16, 32, 64, 128, 256, 512, 1024]):
        print("--- Benchmark Start ---")

        # 結果保存用辞書
        self.benchmark_results = {
            'bench_type' : benc_type,
        }

        if benc_type == "spatial":
            self._run_benchmark_space(grid_list)
        elif benc_type == "velocity":
            self._run_benchmark_velocity(grid_list)
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
                    self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == "cuSOLVER":
                        self._implicit_cusolver_update()
                    elif self.implicit_solver == "tdma":
                        self._implicit_TDMA_update()
                    elif self.implicit_solver == "full":
                        self._implicit_update()
                    else:
                        raise ValueError(f"Unknown implicit solver: {self.implicit_solver}")
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
                    self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == "cuSOLVER":
                        self._implicit_cusolver_update()
                    elif self.implicit_solver == "tdma":
                        self._implicit_TDMA_update()
                    elif self.implicit_solver == "full":
                        self._implicit_update()
                    else:
                        raise ValueError(f"Unknown implicit solver: {self.implicit_solver}")
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

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

        #print("Allocated successfully")

    # モーメント計算メソッド
    def calculate_moments(self, f=None):
        # 引数がなければ、現在の分布関数を使用
        if f is None:
            f = self.f

        # それぞれの空間座標における粒子数密度、平均速度、温度を計算して
        # 物理空間分割点をインデックスとしたモーメント配列を作成
        n = torch.sum(f, dim=1) * self.dv
        u = torch.sum(f * self.v[None, : ], dim=1) * self.dv / n
        T = torch.sum(f * (self.v[None, : ] - u[:, None])**2, dim=1) * self.dv / n

        return n, u, T

    # マックスウェル分布関数計算メソッド
    def Maxwellian(self, n, u, T):
        # マックスウェル分布関数を計算
        coeff = n / torch.sqrt(2 * torch.pi * T)
        exponet = -(self.v[None, :] - u[:, None])**2 / (2 * T[:, None])
        return coeff[:, None] * torch.exp(exponet)

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

    # 陽解法更新メソッド
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

    #陰解法更新メソッド(フル行列)
    def _implicit_update(self):
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

            # 要素c
            c_coeff = -self.dt / self.dx * torch.clamp(-self.v, min=0)

            # 要素b
            b_coeff = 1.0 + (-a_coeff)[:, None] + (-c_coeff)[:, None] + self.dt / tau[1:-1][None, :]

            # 係数行列Aを構築
            A_batch = torch.zeros(self.nv, self.nx -2 , self.nx -2, dtype=self.dtype, device=self.device)
            A_batch.diagonal(dim1=1, dim2=2).copy_(b_coeff)

            A_batch.diagonal(offset=-1, dim1=1, dim2=2).copy_(a_coeff[:, None])
            A_batch.diagonal(offset=1, dim1=1, dim2=2).copy_(c_coeff[:, None])

            # ソース項行列を構築
            B_batch = (self.f[1:-1, :] + (self.dt / tau[1:-1][:, None]) * f_maxwell[1:-1, :]).T

            #境界からの移流
            B_batch[:, 0] += self.dt / self.dx * torch.clamp(self.v, min=0) *f_maxwell[0, :]
            B_batch[:, -1] += self.dt / self.dx * torch.clamp(-self.v, min=0) *f_maxwell[-1, :]

            # 線形方程式を構築, 計算
            solution = torch.linalg.solve(A_batch, B_batch)
            f_z_new[1:-1, :] = solution.T


            # 収束判定
            residual = torch.max(torch.abs(f_z_new - f_z))
            if residual < self.picard_tol:
                break
            f_z = f_z_new.clone()

        self.f_new = f_z_new.clone()
        return z + 1, residual

    #陰解法更新メソッド(TDMA)
    def _implicit_TDMA_update(self):
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
            solution = self.tdma_vec(a_batch, b_batch, c_batch, B_batch)
            f_z_new[1:-1, :] = solution.T


            # 収束判定
            residual = torch.max(torch.abs(f_z_new - f_z))
            if residual < self.picard_tol:
                break
            f_z = f_z_new.clone()

        self.f_new = f_z_new.clone()
        return z + 1, residual

    # TDMAベクトルソルバー呼び出し
    @torch.no_grad()
    def tdma_vec(self, a, b, c, d):
        return _tdma_vec_impl(a, b, c, d)

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
    def _compute_streaming(self, f=None):
        # 引数がなければ、現在の分布関数を使用
        if f is None:
            f = self.f

        # v の符号マスク
        pos = self.v > 0   # (nv,)
        neg = ~pos
        # 前向き・後向き差分
        df_forward  = f[1:, :] - f[:-1, :]
        streaming   = torch.zeros_like(f, dtype=self.dtype, device=self.device)
        streaming[1:,  pos] = - self.v[pos] * df_forward[:,  pos] / self.dx
        streaming[:-1, neg] = - self.v[neg] * df_forward[:,  neg] / self.dx
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
                if self.implicit_solver == 'tdma':
                    Picard_iter, residual = self._implicit_TDMA_update()
                elif self.implicit_solver == 'cuSOLVER':
                    Picard_iter, residual = self._implicit_cusolver_update()
                elif self.implicit_solver == 'full':
                    Picard_iter, residual = self._implicit_update()
                else:
                    raise ValueError(f"Unknown implicit solver: {self.implicit_solver}")

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

# 可視化関数群
class BGK1DPlotMixin:
    """可視化, 解析用の関数群"""
    #状態可視化メソッド
    def plot_state(self):
        """状態の可視化"""
        import matplotlib.pyplot as plt
        # CPUに転送（matplotlib用）
        f_cpu = self.f.cpu().numpy()
        x_cpu = self.x.cpu().numpy()
        v_cpu = self.v.cpu().numpy()

        # モーメント計算
        n, u, T = self.calculate_moments()
        n_cpu = n.cpu().numpy()
        u_cpu = u.cpu().numpy()
        T_cpu = T.cpu().numpy()

        # 4つのサブプロット作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 分布関数f(x,v)のヒートマップ
        im1 = ax1.imshow(f_cpu.T, aspect='auto', origin='lower', 
                     extent=[x_cpu[0], x_cpu[-1], v_cpu[0], v_cpu[-1]],
                     cmap='viridis')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Velocity v')
        ax1.set_title('Distribution Function f(x,v)')
        plt.colorbar(im1, ax=ax1)

        # 2. 密度分布
        ax2.plot(x_cpu, n_cpu, 'b-', linewidth=2)
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Density n')
        ax2.set_title('Density Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. 速度分布
        ax3.plot(x_cpu, u_cpu, 'r-', linewidth=2)
        ax3.set_xlabel('Position x')
        ax3.set_ylabel('Mean Velocity u')
        ax3.set_title('Velocity Distribution')
        ax3.grid(True, alpha=0.3)

        # 4. 温度分布
        ax4.plot(x_cpu, T_cpu, 'g-', linewidth=2)
        ax4.set_xlabel('Position x')
        ax4.set_ylabel('Temperature T')
        ax4.set_title('Temperature Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 統計情報表示
        print(f"Density: mean={n_cpu.mean():.4f}, min={n_cpu.min():.4f}, max={n_cpu.max():.4f}")
        print(f"Velocity: mean={u_cpu.mean():.4f}, min={u_cpu.min():.4f}, max={u_cpu.max():.4f}")
        print(f"Temperature: mean={T_cpu.mean():.4f}, min={T_cpu.min():.4f}, max={T_cpu.max():.4f}")

    #GIF作成メソッド
    def create_gif(self, filename='bgk_simulation.gif', fps=2):
        """GIF作成（バグ修正版）"""
        if not self.animation_data:
            print("No animation data to create GIF")
            return

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            import io

            print(f"Creating GIF with {len(self.animation_data)} frames...")

            frames = []
            x_cpu = self.x.cpu().numpy()
            v_cpu = self.v.cpu().numpy()

            # カラーマップの範囲を事前計算
            f_max = max([np.max(d['f']) for d in self.animation_data])
            n_max = max([np.max(d['n']) for d in self.animation_data]) * 1.1
            u_max = max([np.max(np.abs(d['u'])) for d in self.animation_data]) * 1.1
            T_max = max([np.max(d['T']) for d in self.animation_data]) * 1.1

            for i, data in enumerate(self.animation_data):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                # 1. 分布関数
                im1 = ax1.imshow(data['f'].T, aspect='auto', origin='lower',
                                extent=[x_cpu[0], x_cpu[-1], v_cpu[0], v_cpu[-1]],
                                cmap='viridis', vmin=0, vmax=f_max)
                ax1.set_xlabel('Position x')
                ax1.set_ylabel('Velocity v')
                ax1.set_title(f'Distribution f(x,v) at t={data["time"]:.3f}')

                # 2. 密度
                ax2.plot(x_cpu, data['n'], 'b-', linewidth=2)
                ax2.set_xlabel('Position x')
                ax2.set_ylabel('Density n')
                ax2.set_title('Density')
                ax2.set_ylim([0, n_max])
                ax2.grid(True, alpha=0.3)

                # 3. 速度
                ax3.plot(x_cpu, data['u'], 'r-', linewidth=2)
                ax3.set_xlabel('Position x')
                ax3.set_ylabel('Velocity u')
                ax3.set_title('Velocity')
                ax3.set_ylim([-u_max, u_max])
                ax3.grid(True, alpha=0.3)

                # 4. 温度
                ax4.plot(x_cpu, data['T'], 'g-', linewidth=2)
                ax4.set_xlabel('Position x')
                ax4.set_ylabel('Temperature T')
                ax4.set_title('Temperature')
                ax4.set_ylim([0, T_max])
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()

                # PNG画像として保存（修正版）
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                buf.seek(0)

                # 画像データを完全にメモリにコピー
                img = Image.open(buf)
                img_copy = img.copy()  # ← 重要：コピーを作成
                frames.append(img_copy)

                # リソース解放
                plt.close(fig)
                img.close()
                buf.close()

                if (i + 1) % 3 == 0:
                    print(f"  Frame {i+1}/{len(self.animation_data)} processed")

            # GIF保存
            if frames:
                frames[0].save(filename, save_all=True, append_images=frames[1:], 
                               duration=int(1000/fps), loop=0)
                print(f"GIF saved: '{filename}' ({len(frames)} frames)")
            else:
                print("No frames to save")

        except ImportError:
            print("PIL required for GIF creation: pip install pillow")
        except Exception as e:
            print(f"GIF creation failed: {e}")
            import traceback
            traceback.print_exc()  # デバッグ用

    # ベンチマーク結果plotメソッド
    def plot_benchmark_results(
        self,
        bench_results: dict[str, Union[str, dict[int, dict[str, Any]]]],
        error_dict: dict[int, dict[str, dict[str, float]]],
        fname_moment: str = "moments.png",
        fname_error: str = "errors.png",
        logscale: bool = True,
        show_plots: bool = False,
        ) -> dict[str, Any]:
        """
        ベンチマーク結果を可視化する（横軸＝格子点数）。

        Parameters
        ----------
        bench_results : dict
            run_benchmark の戻り値。キーは格子点数（nx / nv）。
        error_dict : dict
            compute_error の戻り値。
        fname_moment : str
            モーメント比較図の保存名。
        fname_error : str
            誤差収束図の保存名。
        logscale : bool
            True: log‐log プロット、False: 線形軸。
        show_plots : bool
            True なら画面にも表示。

        Returns
        -------
        dict
            convergence_orders, 保存ファイル名などの情報。
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        import warnings

        # --------------------------------------------------------
        # 入力検査
        # --------------------------------------------------------
        if not bench_results:
            raise ValueError("bench_results が空です")
        if not error_dict:
            raise ValueError("error_dict が空です")

        benchmark_type = bench_results.get("bench_type", "spatial")
        results = {k: v for k, v in bench_results.items() if k != "bench_type"}

        if len(results) < 2:
            raise ValueError(f"格子レベルが {len(results)} 個しかありません（最低 2 個必要）")

        # 参照解（最細格子）はキーが最大のものとする
        ref_key = max(results.keys())

        # --------------------------------------------------------
        # モーメント比較図
        # --------------------------------------------------------
        tol_colors = [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ]
        line_styles = ["-", "--", "-.", ":"]
        color_cycle = cycle(tol_colors)
        style_cycle = cycle(line_styles)

        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

        grid_keys = [ref_key] + [k for k in sorted(results.keys(), reverse=True) if k != ref_key]

        legend_handles, legend_labels = [], []

        for grid_key in grid_keys:
            color = next(color_cycle)
            linestyle = next(style_cycle)
            lw = 3 if grid_key == ref_key else 2

            res = results[grid_key]
            x = res["x"]

            if benchmark_type == "spatial":
                grid_info = f"nx={grid_key} (dx={res.get('dx', 1.0/grid_key):.3f})"
            else:
                grid_info = f"nv={grid_key} (dv={res.get('dv', 10.0/grid_key):.3f})"

            if grid_key == ref_key:
                grid_info += " (REF)"

            ln, = axes1[0].plot(x, res["n"], color=color, linestyle=linestyle, lw=lw, label=grid_info)
            axes1[1].plot(x, res["u"], color=color, linestyle=linestyle, lw=lw)
            axes1[2].plot(x, res["T"], color=color, linestyle=linestyle, lw=lw)

            legend_handles.append(ln)
            legend_labels.append(grid_info)

        titles = ["Density Distribution", "Velocity Distribution", "Temperature Distribution"]
        ylabels = ["Density n", "Mean Velocity u", "Temperature T"]

        for ax, title, ylabel in zip(axes1, titles, ylabels):
            ax.set_xlabel("Position x")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        fig1.legend(legend_handles, legend_labels, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig1.savefig(fname_moment, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        if show_plots:
            plt.show()
        plt.close(fig1)

        # --------------------------------------------------------
        # 誤差収束図（横軸＝格子点数）
        # --------------------------------------------------------
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        # 参照解以外のキー（粗 → 細）
        test_keys = sorted([k for k in error_dict.keys() if k != ref_key])

        # 横軸データ：格子点数
        grid_counts = [int(k) for k in test_keys]
        if benchmark_type == "spatial":
            x_label = "Grid points nx"
        else:
            x_label = "Grid points nv"

        variables = ["f", "n", "u", "T"]
        norms = ["L1", "L2", "Linf"]
        var_colors = {"f": "#1f77b4", "n": "#2ca02c", "u": "#d62728", "T": "#9467bd"}
        markers = ["o", "s", "^", "D"]

        convergence_orders: dict[str, dict[str, float]] = {}

        for norm_idx, norm in enumerate(norms):
            ax = axes2[norm_idx]
            convergence_orders[norm] = {}

            for var_idx, var in enumerate(variables):
                errors = []
                counts = []

                for i, key in enumerate(test_keys):
                    try:
                        val = error_dict[key][norm][var]
                        if val <= 0 or np.isnan(val) or np.isinf(val):
                            warnings.warn(f"無効なエラー値をスキップ: {var} {norm} grid={key} error={val}")
                            continue
                        errors.append(val)
                        counts.append(grid_counts[i])
                    except (KeyError, TypeError):
                        warnings.warn(f"エラー取得失敗: {var} {norm} grid={key}")
                        continue

                if len(errors) < 2:
                    convergence_orders[norm][var] = np.nan
                    continue

                # 粗 → 細 でソート
                idx = np.argsort(counts)
                counts_sorted = np.array(counts)[idx]
                errs_sorted = np.array(errors)[idx]

                # 収束次数 p:  error ≈ C * N^{-p}
                slope = np.polyfit(np.log(counts_sorted), np.log(errs_sorted), 1)[0]
                p_mean = -slope
                convergence_orders[norm][var] = p_mean

                label = f"{var} (p̅={p_mean:.2f})"
                plot = ax.loglog if logscale else ax.plot
                plot(
                    counts_sorted,
                    errs_sorted,
                    marker=markers[var_idx],
                    color=var_colors[var],
                    linewidth=2,
                    markersize=8,
                    label=label,
                )

            ax.set_xlabel(f"{x_label}  (coarse → fine)")
            ax.set_ylabel(f"Error ({norm.replace('Linf', 'L∞')})")
            ax.set_title(f"{norm.replace('Linf', 'L∞')} Norm Convergence")
            ax.grid(True, alpha=0.3)
            ax.legend()
            if logscale:
                ax.set_xscale("log")
                ax.set_yscale("log")

        plt.tight_layout()
        fig2.savefig(fname_error, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        if show_plots:
            plt.show()
        plt.close(fig2)

        print(f"モーメント比較図を保存: {fname_moment}")
        print(f"誤差収束図を保存: {fname_error}")
        if show_plots:
            print("図を画面に表示しました")

        return {
            "convergence_orders": convergence_orders,
            "figures_saved": [fname_moment, fname_error],
            "benchmark_type": benchmark_type,
            "ref_grid": ref_key,
        }

    # ベンチマーク結果の分布関数プロット
    def plot_distribution_heatmaps(
        self,
        bench_results: dict,
        show_plots: bool = True,
        save_individual: bool = False,
        fname_base: str = "distribution_heatmaps"
        ) -> dict:
        """
        f(x,v) と |f − f_ref| をヒートマップで可視化。
        補間は compute_error と同じく nearest。引数と戻り値は従来どおり。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from scipy.interpolate import interp1d

        # ────────────── 入力チェック ──────────────
        if not bench_results:
            raise ValueError("ベンチマーク結果が空です")

        bench_type = bench_results["bench_type"]
        bench_dict = {k: v for k, v in bench_results.items() if isinstance(k, int)}
        if not bench_dict:
            raise ValueError("数値格子キーが見つかりません")

        ref_key     = max(bench_dict.keys())
        ref_result  = bench_dict[ref_key]
        sorted_keys = sorted(bench_dict.keys())
        n_grids     = len(sorted_keys)

        # ────────────── 補間と誤差計算 ──────────────
        def _nearest_interp(ref_arr, ref_axis, tgt_axis):
            return interp1d(ref_axis, ref_arr,
                            kind="nearest", assume_sorted=True)(tgt_axis)

        def _get_error(coarse_res):
            """粗格子に合わせた |f − f_ref|"""
            if bench_type == "spatial":
                ref_f = np.zeros_like(coarse_res["f"])
                for v_idx in range(len(coarse_res["v"])):
                    ref_f[:, v_idx] = _nearest_interp(
                        ref_result["f"][:, v_idx],
                        ref_result["x"], coarse_res["x"])
            else:  # velocity
                ref_f = np.zeros_like(coarse_res["f"])
                for x_idx in range(len(coarse_res["x"])):
                    ref_f[x_idx, :] = _nearest_interp(
                        ref_result["f"][x_idx, :],
                        ref_result["v"], coarse_res["v"])
            return np.abs(coarse_res["f"] - ref_f)

        # ────────────── 保存ファイル管理 ──────────────
        saved_files = []

        # ─────────────────────────────────────────────
        # ① 個別保存モード
        # ─────────────────────────────────────────────
        if save_individual:
            for key in sorted_keys:
                res  = bench_dict[key]
                f    = np.asarray(res["f"])
                err  = _get_error(res)

                fig, axes = plt.subplots(2, 1, figsize=(8, 10),
                                         constrained_layout=True)

                im0 = axes[0].imshow(f, origin="lower", aspect="auto",
                                     cmap="cividis")
                axes[0].set_title(f"f(x,v) – Grid {key}")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(err, origin="lower", aspect="auto",
                                     cmap="magma",
                                     norm=Normalize(vmin=0, vmax=err.max()))
                axes[1].set_title(r"|f − f$_{\mathrm{ref}}$|")
                fig.colorbar(im1, ax=axes[1])

                for ax in axes:
                    ax.set_xlabel("v-index")
                    ax.set_ylabel("x-index")

                if show_plots:
                    plt.show()
                fname = f"{fname_base}_grid_{key}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(fname)

        # ─────────────────────────────────────────────
        # ② 統合保存モード（3 列固定）
        # ─────────────────────────────────────────────
        else:
            # サブプロット行列サイズ
            n_cols  = 3
            n_rows  = 2 * int(np.ceil(n_grids / n_cols))   # f と err で 2 倍

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(12, 3 * n_rows),
                constrained_layout=True
            )

            # f 用カラースケール（共有）
            f_all  = np.concatenate(
                [np.asarray(bench_dict[k]["f"]).ravel() for k in sorted_keys]
            )
            f_norm = Normalize(vmin=f_all.min(), vmax=f_all.max())

            # err 用カラースケール（共有）
            err_all  = np.concatenate(
                [_get_error(bench_dict[k]).ravel() for k in sorted_keys]
            )
            err_norm = Normalize(vmin=0, vmax=err_all.max())

            # --------------------- 描画ループ ---------------------
            for idx, key in enumerate(sorted_keys):
                pair_row = (idx // n_cols) * 2     # f 用行
                col      = idx % n_cols            # 列

                res  = bench_dict[key]
                f    = np.asarray(res["f"])
                err  = _get_error(res)

                # f(x,v)
                ax_f = axes[pair_row, col]
                im_f = ax_f.imshow(f, origin="lower", aspect="auto",
                                   cmap="cividis", norm=f_norm)
                ax_f.set_title(f"f – N={key}")
                ax_f.set_xlabel("v")
                ax_f.set_ylabel("x")

                # |f − f_ref|
                ax_e = axes[pair_row + 1, col]
                im_e = ax_e.imshow(err, origin="lower", aspect="auto",
                                   cmap="magma", norm=err_norm)
                ax_e.set_title("|f − f_ref|")
                ax_e.set_xlabel("v")
                ax_e.set_ylabel("x")

            # カラーバー（左: f, 右: 誤差）を 1 本ずつ
            fig.colorbar(im_f, ax=axes[0::2, :].ravel().tolist(),
                         location="left", shrink=0.6, pad=0.02)
            fig.colorbar(im_e, ax=axes[1::2, :].ravel().tolist(),
                         location="right", shrink=0.6, pad=0.02)

            if show_plots:
                plt.show()

            fname = f"{fname_base}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(fname)

        print("=== ヒートマップ保存完了 ===")
        for i, f in enumerate(saved_files, 1):
            print(f"{i}. {f}")

        return {"saved_files": saved_files, "grid_keys": sorted_keys}

    # ベンチマーク結果の分布関数インタラクティブ3次元プロット
    def plot_distribution_interactive(
        self,
        bench_results: dict,
        keys: list[int] | None = None,          # ← 追加：描画対象の格子キー
        show_plots: bool = True,
        save_html: bool = False,
        fname_base: str = "distribution_interactive"
        ) -> dict:
        """
        指定された格子キーについて
        * f(x,v) の 3D サーフェス
        * |f − f_ref| の 3D サーフェス
        を横並びで表示・保存する。

        Notes
        -----
        - `keys` が None の場合は最細格子（max key）のみ描画
        - 誤差は `bench_type` を参照し、compute_error と同じ最近接補間で計算
        """
        import numpy as np
        from matplotlib.colors import Normalize
        from scipy.interpolate import interp1d
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly が必要です: pip install plotly")

        # ─── 入力チェック ───
        if not bench_results:
            raise ValueError("ベンチマーク結果が空です")

        bench_type = bench_results["bench_type"]
        bench_dict = {k: v for k, v in bench_results.items() if isinstance(k, int)}
        if not bench_dict:
            raise ValueError("数値格子キーが見つかりません")

        ref_key    = max(bench_dict.keys())
        ref_result = bench_dict[ref_key]

        # 対象キー決定
        if keys is None:
            target_keys = [ref_key]           # デフォルト: 最細格子
        else:
            target_keys = [k for k in keys if k in bench_dict]
            if not target_keys:
                raise ValueError("指定 keys が bench_results に存在しません")

        # ─── 関数定義 ───
        def _nearest_interp(ref_arr, ref_axis, tgt_axis):
            return interp1d(ref_axis, ref_arr,
                            kind="nearest", assume_sorted=True)(tgt_axis)

        def _get_error(coarse_res):
            """粗格子に合わせた |f − f_ref|"""
            if bench_type == "spatial":
                ref_f = np.zeros_like(coarse_res["f"])
                for v_idx in range(len(coarse_res["v"])):
                    ref_f[:, v_idx] = _nearest_interp(
                        ref_result["f"][:, v_idx],
                        ref_result["x"], coarse_res["x"])
            else:  # velocity
                ref_f = np.zeros_like(coarse_res["f"])
                for x_idx in range(len(coarse_res["x"])):
                    ref_f[x_idx, :] = _nearest_interp(
                        ref_result["f"][x_idx, :],
                        ref_result["v"], coarse_res["v"])
            return np.abs(coarse_res["f"] - ref_f)

        # f, err のグローバル正規化
        all_f   = np.concatenate([bench_dict[k]["f"].ravel() for k in target_keys])
        f_min   = all_f.min()
        f_max   = all_f.max()
        all_err = np.concatenate([_get_error(bench_dict[k]).ravel() for k in target_keys])
        err_max = all_err.max()

        saved_files = []

        # ─── 描画ループ ───
        for key in target_keys:
            res  = bench_dict[key]
            f    = np.asarray(res["f"])
            err  = _get_error(res)
            x    = np.asarray(res["x"])
            v    = np.asarray(res["v"])

            # Figure with 2 surfaces side-by-side
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                column_widths=[0.5, 0.5],
                horizontal_spacing=0.05,
                subplot_titles=("f(x,v)", "|f − f_ref|")
            )

            # f surface
            fig.add_trace(
                go.Surface(
                    x=x, y=v, z=f.T,
                    colorscale="Viridis",
                    cmin=f_min, cmax=f_max,
                    showscale=False,
                    hovertemplate='x:%{x:.3f}<br>v:%{y:.3f}<br>f:%{z:.6f}<extra></extra>'
                ),
                row=1, col=1
            )

            # error surface
            fig.add_trace(
                go.Surface(
                    x=x, y=v, z=err.T,
                    colorscale="Magma",
                    cmin=0, cmax=err_max,
                    showscale=False,
                    hovertemplate='x:%{x:.3f}<br>v:%{y:.3f}<br>|err|:%{z:.6e}<extra></extra>'
                ),
                row=1, col=2
            )

            # レイアウト
            fig.update_layout(
                title=f"Grid {key} – nx×nv = {f.shape[0]}×{f.shape[1]}",
                scene=dict(
                    xaxis_title='x', yaxis_title='v', zaxis_title='f',
                    aspectmode='cube'
                ),
                scene2=dict(
                    xaxis_title='x', yaxis_title='v', zaxis_title='|err|',
                    aspectmode='cube'
                ),
                width=1100, height=550,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # 保存
            if save_html:
                fname = f"{fname_base}_grid{key}.html"
                fig.write_html(fname)
                saved_files.append(fname)
                print(f"Grid {key}: {fname} を保存")

            if show_plots:
                fig.show()

    # ベンチマーク結果の保存・読み込みユーティリティ
    def save_benchmark_results(self, bench_results: dict | None = None, filename: str = "benchmark_results.pkl") -> str:
        """ベンチマーク結果 dict を pickle 形式で保存

        Parameters
        ----------
        bench_results : dict | None
            run_benchmark の戻り値。None のときは self.benchmark_results を使用。
        filename : str, default "benchmark_results.pkl"
            保存先ファイル名。

        Returns
        -------
        str
            実際に保存したファイルパス。
        """
        import pickle, os, datetime, platform, torch

        if bench_results is None:
            if not hasattr(self, "benchmark_results"):
                raise ValueError("bench_results が None で、self.benchmark_results も存在しません")
            bench_results = self.benchmark_results

        # 追加メタデータ
        meta = {
            "saved_at"      : datetime.datetime.now().isoformat(timespec="seconds"),
            "hostname"      : platform.node(),
            "torch_version" : torch.__version__,
            "device"        : str(self.device),
            "dtype"         : str(self.dtype),
            "solver"        : self.solver,
        }
        data_to_save = {
            "meta" : meta,
            "results" : bench_results,
        }

        # ディレクトリが無ければ作成
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(filename) / (1024 ** 2)
        print(f"ベンチマーク結果を保存: {filename}  ({size_mb:.2f} MB)")
        return filename

    @staticmethod
    def load_benchmark_results(filename: str) -> dict:
        """pickle 形式のベンチマーク結果ファイルを読み込む"""
        import pickle, os
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"ベンチマーク結果を読み込み: {filename}  (meta: {data.get('meta', {})})")
        return data.get("results", {})

    @staticmethod
    def list_benchmark_files(pattern: str = "*.pkl", directory: str | None = None) -> list[str]:
        """指定ディレクトリ内の pickle ファイル一覧を返す"""
        import glob, os
        directory = directory or os.getcwd()
        files = glob.glob(os.path.join(directory, pattern))
        for f in files:
            print(f" - {f} ({os.path.getsize(f)/(1024**2):.2f} MB)")
        return files

# クラスの統合
class BGK1D(BGK1DBase, BGK1DPlotMixin):
    pass
"""
ここまでがシミュレーションクラスやでぇ！
"""

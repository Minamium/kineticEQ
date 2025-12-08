import torch
import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Any, Union
import time

from .progress_bar import get_progress_bar, progress_write

class BGK2D2V_core:
    """
    2D2V-BGK数値計算coreクラス
    """
    def __init__(self,
                 
                 # 数値計算パラメータ
                 nx: int,   
                 ny: int,   
                 nv_x: int, 
                 nv_y: int, 
                 dt: float,
                 T_total: float,

                 # 物理スケール
                 Lx: float,
                 Ly: float,
                 vx_max: float,
                 vy_max: float,

                 # ハイパーパラメータ
                 tau_tilde: float,

                 # 初期状態のモーメント場配列(二次元)
                 n: torch.Tensor,
                 u_x: torch.Tensor,
                 u_y: torch.Tensor,
                 T: torch.Tensor,

                 # スキーム選択
                 explicit_advection_scheme: str = "MUSCL2",

                 # デバイス, 精度設定
                 device: str = 'cuda',
                 dtype: str ='float64',

                 # 表示, 保存設定
                 use_tqdm: bool = True,
                 flag_record_state: bool = False,
                 ):

        # 数値計算設定
        # 時間ステップ計算
        self.T_total = T_total
        self.dt = dt
        self.nt = int(T_total / dt)
        # 格子サイズ
        self.nx = nx
        self.ny = ny
        self.nv_x = nv_x
        self.nv_y = nv_y

        # 物理サイズ
        self.Lx = Lx
        self.Ly = Ly
        self.vx_max = vx_max
        self.vy_max = vy_max

        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dv_x = 2.0 * vx_max / nv_x
        self.dv_y = 2.0 * vy_max / nv_y

        # tau_tilde
        self.tau_tilde = tau_tilde

        # スキーム設定
        self.explicit_advection_scheme = explicit_advection_scheme

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

        # 使用するプログレスバー
        self.use_tqdm = use_tqdm
        self.flag_record_state = flag_record_state

        self.animation_data = []

        # 初期モーメント場をデバイスに移動
        self.n = n.to(device=self.device, dtype=self.dtype).clone()
        self.u_x = u_x.to(device=self.device, dtype=self.dtype).clone()
        self.u_y = u_y.to(device=self.device, dtype=self.dtype).clone()
        self.T = T.to(device=self.device, dtype=self.dtype).clone()

        # インスタンス情報表示
        print(f"  nx={self.nx}, ny={self.ny}, nv_x={self.nv_x}, nv_y={self.nv_y}")
        print(f"  dx={self.dx}, dy={self.dy}, dv_x={self.dv_x}, dv_y={self.dv_y}")
        print(f"  dt={self.dt}, T_total={self.T_total}, nt={self.nt}")
        print(f"  tau_tilde={self.tau_tilde}")
        print(f"  explicit_advection_scheme={self.explicit_advection_scheme}")
        print(f"  dtype: {self.dtype_str}")

        # デバイス情報
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"  device: {self.device}, GPU name: {gpu_name}, Total memory: {gpu_memory:.2f} GB")
        else:
            print(f"  device: {self.device}, CPU")


    # シミュレーション実行メソッド
    def run_simulation(self):
        """
        シミュレーションを実行するメインメソッド
        """
        # 配列を確保
        print(" --- Array allocation ---")
        self.Array_allocation()
        
        # 初期条件を設定
        print(" --- Initialize fields ---")
        self.initialize_fields()

        # CFL条件確認
        self.check_cfl_condition()
        
        # シミュレーションループ
        print(" --- run simulation ---")

        progress_interval = max(1, self.nt // 50)

        # 初期状態を記録
        if self.flag_record_state:
            self.compute_moments()
            self._record_state(0.0)

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Explicit Evolution", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                # 時間発展ステップ
                self._explicit_update()

                # 配列交換
                self.f, self.f_new = self.f_new, self.f

                # 新しい f からモーメントを再計算
                self.compute_moments()

                if self.flag_record_state:
                    if step % progress_interval == 0:
                        # 状態記録
                        self._record_state((step + 1) * self.dt)

                # プログレスバー更新
                pbar.update(1)

        print("--- run simulation complete, Result is saved in self.f, self.u_x, self.u_y, self.T ---")
    
    #状態記録メソッド
    def _record_state(self, time):
        # CPUに転送してnumpy配列として保存
        n_cpu = self.n.cpu().numpy()
        ux_cpu = self.u_x.cpu().numpy()
        uy_cpu = self.u_y.cpu().numpy()
        T_cpu = self.T.cpu().numpy()

        state_data = {
            'time': time,
            'n': n_cpu,
            'u_x': ux_cpu,
            'u_y': uy_cpu,
            'T': T_cpu,
        }
        self.animation_data.append(state_data)
            
    
    # 配列確保メソッド
    def Array_allocation(self):
        """
        CUDAデバイス上に必要なテンソルを確保する。
        """

        # 分布関数配列
        self.f = torch.zeros((self.nx, self.ny, self.nv_x, self.nv_y),dtype=self.dtype, device=self.device)
        self.f_new = torch.zeros_like(self.f)

        # 座標系配列
        # x, y
        i = torch.arange(self.nx, dtype=self.dtype, device=self.device)
        j = torch.arange(self.ny, dtype=self.dtype, device=self.device)
        self.x = (i + 0.5) * self.dx
        self.y = (j + 0.5) * self.dy

        # v_x, v_y
        p = torch.arange(self.nv_x, dtype=self.dtype, device=self.device)
        q = torch.arange(self.nv_y, dtype=self.dtype, device=self.device)
        self.vx = -self.vx_max + (p + 0.5) * self.dv_x
        self.vy = -self.vy_max + (q + 0.5) * self.dv_y

    # 初期条件適用メソッド
    def initialize_fields(self):
        """
        初期条件として、__init__ で渡されたモーメント場
        (self.n, self.u_x, self.u_y, self.T) から
        Maxwell 分布を構成し、self.f に設定する。
        """
        if self.f is None:
            raise RuntimeError("Call Array_allocation() before initialize_fields().")

        self.f = self.Maxwellian(self.n, self.u_x, self.u_y, self.T)
        return self.f

    # モーメント計算メソッド
    def compute_moments(self):
        """
        分布関数 self.f からモーメント (n, u_x, u_y, T) を計算し、
        self.n, self.u_x, self.u_y, self.T を更新する。
        戻り値として (n, u_x, u_y, T) も返す。

        self.f : shape = (nx, ny, nv_x, nv_y)
        """
        if self.f is None:
            raise RuntimeError("Distribution f is not allocated. Call Array_allocation() and initialize_fields().")

        f = self.f
        dv_x = self.dv_x
        dv_y = self.dv_y

        # 速度格子
        vx = self.vx.view(1, 1, self.nv_x, 1)   # (1,1,nv_x,1)
        vy = self.vy.view(1, 1, 1, self.nv_y)   # (1,1,1,nv_y)

        # --- 密度 n ---
        # sum over v_x, v_y
        n = torch.sum(f, dim=(2, 3)) * dv_x * dv_y      # (nx, ny)

        # --- 運動量密度 ---
        nux = torch.sum(vx * f, dim=(2, 3)) * dv_x * dv_y
        nuy = torch.sum(vy * f, dim=(2, 3)) * dv_x * dv_y

        # --- 流速 ---
        n_safe = n + 1e-30
        u_x = nux / n_safe
        u_y = nuy / n_safe

        # --- エネルギー密度 ---
        vx2 = vx * vx
        vy2 = vy * vy
        U = 0.5 * torch.sum((vx2 + vy2) * f, dim=(2, 3)) * dv_x * dv_y

        # --- 温度 (2D velocity) ---
        T = U / n_safe - 0.5 * (u_x * u_x + u_y * u_y)
        T = torch.clamp(T, min=1e-30)

        # メンバに反映
        self.n = n
        self.u_x = u_x
        self.u_y = u_y
        self.T = T

        return n, u_x, u_y, T

    # マクスウェル分布計算メソッド
    def Maxwellian(self,
                   n: torch.Tensor,
                   u_x: torch.Tensor,
                   u_y: torch.Tensor,
                   T: torch.Tensor) -> torch.Tensor:
        """
        マクスウェル分布を計算
        入力:
            n, u_x, u_y, T : shape = (nx, ny)
        出力:
            f_M : shape = (nx, ny, nv_x, nv_y)
        """
        # 速度格子を (1,1,nv_x,1), (1,1,1,nv_y) に拡張
        vx = self.vx.view(1, 1, self.nv_x, 1)
        vy = self.vy.view(1, 1, 1, self.nv_y)

        # 空間モーメントを (nx,ny,1,1) に拡張
        n_ = n.view(self.nx, self.ny, 1, 1)
        ux_ = u_x.view(self.nx, self.ny, 1, 1)
        uy_ = u_y.view(self.nx, self.ny, 1, 1)
        T_ = T.view(self.nx, self.ny, 1, 1)

        # ガウス核
        coeff = n_ / (2.0 * math.pi * T_)
        exponent = -((vx - ux_) ** 2 + (vy - uy_) ** 2) / (2.0 * T_)

        f_M = coeff * torch.exp(exponent)
        return f_M

    def check_cfl_condition(self):
        """
        CFL条件を確認する
        """
        C_x = self.vx_max * self.dt / self.dx
        C_y = self.vy_max * self.dt / self.dy
        print(f"  CFL numbers: Cx={C_x:.6f}, Cy={C_y:.6f}, Cx+Cy={C_x+C_y:.6f}")
        if C_x + C_y > 1.0:
            print("Warning: CFL condition violated: Cx + Cy =", C_x + C_y)
        else:
            print("  CFL condition satisfied.")

    # 陽な輸送項計算
    def _compute_explicit_advection(self):
        """
        陽な任意の空間差分スキームを選択し呼びだす
        """
        if self.explicit_advection_scheme == "upwind":
            adv = self._compute_explicit_advection_upwind()
        elif self.explicit_advection_scheme == "MUSCL2":
            adv = self._compute_explicit_advection_muscl2()
        else:
            raise ValueError(f"Unknown explicit advection scheme: {self.explicit_advection_scheme}")
        
        return adv

    #  minmod リミタ（MUSCL 用）
    def _minmod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        minmod(a,b) = 0.5 * (sign(a)+sign(b)) * min(|a|,|b|)
        a,b: 同じ shape の Tensor
        """
        s = torch.sign(a) + torch.sign(b)
        return 0.5 * s * torch.min(torch.abs(a), torch.abs(b))
    
    # 二次精度 MUSCL
    def _compute_explicit_advection_muscl2(self) -> torch.Tensor:
        """
        2次精度 MUSCL-TVD upwind による
        v_x ∂_x f + v_y ∂_y f の評価。
        境界条件は x, y ともに周期境界（torch.roll）を仮定。

        戻り値:
            adv : same shape as self.f, (nx, ny, nv_x, nv_y)
        """
        if self.f is None:
            raise RuntimeError("Distribution f is not allocated.")

        f  = self.f
        dx = self.dx
        dy = self.dy

        # 速度格子をブロードキャスト形状に
        vx = self.vx.view(1, 1, self.nv_x, 1)  # (1,1,nv_x,1)
        vy = self.vy.view(1, 1, 1, self.nv_y)  # (1,1,1,nv_y)

        # ========================================
        # x 方向 MUSCL
        # ========================================
        # 周期境界で隣接セルを取得
        f_ip1 = torch.roll(f, shifts=-1, dims=0)  # i+1
        f_im1 = torch.roll(f, shifts= 1, dims=0)  # i-1

        # 傾き（slope）を minmod で制限
        delta_plus_x  = f_ip1 - f       # f_{i+1} - f_i
        delta_minus_x = f - f_im1       # f_i - f_{i-1}
        slope_x = self._minmod(delta_minus_x, delta_plus_x)  # (nx,ny,nvx,nvy)

        # 界面 i+1/2 における左・右状態
        # 左: f_L(i+1/2) = f_i + 0.5 * slope_i
        # 右: f_R(i+1/2) = f_{i+1} - 0.5 * slope_{i+1}
        f_L = f + 0.5 * slope_x
        slope_x_ip1 = torch.roll(slope_x, shifts=-1, dims=0)
        f_R = f_ip1 - 0.5 * slope_x_ip1

        # upwind flux: F_{i+1/2} = v_x * f_up
        vx_pos_mask = (vx > 0.0)
        f_up_x = torch.where(vx_pos_mask, f_L, f_R)
        F_iphalf = vx * f_up_x                   # (nx,ny,nvx,nvy)

        # F_{i-1/2} は roll で生成
        F_imhalf = torch.roll(F_iphalf, shifts=1, dims=0)
        adv_x = (F_iphalf - F_imhalf) / dx       # ≈ ∂_x (v_x f)

        # ========================================
        # y 方向 MUSCL
        # ========================================
        f_jp1 = torch.roll(f, shifts=-1, dims=1)   # j+1
        f_jm1 = torch.roll(f, shifts= 1, dims=1)   # j-1

        delta_plus_y  = f_jp1 - f
        delta_minus_y = f - f_jm1
        slope_y = self._minmod(delta_minus_y, delta_plus_y)

        # 界面 j+1/2 における下・上状態
        # 下: f_D(j+1/2) = f_j + 0.5 * slope_j
        # 上: f_U(j+1/2) = f_{j+1} - 0.5 * slope_{j+1}
        f_D = f + 0.5 * slope_y
        slope_y_jp1 = torch.roll(slope_y, shifts=-1, dims=1)
        f_U = f_jp1 - 0.5 * slope_y_jp1

        vy_pos_mask = (vy > 0.0)
        f_up_y = torch.where(vy_pos_mask, f_D, f_U)
        G_jphalf = vy * f_up_y                    # (nx,ny,nvx,nvy)

        G_jmhalf = torch.roll(G_jphalf, shifts=1, dims=1)
        adv_y = (G_jphalf - G_jmhalf) / dy        # ≈ ∂_y (v_y f)

        # ========================================
        # 合成
        # ========================================
        adv = adv_x + adv_y
        return adv

    # 一次風上差分
    def _compute_explicit_advection_upwind(self):
        """
        v_x ∂_x f + v_y ∂_y f を 1次風上差分で評価する。
        境界条件は x, y ともに周期境界を仮定（torch.roll 使用）。
        戻り値:
            adv : same shape as self.f, (nx, ny, nv_x, nv_y)
        """
        if self.f is None:
            raise RuntimeError("Distribution f is not allocated.")

        f = self.f
        dx = self.dx
        dy = self.dy

        # 速度格子をブロードキャスト形状に
        vx = self.vx.view(1, 1, self.nv_x, 1)  # (1,1,nv_x,1)
        vy = self.vy.view(1, 1, 1, self.nv_y)  # (1,1,1,nv_y)

        # -------- x 方向の 1次風上差分 --------
        f_ip1 = torch.roll(f, shifts=-1, dims=0)  # i+1
        f_im1 = torch.roll(f, shifts=1,  dims=0)  # i-1

        dfdx_pos = (f - f_im1) / dx
        dfdx_neg = (f_ip1 - f) / dx

        # v_x の符号で切り替え
        # vx > 0 → 上流は i-1, vx < 0 → 上流は i+1
        vx_pos_mask = (vx > 0.0)
        dfdx = torch.where(vx_pos_mask, dfdx_pos, dfdx_neg)

        adv_x = vx * dfdx  # v_x ∂_x f

        # -------- y 方向の 1次風上差分 --------
        f_jp1 = torch.roll(f, shifts=-1, dims=1)  # j+1
        f_jm1 = torch.roll(f, shifts=1,  dims=1)  # j-1

        dfdy_pos = (f - f_jm1) / dy
        dfdy_neg = (f_jp1 - f) / dy

        vy_pos_mask = (vy > 0.0)
        dfdy = torch.where(vy_pos_mask, dfdy_pos, dfdy_neg)

        adv_y = vy * dfdy  # v_y ∂_y f

        # 合成
        adv = adv_x + adv_y
        return adv

    # 陽な衝突項の計算
    def _compute_explicit_collision(self, n, u_x, u_y, T):
        """
        BGK 衝突項 C(f) = (f_M - f) / tau を計算する。
        tau = tau_tilde / (n * sqrt(T))
        """
        if self.f is None:
            raise RuntimeError("Distribution f is not allocated.")

        # 緩和時間 tau(n,T)
        tau = self.tau_tilde / (n * torch.sqrt(T))
        tau = torch.clamp(tau, min=1e-30)  # 安全のための下限

        # Maxwell 分布
        f_M = self.Maxwellian(n, u_x, u_y, T)

        # 衝突項
        # tau: (nx,ny) → (nx,ny,1,1) に拡張
        tau_ = tau.view(self.nx, self.ny, 1, 1)
        f_coll = (f_M - self.f) / tau_

        return f_coll
    
    # 陽解法による時間発展メソッド
    def _explicit_update(self):
        """
        Explicitによる分布関数の時間発展
        """
        # モーメント計算
        n, u_x, u_y, T = self.compute_moments()

        # 輸送項計算
        f_adv = self._compute_explicit_advection()
        
        # 衝突項計算
        f_coll = self._compute_explicit_collision(n, u_x, u_y, T)
        
        # 分布関数の更新
        self.f_new = self.f + self.dt * ( (-f_adv) + f_coll)

        return self.f_new

    
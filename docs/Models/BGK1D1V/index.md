---
title: BGK1D1V
parent: Models
nav_order: 21
has_children: true
lang: ja
---

# BGK1D1V

BGK1D1V は、一次元空間 `x` と一次元速度 `v` をもつ BGK 方程式

$$
\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x}
= \frac{1}{\tau}(f_M - f)
$$

を対象とする。実装では緩和時間を直接 `\tau` として与えるのではなく、

$$
\tau(x,t) = \frac{\tilde{\tau}}{n(x,t)\sqrt{T(x,t)}}
$$

として局所密度・温度から再構成する構成を採る。

## Maxwell 分布とモーメント

Maxwell 分布は

$$
f_M(x,v) = \frac{n(x)}{\sqrt{2\pi T(x)}}
\exp\left(-\frac{(v-u(x))^2}{2T(x)}\right)
$$

であり、コード上の基本モーメントは矩形則により

$$
n = \int f\,dv,\qquad
\nu = \int vf\,dv,\qquad
\nu = n u,
$$

であり、実装では

$$
u = \frac{\nu}{n}, \qquad
T = \frac{1}{n}\int v^2 f\,dv - u^2
$$

という順序で計算される。`calculate_moments()` は `f.sum(dim=1) * dv` による直接積分であり、特別な高次求積は用いていない。

## 離散化と state 表現

`State1D1V` は主として以下のテンソル群を保持する。

- `f`, `f_tmp`, `f_m`: 現在の分布、作業配列、Maxwell 分布
- `x`, `v`: 一様格子点
- `n`, `u`, `T`: マクロモーメント
- `dx`, `dv`: 格子幅
- `pos_mask`, `neg_mask`, `v_coeff`, `k0`: 移流と kernel 呼び出しのためのキャッシュ

`allocate_state_1d1v()` は一様格子を生成し、`set_initial_condition()` は区分一定な `(n,u,T)` から初期 Maxwell 分布を構成する。最後の区間のみ右閉区間として扱うため、端点の重複割当ては生じない。

## 境界条件の取り扱い

現行 BGK1D1V 実装では、境界処理は `bgk1d_apply_BC.py` に集約されている。端点行 `i=0` と `i=N_x-1` は内部未知量ではなく、境界 trace として扱う。

- `fixed_maxwellian`: 初期端点行に入っている Maxwellian 分布を固定境界として保持する。
- `reflective`: 隣接内部行の速度反転により鏡面反射境界を作る。

理論式と現行の離散式は [境界条件](boundary_conditions.md) に整理している。拡散反射は同ページに設計式を記述しているが、実装はまだ入っていない。

## backend とスキーム対応

| スキーム | `torch` | `cuda_kernel` | `cpu_kernel` | 実装の要点 |
|---|---|---|---|---|
| `explicit` | 対応 | 対応 | 未登録 | 一次風上 + 局所衝突 |
| `implicit` | 未登録 | 対応 | 対応 | Picard + batched tridiagonal |
| `holo` | 未登録 | 対応 | 未登録 | HO/LO 分離反復 |

## explicit スキーム

`bgk1d_explicit_torch.py` は、以下の手順を一ステップとして実装する。

1. `calculate_moments()` により `n,u,T` を計算する。
2. `maxwellian()` により `f_M` を再構成する。
3. `\tau = \tilde{\tau}/(n\sqrt{T})` を計算する。
4. `_compute_streaming_upwind()` により一次風上差分の移流項を構成する。
5. `(f_M - f)/\tau` を衝突項として加える。
6. 内点のみを陽 Euler で更新し、境界セルを固定する。

CFL 条件は `bgk1d_check_CFL()` により検査され、`v_max dt / dx <= 0.9` を要求する。

`bgk1d_explicit_cuda_kernel.py` では、モーメント計算、Maxwell 再構成、移流・衝突計算の主要部を `explicit_step` カーネルへ委譲するため、Python 側は boundary fix と swap を担当するだけである。

## implicit スキーム

implicit 系は `bgk1d_implicit_cuda_kernel.py` および `bgk1d_implicit_cpu_kernel.py` によって実装される。両者のアルゴリズム構造は同一であり、差異は fused operator と三重対角ソルバの backend にある。

### Picard 反復の流れ

1. 現在の `state.f` を `ws.fz` にコピーする。
2. `moments_n_nu_T()` により初期モーメント `W=(n, \nu, T)` を計算する。
3. 必要に応じて external `W` 注入、または CNN warm-start により内部セルの初期モーメントを補正する。
4. `build_system_from_moments()` により、各速度に対する三重対角係数 `(dl, dd, du, B)` を構築する。
5. `gtsv_strided_inplace()` により batched 三重対角系を解く。
6. 更新後の `f` から再びモーメントを計算し、収束判定を行う。
7. 未収束であれば、`W` をそのまま更新するか、Anderson Acceleration で再加速して次反復へ進む。

内部 unknown は `x` 方向の内点 `1:-1` であり、速度方向はバッチ次元として扱われる。このため、計算の本質は「各速度に対し独立な一次元三重対角系をまとめて解く」構造である。

### 収束判定

`conv_type="f"` では分布関数そのものに対して

$$
\max \frac{|f^{(k+1)} - f^{(k)}|}{\text{abs\_tol} + \text{picard\_tol}\max(|f^{(k+1)}|, |f^{(k)}|)} \le 1
$$

を用いる。`conv_type="w"` では `n`, `\nu`, `T` に対して同様の正規化残差を計算し、その最大値で判定する。

### Anderson Acceleration

`aa_enable=True` の場合、加速対象は分布関数ではなく `W=(n,\nu,T)` である。したがって、三重対角 solve 後に得られた `W_new` を履歴バッファへ格納し、既往 residual の線形結合として新しい予測点を形成する。CUDA 版は `implicit_AA` 拡張、CPU 版は `bgk1d_aa_cpu` を用いる。

### dtype と device

- CUDA 版 fused kernel は `float64` を要求する。
- CPU 版 fused kernel と CPU gtsv も `float64` を要求する。
- CUDA 版 Anderson Acceleration 自体は `float32/float64` を受け付けるが、implicit 主経路が `float64` 制約を持つため、実運用上は `float64` に揃う。

## CNN warm-start

implicit stepper では `moments_cnn_modelpath` が与えられると checkpoint を読み込み、モーメント初期値を CNN で予測できる。

### 入力構成

`build_model_input()` は以下を入力チャンネルとして構成する。

- `input_temporal_mode="none"`: `(n, u\text{ or }\nu, T, \log_{10}dt, \log_{10}\tilde{\tau})`
- `input_temporal_mode="prev_delta"`: 上記に加えて直前状態・直前増分・`has_prev`

`input_state_type="nut"` では第 2 チャンネルが `u`、`input_state_type="nnuT"` では `\nu` になる。

### 出力の解釈

checkpoint metadata の `delta_type` に従い、出力は

- `dw`: `(\Delta n, \Delta u, \Delta T)`
- `dnu`: `(\Delta n, \Delta \nu, \Delta T)`

として解釈される。推論後は `n` と `T` に floor を課し、内部セル `1:-1` のみを更新する。さらに `warm_delta_weight_mode="w_grad"` では、空間勾配に基づく soft gate で flat 領域の増分を減衰できる。

## holo スキーム

HOLO 実装は `bgk1d_holo_cuda_kernel.py` に存在し、`theta=0.5` の Crank-Nicolson 型分割を前提に組まれている。

### 外側 HO 反復

外側反復では、現行分布 `f_z` から

- 界面フラックスモーメント `S_1, S_2, S_3`
- 熱流束相当量 `Q_i = \tfrac12 \int (v-u_i)^3 f_i(v)\,dv`

を計算し、高次情報に基づく整合項 `Y_I_terms` を構成する。

### 内側 LO 反復

LO 系では未知量を

$$
W = (n, \nu, U)
$$

とし、3x3 block-tridiagonal 系を Picard 反復で解く。LO 反復の終了後、

$$
u = \frac{\nu}{n}, \qquad U = \frac12 n(u^2 + T)
$$

から `u`, `T`, `\tau` を再構成し、これに基づく Maxwell 分布 `f_M` を得る。

### 分布関数更新

最後に、LO 側で得た `\tau_lo` と `fM_lo` を用いて、分布関数側の三重対角系を `gtsv_strided` で解く。HO 残差は `f` の正規化差、LO 残差は `W` の正規化差で判定する。

## 実務上の読み替え

- 陽解法は参照実装として明快であるが、stiff 領域では implicit / holo が主役となる。
- implicit は「モーメントから係数を組み立てる反復型線形 solve」と読むと理解しやすい。
- holo は「高次分布情報で低次保存則を補正しつつ、低次系で閉じたモーメント解を作る二層反復」である。

---
title: Core Layer
parent: Implementations
nav_order: 33
lang: ja
---

# Core Layer

## 対象ディレクトリ

- `src/kineticEQ/core/states/`
- `src/kineticEQ/core/schemes/`

## states

### `state_1d.py`

`State1D1V` は BGK1D の主 state であり、以下を保持する。

- 分布関数: `f`, `f_tmp`, `f_m`
- 物理格子: `x`, `v`, `dx`, `dv`
- マクロ量: `n`, `u`, `T`
- shared cache: `v_col`, `inv_sqrt_2pi`, `pos_mask`, `neg_mask`, `v_coeff`, `k0`

### `state_2d2v.py`

`State2D2V` は `f`, `f_new`, `x`, `y`, `vx`, `vy` を保持する。state 自体は構築可能だが、stepper 側の完成度が不足している。

### `states/registry.py`

`Model` ごとに state builder を切り替える registry である。

- `BGK1D1V -> state_1d.build_state`
- `BGK2D2V -> state_2d2v.build_state`

## schemes registry

`core/schemes/registry.py` は `(model, scheme, backend)` をキーに stepper builder を登録する。

現行登録:

- `BGK1D1V / explicit / torch`
- `BGK1D1V / explicit / cuda_kernel`
- `BGK1D1V / implicit / cpu_kernel`
- `BGK1D1V / implicit / cuda_kernel`
- `BGK1D1V / holo / cuda_kernel`
- `BGK2D2V / explicit / torch`（ただし stepper 本体は TODO）

## BGK1D explicit

### `bgk1d_explicit_torch.py`

- モーメント計算
- Maxwell 分布再構成
- 一次風上差分移流項
- 局所衝突項
- 境界固定
- CFL 条件検査

### `bgk1d_explicit_cuda_kernel.py`

- `load_explicit_fused()` による JIT ロード
- fused kernel 呼び出し
- Python 側では主として境界固定と NaN/Inf 監視を担当

## BGK1D implicit

### 共通構造

- workspace: `bgk1d_implicit_ws.py`
- 収束判定: `bgk1d_conv_util.py`
- CNN warm-start: `bgk1d_momentCNN_util.py`
- 初期条件設定: `bgk1d_set_initial_condition.py`

### `bgk1d_implicit_cuda_kernel.py`

- Picard 反復
- `moments_n_nu_T()` と `build_system_from_moments()` を fused CUDA kernel で実行
- `gtsv_strided_inplace()` による batched 三重対角 solve
- optional AA (`implicit_AA`)
- optional CNN warm-start
- `conv_type="f"` / `"w"` の収束判定

### `bgk1d_implicit_cpu_kernel.py`

- アルゴリズム構造は CUDA 版と同一
- fused 演算子と gtsv を CPU C++ extension に置き換え
- `device='cpu'` を明示要求

## BGK1D holo

### `bgk1d_holo_cuda_kernel.py`

- HO 外側反復
- `S_1,S_2,S_3` と `Q` の計算
- `Y_I_terms` による整合項の構成
- LO 側 block-tridiagonal Picard solve
- HO 側三重対角 solve
- `benchlog` として HO/LO 両残差を報告

## BGK2D2V

### `bgk2d2v_explicit_torch.py`

- 現状では `step()` が空実装であり、registry 上の記述は将来実装の placeholder とみなすべきである。

---
title: Core Layer
parent: Implementations
nav_order: 33
---

# Core Layer

## 対象ディレクトリ

- `src/kineticEQ/core/states/`
- `src/kineticEQ/core/schemes/`

## states

- `state_1d.py`
  - `State1D1V`:
    - `f`, `f_tmp`, `f_m`
    - `x`, `v`, `n`, `u`, `T`, `dx`, `dv`
    - cache: `pos_mask`, `neg_mask`, `v_coeff`, `k0`, `inv_sqrt_2pi`
- `state_2d2v.py`
  - `State2D2V`（`f`, `f_new`, `x,y,vx,vy`）

## schemes registry

`core/schemes/registry.py` に `(model, scheme, backend)` -> builder を登録。

現行登録:

- `BGK1D1V/explicit/torch`
- `BGK1D1V/explicit/cuda_kernel`
- `BGK1D1V/implicit/cuda_kernel`
- `BGK1D1V/holo/cuda_kernel`
- `BGK2D2V/explicit/torch`（実体は TODO）

## BGK1D explicit

- モーメント計算 -> Maxwellian -> 上流差分 streaming -> collision
- 境界セル固定（左右コピー）
- `bgk1d_check_CFL` で `vmax*dt/dx <= 0.9` を要求

## BGK1D implicit

- Picard 反復
- `moments_n_nu_T` と `build_system_from_moments` を CUDA 拡張で構築
- cuSPARSE `gtsv_strided_inplace` で batched 三重対角を解く
- 収束判定 `conv_type`:
  - `f`: 分布関数基準
  - `w`: moments 基準
- optional:
  - Anderson Acceleration (`aa_*`)
  - CNN warmstart (`moments_cnn_modelpath`)

## BGK1D holo

- HO/LO 外内反復
- `Y_I_terms`（整合項）
- LO 側は 3x3 block-tridiag を PCR カーネルで解く
- HO 側の分布更新は gtsv で解く

---
title: CUDA Kernels
parent: Implementations
grand_parent: Japanese
nav_order: 34
lang: ja
---

# CUDA Kernels

## 対象ディレクトリ

- `src/kineticEQ/cuda_kernel/`

## `compile.py`

JIT ローダとして以下が提供される。

- `load_explicit_fused`
- `load_implicit_fused`
- `load_gtsv`
- `load_lo_blocktridiag`
- `load_implicit_AA`

## BGK1D1V kernels

### `explicit_fused/`

- explicit 一歩更新を fused 化
- binding は CUDA tensor かつ `torch.float64` を要求

### `implicit_fused/`

- `moments_n_nu_T`
- `build_system_from_moments`
- binding は CUDA tensor かつ `torch.float64` を要求

### `gtsv/`

- cuSPARSE `gtsv2StridedBatch` wrapper
- batched 三重対角系の solve / workspace サイズ問い合わせを提供

### `lo_blocktridiag/`

- HOLO の LO 系で用いる 3x3 block-tridiagonal solver
- CUDA tensor 上で block system を解く

### `implicit_AA/`

- implicit Picard 用 Anderson Acceleration
- `float32` / `float64` の両方を受理するが、主経路が `float64` を要求するため実運用上は `float64` が標準
- cuSOLVER を用いた Gram 系解法と補助 kernel を含む

## BGK2D2V kernels

- `BGK2D2V/explicit_2d2v/` に CUDA 実装が存在する
- ただし現行 registry へは接続されていないため、トップレベル API では未使用である

## 実務上の注意

- fused binding は dtype・device・contiguous 条件を厳密に検査する。
- JIT ビルドは `TORCH_EXTENSIONS_DIR` の影響を受ける。
- 運用上の安定性は `float64` 前提で評価すべきである。

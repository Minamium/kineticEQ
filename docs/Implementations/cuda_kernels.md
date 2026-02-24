---
title: CUDA Kernels
parent: Implementations
nav_order: 34
---

# CUDA Kernels

## 対象ディレクトリ

- `src/kineticEQ/cuda_kernel/`

## `compile.py`

JIT ローダ:

- `load_explicit_fused`
- `load_implicit_fused`
- `load_gtsv`
- `load_lo_blocktridiag`
- `load_implicit_AA`

## BGK1D1V kernels

- `explicit_fused/`
  - fused explicit 1step
  - binding は `torch.float64` のみ受理
- `implicit_fused/`
  - moments と tri-diagonal 系の係数構築
  - binding は `torch.float64` のみ受理
- `gtsv/`
  - cuSPARSE `gtsv2StridedBatch` wrapper
- `lo_blocktridiag/`
  - 3x3 block tri-diagonal PCR solver
- `implicit_AA/`
  - implicit 用 Anderson Acceleration
  - cuSOLVER potrf/potrs + CUDA kernels

## BGK2D2V kernels

- `BGK2D2V/explicit_2d2v/` に CUDA 実装あり
- ただし Engine registry へ接続されていないため、現行 API では未使用

## dtype 注意

BGK1D の fused binding は `kDouble` チェックを持つため、`cuda_kernel` 経路では実質 `float64` 運用。

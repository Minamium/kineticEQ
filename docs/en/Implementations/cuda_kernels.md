---
title: English CUDA Kernels
nav_title: CUDA Kernels
parent: English Implementations
grand_parent: English
nav_order: 34
lang: en
---

# CUDA Kernels

## Target directory

- `src/kineticEQ/cuda_kernel/`

## Loaders in `compile.py`

- `load_explicit_fused`
- `load_implicit_fused`
- `load_gtsv`
- `load_lo_blocktridiag`
- `load_implicit_AA`

## BGK1D1V kernels

### `explicit_fused/`

A fused explicit one-step update. The binding requires CUDA tensors with `torch.float64`.

### `implicit_fused/`

Provides `moments_n_nu_T` and `build_system_from_moments`. The binding also requires CUDA tensors with `torch.float64`.

### `gtsv/`

Wraps cuSPARSE batched tridiagonal solves.

### `lo_blocktridiag/`

Solves the 3x3 block-tridiagonal systems used by the HOLO low-order layer.

### `implicit_AA/`

Implements Anderson acceleration for implicit Picard iterations.

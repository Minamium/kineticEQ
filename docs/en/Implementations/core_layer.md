---
title: English Core Layer
nav_title: Core Layer
parent: English Implementations
grand_parent: English
nav_order: 33
lang: en
---

# Core Layer

## Target directories

- `src/kineticEQ/core/states/`
- `src/kineticEQ/core/schemes/`

## States

### `state_1d.py`

`State1D1V` stores the distribution buffers, grids, macroscopic moments, and shared transport/kernel caches.

### `state_2d2v.py`

`State2D2V` allocates the 2D2V distribution and its grids, but the corresponding stepper path is incomplete.

### `states/registry.py`

Maps each `Model` enum to the corresponding state builder.

## Stepper registry

The current registry includes:

- `BGK1D1V / explicit / torch`
- `BGK1D1V / explicit / cuda_kernel`
- `BGK1D1V / implicit / cpu_kernel`
- `BGK1D1V / implicit / cuda_kernel`
- `BGK1D1V / holo / cuda_kernel`
- `BGK2D2V / explicit / torch` (placeholder stepper)

## BGK1D explicit

- pure PyTorch reference implementation in `bgk1d_explicit_torch.py`
- fused CUDA implementation in `bgk1d_explicit_cuda_kernel.py`

## BGK1D implicit

- shared workspace in `bgk1d_implicit_ws.py`
- convergence logic in `bgk1d_conv_util.py`
- CNN warm-start utilities in `bgk1d_momentCNN_util.py`
- CUDA and CPU backends with the same high-level Picard structure

## BGK1D holo

`bgk1d_holo_cuda_kernel.py` implements the HO/LO coupled iteration, including the consistency terms and the LO block-tridiagonal solve.

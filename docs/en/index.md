---
title: English
nav_title: English
nav_order: 20
has_children: true
permalink: /en/
lang: en
---

# kineticEQ Documentation

`kineticEQ` is a numerical library for BGK-type kinetic equations. It combines a high-level Python interface built around `Config` and `Engine` with optimized C++/CUDA extensions for the BGK1D production path. This documentation is written against the current `src/kineticEQ` implementation rather than the legacy notebook workflow.

## Current implementation status

| Model | Scheme | `torch` | `cuda_kernel` | `cpu_kernel` | Status |
|---|---|---|---|---|---|
| `BGK1D1V` | `explicit` | supported | supported | not registered | runnable |
| `BGK1D1V` | `implicit` | not registered | supported | supported | runnable |
| `BGK1D1V` | `holo` | not registered | supported | not registered | runnable |
| `BGK2D2V` | `explicit` | registered in the registry | not registered | not registered | incomplete `Engine` path |

Notes:

- The BGK1D `cuda_kernel` path requires `float64` in the fused explicit and implicit bindings.
- The BGK1D `cpu_kernel` path is currently implicit-only and also requires `float64`.
- `BGK2D2V` can allocate state, but the `Engine` path is not operational because the model config lacks `scheme_params` and the stepper itself is unfinished.

## Numerical highlights

- **BGK1D explicit**: direct velocity-space quadrature, Maxwellian reconstruction, first-order upwind transport, and local collision relaxation.
- **BGK1D implicit**: Picard iteration with batched tridiagonal solves, optionally accelerated by Anderson acceleration and CNN warm-starts.
- **BGK1D holo**: coupled HO/LO iterations, with a block-tridiagonal LO solve and a tridiagonal HO distribution update.
- **CNN warm-start**: a 1D CNN predicts moment increments and provides a better initialization for implicit Picard iterations.

## Documentation map

- [Getting Started](getting-started/index.md)
- [Models](Models/index.md)
- [API Reference](api_Reference/index.md)
- [Implementations](Implementations/index.md)

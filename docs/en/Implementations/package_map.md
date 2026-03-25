---
title: English Package Map
nav_title: Package Map
parent: English Implementations
nav_order: 31
lang: en
---

# Package Map

| Directory | Role |
|---|---|
| `api/` | `Config`, `Engine`, `Result`, logging setup |
| `params/` | model/scheme dataclasses and default registries |
| `core/states/` | state allocation and model-specific builders |
| `core/schemes/` | stepper implementations and the `(model, scheme, backend)` registry |
| `cuda_kernel/` | CUDA/C++ extensions and JIT loaders |
| `cpu_kernel/` | CPU-side C++ extensions for the implicit path |
| `CNN/BGK1D1V/` | warm-start data generation, training, and evaluation |
| `analysis/` | benchmark, convergence, and cross-scheme tools |
| `plotting/` | state visualization |
| `utillib/` | device, progress, and formatting helpers |
| `_legacy/` | legacy class-based implementations |
| `tests/` | smoke and utility-level checks |

## Stable execution paths

- `BGK1D1V + explicit + torch`
- `BGK1D1V + explicit + cuda_kernel`
- `BGK1D1V + implicit + cuda_kernel`
- `BGK1D1V + implicit + cpu_kernel`
- `BGK1D1V + holo + cuda_kernel`

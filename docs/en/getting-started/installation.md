---
title: Installation
parent: English Getting Started
nav_order: 11
lang: en
---

# Installation

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- A CUDA-capable GPU and `nvcc` for `cuda_kernel`
- A C++17-capable host compiler for `cpu_kernel`

## Dependencies

The main dependencies declared by the project are:

- `torch>=2.0`
- `numpy`
- `scipy`
- `tqdm`
- `ninja`
- `setuptools`
- `wheel`
- `zarr`
- `numcodecs`

Visualization can be installed with the `viz` extra, which adds `matplotlib` and `plotly>=5`.

## Installation command

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e ".[viz]"
```

## Extension build model

Fast backends are loaded through `torch.utils.cpp_extension.load`, so the first invocation triggers JIT compilation.

### CUDA loaders

`src/kineticEQ/cuda_kernel/compile.py` exposes:

- `load_explicit_fused()`
- `load_implicit_fused()`
- `load_gtsv()`
- `load_lo_blocktridiag()`
- `load_implicit_AA()`

For BGK1D, the fused CUDA bindings require `torch.float64`, so practical use of `backend="cuda_kernel"` should assume `dtype="float64"`.

### CPU loaders

`src/kineticEQ/cpu_kernel/compile.py` exposes:

- `load_implicit_fused_cpu()`
- `load_gtsv_cpu()`

These CPU bindings also require `float64`, and they are currently available only for the implicit BGK1D path.

## Cache location

```bash
export TORCH_EXTENSIONS_DIR="/path/to/torch_extensions"
```

On HPC systems, it is often useful to trigger extension loading once on the login node before launching jobs.

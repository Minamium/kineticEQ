<p align="center">
  <img src="docs/assets/kineticEQ-logo-transparent.png" alt="kineticEQ" width="900">
</p>

<p align="center">
  <a href="https://minamium.github.io/kineticEQ/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-00b8c8?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-1f6feb?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-kernels-76b900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="GCC" src="https://img.shields.io/badge/GCC%2FG%2B%2B-%E2%89%A59%20recommended-004482?style=flat-square&logo=gnu&logoColor=white">
  <img alt="NVHPC" src="https://img.shields.io/badge/NVHPC-optional-76b900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-546a7b?style=flat-square">
  <img alt="Author" src="https://img.shields.io/badge/author-Minamium-0f2233?style=flat-square">
</p>

# kineticEQ

**GPU-accelerated solvers for the Boltzmann-BGK equation.**

kineticEQ is a Python library for numerical computation of the Boltzmann-BGK equation. It provides GPU computation through the `Config` / `Engine` API, using PyTorch and C++/CUDA kernels.

kineticEQ は，Boltzmann-BGK 方程式の数値計算を扱うための Python ライブラリ.`Config` / `Engine` API から PyTorch と C++/CUDA カーネルによる GPU 計算を利用を利用可能.

## Installation

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e ".[viz]"
```

CUDA backends require a CUDA-enabled PyTorch environment, CUDA Toolkit 12.x or newer, and `nvcc`.
GCC/G++ 9 or newer is recommended for JIT compilation. NVHPC may work when supported by the installed CUDA Toolkit, but the recommended configuration for kineticEQ CUDA extensions is `nvcc` with GCC/G++.

```bash
# Optional: adjust these values for your compiler and build-cache environment.
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export TORCH_EXTENSIONS_DIR=/path/to/your/torch_extensions
export MAX_JOBS=8
```

## Quick Start

```python
from kineticEQ import BGK1D, Config, Engine

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="torch",
    device="cpu",
    dtype="float64",
    use_tqdm=False,
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-6, T_total=5e-4),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
    ),
)

result = Engine(cfg).run()
```

Use `backend="cuda_kernel"` and `device="cuda"` for CUDA kernel runs.

## Plotting

Static snapshots are available through `plot_state`.

```python
from kineticEQ.plotting.bgk1d import plot_state

engine = Engine(cfg)
engine.run()
plot_state(engine.state, filename="state.png", output_dir="result")
```

GIF animation and conservation diagnostics are available through `animate_state_run`.

```python
from kineticEQ.plotting.bgk1d import animate_state_run

paths = animate_state_run(
    cfg,
    frames=80,
    filename="state.gif",
    output_dir="result",
    fps=10,
)
```

## Engine Arguments

| Argument | Where | Default | Description |
|---|---|---|---|
| `config` | `Engine` | required | `Config` object defining the simulation. |
| `apply_logging_flag` | `Engine` | `True` | Apply logging settings during initialization. |
| `model` | `Config` | `BGK1D1V` | Target model. Current production path is `BGK1D1V`. |
| `scheme` | `Config` | `explicit` | Time-evolution scheme: `explicit`, `implicit`, or `holo`. |
| `backend` | `Config` | `torch` | Compute backend: `torch`, `cuda_kernel`, or `cpu_kernel`. |
| `device` | `Config` | `cuda` | Execution device such as `cpu`, `cuda`, or `mps`. |
| `dtype` | `Config` | `float64` | Tensor precision. Kernel backends currently assume `float64`. |
| `log_level` | `Config` | `info` | Logging level: `debug`, `info`, `warning`, or `error`. |
| `model_cfg` | `Config` | `None` | Model-specific dataclass such as `BGK1D.ModelConfig`. |
| `use_tqdm` | `Config` | `true` | Show progress bars during time evolution. |

## Documentation

- Documentation: https://minamium.github.io/kineticEQ/
- [`docs/getting-started/`](https://minamium.github.io/kineticEQ/getting-started/)
- [`docs/Models/`](https://minamium.github.io/kineticEQ/Models/)
- [`docs/Implementations/`](https://minamium.github.io/kineticEQ/Implementations/)
- [`docs/api_Reference/`](https://minamium.github.io/kineticEQ/api_Reference/)
- [`docs/en/`](https://minamium.github.io/kineticEQ/en/)

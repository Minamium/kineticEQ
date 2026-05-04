---
title: Examples
parent: English Getting Started
nav_order: 13
lang: en
---

# Examples

## 1. BGK1D explicit with `torch`

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="torch",
    device="cpu",
    dtype="float64",
    use_tqdm="false",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-6, T_total=5e-4),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
    ),
)

Engine(cfg).run()
```

## 2. BGK1D implicit with `cpu_kernel`

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="implicit",
    backend="cpu_kernel",
    device="cpu",
    dtype="float64",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=257, nv=129, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-5, T_total=5e-3),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-7),
        scheme_params=BGK1D.implicit.Params(
            picard_iter=200,
            picard_tol=1e-6,
            abs_tol=1e-13,
            conv_type="w",
            aa_enable=True,
        ),
    ),
)

Engine(cfg).run()
```

## 3. BGK1D implicit with `cuda_kernel`, AA, and CNN warm-start

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="implicit",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=512, nv=256, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-5, T_total=5e-3),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-7),
        scheme_params=BGK1D.implicit.Params(
            picard_iter=400,
            picard_tol=1e-7,
            abs_tol=1e-13,
            conv_type="f",
            aa_enable=True,
            aa_m=6,
            moments_cnn_modelpath="/path/to/checkpoint.pt",
            warm_delta_weight_mode="w_grad",
        ),
    ),
)

Engine(cfg).run()
```

## 4. BGK1D holo with `cuda_kernel`

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="holo",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=257, nv=129, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=2e-5, T_total=2e-3),
        params=BGK1D.BGK1D1VParams(tau_tilde=1e-5),
        scheme_params=BGK1D.holo.Params(
            ho_iter=8,
            ho_tol=1e-5,
            lo_iter=16,
            lo_tol=1e-5,
            Con_Terms_do=True,
            flux_consistency_do=True,
        ),
    ),
)

Engine(cfg).run()
```

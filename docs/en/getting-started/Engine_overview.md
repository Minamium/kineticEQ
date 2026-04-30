---
title: Engine Overview
parent: English Getting Started
grand_parent: English
nav_order: 12
lang: en
---

# Engine Overview

The standard kineticEQ execution path has three layers: configuration through `Config`, state/stepper construction through `Engine`, and time integration through `run()`. `Engine` is not merely a loop wrapper; it also fills missing model configuration, resolves the registered stepper, configures logging, and validates the requested device.

## 1. Config

```python
from kineticEQ import Config

cfg = Config(
    model="BGK1D1V",
    scheme="implicit",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",
    log_level="info",
    use_tqdm="true",
    model_cfg=None,
)
```

### Normalized fields

| Field | Default | Accepted values in the current implementation |
|---|---:|---|
| `model` | `"BGK1D1V"` | `BGK1D1V`, `BGK2D2V`, aliases `bgk1d`, `bgk1d1v`, `bgk2d2v` |
| `scheme` | `"explicit"` | `explicit`, `implicit`, `holo`, aliases `exp`, `imp`, `hl` |
| `backend` | `"torch"` | `torch`, `cuda_kernel`, `cpu_kernel`, aliases `pytorch`, `cuda_backend`, `cpu_backend` |
| `device` | `"cuda"` | `cuda`, `cpu`, `mps` |
| `dtype` | `"float64"` | `float32`, `float64`, aliases `fp32`, `fp64` |
| `log_level` | `"info"` | `debug`, `info`, `warning`, `error` |
| `use_tqdm` | `"true"` | `true`, `false` |

## 2. Model configuration

If `model_cfg` is omitted, `Engine` requests a default dataclass from `params.registry.default_model_cfg()`. For BGK1D, a representative instance is:

```python
from kineticEQ import BGK1D

model_cfg = BGK1D.ModelConfig(
    grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
    time=BGK1D.TimeConfig(dt=5e-6, T_total=5e-4),
    params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
    initial=BGK1D.InitialCondition1D(),
    scheme_params=None,
)
```

`TimeConfig.n_steps` is computed as `ceil(T_total / dt)`.

## 3. Scheme parameters

If `model_cfg.scheme_params` is `None`, `Engine` requests a scheme-specific default.

### `explicit`

`BGK1D.explicit.Params` is an empty dataclass.

### `implicit`

`BGK1D.implicit.Params` contains Picard tolerances, convergence mode, Anderson-acceleration parameters, and CNN warm-start options.

### `holo`

`BGK1D.holo.Params` contains HO and LO iteration limits/tolerances together with consistency flags.

## 4. What `Engine` actually does

`Engine(config)` performs the following steps.

1. Fill `model_cfg` if it is missing and validate its type.
2. Fill `scheme_params` if they are missing.
3. Apply logging settings.
4. Validate the requested device through `resolve_device()`.
5. Allocate the state through `core.states.registry.build_state()`.
6. Build the stepper through `core.schemes.registry.build_stepper()`.

A subtle but important implementation detail is that initial conditions are applied inside the stepper builder, not in `Engine` itself.

## 5. Runtime loop

```python
from kineticEQ import Engine

engine = Engine(cfg)
result = engine.run()
```

`Engine.run()` advances the simulation for `n_steps` iterations and calls `self.stepper(step)` on each iteration. Implicit and holo steppers may attach a `benchlog` attribute, which is printed periodically during the run.

The top-level return type is currently a placeholder:

```python
Result(metrics=None, payload=None)
```

## 6. Practical limitations

- `BGK1D1V + explicit + torch` is the simplest reference implementation.
- `BGK1D1V + implicit + cpu_kernel` uses CPU-side C++ extensions.
- `BGK1D1V + cuda_kernel` should be treated as a `float64` path.
- `BGK2D2V` is not yet an operational Engine target.

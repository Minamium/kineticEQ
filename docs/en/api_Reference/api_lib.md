---
title: kineticEQ.api (English)
nav_title: kineticEQ.api
parent: English API Reference
grand_parent: English
nav_order: 1
lang: en
---

# API Reference

## Top-level exports

```python
from kineticEQ import Config, Engine, run, Result, BGK1D, BGK2D2V
```

## Config

```python
Config(
    model: str = "BGK1D1V",
    scheme: str = "explicit",
    backend: str = "torch",
    device: str = "cuda",
    dtype: str = "float64",
    log_level: str = "info",
    model_cfg: Any | None = None,
    use_tqdm: str = "true",
)
```

### Core fields

- `model`: `BGK1D1V` / `BGK2D2V`
- `scheme`: `explicit` / `implicit` / `holo`
- `backend`: `torch` / `cuda_kernel` / `cpu_kernel`
- `device`: `cuda` / `cpu` / `mps`
- `dtype`: `float32` / `float64`
- `model_cfg`: model-specific dataclass
- `use_tqdm`: `true` / `false`

### Aliases

- `model`: `bgk1d`, `bgk1d1v`, `bgk2d2v`
- `scheme`: `exp`, `imp`, `hl`
- `backend`: `pytorch`, `cuda_backend`, `cpu_backend`
- `dtype`: `fp32`, `fp64`

### Convenience accessors

- `model_name`
- `scheme_name`
- `backend_name`
- `dtype_name`
- `log_level_name`
- `use_tqdm_name`
- `use_tqdm_bool`
- `as_dict`

## Engine

```python
Engine(config: Config, apply_logging_flag: bool = True)
```

### Responsibilities

- fill or validate `model_cfg`
- fill default `scheme_params`
- apply logging
- validate the requested device
- build the state
- build the registered stepper

### `run`

```python
engine.run() -> Result
```

The runtime loop advances `model_cfg.time.n_steps` iterations. If the stepper exposes a `benchlog` attribute, it is periodically written to the progress output.

## `run`

```python
run(config: Config) -> Result
```

This is a thin shortcut for `Engine(config).run()`.

## `Result`

```python
@dataclass
class Result:
    metrics: dict[str, float] | None = None
    payload: dict[str, Any] | None = None
```

## BGK1D dataclasses

### `BGK1D.ModelConfig`

```python
BGK1D.ModelConfig(
    grid: Grid1D1V = Grid1D1V(),
    time: TimeConfig = TimeConfig(),
    params: BGK1D1VParams = BGK1D1VParams(),
    initial: InitialCondition1D = InitialCondition1D(),
    scheme_params: Any = None,
)
```

### `BGK1D.Grid1D1V`

- `nx: int = 124`
- `nv: int = 64`
- `Lx: float = 1.0`
- `v_max: float = 10.0`

### `BGK1D.TimeConfig`

- `dt: float = 5e-4`
- `T_total: float = 0.05`
- `n_steps = ceil(T_total / dt)`

### `BGK1D.BGK1D1VParams`

- `tau_tilde: float = 0.5`

### `BGK1D.InitialCondition1D`

- `initial_regions: tuple[Any, ...]`
- each element provides `x_range`, `n`, `u`, and `T`

### `BGK1D.implicit.Params`

Contains Picard tolerances, convergence mode, Anderson-acceleration settings, and CNN warm-start options.

### `BGK1D.holo.Params`

Contains HO/LO iteration counts and tolerances together with consistency flags.

## BGK2D2V dataclasses

### `BGK2D2V.Grid2D2V`

- `nx: int = 124`
- `ny: int = 124`
- `nx_v: int = 16`
- `ny_v: int = 16`
- `Lx: float = 1.0`
- `Ly: float = 1.0`
- `v_max: float = 10.0`

### `BGK2D2V.TimeConfig`

- `dt: float = 5e-3`
- `T_total: float = 0.05`
- `n_steps = int(T_total / dt)`

### `BGK2D2V.BGK2D2VParams`

- `tau_tilde: float = 0.5`

Note:

- the current `BGK2D2V.ModelConfig` does not define `scheme_params`, so the `Engine` path is not operational.

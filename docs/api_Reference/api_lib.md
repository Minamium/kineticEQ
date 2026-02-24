---
title: kineticEQ.api
parent: API Reference
nav_order: 1
---

# API リファレンス

## トップレベル

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

- `model`: `BGK1D1V` / `BGK2D2V`
- `scheme`: `explicit` / `implicit` / `holo` / `holo_nn`(予約)
- `backend`: `torch` / `cuda_kernel`

注記:
- `holo_nn` は stepper 未登録
- `BGK2D2V` は現行 Engine 経路では未対応

## Engine

```python
Engine(config: Config, apply_logging_flag: bool = True)
```

### Engine.run

```python
engine.run() -> Result
```

時間発展ループを実行。

## run

```python
run(config: Config) -> Result
```

`Engine(config).run()` のショートカット。

## Result

```python
@dataclass
class Result:
    metrics: dict[str, float] | None = None
    payload: dict[str, Any] | None = None
```

---

## パラメータ: BGK1D

### ModelConfig

```python
BGK1D.ModelConfig(
    grid: Grid1D1V = Grid1D1V(),
    time: TimeConfig = TimeConfig(),
    params: BGK1D1VParams = BGK1D1VParams(),
    initial: InitialCondition1D = InitialCondition1D(),
    scheme_params: Any = None,
)
```

### Grid1D1V

- `nx: int = 124`
- `nv: int = 64`
- `Lx: float = 1.0`
- `v_max: float = 10.0`

### TimeConfig

- `dt: float = 5e-4`
- `T_total: float = 0.05`
- `n_steps = ceil(T_total / dt)`

### BGK1D1VParams

- `tau_tilde: float = 0.5`

### InitialCondition1D

- `initial_regions: tuple[dict, ...]`
- 各 dict は `x_range`, `n`, `u`, `T`

### explicit.Params

- 追加パラメータなし

### implicit.Params

- `picard_iter: int = 16`
- `picard_tol: float = 1e-4`
- `abs_tol: float = 1e-16`
- `conv_type: str = "f"`
- `aa_enable: bool = False`
- `aa_m: int = 6`
- `aa_beta: float = 1.0`
- `aa_stride: int = 1`
- `aa_start_iter: int = 2`
- `aa_reg: float = 1e-10`
- `aa_alpha_max: float = 50.0`
- `moments_cnn_modelpath: str | None = None`

### holo.Params

- `ho_iter: int = 8`
- `ho_tol: float = 1e-4`
- `ho_abs_tol: float = 1e-12`
- `lo_iter: int = 16`
- `lo_tol: float = 1e-4`
- `lo_abs_tol: float = 1e-12`
- `Con_Terms_do: bool = True`
- `flux_consistency_do: bool = True`

---

## パラメータ: BGK2D2V

### Grid2D2V

- `nx: int = 124`
- `ny: int = 124`
- `nx_v: int = 16`
- `ny_v: int = 16`
- `Lx: float = 1.0`
- `Ly: float = 1.0`
- `v_max: float = 10.0`

### TimeConfig

- `dt: float = 5e-3`
- `T_total: float = 0.05`
- `n_steps = int(T_total / dt)`

### BGK2D2VParams

- `tau_tilde: float = 0.5`

注記:
- 現行 `BGK2D2V.ModelConfig` には `scheme_params` フィールドがないため、`Engine` 経路は未対応。

---

## analysis.BGK1D

実行関数:
- `run_benchmark(...)`
- `run_convergence_test(...)`
- `run_scheme_comparison_test(...)`

plotting:
- `plot_benchmark_results(...)`
- `plot_convergence_results(...)`
- `plot_cross_scheme_results(...)`
- `plot_timing_benchmark(...)`

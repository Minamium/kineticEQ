---
title: kineticEQ.api
parent: API Reference
grand_parent: Japanese
nav_order: 1
lang: ja
---

# API リファレンス

## トップレベル export

```python
from kineticEQ import Config, Engine, run, Result, BGK1D, BGK2D2V
```

`BGK1D` と `BGK2D2V` は、それぞれモデル固有の dataclass 群を束ねた namespace 的モジュールとして振る舞う。

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

### 主要フィールド

- `model`: `BGK1D1V` / `BGK2D2V`
- `scheme`: `explicit` / `implicit` / `holo`
- `backend`: `torch` / `cuda_kernel` / `cpu_kernel`
- `device`: `cuda` / `cpu` / `mps`
- `dtype`: `float32` / `float64`
- `model_cfg`: モデル固有 dataclass
- `use_tqdm`: `true` / `false`

### alias

- `model`: `bgk1d`, `bgk1d1v`, `bgk2d2v`
- `scheme`: `exp`, `imp`, `hl`
- `backend`: `pytorch`, `cuda_backend`, `cpu_backend`
- `dtype`: `fp32`, `fp64`

### アクセサ

`Config` は正規化済み Enum に対する以下のプロパティを持つ。

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

### 役割

- `model_cfg` の補完と型検証
- `scheme_params` の既定値補完
- logger 設定
- device 検証
- state 構築
- stepper 構築

### run

```python
engine.run() -> Result
```

`run()` は `model_cfg.time.n_steps` 回の時間発展を実行し、progress bar と benchlog を扱う。stepper が `benchlog` 属性を持つ場合、一定間隔で進捗出力に反映される。

## run

```python
run(config: Config) -> Result
```

`Engine(config).run()` の薄いショートカットである。

## Result

```python
@dataclass
class Result:
    metrics: dict[str, float] | None = None
    payload: dict[str, Any] | None = None
```

現時点ではコンテナのみが定義されており、トップレベル実行結果の自動格納は未実装である。

## BGK1D パラメータ

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
- 各要素は `x_range`, `n`, `u`, `T` を持つ dict もしくは同等属性をもつ object

### `BGK1D.explicit.Params`

- 追加パラメータなし

### `BGK1D.implicit.Params`

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
- `warm_enable: bool | None = None`
- `moments_cnn_modelpath: str | None = None`
- `warm_delta_weight_mode: str = "none"`
- `warm_delta_weight_floor: float = 0.2`
- `warm_delta_weight_center: float = 0.5`
- `warm_delta_weight_sharpness: float = 10.0`
- `warm_delta_weight_sigma: float = 3.0`
- `warm_delta_exclude_cells: int = 0`

### `BGK1D.holo.Params`

- `ho_iter: int = 8`
- `ho_tol: float = 1e-4`
- `ho_abs_tol: float = 1e-12`
- `lo_iter: int = 16`
- `lo_tol: float = 1e-4`
- `lo_abs_tol: float = 1e-12`
- `Con_Terms_do: bool = True`
- `flux_consistency_do: bool = True`

## BGK2D2V パラメータ

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

注記:

- 現行 `BGK2D2V.ModelConfig` には `scheme_params` が存在しないため、`Engine` 経路では未対応である。

## 実務上の注意

- `BGK1D + cuda_kernel` は fused binding の都合で `float64` を前提に考えるべきである。
- `BGK1D + cpu_kernel` は implicit 専用である。
- `BGK2D2V` は API 名としては公開されているが、実行可能モデルとしては扱わない方がよい。

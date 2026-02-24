---
title: Engine Overview
parent: Getting Started
nav_order: 12
---

# Engine Overview

kineticEQ の基本フローは `Config` -> `Engine` -> `run()`。

## 1. Config

```python
from kineticEQ import Config

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="torch",
    device="cpu",
    dtype="float64",
    log_level="info",
    use_tqdm="true",
    model_cfg=None,  # None なら model の既定値を補完
)
```

### Config フィールド

| フィールド | デフォルト | 備考 |
|---|---:|---|
| `model` | `"BGK1D1V"` | `BGK1D1V` / `BGK2D2V` |
| `scheme` | `"explicit"` | `explicit` / `implicit` / `holo` / `holo_nn`(予約) |
| `backend` | `"torch"` | `torch` / `cuda_kernel` |
| `device` | `"cuda"` | `resolve_device` で検証 |
| `dtype` | `"float64"` | `float32` / `float64` |
| `log_level` | `"info"` | `debug` / `info` / `warning` / `error` |
| `model_cfg` | `None` | model ごとの dataclass |
| `use_tqdm` | `"true"` | 文字列 Enum |

### エイリアス

- `model`: `bgk1d`, `bgk1d1v`, `bgk2d2v`
- `scheme`: `exp`, `imp`, `hl`, `hl_nn`
- `backend`: `pytorch`, `cuda_backend`
- `dtype`: `fp32`, `fp64`

## 2. model_cfg (BGK1D1V)

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

- `TimeConfig.n_steps = ceil(T_total / dt)`
- `initial_regions` は区分一定な `(x_range, n, u, T)` 定義

## 3. scheme_params

### explicit

- 追加パラメータなし (`BGK1D.explicit.Params` は空 dataclass)

### implicit

```python
BGK1D.implicit.Params(
    picard_iter=16,
    picard_tol=1e-4,
    abs_tol=1e-16,
    conv_type="f",            # "f" or "w"
    aa_enable=False,
    aa_m=6,
    aa_beta=1.0,
    aa_stride=1,
    aa_start_iter=2,
    aa_reg=1e-10,
    aa_alpha_max=50.0,
    moments_cnn_modelpath=None,
)
```

### holo

```python
BGK1D.holo.Params(
    ho_iter=8, ho_tol=1e-4, ho_abs_tol=1e-12,
    lo_iter=16, lo_tol=1e-4, lo_abs_tol=1e-12,
    Con_Terms_do=True, flux_consistency_do=True,
)
```

## 4. Engine 初期化

`Engine(config)` は次を実行する:

1. `model_cfg` 補完 (`params.registry.default_model_cfg`)
2. `scheme_params` 補完 (`params.registry.default_scheme_params`)
3. ロギング設定
4. デバイス検証
5. state 構築
6. stepper 構築

## 5. run

```python
from kineticEQ import Engine

engine = Engine(cfg)
result = engine.run()
```

- `run()` は `n_steps` 回ループで `self.stepper(step)` を呼ぶ
- return は `Result(metrics=None, payload=None)`（現時点で中身は未実装）

## 6. 注意点（実装準拠）

- `BGK1D1V + cuda_kernel` は `float64` 前提
- `scheme="holo_nn"` は Config 受理するが、現時点で stepper 未登録
- `BGK2D2V` は Engine 経路が未整備で、現状は実行不可

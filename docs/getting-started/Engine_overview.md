---
title: Engine Overview
parent: Getting Started
nav_order: 12
---

# Engine Overview

kineticEQ のシミュレーションは `Config` → `Engine` → `run()` の 3 ステップで実行する。

## 1. Config

`Config` はシミュレーション全体の設定を保持する frozen dataclass。

```python
from kineticEQ import Config

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",
    log_level="info",
    use_tqdm="true",
    model_cfg=...,        # モデル固有の設定（後述）
)
```

### Config パラメータ一覧

| パラメータ | 型 | デフォルト | 選択肢 |
|-----------|-----|-----------|--------|
| `model` | str | `"BGK1D1V"` | `"BGK1D1V"`, `"BGK2D2V"` |
| `scheme` | str | `"explicit"` | `"explicit"`, `"implicit"`, `"holo"` |
| `backend` | str | `"torch"` | `"torch"`, `"cuda_kernel"` |
| `device` | str | `"cuda"` | `"cuda"`, `"cpu"`, `"mps"` |
| `dtype` | str | `"float64"` | `"float32"`, `"float64"` |
| `log_level` | str | `"info"` | `"debug"`, `"info"`, `"warning"`, `"error"` |
| `use_tqdm` | str | `"true"` | `"true"`, `"false"` |
| `model_cfg` | ModelConfig | None | モデル固有の設定 |

`model_cfg` を省略すると、選択した `model` のデフォルト値が自動設定される。
`scheme_params` を省略すると、選択した `(model, scheme)` の組に対応するデフォルトが自動設定される。

### エイリアス

`model`, `scheme`, `backend` にはエイリアスが使える:

| パラメータ | エイリアス |
|-----------|-----------|
| `model` | `"bgk1d"` → `BGK1D1V`, `"bgk1d1v"` → `BGK1D1V`, `"bgk2d2v"` → `BGK2D2V` |
| `scheme` | `"exp"` → `explicit`, `"imp"` → `implicit`, `"hl"` → `holo` |
| `backend` | `"pytorch"` → `torch`, `"cuda_backend"` → `cuda_kernel` |
| `dtype` | `"fp32"` → `float32`, `"fp64"` → `float64` |

## 2. model_cfg (BGK1D1V の場合)

```python
from kineticEQ import BGK1D

model_cfg = BGK1D.ModelConfig(
    grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
    time=BGK1D.TimeConfig(dt=1e-6, T_total=0.01),
    params=BGK1D.BGK1D1VParams(tau_tilde=1e-5),
    initial=BGK1D.InitialCondition1D(
        initial_regions=(
            {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
            {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
        )
    ),
    scheme_params=...,  # スキーム固有パラメータ（省略可）
)
```

### Grid1D1V

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `nx` | int | 124 | 空間格子点数 |
| `nv` | int | 64 | 速度格子点数 |
| `Lx` | float | 1.0 | 空間領域長 |
| `v_max` | float | 10.0 | 速度空間の上限 ([-v_max, v_max]) |

### TimeConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `dt` | float | 5e-4 | 時間刻み幅 |
| `T_total` | float | 0.05 | 総シミュレーション時間 |

ステップ数は `n_steps = ceil(T_total / dt)` で自動計算される。

### BGK1D1VParams

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `tau_tilde` | float | 0.5 | 無次元化された緩和時間 |

## 3. scheme_params

スキームごとに固有のパラメータを持つ。`model_cfg.scheme_params` に設定する。
省略した場合は各スキームのデフォルト値が自動的に適用される。

### explicit

追加パラメータなし。

### implicit (Picard 反復)

```python
BGK1D.implicit.Params(
    picard_iter=16,
    picard_tol=1e-4,
    abs_tol=1e-16,
    moments_cnn_modelpath=None,  # CNN warmstart のチェックポイントパス
)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `picard_iter` | int | 16 | Picard 反復の最大回数 |
| `picard_tol` | float | 1e-4 | 相対収束判定閾値 |
| `abs_tol` | float | 1e-16 | 絶対収束判定閾値 |
| `moments_cnn_modelpath` | str or None | None | CNN warmstart モデルのパス。指定するとモーメント初期推定を CNN で生成する |

### holo (HOLO 法)

```python
BGK1D.holo.Params(
    ho_iter=8, ho_tol=1e-4, ho_abs_tol=1e-12,
    lo_iter=16, lo_tol=1e-4, lo_abs_tol=1e-12,
    Con_Terms_do=True, flux_consistency_do=True,
)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `ho_iter` | int | 8 | High-Order ループ最大反復数 |
| `ho_tol` | float | 1e-4 | HO 相対収束閾値 |
| `ho_abs_tol` | float | 1e-12 | HO 絶対収束閾値 |
| `lo_iter` | int | 16 | Low-Order ループ最大反復数 |
| `lo_tol` | float | 1e-4 | LO 相対収束閾値 |
| `lo_abs_tol` | float | 1e-12 | LO 絶対収束閾値 |
| `Con_Terms_do` | bool | True | 整合項 (consistency terms) を有効にする |
| `flux_consistency_do` | bool | True | フラックス整合を有効にする |

## 4. Engine の実行

```python
from kineticEQ import Engine

engine = Engine(cfg)
result = engine.run()   # -> Result
```

`Engine.__init__` では以下が実行される:
1. `model_cfg` / `scheme_params` の自動補完
2. ロギング設定
3. デバイス検証
4. State（分布関数・モーメント等のテンソル群）の構築
5. Stepper（時間発展関数）の構築

`engine.run()` は時間発展ループを実行し `Result` を返す。

## 5. Result

```python
@dataclass
class Result:
    metrics: dict[str, float] | None = None
    payload: dict[str, Any] | None = None
```

## 6. ワンライナー実行

`Config` の作成と `Engine.run()` をまとめて実行する関数:

```python
from kineticEQ import run

result = run(cfg)
```
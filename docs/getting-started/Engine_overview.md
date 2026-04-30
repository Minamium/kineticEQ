---
title: Engine Overview
parent: Getting Started
grand_parent: Japanese
nav_order: 12
lang: ja
---

# Engine Overview

`kineticEQ` の標準的な実行経路は `Config` の生成、`Engine` による state / stepper 構築、`run()` による時間発展、という三段階から成る。`Engine` は単なるループ実行器ではなく、モデル設定の補完、stepper の registry 解決、ロギング設定、device 検証を一括して担う。

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

### 正規化される項目

| フィールド | 既定値 | 現行実装で受理される代表値 |
|---|---:|---|
| `model` | `"BGK1D1V"` | `BGK1D1V`, `BGK2D2V`, alias: `bgk1d`, `bgk1d1v`, `bgk2d2v` |
| `scheme` | `"explicit"` | `explicit`, `implicit`, `holo`, alias: `exp`, `imp`, `hl` |
| `backend` | `"torch"` | `torch`, `cuda_kernel`, `cpu_kernel`, alias: `pytorch`, `cuda_backend`, `cpu_backend` |
| `device` | `"cuda"` | `cuda`, `cpu`, `mps` |
| `dtype` | `"float64"` | `float32`, `float64`, alias: `fp32`, `fp64` |
| `log_level` | `"info"` | `debug`, `info`, `warning`, `error` |
| `use_tqdm` | `"true"` | `true`, `false` |

`Config.__post_init__` はこれらを Enum に正規化し、`model_name`, `scheme_name`, `backend_name` などのアクセサを提供する。

## 2. model_cfg の補完

`Engine.__init__` は `model_cfg is None` のとき `params.registry.default_model_cfg()` を呼び出し、モデルごとの dataclass を補う。BGK1D の典型例は以下である。

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

`TimeConfig.n_steps` は `ceil(T_total / dt)` で計算される。

## 3. scheme_params の補完

`model_cfg.scheme_params is None` であれば、`params.registry.default_scheme_params(model, scheme)` が呼び出される。

### `explicit`

`BGK1D.explicit.Params` は空 dataclass であり、追加設定は無い。

### `implicit`

```python
BGK1D.implicit.Params(
    picard_iter=16,
    picard_tol=1e-4,
    abs_tol=1e-16,
    conv_type="f",
    aa_enable=False,
    aa_m=6,
    aa_beta=1.0,
    aa_stride=1,
    aa_start_iter=2,
    aa_reg=1e-10,
    aa_alpha_max=50.0,
    warm_enable=None,
    moments_cnn_modelpath=None,
    warm_delta_weight_mode="none",
)
```

### `holo`

```python
BGK1D.holo.Params(
    ho_iter=8,
    ho_tol=1e-4,
    ho_abs_tol=1e-12,
    lo_iter=16,
    lo_tol=1e-4,
    lo_abs_tol=1e-12,
    Con_Terms_do=True,
    flux_consistency_do=True,
)
```

## 4. Engine が行う初期化

`Engine(config)` は概ね以下の順序で処理する。

1. `model_cfg` の補完と型検証
2. `scheme_params` の既定値補完
3. `apply_logging()` による `kineticEQ` logger の設定
4. `resolve_device()` による device 妥当性確認
5. `core.states.registry.build_state()` による state テンソル確保
6. `core.schemes.registry.build_stepper()` による stepper 構築

重要なのは、初期条件の実際の書き込みは stepper builder 側で `set_initial_condition()` を呼ぶことにより行われる点である。すなわち、`Engine` 自身は state をゼロで確保した後、登録済み stepper に初期化の責務を委譲している。

## 5. run の実行フロー

```python
from kineticEQ import Engine

engine = Engine(cfg)
result = engine.run()
```

`Engine.run()` は `n_steps` 回ループし、各ステップで `self.stepper(step)` を呼ぶ。implicit / holo の stepper は内部に `benchlog` 属性を持ち、進捗出力時に反復回数や残差が表示される。

返り値は現時点では

```python
Result(metrics=None, payload=None)
```

であり、計算結果の永続化や snapshot 返却はトップレベル API にはまだ統合されていない。

## 6. 現行仕様上の注意

- `BGK1D1V + explicit + torch` は最も単純な参照実装である。
- `BGK1D1V + implicit + cpu_kernel` は CPU 上で C++ 拡張を用いる。
- `BGK1D1V + explicit/implicit/holo + cuda_kernel` は高速経路だが、BGK1D fused kernel は実質 `float64` 前提である。
- `BGK2D2V` は registry 上の記述と `Engine` 経路が整合しておらず、現行版では実行対象に含めるべきではない。

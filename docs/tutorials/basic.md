---
title: 基本チュートリアル
nav_order: 3
parent: kineticEQ Docs
---

# 基本チュートリアル

## 1. 設定の作成

```python
from kineticEQ import Config, Engine, BGK1D

# モデル設定
model_cfg = BGK1D.ModelConfig(
    grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
    time=BGK1D.TimeConfig(dt=1e-6, T_total=0.01),
    params=BGK1D.BGK1D1VParams(tau_tilde=1e-5),
)

# 全体設定
cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    model_cfg=model_cfg,
)
```

## 2. エンジンの起動とシミュレーション実行

```python
engine = Engine(cfg)
engine.run()
```

## 3. 結果の可視化

```python
from kineticEQ.plotting.bgk1d import plot_state

plot_state(engine.state, filename="result.png")
```

## 4. スキームの選択

### 陽解法 (Explicit)

```python
cfg = Config(scheme="explicit", ...)
```

### 陰解法 (Implicit)

```python
from kineticEQ.params.BGK1D import implicit

cfg = Config(
    scheme="implicit",
    model_cfg=BGK1D.ModelConfig(
        ...,
        scheme_params=implicit.Params(picard_iter=10, picard_tol=1e-8),
    )
)
```

### HOLO法

```python
from kineticEQ.params.BGK1D import holo

cfg = Config(
    scheme="holo",
    model_cfg=BGK1D.ModelConfig(
        ...,
        scheme_params=holo.Params(ho_iter=5, lo_iter=3),
    )
)
```

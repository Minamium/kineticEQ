---
title: Engine Overview
parent: Getting Started
nav_order: 12
---

# 基本チュートリアル

## 1. 設定の作成, 以下はBGK1D1Vモデルの設定例

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

## 2. Engineインスタンスの作成とシミュレーション実行

```python
engine = Engine(cfg)
engine.run()
```
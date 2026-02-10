---
title: kineticEQ Docs
nav_order: 1
---

# kineticEQ

PyTorch ベースの運動論方程式（BGK 方程式）ソルバー。
カスタム CUDA/C++ カーネルによる高速計算と、Python API による柔軟な設定を両立する。

## 特徴

- **複数の数値スキーム** -- 陽解法 (`explicit`)、陰解法 (`implicit`, Picard 反復)、HOLO 法 (`holo`)
- **2 種類の計算バックエンド** -- 純 PyTorch (`torch`) およびカスタム CUDA カーネル (`cuda_kernel`)
- **CNN Warmstart** -- 陰解法の初期推定値を CNN で予測し、Picard 反復回数を削減
- **組み込みベンチマーク・収束テスト** -- `analysis` モジュールによるスキーム比較、収束次数検証、タイミング計測

## 対応モデル

| モデル | 説明 |
|--------|------|
| `BGK1D1V` | 1 次元空間 + 1 次元速度空間の BGK 方程式 |
| `BGK2D2V` | 2 次元空間 + 2 次元速度空間の BGK 方程式 |

## Quick Start

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=1e-6, T_total=0.01),
        params=BGK1D.BGK1D1VParams(tau_tilde=1e-5),
    )
)

engine = Engine(cfg)
result = engine.run()
```

## Documentation

- [Installation](getting-started/installation.md)
- [Engine Overview](getting-started/Engine_overview.md)
- [Models](Models/index.md)
- [API Reference](api_Reference/api_lib.md)

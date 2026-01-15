---
title: kineticEQ Docs
nav_order: 1
---

# kineticEQ ドキュメント

高性能運動論方程式ソルバー（BGK方程式）

## クイックスタート

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128),
        time=BGK1D.TimeConfig(dt=1e-6, T_total=0.01),
    )
)

engine = Engine(cfg)
engine.run()
```

## ドキュメント

- [インストール](getting-started/installation.md)
- [チュートリアル](tutorials/basic.md)
- [APIリファレンス](api/index.md)

## 特徴

- **高速計算**: PyTorch + カスタムCUDAカーネルによる最適化
- **複数スキーム**: 陽解法 / 陰解法 / HOLO法
- **モジュラー設計**: 拡張可能なアーキテクチャ

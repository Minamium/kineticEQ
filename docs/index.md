---
title: kineticEQ Docs
nav_order: 1
---

# kineticEQ

PyTorch ベースの運動論方程式（BGK 方程式）ソルバー。
Python API (`Config` / `Engine`) と、BGK1D 向けの CUDA/C++ 拡張カーネルを統合している。

## 現在の実装ステータス

| モデル | スキーム | backend=`torch` | backend=`cuda_kernel` | 状態 |
|---|---|---|---|---|
| `BGK1D1V` | `explicit` | 対応 | 対応 | 実運用可能 |
| `BGK1D1V` | `implicit` | 未対応 | 対応 | 実運用可能 |
| `BGK1D1V` | `holo` | 未対応 | 対応 | 実運用可能 |
| `BGK2D2V` | `explicit` | 実装未完了 | 未登録 | 現状は Engine 実行不可 |

補足:
- `Config` では `scheme="holo_nn"` を受理するが、stepper 未登録のため実行は不可。
- BGK1D の `cuda_kernel` 経路は実質 `float64` 前提（拡張側の dtype チェックが `kDouble` 固定）。

## 主要機能

- **BGK1D の複数スキーム**: `explicit` / `implicit(Picard)` / `holo(HOLO)`
- **CUDA 拡張**: fused explicit、implicit 系列、cuSPARSE gtsv、AA、LO block-tridiag
- **CNN warmstart**: implicit Picard の初期モーメント推定
- **分析モジュール**: benchmark / convergence / scheme comparison + plotting

## Documentation

- [Installation](getting-started/installation.md)
- [Engine Overview](getting-started/Engine_overview.md)
- [Examples](getting-started/examples.md)
- [Models](Models/index.md)
- [API Reference](api_Reference/api_lib.md)
- [Implementations](Implementations/index.md)

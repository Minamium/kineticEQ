---
title: Japanese
nav_exclude: true
permalink: /ja/
lang: ja
---

# 日本語ドキュメント

{% include keq_banner.html %}

`kineticEQ` は、Boltzmann-BGK 方程式を対象とする数値計算ライブラリであり、`Config` / `Engine` による高水準 API と、BGK1D1V 向けに最適化された C++/CUDA 拡張を単一の実行系として統合している。

## 現行実装の対応状況

| モデル | スキーム | `torch` | `cuda_kernel` | `cpu_kernel` | 評価 |
|---|---|---|---|---|---|
| `BGK1D1V` | `explicit` | 対応 | 対応 | 未登録 | 実行可能 |
| `BGK1D1V` | `implicit` | 未登録 | 対応 | 対応 | 実行可能 |
| `BGK1D1V` | `holo` | 未登録 | 対応 | 未登録 | 実行可能 |
| `BGK2D2V` | `explicit` | registry には登録 | 未登録 | 未登録 | `Engine` 経路未完 |

## 主要ページ

- [Getting Started](../getting-started/index.md)
- [Models](../Models/index.md)
- [BGK1D1V](../Models/BGK1D1V/index.md)
- [BGK1D1V Boundary Conditions](../Models/BGK1D1V/boundary_conditions.md)
- [API Reference](../api_Reference/index.md)
- [Implementations](../Implementations/index.md)

## 数値実装の要点

- **BGK1D1V explicit**: 速度空間モーメントの直接積分、Maxwell 分布の再構成、一次風上差分による移流項、局所緩和項を組み合わせた陽解法である。
- **BGK1D1V implicit**: Picard 反復によりモーメントを更新しつつ、速度ごとに batched 三重対角系を解く。収束判定は分布関数基準 `conv_type="f"` とモーメント基準 `conv_type="w"` を選択できる。
- **BGK1D1V holo**: 高次分布関数更新と低次モーメント更新を HO/LO に分離し、LO 側では 3x3 block-tridiagonal 系、HO 側では通常の三重対角系を解く構成である。
- **MIWS-Net warm-start**: implicit Picard の初期モーメント推定に 1D CNN を組み込み、`W=(n, u, T)` あるいは `W=(n, \nu, T)` の増分予測を用いて反復開始点を補正する。

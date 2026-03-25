---
title: kineticEQ Docs
nav_order: 1
lang: ja
---

# kineticEQ

`kineticEQ` は、BGK 型運動論方程式を対象とする数値計算ライブラリであり、`Config` / `Engine` による高水準 API と、BGK1D 向けに最適化された C++/CUDA 拡張を単一の実行系として統合している。本ドキュメントは、現行の `src/kineticEQ` 実装に基づいて記述しており、旧ノートブック群ではなく現在の API・stepper・学習補助系を基準とする。

## 現行実装の対応状況

| モデル | スキーム | `torch` | `cuda_kernel` | `cpu_kernel` | 評価 |
|---|---|---|---|---|---|
| `BGK1D1V` | `explicit` | 対応 | 対応 | 未登録 | 実行可能 |
| `BGK1D1V` | `implicit` | 未登録 | 対応 | 対応 | 実行可能 |
| `BGK1D1V` | `holo` | 未登録 | 対応 | 未登録 | 実行可能 |
| `BGK2D2V` | `explicit` | registry には登録 | 未登録 | 未登録 | `Engine` 経路未完 |

注記:

- BGK1D の `cuda_kernel` 経路は、`explicit_fused` と `implicit_fused` の binding が `float64` を要求する。
- BGK1D の `cpu_kernel` 経路も、現状は implicit 専用であり、C++ binding 側で `float64` を要求する。
- `BGK2D2V` は state 生成までは実装されているが、`ModelConfig` に `scheme_params` が無く、さらに stepper 本体も未完成であるため、現行 `Engine` からは実運用できない。

## 数値実装の要点

- **BGK1D explicit**: 速度空間モーメントの直接積分、Maxwell 分布の再構成、一次風上差分による移流項、局所緩和項を組み合わせた陽解法である。
- **BGK1D implicit**: Picard 反復によりモーメントを更新しつつ、速度ごとに batched 三重対角系を解く。収束判定は分布関数基準 `conv_type="f"` とモーメント基準 `conv_type="w"` を選択できる。
- **BGK1D holo**: 高次分布関数更新と低次モーメント更新を HO/LO に分離し、LO 側では 3x3 block-tridiagonal 系、HO 側では通常の三重対角系を解く構成である。
- **CNN warm-start**: implicit Picard の初期モーメント推定に 1D CNN を組み込み、`W=(n, u, T)` あるいは `W=(n, \nu, T)` の増分予測を用いて反復開始点を補正する。

## ドキュメント構成

- [Getting Started](getting-started/index.md): インストール、`Engine` の実行フロー、基本例
- [Models](Models/index.md): 支配方程式、離散化、スキームごとの数値ロジック
- [API Reference](api_Reference/index.md): `Config` / `Engine` / パラメータ dataclass
- [Implementations](Implementations/index.md): ディレクトリごとの責務と接続関係

---
title: Package Map
parent: Implementations
nav_order: 31
lang: ja
---

# Package Map

## `src/kineticEQ` の主要ディレクトリ

| ディレクトリ | 役割 |
|---|---|
| `api/` | `Config`, `Engine`, `Result`, logging 設定 |
| `params/` | モデル・スキーム設定 dataclass と既定値 registry |
| `core/states/` | state テンソルの確保とモデル別 state builder |
| `core/schemes/` | stepper 本体と `(model, scheme, backend)` registry |
| `cuda_kernel/` | CUDA/C++ 拡張の JIT ローダと kernel 実装 |
| `cpu_kernel/` | CPU C++ 拡張と implicit 用ソルバ |
| `CNN/BGK1D1V/` | warm-start 用データ生成、学習、評価 |
| `analysis/` | benchmark / convergence / scheme comparison |
| `plotting/` | state 可視化 |
| `utillib/` | device / progress / pretty print |
| `_legacy/` | 旧クラスベース実装 |
| `tests/` | smoke, plotting, benchmark などの確認コード |

## 現在の安定実行経路

- `BGK1D1V + explicit + torch`
- `BGK1D1V + explicit + cuda_kernel`
- `BGK1D1V + implicit + cuda_kernel`
- `BGK1D1V + implicit + cpu_kernel`
- `BGK1D1V + holo + cuda_kernel`

## 開発中あるいは未整備の領域

- `BGK2D2V` の `Engine` 経路
- BGK1D `torch` backend における implicit / holo
- BGK1D CPU backend における explicit / holo

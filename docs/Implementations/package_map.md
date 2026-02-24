---
title: Package Map
parent: Implementations
nav_order: 31
---

# Package Map

## `src/kineticEQ` の主要ディレクトリ

| ディレクトリ | 役割 |
|---|---|
| `api/` | `Config` / `Engine` / `Result` |
| `params/` | モデル・スキーム設定 dataclass |
| `core/states/` | state テンソル確保 |
| `core/schemes/` | 時間発展 stepper |
| `cuda_kernel/` | C++/CUDA 拡張の JIT ビルドと実装 |
| `analysis/` | benchmark / convergence / scheme 比較 |
| `plotting/` | state の可視化 |
| `CNN/BGK1D1V/` | warmstart 用学習・評価コード |
| `utillib/` | progress / pretty / device ユーティリティ |
| `_legacy/` | 旧クラスベース実装 |

## 現在の安定実行経路

- `BGK1D1V + explicit + torch`
- `BGK1D1V + explicit/implicit/holo + cuda_kernel`

## 未整備領域

- `BGK2D2V` の Engine 実行経路
- `scheme="holo_nn"` の stepper

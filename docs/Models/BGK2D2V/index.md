---
title: BGK2D2V
parent: Models
nav_order: 22
has_children: true
---

# BGK2D2V

2 次元空間 + 2 次元速度空間の BGK モデル。

## 現在の実装状態

現行コードベースでは **Engine 経路は未完成**。

理由:

1. `core/schemes/BGK2D2V/bgk2d2v_explicit_torch.py` の `step` は TODO ダミー実装。
2. `params/BGK2D2V/ModelConfig` に `scheme_params` がなく、`Engine` 初期化時に `AttributeError` になる。

したがって、ドキュメント上の「対応モデル」には含まれているが、現バージョンでは実行対象としては扱えない。

## 関連実装ファイル

- `src/kineticEQ/params/BGK2D2V/BGK2D2V_params.py`
- `src/kineticEQ/core/states/state_2d2v.py`
- `src/kineticEQ/core/schemes/BGK2D2V/bgk2d2v_explicit_torch.py`
- `src/kineticEQ/cuda_kernel/BGK2D2V/explicit_2d2v/*`（拡張コードはあるが Engine registry 未接続）

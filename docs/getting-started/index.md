---
title: Getting Started
parent: Japanese
nav_order: 10
has_children: true
lang: ja
---

# Getting Started

本節では、`kineticEQ` の最小実行単位である `Config -> Engine -> run()` の流れを概説する。実装上、初期条件の設定は `Engine` 本体ではなく各 stepper の `build_stepper` 内で行われるため、API 仕様と内部実装の対応を理解しておくと挙動を追いやすい。

- [Installation](installation.md): 実行要件、依存パッケージ、CUDA/CPU 拡張の JIT ビルド
- [Engine Overview](Engine_overview.md): `Config` 正規化、`model_cfg` 補完、stepper 構築の順序
- [Examples](examples.md): explicit / implicit / holo / CNN warm-start の典型例

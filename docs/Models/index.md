---
title: Models
parent: Japanese
nav_order: 20
has_children: true
lang: ja
---

# Models

本節では、支配方程式の記述だけでなく、現行コードがどの離散化と backend を実装しているかを併せて整理する。とくに BGK1D では、同じモデルに対して explicit / implicit / holo の三系統が実装されており、各 stepper の数値的責務は明確に分かれている。

- [BGK1D1V](BGK1D1V/index.md): 現行の主力実装。`explicit` / `implicit` / `holo` をサポートする。
- [BGK1D1V Boundary Conditions](BGK1D1V/boundary_conditions.md): 固定 Maxwellian、鏡面反射、拡散反射の連続式と離散式。
- [BGK2D2V](BGK2D2V/index.md): モデル定義と state は存在するが、`Engine` 実行経路は未完成である。

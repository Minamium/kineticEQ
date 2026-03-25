---
title: BGK2D2V
parent: Models
nav_order: 22
has_children: true
lang: ja
---

# BGK2D2V

BGK2D2V は、二次元空間 `(x,y)` と二次元速度 `(v_x,v_y)` を持つ BGK モデルである。state dataclass とパラメータ dataclass は整備されているが、現行版では `Engine` 実行経路が未完成である。

## 支配方程式

概念的には

$$
\frac{\partial f}{\partial t} + v_x \frac{\partial f}{\partial x} + v_y \frac{\partial f}{\partial y}
= \frac{1}{\tau}(f_M - f)
$$

を想定している。しかし、現行実装はこの方程式の完全な時間発展器を提供していない。

## 実装の現状

1. `State2D2V` は `f`, `f_new`, `x`, `y`, `vx`, `vy` を確保する。
2. `core/schemes/BGK2D2V/bgk2d2v_explicit_torch.py` は registry には接続されているが、`step()` が TODO ダミーである。
3. `params/BGK2D2V/BGK2D2V_params.py` の `ModelConfig` には `scheme_params` フィールドが無く、`Engine` 初期化中に `config.model_cfg.scheme_params` を参照する箇所で破綻する。
4. `cuda_kernel/BGK2D2V/explicit_2d2v/` には拡張コードが存在するが、現行 registry には結線されていない。

## 結論

`BGK2D2V` は、将来的な拡張の骨格としては存在するものの、現時点では API 公開対象というより開発中のコードパスと解釈すべきである。研究・検証の主対象は BGK1D 系に置くのが妥当である。

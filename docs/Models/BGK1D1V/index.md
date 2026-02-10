---
title: BGK1D1V
parent: Models
nav_order: 21
has_children: true
---

# BGK1D1V

1 次元空間 (x) + 1 次元速度空間 (v) の BGK 方程式を解くモデル。

## 支配方程式

$$
\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x}
= \frac{1}{\tilde{\tau}} \bigl( f_M - f \bigr)
$$

ここで:
- $f(t, x, v)$: 分布関数
- $f_M$: 局所 Maxwellian 平衡分布
- $\tilde{\tau}$: 無次元化された緩和時間

Maxwellian は巨視的モーメント (数密度 $n$, 流速 $u$, 温度 $T$) から構成される:

$$
f_M(x, v) = \frac{n(x)}{\sqrt{2 \pi T(x)}}
\exp\!\left( -\frac{(v - u(x))^2}{2 T(x)} \right)
$$

## 対応スキームとバックエンド

| スキーム | `torch` | `cuda_kernel` | 備考 |
|---------|---------|---------------|------|
| `explicit` | 対応 | 対応 | 1 次風上法。CFL 条件あり |
| `implicit` | -- | 対応 | Picard 反復 + cuSPARSE 三重対角ソルバー。CNN warmstart 対応 |
| `holo` | -- | 対応 | High-Order/Low-Order 分離法。ブロック三重対角 PCR ソルバー |

## 初期条件

`InitialCondition1D` で区分的に定数な初期モーメントを指定する。
各領域は `x_range`, `n`, `u`, `T` を持つ辞書で定義する。

```python
BGK1D.InitialCondition1D(
    initial_regions=(
        {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    )
)
```

デフォルトは Sod 衝撃波管問題に類似した 2 領域設定。

## CNN Warmstart (implicit スキーム)

陰解法の Picard 反復における初期推定値を CNN (`MomentCNN1D`) で予測する機能。
`moments_cnn_modelpath` にチェックポイントパスを指定すると有効になる。

```python
BGK1D.implicit.Params(
    picard_iter=1000,
    picard_tol=1e-4,
    moments_cnn_modelpath="path/to/best_speed.pt",
)
```

CNN は入力 $(n, u, T, \log_{10}\Delta t, \log_{10}\tilde{\tau})$ からモーメントの差分を予測する。

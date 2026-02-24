---
title: BGK1D1V
parent: Models
nav_order: 21
has_children: true
---

# BGK1D1V

1 次元空間 (`x`) + 1 次元速度空間 (`v`) の BGK 方程式モデル。

## 支配方程式

$$
\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x}
= \frac{1}{\tilde{\tau}} (f_M - f)
$$

$$
f_M(x, v) = \frac{n(x)}{\sqrt{2 \pi T(x)}}
\exp\left(-\frac{(v-u(x))^2}{2T(x)}\right)
$$

## 実装対応表

| スキーム | `torch` | `cuda_kernel` | 備考 |
|---|---|---|---|
| `explicit` | 対応 | 対応 | 1 次風上 + CFL チェック |
| `implicit` | 未対応 | 対応 | Picard + cuSPARSE gtsv |
| `holo` | 未対応 | 対応 | HO/LO 分離 + block tridiag |

## implicit の追加パラメータ（実装準拠）

`BGK1D.implicit.Params` は以下を持つ:

- `picard_iter`, `picard_tol`, `abs_tol`
- `conv_type` (`"f"` / `"w"`)
- `aa_enable`, `aa_m`, `aa_beta`, `aa_stride`, `aa_start_iter`, `aa_reg`, `aa_alpha_max`
- `moments_cnn_modelpath`

## CNN warmstart

`moments_cnn_modelpath` を指定すると、implicit の初期モーメント推定を CNN で置換する。

入力:

$$(n, u, T, \log_{10}\Delta t, \log_{10}\tilde{\tau})$$

## 初期条件

`InitialCondition1D.initial_regions` に区分一定なモーメントを指定。

```python
BGK1D.InitialCondition1D(
    initial_regions=(
        {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    )
)
```

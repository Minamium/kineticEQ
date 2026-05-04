---
title: English BGK1D1V Numerical Notes
nav_title: BGK1D1V Numerical Notes
parent: English Models
nav_order: 3
lang: en
---

# Numerical Notes

This page collects the quantities and update structures that appear repeatedly across the BGK1D steppers.

## Basic moments

The fused kernels and the pure PyTorch utilities use the same discrete moments:

$$
n_i = \sum_k f_i(v_k)\,\Delta v, \qquad
\nu_i = \sum_k v_k f_i(v_k)\,\Delta v, \qquad
T_i = \frac{1}{n_i}\sum_k v_k^2 f_i(v_k)\,\Delta v - \left(\frac{\nu_i}{n_i}\right)^2.
$$

## Implicit system

For each velocity node, the implicit stepper builds a tridiagonal system along the spatial direction. In code, the three diagonals and the right-hand side appear as `(dl, dd, du, B)`.

## Convergence metrics

### `conv_type="f"`

$$
r_f = \max_{i,k}
\frac{|f_{i,k}^{(m+1)} - f_{i,k}^{(m)}|}
{\mathrm{abs\_tol} + \mathrm{picard\_tol}\max(|f_{i,k}^{(m+1)}|, |f_{i,k}^{(m)}|)}.
$$

### `conv_type="w"`

$$
r_W = \max(r_n, r_{\nu}, r_T).
$$

## HOLO low-order state

The low-order unknown is

$$
W = (n, \nu, U), \qquad U = \frac12 n(u^2 + T).
$$

This representation makes it natural to build a 3x3 block-tridiagonal system associated with mass, momentum, and energy transport.

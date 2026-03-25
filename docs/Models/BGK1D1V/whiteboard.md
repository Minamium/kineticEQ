---
title: Numerical Notes
parent: BGK1D1V
nav_order: 1
lang: ja
---

# 数値実装ノート

本ページでは、`BGK1D1V` の主要 stepper に共通する量を、実装で用いられている記号系に揃えて整理する。

## 1. 基本モーメント

`calculate_moments()` および fused kernel の `moments_n_nu_T()` は、離散分布 `f_i(v_k)` から以下を計算する。

$$
n_i = \sum_k f_i(v_k)\,\Delta v,
$$

$$
\nu_i = \sum_k v_k f_i(v_k)\,\Delta v,
$$

$$
T_i = \frac{1}{n_i}\sum_k v_k^2 f_i(v_k)\,\Delta v - \left(\frac{\nu_i}{n_i}\right)^2.
$$

実装内部では、速度 `u_i` を直接保持する場面と、運動量 `\nu_i` を直接保持する場面が混在する。implicit / holo の線形 solve では、保存量との整合を取りやすい `\nu` が優先される。

## 2. implicit の三重対角系

implicit では、各速度 `v_k` ごとに `x` 方向内点 `i=1,\dots,n_x-2` の未知量を並べ、

$$
A_k f_{k}^{n+1} = b_k(W)
$$

の形で三重対角系を組み立てる。ここで `A_k` の係数は Picard 反復中のモーメント `W=(n,\nu,T)` に依存し、`b_k` は前時刻分布 `B0` と境界分布 `f_{bc}` を含む。実装上は `(dl, dd, du, B)` がそれぞれ下対角、対角、上対角、右辺に対応する。

## 3. 収束規準

### `conv_type="f"`

分布関数そのものの変化率を評価する。

$$
r_f = \max_{i,k}
\frac{|f_{i,k}^{(m+1)} - f_{i,k}^{(m)}|}
{\mathrm{abs\_tol} + \mathrm{picard\_tol}\max(|f_{i,k}^{(m+1)}|, |f_{i,k}^{(m)}|)}.
$$

### `conv_type="w"`

モーメントごとに同じ規格化を行い、その最大値を採る。

$$
r_W = \max(r_n, r_{\nu}, r_T).
$$

`picard_tol` を極端に小さくしても `abs_tol` が支配的になる領域があるため、反復回数の減少が頭打ちに見える場合は、両者の相対スケールを併せて評価する必要がある。

## 4. HOLO の LO 変数

HOLO では、低次系の未知量を

$$
W = (n, \nu, U), \qquad U = \frac12 n(u^2 + T)
$$

として持つ。これにより、質量・運動量・エネルギーに対応する 3 成分系として block-tridiagonal solve を構成できる。

界面フラックスには

$$
F^{HO}_{i+1/2} = (S_1, S_2, S_3 + 2Q_{i+1/2})
$$

が現れ、`Q` は高次分布 `f_z` から計算される熱流束補正である。

## 5. 整合項 `Y_I_terms`

`Y_I_terms` は二種類の寄与から成る。

- **flux consistency**: HO フラックスと LO フラックスの差 `\gamma_{i+1/2}` の空間差分
- **collision consistency**: 混合衝突項 `C_mix` のモーメント residual

これにより、LO 側だけでは失われる高次分布情報を、保存則を崩さない形で右辺へ押し戻している。

## 6. CNN warm-start と時間履歴

CNN warm-start は、現時刻の `W_t` だけでなく、`prev_delta` モードでは `W_{t-1}` と `W_t-W_{t-1}` も入力に加える。実装では、stepper workspace に `_warm_prev_W` を保持し、次ステップの推論時にこれを再利用する。したがって、warm-start は単発の初期値推定ではなく、時系列的一貫性を持つ predictor として振る舞う。

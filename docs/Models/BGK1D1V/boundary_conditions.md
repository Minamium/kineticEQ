---
title: BGK1D1V Boundary Conditions
nav_title: BGK1D1V Boundary Conditions
parent: Models
nav_order: 2
lang: ja
---

# BGK1D1V の境界条件

本ページでは、`BGK1D1V` で使う境界条件を、連続式と現行実装の離散式に分けて整理する。対象領域は $x\in[0,L]$、速度は $v\in[-v_{\max},v_{\max}]$ とする。

## 境界での流入・流出

一次元では、左壁 $x=0$ の外向き法線は $n_x=-1$、右壁 $x=L$ の外向き法線は $n_x=+1$ である。したがって、壁へ出ていく速度集合と、壁から計算領域へ入ってくる速度集合は次のように分かれる。

| 境界 | 壁へ出る速度 | 領域へ戻る速度 |
|---|---:|---:|
| 左壁 $x=0$ | $v<0$ | $v>0$ |
| 右壁 $x=L$ | $v>0$ | $v<0$ |

境界条件とは、基本的には「領域へ戻る半空間の分布関数」をどう与えるか、という規則である。

## 現行実装の境界行

現行の `State1D1V.f` は `x` 方向に `0,\dots,N_x-1` の行を持つ。このうち、内部更新の未知量は主に

$$
i = 1,\dots,N_x-2
$$

であり、端点の `i=0` と `i=N_x-1` は境界 trace として使われる。これは厳密な有限体積ゴーストセルとはまだ完全には分離されていないが、保存量や時系列診断では物理体積セルではなく境界情報として扱うのが自然である。

速度格子は $v_k$、対応する分布を $f_{i,k}$ と書く。速度格子が原点対称であるとき、反転速度の index を

$$
k^\star = N_v - 1 - k
$$

で表す。実装上は `torch.flip(..., dims=(0,))` がこの写像に対応する。

## 固定 Maxwellian 境界

`bc_type="fixed_maxwellian"` は、端点行に入っている初期 Maxwellian をそのまま境界値として保持する境界条件である。連続的には、境界から領域へ流入する半空間に所与の Maxwellian を課すと読む。

左壁では

$$
f(t,0,v) = M_L(v), \qquad v>0,
$$

右壁では

$$
f(t,L,v) = M_R(v), \qquad v<0
$$

である。現行実装では端点行全体を固定するため、離散的には

$$
f_{0,k}^{n+1}=f_{0,k}^{n}, \qquad
f_{N_x-1,k}^{n+1}=f_{N_x-1,k}^{n}
$$

として扱われる。これは開いた境界や壁反射ではなく、端点分布を固定した reservoir に近い。

## 鏡面反射境界

鏡面反射では、壁に入射した粒子が法線方向速度だけを反転して戻る。一次元では速度反転そのものなので、連続式は次で与えられる。

左壁では

$$
f(t,0,v) = f(t,0,-v), \qquad v>0,
$$

右壁では

$$
f(t,L,v) = f(t,L,-v), \qquad v<0.
$$

この条件では、壁を通過する質量フラックスは打ち消される。

$$
\int_{v>0} v f(t,0,v)\,dv
=
\int_{v<0} (-v) f(t,0,v)\,dv,
$$

$$
\int_{v<0} (-v) f(t,L,v)\,dv
=
\int_{v>0} v f(t,L,v)\,dv.
$$

したがって、閉じた領域としては質量保存と整合する。一方で壁は運動量を反転させるため、気体単独の運動量は一般には保存しない。壁からの力積まで含めた系で保存を考えるべきである。

現行実装では、境界 trace は隣接内部行から作る。左境界は `i=1`、右境界は `i=N_x-2` を参照し、領域へ戻る半空間だけ速度反転値で置き換える。

左境界:

$$
f_{0,k}^{bc}
=
\begin{cases}
f_{1,k^\star}, & v_k>0,\\
f_{1,k}, & v_k<0.
\end{cases}
$$

右境界:

$$
f_{N_x-1,k}^{bc}
=
\begin{cases}
f_{N_x-2,k}, & v_k>0,\\
f_{N_x-2,k^\star}, & v_k<0.
\end{cases}
$$

$v_k=0$ は壁を横切るフラックスを持たないため、実装では隣接内部値のコピーとして残す。

## 拡散反射境界

拡散反射では、壁に入射した粒子は壁温度 $T_w$ の Maxwellian として再放出される。ただし、返す質量フラックスは壁へ入った質量フラックスと一致させる。実装では静止壁 $u_w=0$ を仮定する。一般式として壁速度を $u_w$ とすると、単位密度 Maxwellian を

$$
M_w^{(1)}(v)
=
\frac{1}{\sqrt{2\pi T_w}}
\exp\left(-\frac{(v-u_w)^2}{2T_w}\right)
$$

と書ける。

左壁へ入る質量フラックスは

$$
J_L^{out}
=
\int_{v<0} (-v) f(t,0,v)\,dv
$$

である。左壁から返す分布は $v>0$ に対して

$$
f(t,0,v) = \rho_L^w M_{w,L}^{(1)}(v),
$$

ただし

$$
\rho_L^w
=
\frac{J_L^{out}}
{\int_{v>0} v M_{w,L}^{(1)}(v)\,dv}.
$$

右壁も同様に、

$$
J_R^{out}
=
\int_{v>0} v f(t,L,v)\,dv,
$$

$$
f(t,L,v) = \rho_R^w M_{w,R}^{(1)}(v), \qquad v<0,
$$

$$
\rho_R^w
=
\frac{J_R^{out}}
{\int_{v<0} (-v) M_{w,R}^{(1)}(v)\,dv}.
$$

この規格化により、壁を横切る正味質量フラックスは 0 になる。対して、運動量とエネルギーは壁温度・壁速度に応じて壁と交換されるため、気体側だけでは一般に保存しない。

現行の境界行規約に沿った離散式では、左壁の入射フラックスを隣接内部行から

$$
J_{L,h}^{out}
=
\Delta v \sum_{v_k<0} (-v_k) f_{1,k}
$$

と評価し、

$$
D_{L,h}
=
\Delta v \sum_{v_k>0} v_k M_{w,L,k}^{(1)},
\qquad
\rho_{L,h}^w
=
\frac{J_{L,h}^{out}}{D_{L,h}}
$$

として、境界行を

$$
f_{0,k}^{bc}
=
\begin{cases}
\rho_{L,h}^w M_{w,L,k}^{(1)}, & v_k>0,\\
f_{1,k}, & v_k<0
\end{cases}
$$

で与える。すなわち、領域へ戻る半空間だけを壁 Maxwellian で置き換え、壁へ出る半空間は隣接内部行の trace として残す。右壁は

$$
J_{R,h}^{out}
=
\Delta v \sum_{v_k>0} v_k f_{N_x-2,k},
$$

$$
D_{R,h}
=
\Delta v \sum_{v_k<0} (-v_k) M_{w,R,k}^{(1)},
\qquad
\rho_{R,h}^w
=
\frac{J_{R,h}^{out}}{D_{R,h}},
$$

$$
f_{N_x-1,k}^{bc}
=
\begin{cases}
f_{N_x-2,k}, & v_k>0,\\
\rho_{R,h}^w M_{w,R,k}^{(1)}, & v_k<0
\end{cases}
$$

で与える。

実装上は `bc_type="diffuse"`、`"diffuse_reflective"`、`"diffuse_reflection"` を同義に扱う。左右の壁温度は `BoundaryCondition1D.Lwall_temperature` と `BoundaryCondition1D.Rwall_temperature` から取得する。

## implicit stepper での注意

implicit stepper は各速度ごとに空間方向の三重対角系を解く。鏡面反射や拡散反射を完全に時刻 $n+1$ の未知量で課すと、境界で $v_k$ と $v_{k^\star}$ が結合し、速度ごとに独立な三重対角 solve という構造が崩れる。

そのため現行方針では、境界行は stepper が受け取った `source` 分布から明示的に作る。これは境界条件を一段 lag させる実装であり、安定性と精度の評価では `dt` 依存性を別途確認する必要がある。

## 保存量診断との関係

端点行を境界 trace と見る場合、質量・運動量・エネルギーの体積積分は内部セル `1:-1` を主対象にするのが自然である。現在の `animate_state_run()` も既定では内部セルだけを診断する。

BGK1D1V の代表的な離散診断量は

$$
\mathcal{M}
=
\Delta x\,\Delta v \sum_i \sum_k f_{i,k},
$$

$$
\mathcal{P}
=
\Delta x\,\Delta v \sum_i \sum_k v_k f_{i,k},
$$

$$
\mathcal{E}
=
\frac12 \Delta x\,\Delta v \sum_i \sum_k v_k^2 f_{i,k}
$$

である。固定 Maxwellian や拡散反射では境界 reservoir や壁との交換があるため、これらを気体単独で保存量とみなすべきではない。鏡面反射では質量とエネルギーは壁交換なしに保存されるのが期待されるが、数値粘性、有限速度幅、境界行の lag、端点積分の扱いによって離散誤差が残る。

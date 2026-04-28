---
title: Boundary Conditions
parent: English BGK1D1V
nav_order: 2
lang: en
---

# BGK1D1V Boundary Conditions

This page records the boundary-condition formulas used by `BGK1D1V`, at both the continuous and discrete implementation levels. The physical domain is $x\in[0,L]$, with velocity $v\in[-v_{\max},v_{\max}]$.

## Incoming and outgoing half spaces

At the left wall $x=0$, the outward normal is $n_x=-1$. At the right wall $x=L$, the outward normal is $n_x=+1$.

| Boundary | Outgoing to wall | Incoming to gas |
|---|---:|---:|
| Left wall $x=0$ | $v<0$ | $v>0$ |
| Right wall $x=L$ | $v>0$ | $v<0$ |

A kinetic boundary condition specifies the incoming half-space distribution.

## Boundary rows in the current code

`State1D1V.f` stores rows `0,\dots,N_x-1`. The updated interior unknowns are primarily

$$
i = 1,\dots,N_x-2.
$$

Rows `0` and `N_x-1` are boundary traces. They are not yet stored as separate finite-volume ghost cells, but conservation diagnostics should usually treat them as boundary data rather than physical volume cells.

For a symmetric velocity grid, the reversed-velocity index is

$$
k^\star = N_v - 1 - k.
$$

The implementation realizes this map with `torch.flip(..., dims=(0,))`.

## Fixed Maxwellian boundary

`bc_type="fixed_maxwellian"` keeps the endpoint rows fixed. Continuously, this can be read as prescribing Maxwellian data on the incoming half space:

$$
f(t,0,v)=M_L(v), \qquad v>0,
$$

$$
f(t,L,v)=M_R(v), \qquad v<0.
$$

In the present discrete implementation the whole endpoint row is held fixed:

$$
f_{0,k}^{n+1}=f_{0,k}^{n}, \qquad
f_{N_x-1,k}^{n+1}=f_{N_x-1,k}^{n}.
$$

This is closer to a fixed reservoir than to a closed reflecting wall.

## Specular reflection

Specular reflection reverses the normal velocity component. In 1D this is simply velocity reversal:

$$
f(t,0,v)=f(t,0,-v), \qquad v>0,
$$

$$
f(t,L,v)=f(t,L,-v), \qquad v<0.
$$

The mass flux through the wall cancels:

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

Gas momentum is not generally conserved by itself, because the wall reverses momentum through an impulse.

The current discrete boundary trace is built from the adjacent interior row:

$$
f_{0,k}^{bc}
=
\begin{cases}
f_{1,k^\star}, & v_k>0,\\
f_{1,k}, & v_k<0,
\end{cases}
$$

$$
f_{N_x-1,k}^{bc}
=
\begin{cases}
f_{N_x-2,k}, & v_k>0,\\
f_{N_x-2,k^\star}, & v_k<0.
\end{cases}
$$

The zero-velocity node has no wall-normal flux and is left as the adjacent interior copy.

## Diffuse reflection

Diffuse reflection re-emits particles as a wall Maxwellian at wall temperature $T_w$, while normalizing the emitted density so that the returned mass flux equals the incoming mass flux. With wall velocity $u_w$, the unit-density Maxwellian is

$$
M_w^{(1)}(v)
=
\frac{1}{\sqrt{2\pi T_w}}
\exp\left(-\frac{(v-u_w)^2}{2T_w}\right).
$$

At the left wall,

$$
J_L^{out}
=
\int_{v<0} (-v) f(t,0,v)\,dv,
$$

$$
f(t,0,v)=\rho_L^w M_{w,L}^{(1)}(v), \qquad v>0,
$$

$$
\rho_L^w
=
\frac{J_L^{out}}
{\int_{v>0} v M_{w,L}^{(1)}(v)\,dv}.
$$

At the right wall,

$$
J_R^{out}
=
\int_{v>0} v f(t,L,v)\,dv,
$$

$$
f(t,L,v)=\rho_R^w M_{w,R}^{(1)}(v), \qquad v<0,
$$

$$
\rho_R^w
=
\frac{J_R^{out}}
{\int_{v<0} (-v) M_{w,R}^{(1)}(v)\,dv}.
$$

The corresponding discrete formula, using the current boundary-row convention, evaluates the incoming wall flux from the adjacent interior row:

$$
J_{L,h}^{out}
=
\Delta v \sum_{v_k<0} (-v_k) f_{1,k},
\qquad
D_{L,h}
=
\Delta v \sum_{v_k>0} v_k M_{w,L,k}^{(1)},
$$

$$
\rho_{L,h}^w
=
\frac{J_{L,h}^{out}}{D_{L,h}},
\qquad
f_{0,k}^{bc}
=
\begin{cases}
\rho_{L,h}^w M_{w,L,k}^{(1)}, & v_k>0,\\
f_{1,k}, & v_k<0.
\end{cases}
$$

Only the incoming half space is replaced by the wall Maxwellian. The outgoing half space remains the adjacent interior trace.

For the right wall,

$$
J_{R,h}^{out}
=
\Delta v \sum_{v_k>0} v_k f_{N_x-2,k},
\qquad
D_{R,h}
=
\Delta v \sum_{v_k<0} (-v_k) M_{w,R,k}^{(1)},
$$

$$
\rho_{R,h}^w
=
\frac{J_{R,h}^{out}}{D_{R,h}},
\qquad
f_{N_x-1,k}^{bc}
=
\begin{cases}
f_{N_x-2,k}, & v_k>0,\\
\rho_{R,h}^w M_{w,R,k}^{(1)}, & v_k<0.
\end{cases}
$$

Diffuse reflection enforces zero net mass flux, but the wall may exchange momentum and energy with the gas.

## Note on implicit steppers

The implicit stepper solves independent tridiagonal systems for each velocity node. A fully implicit specular or diffuse boundary condition would couple $v_k$ and $v_{k^\star}$ at the boundary and break that independent velocity-batch structure.

The current policy therefore builds boundary rows explicitly from the `source` distribution passed to the boundary utility. This is a lagged boundary treatment, so time-step sensitivity should be checked when assessing accuracy.

## Conservation diagnostics

When endpoint rows are interpreted as boundary traces, volume-integrated diagnostics should usually use interior rows `1:-1`. The representative discrete diagnostics are

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
\frac12 \Delta x\,\Delta v \sum_i \sum_k v_k^2 f_{i,k}.
$$

Fixed Maxwellian and diffuse reflection boundaries exchange quantities with a reservoir or wall. Specular reflection is expected to preserve mass and kinetic energy for a closed domain, while gas momentum alone need not be constant.

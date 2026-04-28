---
title: English BGK1D1V
nav_title: BGK1D1V
parent: English Models
nav_order: 21
has_children: true
lang: en
---

# BGK1D1V

BGK1D1V is the one-dimensional-in-space, one-dimensional-in-velocity BGK model. In the implementation, the local relaxation time is reconstructed as

$$
\tau(x,t) = \frac{\tilde{\tau}}{n(x,t)\sqrt{T(x,t)}}
$$

rather than being prescribed directly as a field.

## State and quadrature

`State1D1V` stores:

- `f`, `f_tmp`, `f_m`
- uniform grids `x`, `v`
- macroscopic moments `n`, `u`, `T`
- cached masks and coefficients used by transport kernels

Moment evaluation is performed by direct discrete quadrature,

$$
n = \int f\,dv, \qquad \nu = \int vf\,dv, \qquad u = \frac{\nu}{n}, \qquad T = \frac{1}{n}\int v^2 f\,dv - u^2.
$$

## Boundary handling

Current BGK1D1V boundary handling is centralized in `bgk1d_apply_BC.py`. Endpoint rows `i=0` and `i=N_x-1` are treated as boundary traces rather than interior unknowns.

- `fixed_maxwellian` keeps the endpoint Maxwellian rows fixed.
- `reflective` builds a specular boundary trace by reversing the velocity components of the adjacent interior rows.
- `diffuse` builds a wall-temperature Maxwellian trace normalized by the outgoing mass flux into the wall.

The continuous and discrete formulas are summarized in [Boundary Conditions](boundary_conditions.md).

## Backend support

| Scheme | `torch` | `cuda_kernel` | `cpu_kernel` |
|---|---|---|---|
| `explicit` | supported | supported | not registered |
| `implicit` | not registered | supported | supported |
| `holo` | not registered | supported | not registered |

## Explicit stepper

The explicit BGK1D stepper performs:

1. moment evaluation,
2. Maxwellian reconstruction,
3. local relaxation-time construction,
4. first-order upwind transport,
5. collision update,
6. boundary fixation and buffer swap.

The `torch` version is a transparent reference implementation; the `cuda_kernel` version delegates the heavy work to a fused extension.

## Implicit stepper

The implicit stepper is implemented in both CUDA and CPU extension variants. The algorithmic structure is the same in both cases.

1. copy the current distribution into the implicit workspace,
2. compute initial moments `W=(n, \nu, T)`,
3. optionally inject a prescribed initial `W` or a CNN-based warm-start,
4. build the tridiagonal coefficients from the current moments,
5. solve the batched tridiagonal systems along the spatial direction,
6. recompute moments and test convergence,
7. update `W` directly or through Anderson acceleration.

The convergence metric can be chosen as either distribution-based (`conv_type="f"`) or moment-based (`conv_type="w"`).

## CNN warm-start

When `moments_cnn_modelpath` is set, the implicit stepper loads a checkpoint and predicts moment increments before Picard iteration begins. The current implementation supports:

- primitive or conservative input channels,
- optional temporal-history augmentation through `prev_delta`,
- `dw` and `dnu` output conventions,
- gradient-based attenuation through `warm_delta_weight_mode="w_grad"`.

## HOLO stepper

The HOLO implementation uses an outer high-order iteration and an inner low-order moment solve.

- The HO layer computes face moments and a heat-flux-like term `Q` from the current distribution.
- The LO layer solves a 3x3 block-tridiagonal system for `W=(n, \nu, U)`.
- The distribution update is then obtained from a tridiagonal solve using the LO moments.

The implementation uses `theta=0.5`, i.e. a Crank-Nicolson-like split.

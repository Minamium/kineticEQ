// src/kineticEQ/backends/imp_picard/imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>

namespace cg = cooperative_groups;

// ---- utility: atomic max for double (non-negative domain) -------------------
__device__ inline void atomicMaxDouble(double* addr, double val) {
    // assumes val >= 0
#if __CUDA_ARCH__ >= 600
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old    = *uaddr, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;
        old = atomicCAS(uaddr, assumed, __double_as_longlong(val));
    } while (assumed != old);
#else
    // Fallback: coarse-grained via atomicCAS on 64-bit
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old    = *uaddr, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;
        old = atomicCAS(uaddr, assumed, __double_as_longlong(val));
    } while (assumed != old);
#endif
}

// ---- helpers ----------------------------------------------------------------
__device__ __forceinline__ int gtid() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ __forceinline__ int gsize() {
    return gridDim.x * blockDim.x;
}

// ---- The cooperative kernel -------------------------------------------------
// f  : [nx * nv] (previous time step, read-only)
// fn : [nx * nv] (Picard iterate buffer, read/write; initialized to f inside)
// v  : [nv]
// dl,dd,du,B : [nv * (nx-2)] workspaces (in-place Thomas; solution overwrites B)
// n_arr,u_arr,T_arr,s0_arr,s1_arr,s2_arr : [nx] workspaces for moments
// res_dev : [1] device scalar (residual L∞ over interior)
// iters_dev : [1] device int (non-zero when converged)
extern "C" __global__
void picard_kernel_double(
    const double* __restrict__ f,
    double* __restrict__ fn,
    const double* __restrict__ v,
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    double* __restrict__ dl,
    double* __restrict__ dd,
    double* __restrict__ du,
    double* __restrict__ B,
    double* __restrict__ n_arr,
    double* __restrict__ u_arr,
    double* __restrict__ T_arr,
    double* __restrict__ s0_arr,
    double* __restrict__ s1_arr,
    double* __restrict__ s2_arr,
    double* __restrict__ res_dev,
    int*    __restrict__ iters_dev
){
    cg::grid_group grid = cg::this_grid();
    const int tid  = gtid();
    const int nT   = gsize();
    const int n_inner = (nx > 2) ? (nx - 2) : 0;

    // --- initialize fn = f (all cells, all velocities)
    for (int idx = tid; idx < nx * nv; idx += nT) {
        fn[idx] = f[idx];
    }
    // boundaries are therefore already equal to f and will remain untouched
    grid.sync();

    // constants reused
    const double dt_over_dx = dt / dx;
    const double eps_n = 1e-300;    // tiny floor to avoid 0-division
    const double eps_T = 1e-300;

    // Picard loop
    for (int it = 0; it < max_iters; ++it) {

        // clear reductions and residual flag by all threads (grid-stride)
        for (int i = tid; i < nx; i += nT) {
            s0_arr[i] = 0.0;
            s1_arr[i] = 0.0;
            s2_arr[i] = 0.0;
        }
        if (tid == 0) { *res_dev = 0.0; *iters_dev = 0; }
        grid.sync();

        // (1) accumulate moments: s0 = sum_j fn(i,j), s1 = sum_j v_j fn, s2 = sum_j v_j^2 fn
        for (int j = tid; j < nv; j += nT) {
            const double vj  = v[j];
            const double vj2 = vj * vj;
            const int base_j = j;           // fn[i*nv + j]
            for (int i = 0; i < nx; ++i) {
                const double fij = fn[i * nv + base_j];
#if __CUDA_ARCH__ >= 600
                atomicAdd(&s0_arr[i], fij);
                atomicAdd(&s1_arr[i], fij * vj);
                atomicAdd(&s2_arr[i], fij * vj2);
#else
                // (rare) for very old arch, CAS-based add could be implemented
                // but this codepath is practically unused on modern GPUs
                atomicAdd(&s0_arr[i], fij);
                atomicAdd(&s1_arr[i], fij * vj);
                atomicAdd(&s2_arr[i], fij * vj2);
#endif
            }
        }
        grid.sync();

        // (2) finalize moments and per-cell (n,u,T)
        for (int i = tid; i < nx; i += nT) {
            const double n  = s0_arr[i] * dv;
            const double s1 = s1_arr[i] * dv;
            const double s2 = s2_arr[i] * dv;

            const double n_safe = (n > eps_n) ? n : eps_n;
            const double u  = s1 / n_safe;
            double T  = s2 / n_safe - u * u;
            if (T < eps_T) T = eps_T;

            n_arr[i] = n_safe;
            u_arr[i] = u;
            T_arr[i] = T;
        }
        grid.sync();

        // (3) build tri-diagonal per-velocity (interior i=1..nx-2) and solve with Thomas in-place
        double local_max = 0.0;

        for (int j = tid; j < nv; j += nT) {
            if (n_inner <= 0) {
                // nothing to solve; residual stays 0 (only boundaries exist)
                continue;
            }

            const double vj   = v[j];
            const double alpha = dt_over_dx * (vj > 0.0 ? vj : 0.0);
            const double beta  = dt_over_dx * (vj < 0.0 ? -vj : 0.0);

            // row pointers for this velocity
            double* dl_row = dl + j * n_inner;
            double* dd_row = dd + j * n_inner;
            double* du_row = du + j * n_inner;
            double* B_row  = B  + j * n_inner;

            // fill coefficients and RHS
            for (int il = 0; il < n_inner; ++il) {
                const int i = il + 1; // interior cell index

                // dt/tau_i = dt * n * sqrt(T) / tau_tilde
                const double n_i = n_arr[i];
                const double T_i = T_arr[i];
                const double u_i = u_arr[i];
                const double dt_over_tau = dt * n_i * sqrt(T_i) / tau_tilde;

                // main diag and off-diags (signs follow the discrete eq):
                // -beta f_{i+1} + (1+alpha+beta+dt/tau) f_i - alpha f_{i-1} = rhs
                dd_row[il] = 1.0 + alpha + beta + dt_over_tau;
                dl_row[il] = -alpha;   // lower diag (i-1)
                du_row[il] = -beta;    // upper diag (i+1)

                // RHS = f^k_{i,j} + (dt/tau) * fM_{i,j}
                const double coeffM = n_i * inv_sqrt_2pi / sqrt(T_i);
                const double diff   = vj - u_i;
                const double expo   = -0.5 * (diff * diff) / T_i;
                double fM = coeffM * exp(expo);

                double rhs = f[i * nv + j] + dt_over_tau * fM;

                // boundary contributions (Dirichlet: fixed boundary cells from f)
                if (il == 0) {
                    // i = 1 uses f_{0,j}
                    rhs += alpha * f[0 * nv + j];
                }
                if (il == n_inner - 1) {
                    // i = nx-2 uses f_{nx-1,j}
                    rhs += beta * f[(nx - 1) * nv + j];
                }
                B_row[il] = rhs;
            }

            // Thomas forward elimination (in-place)
            for (int il = 1; il < n_inner; ++il) {
                const double w = dl_row[il] / dd_row[il - 1];
                dd_row[il]    -= w * du_row[il - 1];
                B_row[il]     -= w * B_row[il - 1];
            }

            // Thomas backward substitution (solution overwrites B_row)
            B_row[n_inner - 1] = B_row[n_inner - 1] / dd_row[n_inner - 1];
            for (int il = n_inner - 2; il >= 0; --il) {
                B_row[il] = (B_row[il] - du_row[il] * B_row[il + 1]) / dd_row[il];
            }

            // write back to fn (interior only) & compute local residual
            for (int il = 0; il < n_inner; ++il) {
                const int i = il + 1;
                const double newv = B_row[il];
                const double oldv = fn[i * nv + j];
                const double diff = fabs(newv - oldv);
                if (diff > local_max) local_max = diff;
                fn[i * nv + j] = newv;
            }
        }

        // reduce residual to global
        atomicMaxDouble(res_dev, local_max);
        grid.sync();

        // (4) convergence check (single writer)
        if (tid == 0) {
            if (*res_dev < tol) {
                *iters_dev = it + 1;
            }
        }
        grid.sync();

        if (*iters_dev != 0) break;
    }

    // boundaries must remain as input f (already true because we never overwrote them)
}

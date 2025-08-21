// src/kineticEQ/backends/imp_picard/imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// ---- helpers ----
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) { return (x > eps) ? x : eps; }

template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tg, T vj, T inv_sqrt_2pi) {
    T inv_sqrtT = rsqrt(clamp_pos(Tg, T(1e-300)));
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / clamp_pos(Tg, T(1e-300));
    return coeff * exp(expo);
}

// atomicMax for positive double
__device__ inline double atomicMaxDouble(double* addr, double val) {
    unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old  = *ull, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur >= val) break;
        old = atomicCAS(ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// 1 block per velocity j
template <typename T>
__global__ void picard_coop_kernel(
    // in/out
    const T* __restrict__ f_in,   // (nx,nv)
    T* __restrict__ fA,           // (nx,nv) ping
    T* __restrict__ fB,           // (nx,nv) pong
    const T* __restrict__ v,      // (nv)
    // const
    int nx, int nv, T dv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    int picard_iter, T picard_tol,
    // scratch (global)
    T* __restrict__ s0, T* __restrict__ s1, T* __restrict__ s2,
    T* __restrict__ n_arr, T* __restrict__ u_arr, T* __restrict__ T_arr,
    // outputs
    int* __restrict__ iters_out, T* __restrict__ resid_out)
{
    cg::grid_group grid = cg::this_grid();

    const int j  = blockIdx.x;
    if (j >= nv) return;

    const T vj   = v[j];
    const int n_inner = max(nx - 2, 0);
    const T alpha = (vj > T(0)) ? (dt/dx * vj)     : T(0);
    const T beta  = (vj < T(0)) ? (dt/dx * (-vj))  : T(0);

    // dynamic shared memory: dl, dd, du, B (length = n_inner)
    extern __shared__ unsigned char smem_raw[];
    T* dl = reinterpret_cast<T*>(smem_raw);
    T* dd = dl + n_inner;
    T* du = dd + n_inner;
    T*  B = du + n_inner;

    // ping-pong ptr
    T* fz = fA;
    T* fn = fB;

    T last_res = 0;
    int iters = 0;

    for (int z = 0; z < picard_iter; ++z) {
        // 0) clear s0,s1,s2 by grid-stride
        for (int i = threadIdx.x + blockIdx.x; i < nx; i += blockDim.x * gridDim.x) {
            s0[i] = T(0); s1[i] = T(0); s2[i] = T(0);
        }
        grid.sync();

        // 1) accumulate moments from fz
        for (int i = threadIdx.x; i < nx; i += blockDim.x) {
            T fij = fz[i*nv + j];
            atomicAdd(&s0[i], fij);
            atomicAdd(&s1[i], fij * vj);
            atomicAdd(&s2[i], fij * vj * vj);
        }
        grid.sync();

        // 2) compute n,u,T
        for (int i = threadIdx.x + blockIdx.x; i < nx; i += blockDim.x * gridDim.x) {
            T n   = s0[i] * dv;
            T s1d = s1[i] * dv;
            T s2d = s2[i] * dv;
            T u   = s1d / clamp_pos(n, T(1e-300));
            T Tg  = s2d / clamp_pos(n, T(1e-300)) - u*u;
            if (!(Tg > T(0))) Tg = T(1e-300);
            n_arr[i] = n; u_arr[i] = u; T_arr[i] = Tg;
        }
        grid.sync();

        // 3) boundary (keep previous)
        const T fL = f_in[0*nv + j];
        const T fR = f_in[(nx-1)*nv + j];

        // 4) build tri-diagonal and RHS for this j
        for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
            const int i = k + 1;
            const T n   = n_arr[i];
            const T u   = u_arr[i];
            const T Tg  = T_arr[i];
            const T inv_tau = (n * sqrt(Tg)) / tau_tilde; // 1/tau

            dd[k] = T(1) + alpha + beta + dt * inv_tau;
            dl[k] = -alpha;
            du[k] = -beta;

            const T fij = fz[i*nv + j];
            const T fM  = maxwell_1v<T>(n, u, Tg, vj, inv_sqrt_2pi);
            T rhs = fij + dt * inv_tau * fM;
            if (k == 0)           rhs += alpha * fL;
            if (k == n_inner-1)   rhs += beta  * fR;
            B[k] = rhs;
        }
        __syncthreads();

        // 5) Thomas
        if (threadIdx.x == 0) {
            for (int k = 1; k < n_inner; ++k) {
                T m = dl[k] / dd[k-1];
                dd[k] -= m * du[k-1];
                B[k]  -= m * B[k-1];
            }
            if (n_inner > 0) {
                B[n_inner-1] /= dd[n_inner-1];
                for (int k = n_inner - 2; k >= 0; --k) {
                    B[k] = (B[k] - du[k] * B[k+1]) / dd[k];
                }
            }
        }
        __syncthreads();

        // 6) writeback + local residual
        T local_max = 0;
        for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
            const int i = k + 1;
            const T oldv = fz[i*nv + j];
            const T newv = B[k];
            fn[i*nv + j] = newv;
            T diff = fabs(newv - oldv);
            if (diff > local_max) local_max = diff;
        }
        if (threadIdx.x == 0) {
            fn[0*nv + j]      = f_in[0*nv + j];
            fn[(nx-1)*nv + j] = f_in[(nx-1)*nv + j];
        }

        // global max residual
        atomicMaxDouble(reinterpret_cast<double*>(resid_out), (double)local_max);
        __syncthreads();
        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            last_res = *resid_out;
            *resid_out = 0.0; // reset for next iter
        }
        grid.sync();

        ++iters;

        // stop?
        __shared__ int s_done;
        if (threadIdx.x == 0) s_done = (blockIdx.x == 0 && last_res <= picard_tol) ? 1 : 0;
        __syncthreads();
        grid.sync();
        if (s_done) break;

        // swap
        T* tmp = fz; fz = fn; fn = tmp;
        grid.sync();
    }

    // return results
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *iters_out = iters;
        *resid_out = last_res; // ← 最終残差を書き戻す（重要）
    }
}

// 明示的実体化
template __global__ void picard_coop_kernel<double>(
    const double*, double*, double*, const double*,
    int,int,double,double,double,double,double,int,double,
    double*,double*,double*, double*,double*,double*,
    int*, double*);

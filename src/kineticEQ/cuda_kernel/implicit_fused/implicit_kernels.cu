// kineticEQ/src/kineticEQ/cuda_kernel/implicit_fused/implicit_kernels.cu

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace implicit_fused {

// ------------ utilities ---------------
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) {
    return (x > eps) ? x : eps;
}

template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tgas, T vj, T inv_sqrt_2pi) {
    // f_M = n / sqrt(2π T) * exp(-(v-u)^2/(2T))
    T Tg = clamp_pos(Tgas, (T)1e-300);
    T inv_sqrtT = rsqrt(Tg);
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / Tg;
    return coeff * exp(expo);
}

// --------------- moments (n,nu,T): 1 block per x ----------------
template <typename T>
__global__ void moments_n_nu_T_kernel(
    const T* __restrict__ fz,   // (nx*nv), row-major: i*nv + j
    const T* __restrict__ v,    // (nv)
    int nx, int nv, T dv,
    T* __restrict__ n_out,
    T* __restrict__ nu_out,
    T* __restrict__ T_out)
{
    int i = blockIdx.x;
    if (i >= nx) return;

    extern __shared__ unsigned char smem_raw[];
    T* s0 = reinterpret_cast<T*>(smem_raw);
    T* s1 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x);
    T* s2 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x*2);

    T p0 = T(0), p1 = T(0), p2 = T(0);
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        T fij = fz[i*nv + j];
        T vj  = v[j];
        p0 += fij;
        p1 += fij * vj;
        p2 += fij * vj * vj;
    }
    s0[threadIdx.x] = p0;
    s1[threadIdx.x] = p1;
    s2[threadIdx.x] = p2;
    __syncthreads();

    for (int offset = blockDim.x>>1; offset>0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s0[threadIdx.x] += s0[threadIdx.x + offset];
            s1[threadIdx.x] += s1[threadIdx.x + offset];
            s2[threadIdx.x] += s2[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        T n_raw = s0[0] * dv;
        T nu = s1[0] * dv;
        T s2d = s2[0] * dv;

        T n = clamp_pos(n_raw, (T)1e-300);
        T u = nu / n;
        T Tg = s2d / n - u*u;
        if (!(Tg > T(0))) Tg = T(1e-300);

        n_out[i] = n;
        nu_out[i] = nu;
        T_out[i] = Tg;
    }
}

// --------------- build from precomputed moments: 1 block per v ----------------
template <typename T>
__global__ void build_system_from_moments_kernel(
    const T* __restrict__ B0,       // (nv, n_inner) contiguous
    const T* __restrict__ v,        // (nv)
    const T* __restrict__ n,        // (nx)
    const T* __restrict__ nu,       // (nx)
    const T* __restrict__ Tg,       // (nx)
    const T* __restrict__ f_bc_l,   // (nv)
    const T* __restrict__ f_bc_r,   // (nv)
    int nx, int nv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    T* __restrict__ dl,             // (nv, n_inner)
    T* __restrict__ dd,             // (nv, n_inner)
    T* __restrict__ du,             // (nv, n_inner)
    T* __restrict__ B)              // (nv, n_inner)
{
    int j = blockIdx.x;
    if (j >= nv) return;

    const int n_inner = nx - 2;
    const T vj = v[j];

    const T alpha = (vj > T(0)) ? (dt/dx * vj) : T(0);
    const T beta  = (vj < T(0)) ? (dt/dx * (-vj)) : T(0);

    T* dl_j = dl + ((size_t)j) * n_inner;
    T* dd_j = dd + ((size_t)j) * n_inner;
    T* du_j = du + ((size_t)j) * n_inner;
    T*  B_j =  B + ((size_t)j) * n_inner;
    const T* B0_j = B0 + ((size_t)j) * n_inner;

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1;
        const T n_i = clamp_pos(n[i], (T)1e-300);
        const T T_i = clamp_pos(Tg[i], (T)1e-300);
        const T u_i = nu[i] / n_i;

        const T inv_tau = (n_i * sqrt(T_i)) / tau_tilde;

        dd_j[k] = T(1) + alpha + beta + dt * inv_tau;
        dl_j[k] = (k == 0) ? T(0) : (-alpha);
        du_j[k] = (k == n_inner - 1) ? T(0) : (-beta);

        const T fM_ij = maxwell_1v<T>(n_i, u_i, T_i, vj, inv_sqrt_2pi);
        T rhs = B0_j[k] + dt * inv_tau * fM_ij;

        if (k == 0) rhs += alpha * f_bc_l[j];
        if (k == n_inner - 1) rhs += beta * f_bc_r[j];

        B_j[k] = rhs;
    }
}


// ================== launchers ==================
void launch_moments_n_nu_T_double(
    const double* fz, const double* v,
    int nx, int nv, double dv,
    double* n, double* nu, double* T,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nx);
    size_t shmem = sizeof(double) * block * 3;
    moments_n_nu_T_kernel<double><<<grid, block, shmem, stream>>>(
        fz, v, nx, nv, dv, n, nu, T);
}

void launch_build_system_from_moments_double(
    const double* B0, const double* v,
    const double* n, const double* nu, const double* T,
    const double* f_bc_l, const double* f_bc_r,
    int nx, int nv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_system_from_moments_kernel<double><<<grid, block, 0, stream>>>(
        B0, v, n, nu, T, f_bc_l, f_bc_r,
        nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl, dd, du, B);
}

} // namespace implicit_fused

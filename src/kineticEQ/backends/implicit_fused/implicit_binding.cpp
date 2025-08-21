#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <assert.h>

namespace implicit_fused {

// ---------------- Utility ----------------
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) {
    return (x > eps) ? x : eps;
}

template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tgas, T vj, T inv_sqrt_2pi) {
    // fM = n / sqrt(2π T) * exp(-(v-u)^2/(2T))
    T Tpos = clamp_pos(Tgas, T(1e-300));
    T inv_sqrtT = rsqrt(Tpos);
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / Tpos;
    return coeff * exp(expo);
}

// ---------------- Moments kernel ----------------
// 1 block per row (x), reduction in shared memory.
template <typename T>
__global__ void moments_kernel(
    const T* __restrict__ f,   // (nx, nv)
    const T* __restrict__ v,   // (nv)
    int nx, int nv, T dv,
    T* __restrict__ n_out,     // (nx)
    T* __restrict__ u_out,     // (nx)
    T* __restrict__ T_out)     // (nx)
{
    const int i = blockIdx.x;
    if (i >= nx) return;

    extern __shared__ unsigned char smem_raw[];
    T* s0 = reinterpret_cast<T*>(smem_raw);
    T* s1 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x);
    T* s2 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x*2);

    T p0 = T(0), p1 = T(0), p2 = T(0);
    const T* fi = f + i*nv;

    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        T fij = fi[j];
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
        T n  = s0[0] * dv;
        T s1d= s1[0] * dv;
        T s2d= s2[0] * dv;
        T u  = s1d / n;
        T Tg = s2d / n - u*u;
        if (!(Tg > T(0))) Tg = T(1e-300);
        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = Tg;
    }
}

inline void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nx);
    size_t shmem = sizeof(double) * block * 3;
    moments_kernel<double><<<grid, block, shmem, stream>>>(
        f, v, nx, nv, dv, n, u, T);
}

// ---------------- Build A,B,C and RHS (B) ----------------
// 1 block per velocity j. Build lower/diag/upper and RHS for interior cells k=0..n_inner-1 (i=k+1)
template <typename T>
__global__ void build_abcB_kernel(
    const T* __restrict__ f,    // (nx, nv)
    const T* __restrict__ v,    // (nv)
    const T* __restrict__ n,    // (nx)
    const T* __restrict__ u,    // (nx)
    const T* __restrict__ Tg,   // (nx)
    int nx, int nv,
    T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    T* __restrict__ a_out,      // (nv, n_inner)
    T* __restrict__ b_out,      // (nv, n_inner)
    T* __restrict__ c_out,      // (nv, n_inner)
    T* __restrict__ B_out)      // (nv, n_inner)
{
    const int j = blockIdx.x;
    if (j >= nv) return;

    const int n_inner = nx - 2;
    const T vj = v[j];

    // coefficients
    const T a_coeff = -dt/dx * fmax(vj,  T(0.0));  // <= 0
    const T c_coeff = -dt/dx * fmax(-vj, T(0.0));  // <= 0
    const T minus_a = -a_coeff;
    const T minus_c = -c_coeff;

    T* aj = a_out + j * n_inner;
    T* bj = b_out + j * n_inner;
    T* cj = c_out + j * n_inner;
    T* Bj = B_out + j * n_inner;

    // boundary Maxwellians for RHS advection part
    const T fL = maxwell_1v<T>(n[0],      u[0],      Tg[0],      vj, inv_sqrt_2pi);
    const T fR = maxwell_1v<T>(n[nx-1],   u[nx-1],   Tg[nx-1],   vj, inv_sqrt_2pi);

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1; // interior row index

        const T inv_tau = (n[i] * sqrt(Tg[i])) / tau_tilde;

        // diagonals
        bj[k] = T(1) + minus_a + minus_c + dt * inv_tau;
        aj[k] = (k==0)         ? T(0) : a_coeff;
        cj[k] = (k==n_inner-1) ? T(0) : c_coeff;

        // RHS
        const T fij = f[i*nv + j];
        const T fMi = maxwell_1v<T>(n[i], u[i], Tg[i], vj, inv_sqrt_2pi);
        T rhs = fij + dt * inv_tau * fMi;

        // boundary advection contributions
        if (k == 0)         rhs += (dt/dx) * fmax(vj,  T(0.0)) * fL;
        if (k == n_inner-1) rhs += (dt/dx) * fmax(-vj, T(0.0)) * fR;

        Bj[k] = rhs;
    }
}

inline void launch_build_abcB_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* a, double* b, double* c, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_abcB_kernel<double><<<grid, block, 0, stream>>>(
        f, v, n, u, T, nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi, a, b, c, B);
}

} // namespace implicit_fused

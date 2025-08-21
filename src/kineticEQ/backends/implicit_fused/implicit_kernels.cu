#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace implicit_fused {

// clamp positive
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) {
    return (x > eps) ? x : eps;
}

// Maxwellian for one (n,u,T) and one v
template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tgas, T vj, T inv_sqrt_2pi) {
    T inv_sqrtT = rsqrt(Tgas);
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / Tgas;
    return coeff * exp(expo);
}

/* ---------------- moments: 1 block per i ---------------- */
template <typename T>
__global__ void moments_kernel(
    const T* __restrict__ f,
    const T* __restrict__ v,
    int nx, int nv, T dv,
    T* __restrict__ n_out,
    T* __restrict__ u_out,
    T* __restrict__ T_out)
{
    const int i = blockIdx.x;
    if (i >= nx) return;

    extern __shared__ unsigned char smem_raw[];
    T* s0 = reinterpret_cast<T*>(smem_raw);
    T* s1 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x);
    T* s2 = reinterpret_cast<T*>(smem_raw + sizeof(T)*blockDim.x*2);

    T p0 = T(0), p1 = T(0), p2 = T(0);
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        T fij = f[i*nv + j];
        T vj  = v[j];
        p0 += fij;
        p1 += fij * vj;
        p2 += fij * vj * vj;
    }
    s0[threadIdx.x] = p0;
    s1[threadIdx.x] = p1;
    s2[threadIdx.x] = p2;
    __syncthreads();

    for (int off = blockDim.x>>1; off>0; off >>= 1) {
        if (threadIdx.x < off) {
            s0[threadIdx.x] += s0[threadIdx.x + off];
            s1[threadIdx.x] += s1[threadIdx.x + off];
            s2[threadIdx.x] += s2[threadIdx.x + off];
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

void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nx);
    size_t shmem = sizeof(double) * block * 3;
    moments_kernel<double><<<grid, block, shmem, stream>>>(f, v, nx, nv, dv, n, u, T);
}

/* ------------- boundary from current f (no touch) ------------- */
template <typename T>
__global__ void boundary_from_f_kernel(
    const T* __restrict__ f, int nx, int nv,
    T* __restrict__ fL, T* __restrict__ fR)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nv) return;
    fL[j] = f[0*nv + j];
    fR[j] = f[(nx-1)*nv + j];
}

void launch_boundary_from_current_f_double(
    const double* f, int nx, int nv,
    double* fL, double* fR,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (nv + block - 1) / block;
    boundary_from_f_kernel<double><<<grid, block, 0, stream>>>(f, nx, nv, fL, fR);
}

/* ------------- build tri-diagonal & RHS for interior ------------- */
template <typename T>
__global__ void build_tridiag_rhs_kernel(
    const T* __restrict__ f,
    const T* __restrict__ v,
    const T* __restrict__ n,
    const T* __restrict__ u,
    const T* __restrict__ Tg,
    const T* __restrict__ fL,
    const T* __restrict__ fR,
    int nx, int nv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    T* __restrict__ dl,  // (nv, m)
    T* __restrict__ dd,  // (nv, m)
    T* __restrict__ du,  // (nv, m)
    T* __restrict__  B)  // (nv, m)
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int m = nx - 2;
    const T vj = v[j];
    const T ap = -dt/dx * fmax(vj,  T(0));   // <=0
    const T cp = -dt/dx * fmax(-vj, T(0));   // <=0
    const T ma = -ap; // >=0
    const T mc = -cp; // >=0

    T* dl_j = dl + j * m;
    T* dd_j = dd + j * m;
    T* du_j = du + j * m;
    T*  B_j =  B + j * m;

    const T fL_j = fL[j];
    const T fR_j = fR[j];

    for (int k = threadIdx.x; k < m; k += blockDim.x) {
        const int i = k + 1; // interior row (1..nx-2)

        const T Tg_i = clamp_pos(Tg[i], T(1e-300));
        const T inv_tau = (n[i] * sqrt(Tg_i)) / tau_tilde;

        dd_j[k] = T(1) + ma + mc + dt * inv_tau;
        dl_j[k] = (k==0)     ? T(0) : ap;
        du_j[k] = (k==m-1)   ? T(0) : cp;

        const T fij = f[i*nv + j];
        const T fM  = maxwell_1v<T>(n[i], u[i], Tg_i, vj, inv_sqrt_2pi);

        T rhs = fij + dt * inv_tau * fM;
        if (k == 0)     rhs += (dt/dx) * fmax(vj,  T(0)) * fL_j;
        if (k == m-1)   rhs += (dt/dx) * fmax(-vj, T(0)) * fR_j;

        B_j[k] = rhs;
    }
}

void launch_build_tridiag_rhs_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    const double* fL, const double* fR,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* d, double* du, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_tridiag_rhs_kernel<double><<<grid, block, 0, stream>>>(
        f, v, n, u, T, fL, fR, nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi, dl, d, du, B);
}

/* ------------- residual (interior only) ------------- */
template <typename T>
__global__ void residual_interior_kernel(
    const T* __restrict__ f_old,
    const T* __restrict__ f_new,
    int nx, int nv,
    T* __restrict__ res_per_v) // length nv
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    T rmax = T(0);
    for (int i = 1; i < nx-1; ++i) {
        T diff = fabs(f_new[i*nv + j] - f_old[i*nv + j]);
        if (diff > rmax) rmax = diff;
    }
    if (threadIdx.x == 0) res_per_v[j] = rmax;
}

void launch_residual_interior_double(
    const double* f_old, const double* f_new,
    int nx, int nv, double* res_per_v,
    cudaStream_t stream)
{
    dim3 grid(nv), block(1);
    residual_interior_kernel<double><<<grid, block, 0, stream>>>(f_old, f_new, nx, nv, res_per_v);
}

} // namespace implicit_fused

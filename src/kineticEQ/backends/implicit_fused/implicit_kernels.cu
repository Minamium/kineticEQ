#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <assert.h>

namespace implicit_fused {

// ================= Utility =================
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) {
    return (x > eps) ? x : eps;
}

// Maxwellian for one (n,u,T) row and one v
template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tgas, T vj, T inv_sqrt_2pi) {
    // fM = n / sqrt(2π T) * exp(-(v-u)^2/(2T))
    T inv_sqrtT = rsqrt(Tgas);
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / Tgas;
    return coeff * exp(expo);
}

// ================= Moments kernel =================
// 1 block per row (x), reduction in shared memory.
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

// ================= Build tri-diagonal & RHS =================
// 1 block per velocity j. Build dl/d/du/B for interior cells k=0..n_inner-1 (i=k+1)
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
    T* __restrict__ dl,  // (nv, n_inner)
    T* __restrict__ dd,  // (nv, n_inner)
    T* __restrict__ du,  // (nv, n_inner)
    T* __restrict__ B)   // (nv, n_inner)
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int n_inner = nx - 2;
    const T vj = v[j];
    const T ap = -dt/dx * fmax(vj, T(0));   // a_coeff (<=0)
    const T cp = -dt/dx * fmax(-vj, T(0));  // c_coeff (<=0)
    const T minus_a = -ap; // = dt/dx*max(v,0)
    const T minus_c = -cp; // = dt/dx*max(-v,0)

    T* dl_j = dl + j * n_inner;
    T* dd_j = dd + j * n_inner;
    T* du_j = du + j * n_inner;
    T*  B_j =  B + j * n_inner;

    // boundary Maxwellians for this velocity
    const T fL_j = fL[j];
    const T fR_j = fR[j];

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1; // interior row index in [1..nx-2]

        // inv_tau = (n*sqrt(T))/tau_tilde
        const T sqrtT = sqrt(Tg[i]);
        const T inv_tau = (n[i] * sqrtT) / tau_tilde;

        // diagonal
        dd_j[k] = T(1) + minus_a + minus_c + dt * inv_tau;
        // sub & super
        dl_j[k] = (k==0) ? T(0) : ap;
        du_j[k] = (k==n_inner-1) ? T(0) : cp;

        // Maxwellian at interior cell
        const T fM = maxwell_1v<T>(n[i], u[i], Tg[i], vj, inv_sqrt_2pi);

        // RHS
        const T fij = f[i*nv + j];
        T rhs = fij + dt * inv_tau * fM;

        // boundary advection contributions
        if (k == 0)        rhs += (dt/dx) * fmax(vj,  T(0)) * fL_j;
        if (k == n_inner-1)rhs += (dt/dx) * fmax(-vj, T(0)) * fR_j;

        B_j[k] = rhs;
    }
}

// ================= Batched TDMA solve + writeback + residual =================
// 1 block per velocity j. Thread 0 runs sequential Thomas for stability/simplicity.
// Then write back interior to fn and set boundaries from fL/fR.
// Also compute per-velocity residual max |fn - f| across the column.
template <typename T>
__global__ void tdma_solve_and_writeback_kernel(
    const T* __restrict__ f,
    T* __restrict__ fn,
    const T* __restrict__ fL,
    const T* __restrict__ fR,
    T* __restrict__ dl,  // (nv, n_inner)
    T* __restrict__ dd,  // (nv, n_inner)
    T* __restrict__ du,  // (nv, n_inner)
    T* __restrict__ B,   // (nv, n_inner)  -> solution overwritten
    int nx, int nv,
    T* __restrict__ res_per_v)
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int n_inner = nx - 2;

    T local_max = T(0);

    // boundaries
    const T fL_j = fL[j];
    const T fR_j = fR[j];

    // Thomas (sequential in thread 0)
    if (threadIdx.x == 0) {
        T* dl_j = dl + j * n_inner;
        T* dd_j = dd + j * n_inner;
        T* du_j = du + j * n_inner;
        T*  B_j =  B + j * n_inner;

        // forward elimination
        for (int k = 1; k < n_inner; ++k) {
            T m = dl_j[k] / dd_j[k-1];
            dd_j[k] -= m * du_j[k-1];
            B_j[k]  -= m * B_j[k-1];
        }
        // back substitution
        B_j[n_inner - 1] = B_j[n_inner - 1] / dd_j[n_inner - 1];
        for (int k = n_inner - 2; k >= 0; --k) {
            B_j[k] = (B_j[k] - du_j[k] * B_j[k+1]) / dd_j[k];
        }

        // write back interior
        for (int k = 0; k < n_inner; ++k) {
            const int i = k + 1;
            const T oldv = f[i*nv + j];
            const T newv = B_j[k];
            fn[i*nv + j] = newv;
            const T diff = fabs(newv - oldv);
            if (diff > local_max) local_max = diff;
        }

        // boundaries
        {
            const T old0 = f[0*nv + j];
            const T new0 = fL_j;
            fn[0*nv + j] = new0;
            const T diff0 = fabs(new0 - old0);
            if (diff0 > local_max) local_max = diff0;
        }
        {
            const T oldN = f[(nx-1)*nv + j];
            const T newN = fR_j;
            fn[(nx-1)*nv + j] = newN;
            const T diffN = fabs(newN - oldN);
            if (diffN > local_max) local_max = diffN;
        }

        res_per_v[j] = local_max;
    }
}

// ================= Launchers =================
void launch_moments_double(
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

void launch_tdma_solve_and_writeback_double(
    const double* f, double* fn,
    const double* fL, const double* fR,
    const double* dl, const double* d, const double* du, double* B,
    int nx, int nv,
    double* res_per_v,
    cudaStream_t stream)
{
    const int block = 32; // only thread 0 used; small block to keep occupancy reasonable
    dim3 grid(nv);
    tdma_solve_and_writeback_kernel<double><<<grid, block, 0, stream>>>(
        f, fn, fL, fR, const_cast<double*>(dl), const_cast<double*>(d),
        const_cast<double*>(du), B, nx, nv, res_per_v);
}

template <typename T>
__global__ void boundary_maxwell_kernel(
    const T* __restrict__ v, int nv, T inv_sqrt_2pi,
    T nL, T uL, T TL, T nR, T uR, T TR,
    T* __restrict__ fL, T* __restrict__ fR)
{
    TL = clamp_pos(TL, T(1e-300));
    TR = clamp_pos(TR, T(1e-300));
    T inv_sqrtTL = rsqrt(TL);
    T inv_sqrtTR = rsqrt(TR);
    T coeffL = nL * inv_sqrt_2pi * inv_sqrtTL;
    T coeffR = nR * inv_sqrt_2pi * inv_sqrtTR;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nv; j += blockDim.x * gridDim.x) {
        T vj = v[j];
        T eL = exp(-T(0.5) * (vj-uL)*(vj-uL) / TL);
        T eR = exp(-T(0.5) * (vj-uR)*(vj-uR) / TR);
        fL[j] = coeffL * eL;
        fR[j] = coeffR * eR;
    }
}

void launch_boundary_maxwell_double(
    const double* v, int nv, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    double* fL, double* fR,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (nv + block - 1) / block;
    boundary_maxwell_kernel<double><<<grid, block, 0, stream>>>(
        v, nv, inv_sqrt_2pi, nL, uL, TL, nR, uR, TR, fL, fR);
}

} // namespace implicit_fused

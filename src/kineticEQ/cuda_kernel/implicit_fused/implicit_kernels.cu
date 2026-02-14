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

// --------------- moments: 1 block per x ----------------
template <typename T>
__global__ void moments_kernel(
    const T* __restrict__ fz,   // (nx*nv), row-major: i*nv + j
    const T* __restrict__ v,    // (nv)
    int nx, int nv, T dv,
    T* __restrict__ n_out,
    T* __restrict__ u_out,
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
        T n  = s0[0] * dv;
        T s1d= s1[0] * dv;
        T s2d= s2[0] * dv;

        T u   = s1d / n;
        T Tg  = s2d / n - u*u;
        if (!(Tg > T(0))) Tg = T(1e-300);

        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = Tg;
    }
}

// --------------- build (dl,dd,du,B): 1 block per v ----------------
// 内部未知数: i=1..nx-2 → k=i-1, k=0..n_inner-1
// 符号は提示式に厳密に合わせる：
//   dl (=subdiag) = -alpha_j (k>0 のみ; k=0 は 0)
//   du (=superdiag)= -beta_j  (k<n_inner-1 のみ; 末端は 0)
//   dd (=diag)     = 1 + alpha_j + beta_j + dt/τ_i
//   B              = f_prev[i,j] + dt/τ_i * fM_i,j
// 境界寄与：
//   B[k=0]         += alpha_j * fM(i=0,j)      // v_j>0 側（左境界）
//   B[k=n_inner-1] += beta_j  * fM(i=nx-1,j)   // v_j<0 側（右境界）
template <typename T>
__global__ void build_system_fused_kernel(
    const T* __restrict__ f_prev,   // (nx*nv)
    const T* __restrict__ fz,       // (nx*nv)
    const T* __restrict__ v,        // (nv)
    const T* __restrict__ n,        // (nx)
    const T* __restrict__ u,        // (nx)
    const T* __restrict__ Tg,       // (nx)
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

    // α, β（正の係数）
    const T alpha = (vj > T(0)) ? (dt/dx * vj) : T(0);
    const T beta  = (vj < T(0)) ? (dt/dx * (-vj)) : T(0);

    // 境界の Maxwell は fz の i=0, nx-1 から（参照式に合わせる）
    const T fM_L = maxwell_1v<T>(n[0], u[0], Tg[0], vj, inv_sqrt_2pi);
    const T fM_R = maxwell_1v<T>(n[nx-1], u[nx-1], Tg[nx-1], vj, inv_sqrt_2pi);

    T* dl_j = dl + ((size_t)j) * n_inner;
    T* dd_j = dd + ((size_t)j) * n_inner;
    T* du_j = du + ((size_t)j) * n_inner;
    T*  B_j =  B + ((size_t)j) * n_inner;

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1; // interior i = 1..nx-2

        const T inv_tau = (n[i] * sqrt(clamp_pos(Tg[i], (T)1e-300))) / tau_tilde;

        // 対角
        dd_j[k] = T(1) + alpha + beta + dt * inv_tau;

        // 下対角（subdiag）
        dl_j[k] = (k==0) ? T(0) : (-alpha);

        // 上対角（superdiag）
        du_j[k] = (k==n_inner-1) ? T(0) : (-beta);

        // 右辺（f は “前ステップ” f_prev）
        const T fij_prev = f_prev[i*nv + j];
        const T fM_ij = maxwell_1v<T>(n[i], u[i], Tg[i], vj, inv_sqrt_2pi);
        T rhs = fij_prev + dt * inv_tau * fM_ij;

        // 境界由来の移流寄与（参照式）
        if (k == 0)          rhs += alpha * fM_L;
        if (k == n_inner-1)  rhs += beta  * fM_R;

        B_j[k] = rhs;
    }
}


// ================== launchers ==================
void launch_moments_double(
    const double* fz, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nx);
    size_t shmem = sizeof(double) * block * 3;
    moments_kernel<double><<<grid, block, shmem, stream>>>(
        fz, v, nx, nv, dv, n, u, T);
}

void launch_build_system_fused_double(
    const double* f_prev, const double* fz, const double* v,
    const double* n, const double* u, const double* T,
    int nx, int nv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_system_fused_kernel<double><<<grid, block, 0, stream>>>(
        f_prev, fz, v, n, u, T,
        nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl, dd, du, B);
}

} // namespace implicit_fused

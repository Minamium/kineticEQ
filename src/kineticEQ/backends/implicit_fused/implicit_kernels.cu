// implicit_kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace implicit_fused {

// ====================== helpers ======================
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

// ====================== moments: one block per row (x) ======================
template <typename T>
__global__ void moments_kernel(
    const T* __restrict__ f,  // (nx*nv)
    const T* __restrict__ v,  // (nv)
    int nx, int nv, T dv,
    T* __restrict__ n_out,    // (nx)
    T* __restrict__ u_out,    // (nx)
    T* __restrict__ T_out)    // (nx)
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
        T n   = s0[0] * dv;
        T s1d = s1[0] * dv;
        T s2d = s2[0] * dv;
        T u   = s1d / n;
        T Tg  = s2d / n - u*u;
        if (!(Tg > T(0))) Tg = T(1e-300);
        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = Tg;
    }
}

// ====================== build tri-diagonal & RHS: one block per velocity ======================
// NOTE: RHS uses f_prev (前タイムステップ). n,u,T は現在ピカード反復の f_iter から計算して与える。
// 境界流入は f_prev の境界値を使用（Maxwell BC は使わない）。
template <typename T>
__global__ void build_tridiag_rhs_kernel(
    const T* __restrict__ f_prev, // (nx*nv) ← RHS の f^n
    const T* __restrict__ v,      // (nv)
    const T* __restrict__ n,      // (nx)
    const T* __restrict__ u,      // (nx)
    const T* __restrict__ Tg,     // (nx)
    int nx, int nv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    T* __restrict__ dl,           // (nv, n_inner)
    T* __restrict__ dd,           // (nv, n_inner)
    T* __restrict__ du,           // (nv, n_inner)
    T* __restrict__ B)            // (nv, n_inner) ← cuSOLVER 解で上書きされる
{
    const int j = blockIdx.x;
    if (j >= nv) return;

    const int n_inner = nx - 2;
    const T vj = v[j];
    const T ap = -dt/dx * fmax(vj,  T(0));  // 下対角（<=0）
    const T cp = -dt/dx * fmax(-vj, T(0));  // 上対角（<=0）
    const T minus_a = -ap; // dt/dx * max(v,0)
    const T minus_c = -cp; // dt/dx * max(-v,0)

    T* dl_j = dl + j * n_inner;
    T* dd_j = dd + j * n_inner;
    T* du_j = du + j * n_inner;
    T*  B_j =  B + j * n_inner;

    // 入力 f の境界値（RHS の境界流入に使用）
    const T fL_j = f_prev[0 * nv + j];
    const T fR_j = f_prev[(nx-1) * nv + j];

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1; // interior cell index [1..nx-2]

        // inv_tau = (n*sqrt(T))/tau_tilde
        T sqrtT = sqrt(Tg[i]);
        T inv_tau = (n[i] * sqrtT) / tau_tilde;

        // 三重対角
        dd_j[k] = T(1) + minus_a + minus_c + dt * inv_tau;
        dl_j[k] = (k==0)           ? T(0) : ap;
        du_j[k] = (k==n_inner-1)   ? T(0) : cp;

        // Maxwellian at interior cell (i)
        T fM = maxwell_1v<T>(n[i], u[i], Tg[i], vj, inv_sqrt_2pi);

        // RHS = f_prev(i,j) + dt*inv_tau*fM + boundary advection terms
        T fij_prev = f_prev[i*nv + j];
        T rhs = fij_prev + dt * inv_tau * fM;

        if (k == 0)         rhs += (dt/dx) * fmax(vj,  T(0)) * fL_j;
        if (k == n_inner-1) rhs += (dt/dx) * fmax(-vj, T(0)) * fR_j;

        B_j[k] = rhs;
    }
}

// ====================== writeback interior & residual: one block per velocity ======================
// B には cuSOLVER で解が上書き済みとする（形状 (nv, n_inner)）。
// 内部セルだけ fn に書き戻し、残差は max_{i=1..nx-2} |fn(i,j) - f_iter(i,j)| を列毎に計算。
// 境界セル fn[0,:], fn[-1,:] は一切書かない（Python側で最後に上書きする想定）。
template <typename T>
__global__ void writeback_and_residual_kernel(
    const T* __restrict__ f_iter, // (nx*nv) ← ピカード直前反復 f^{(k)}
    const T* __restrict__ B,      // (nv, n_inner) 解
    int nx, int nv,
    T* __restrict__ fn,           // (nx*nv) 出力（内部のみ書く）
    T* __restrict__ res_per_v)    // (nv) 列毎の最大残差
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int n_inner = nx - 2;

    T local_max = T(0);

    if (threadIdx.x == 0) {
        const T* Bj = B + j * n_inner;
        for (int k = 0; k < n_inner; ++k) {
            const int i = k + 1;
            const T newv = Bj[k];
            const T oldv = f_iter[i*nv + j];
            fn[i*nv + j] = newv;               // 内部のみ書き戻す
            T diff = fabs(newv - oldv);
            if (diff > local_max) local_max = diff;
        }
        res_per_v[j] = local_max;
    }
}

// ====================== launchers (double) ======================
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
    const double* f_prev, const double* v,
    const double* n, const double* u, const double* T,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_tridiag_rhs_kernel<double><<<grid, block, 0, stream>>>(
        f_prev, v, n, u, T,
        nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl, dd, du, B);
}

void launch_writeback_and_residual_double(
    const double* f_iter, const double* B,
    int nx, int nv,
    double* fn, double* res_per_v,
    cudaStream_t stream)
{
    const int block = 32;   // thread 0 使用、軽量でOK
    dim3 grid(nv);
    writeback_and_residual_kernel<double><<<grid, block, 0, stream>>>(
        f_iter, B, nx, nv, fn, res_per_v);
}

} // namespace implicit_fused

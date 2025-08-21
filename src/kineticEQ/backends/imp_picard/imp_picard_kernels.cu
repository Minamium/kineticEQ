// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace imp_picard {

__device__ __forceinline__ int idx2(int i, int j, int nv) {
    return i * nv + j; // row-major: (nx, nv)
}
__device__ __forceinline__ int idx2_vn(int j, int i_inner, int n_inner) {
    return j * n_inner + i_inner; // (nv, n_inner)
}

// ---- atomicMax for double (CAS) ----
__device__ inline double atomicMaxDouble(double* address, double val) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr_as_ull, assumed;
    while (true) {
        double old_val = __longlong_as_double(old);
        if (old_val >= val) return old_val;
        assumed = old;
        unsigned long long desired = __double_as_longlong(val);
        old = atomicCAS(addr_as_ull, assumed, desired);
        if (assumed == old) return __longlong_as_double(old);
    }
}

// ======================
// 1) モーメント（s0,s1,s2→ n,u,T）
//    1ブロック=1セル i、threads で v を分担
// ======================
__global__ void moments_kernel_double(
    const double* __restrict__ fz,
    const double* __restrict__ v,
    int nx, int nv, double dv,
    double* __restrict__ n_out,
    double* __restrict__ u_out,
    double* __restrict__ T_out)
{
    int i = blockIdx.x;           // cell
    if (i >= nx) return;

    extern __shared__ double shm[];
    double* s0 = shm;             // blockDim.x
    double* s1 = s0 + blockDim.x; // blockDim.x
    double* s2 = s1 + blockDim.x; // blockDim.x

    double t0 = 0.0, t1 = 0.0, t2 = 0.0;
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        double fij = fz[idx2(i, j, nv)];
        double vj  = v[j];
        t0 += fij;
        t1 += vj * fij;
        t2 += vj * vj * fij;
    }
    s0[threadIdx.x] = t0;
    s1[threadIdx.x] = t1;
    s2[threadIdx.x] = t2;
    __syncthreads();

    // 共有メモリで reduce
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s0[threadIdx.x] += s0[threadIdx.x + offset];
            s1[threadIdx.x] += s1[threadIdx.x + offset];
            s2[threadIdx.x] += s2[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double S0 = s0[0] * dv;             // ∫ f dv
        double n  = fmax(S0, 1e-300);       // avoid zero
        double u  = s1[0] / fmax(s0[0], 1e-300); // average by Σ f (≡ S0/dv)
        double T  = s2[0] / fmax(s0[0], 1e-300) - u*u;
        T = fmax(T, 1e-12);

        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = T;
    }
}

// ======================
// 2) 帯行列と RHS 構築（内部セルのみ）
//    grid.x = nv, grid.y = ceil(n_inner / tile)
// ======================
__global__ void build_system_kernel_double(
    const double* __restrict__ fk,   // (nx, nv) 前ステップ
    const double* __restrict__ v,    // (nv)
    const double* __restrict__ n_arr,// (nx)
    const double* __restrict__ u_arr,// (nx)
    const double* __restrict__ T_arr,// (nx)
    int nx, int nv, int n_inner,     // n_inner = nx-2
    double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* __restrict__ dl,         // (nv, n_inner)
    double* __restrict__ dd,         // (nv, n_inner)
    double* __restrict__ du,         // (nv, n_inner)
    double* __restrict__ B)          // (nv, n_inner)
{
    int j = blockIdx.x;  // velocity
    if (j >= nv) return;

    const int tile = blockDim.x;
    int inner_idx = blockIdx.y * tile + threadIdx.x; // 0..n_inner-1
    if (inner_idx >= n_inner) return;
    int i = inner_idx + 1; // global cell index

    double vj = v[j];
    double alpha = (vj > 0.0) ? (dt/dx)*vj : 0.0;
    double beta  = (vj < 0.0) ? (dt/dx)*(-vj) : 0.0;

    double n  = n_arr[i];
    double u  = u_arr[i];
    double T  = T_arr[i];
    double inv_sqrt_T = rsqrt(T);
    double arg = (vj - u);
    double fM = n * inv_sqrt_2pi * inv_sqrt_T * exp(-0.5 * (arg*arg) / T);

    double tau = tau_tilde / fmax(n, 1e-300);
    double dt_over_tau = dt / tau;

    // 帯行列
    dl[idx2_vn(j, inner_idx, n_inner)] = beta;
    dd[idx2_vn(j, inner_idx, n_inner)] = 1.0 + alpha + beta + dt_over_tau;
    du[idx2_vn(j, inner_idx, n_inner)] = alpha;

    // 右辺
    double rhs = fk[idx2(i, j, nv)] + dt_over_tau * fM;

    // 境界寄与（i=1 なら左、i=nx-2 なら右）
    if (inner_idx == 0) {
        rhs += alpha * fk[idx2(0, j, nv)];
    }
    if (inner_idx == (n_inner - 1)) {
        rhs += beta  * fk[idx2(nx-1, j, nv)];
    }
    B[idx2_vn(j, inner_idx, n_inner)] = rhs;
}

// ======================
// 3) 解の書き戻し＋L∞残差
//    solution: (nv, n_inner) を内部セルへ散布
// ======================
__global__ void scatter_and_residual_kernel_double(
    const double* __restrict__ fz,        // (nx, nv) 旧候補
    const double* __restrict__ solution,  // (nv, n_inner)
    int nx, int nv, int n_inner,
    double* __restrict__ fn_out,          // (nx, nv) 新候補
    double* __restrict__ res_dev)         // 1 スカラ（初期値 0）
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = n_inner * nv;
    if (tid >= N) return;

    int j = tid % nv;
    int inner_idx = tid / nv; // 0..n_inner-1
    int i = inner_idx + 1;    // global i

    double newv = solution[idx2_vn(j, inner_idx, n_inner)];
    double oldv = fz[idx2(i, j, nv)];
    fn_out[idx2(i, j, nv)] = newv;

    double diff = fabs(newv - oldv);
    atomicMaxDouble(res_dev, diff);

    // 端のときに境界もついでにコピー（余分だが軽い）
    if (inner_idx == 0) {
        fn_out[idx2(0, j, nv)]     = fz[idx2(0, j, nv)];
    }
    if (inner_idx == n_inner - 1) {
        fn_out[idx2(nx-1, j, nv)]  = fz[idx2(nx-1, j, nv)];
    }
}

} // namespace imp_picard
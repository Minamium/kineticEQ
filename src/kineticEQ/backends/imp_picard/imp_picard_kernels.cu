// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include <algorithm>

namespace imp_picard {

__device__ __forceinline__
double maxwell_ij(double n, double u, double T, double vj, double inv_sqrt_2pi)
{
    // T は正である前提（上流で clamp 済み推奨）
    double coeff = (n * inv_sqrt_2pi) / sqrt(T);
    double invT2 = 0.5 / T;                       // 1 / (2T)
    double d = vj - u;
    return coeff * exp(-d*d * invT2);
}

// ===== moments: n,u,T を各 i(0..nx-1) で計算 =====
// gridDim.x = nx, blockDim.x = t (e.g., 256)
__global__ void moments_kernel_double(
    const double* __restrict__ f,   // (nx, nv)
    const double* __restrict__ v,   // (nv,)
    int nx, int nv, double dv,
    double* __restrict__ n_out,     // (nx,)
    double* __restrict__ u_out,     // (nx,)
    double* __restrict__ T_out      // (nx,)
){
    int i = blockIdx.x;   // 行（空間）
    if (i >= nx) return;

    // 行先頭ポインタ
    const double* fi = f + (size_t)i * nv;

    // 並列に s0, s1, s2 を集計
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        double fij = fi[j];
        double vj  = v[j];
        s0 += fij;
        s1 += fij * vj;
        s2 += fij * vj * vj;
    }

    // ブロック内 reduce（単純加算）
    __shared__ double sh0[256];
    __shared__ double sh1[256];
    __shared__ double sh2[256];
    sh0[threadIdx.x] = s0;
    sh1[threadIdx.x] = s1;
    sh2[threadIdx.x] = s2;
    __syncthreads();

    // 256 固定の木構造 reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh0[threadIdx.x] += sh0[threadIdx.x + stride];
            sh1[threadIdx.x] += sh1[threadIdx.x + stride];
            sh2[threadIdx.x] += sh2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double n = sh0[0] * dv;
        double s1dv = sh1[0] * dv;
        double s2dv = sh2[0] * dv;

        // 安全側 clamp
        n = n > 1e-300 ? n : 1e-300;
        double u = s1dv / n;
        double T = s2dv / n - u * u;
        T = (T > 1e-300) ? T : 1e-300;

        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = T;
    }
}

// ===== 帯行列 (dl,dd,du) と RHS B を構築 =====
// 1 thread が 1 つの速度 j を担当し、内部セル i=1..nx-2 をループで処理
// grid: blocks = ceil(nv / t), blockDim.x = t (例 256)
__global__ void build_system_kernel_double(
    const double* __restrict__ f_k,     // (nx,nv) 旧ステップ f^k
    const double* __restrict__ v,       // (nv,)
    const double* __restrict__ n_arr,   // (nx,)  f_z からの moments
    const double* __restrict__ u_arr,   // (nx,)
    const double* __restrict__ T_arr,   // (nx,)
    int nx, int nv,
    double dt, double dx,
    double tau_tilde,
    double inv_sqrt_2pi,
    // 出力（nv, n_inner）
    double* __restrict__ dl, double* __restrict__ dd,
    double* __restrict__ du, double* __restrict__ B
){
    const int n_inner = max(nx - 2, 0);
    if (n_inner <= 0) return;

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nv) return;

    // α, β と境界の Maxwell（j 固有）
    double vj   = v[j];
    double alpha = (dt / dx) * (vj > 0.0 ? vj : 0.0);
    double beta  = (dt / dx) * (vj < 0.0 ? -vj : 0.0);

    // 左右境界の Maxwell 値
    double fL = maxwell_ij(n_arr[0], u_arr[0], T_arr[0], vj, inv_sqrt_2pi);
    double fR = maxwell_ij(n_arr[nx-1], u_arr[nx-1], T_arr[nx-1], vj, inv_sqrt_2pi);

    // 内部セルを順に
    for (int ii = 0; ii < n_inner; ++ii) {
        int i = 1 + ii;

        // τ_i: 明示的解法と同じ形（n*sqrt(T)）に合わせる
        double n  = n_arr[i];
        double T  = T_arr[i];
        double tau = tau_tilde / (n * sqrt(T));   // 安全 clamp は moments 側で

        // 対角成分
        double ddv = 1.0 + alpha + beta + (dt / tau);
        double dlv = -alpha;
        double duv = -beta;

        // f_M(i,j)
        double fM_ij = maxwell_ij(n_arr[i], u_arr[i], T_arr[i], vj, inv_sqrt_2pi);

        // 右辺
        double rhs = f_k[(size_t)i * nv + j] + (dt / tau) * fM_ij;

        // 境界寄与
        if (ii == 0)         rhs += alpha * fL;
        if (ii == n_inner-1) rhs += beta  * fR;

        // 書き込み（nv 主導ストライド）
        size_t idx = (size_t)j * n_inner + ii;
        dl[idx] = dlv;
        dd[idx] = ddv;
        du[idx] = duv;
        B [idx] = rhs;
    }
}

// ===== 解の書き戻し + 残差 L∞ を集計 =====
// 1D で nv*n_inner 要素を担当。boundary は tid==0 がまとめてコピー。
__device__ __forceinline__
void atomicMaxDouble(double* address, double val)
{
    // 非負値前提（|diff|）なのでビット順序 = 数値順序
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr, assumed;
    unsigned long long vull = __double_as_longlong(val);
    while (val > __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(addr, assumed, vull);
        if (assumed == old) break;
    }
}

__global__ void scatter_and_residual_kernel_double(
    const double* __restrict__ fz,        // (nx,nv) 旧候補
    const double* __restrict__ solution,  // (nv,n_inner)
    double* __restrict__ fn_tmp,          // (nx,nv) 新候補(一時)
    int nx, int nv,
    double* __restrict__ res_out          // (1,) L∞
){
    const int n_inner = max(nx - 2, 0);
    const size_t N = (size_t)nv * n_inner;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int j  = tid / n_inner;
        int ii = tid % n_inner;
        int i  = 1 + ii;

        double val = solution[(size_t)j * n_inner + ii];
        fn_tmp[(size_t)i * nv + j] = val;

        double diff = fabs(val - fz[(size_t)i * nv + j]);
        atomicMaxDouble(res_out, diff);
    }

    // 最初のスレッドで境界をコピー
    if (tid == 0) {
        for (int j = 0; j < nv; ++j) {
            fn_tmp[j] = fz[j];                               // i=0
            fn_tmp[(size_t)(nx-1)*nv + j] = fz[(size_t)(nx-1)*nv + j];
        }
    }
}

} // namespace imp_picard

// ===== ラッパ関数（.cpp から呼び出す） =====
extern "C" {

void launch_moments_kernel_double(
    const double* f, const double* v, int nx, int nv, double dv,
    double* n_out, double* u_out, double* T_out, cudaStream_t stream)
{
    int t = 256;
    dim3 grid(nx);
    size_t shm = 0;
    imp_picard::moments_kernel_double<<<grid, t, shm, stream>>>(
        f, v, nx, nv, dv, n_out, u_out, T_out);
}

void launch_build_system_kernel_double(
    const double* f_k, const double* v,
    const double* n_arr, const double* u_arr, const double* T_arr,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream)
{
    int t = 256;
    int n_inner = std::max(nx - 2, 0);
    if (n_inner <= 0) return;
    dim3 grid((nv + t - 1) / t);
    imp_picard::build_system_kernel_double<<<grid, t, 0, stream>>>(
        f_k, v, n_arr, u_arr, T_arr, nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl, dd, du, B);
}

void launch_scatter_and_residual_kernel_double(
    const double* fz, const double* solution, double* fn_tmp,
    int nx, int nv, double* res_out, cudaStream_t stream)
{
    int t = 256;
    int n_inner = std::max(nx - 2, 0);
    size_t N = (size_t)std::max(n_inner, 0) * (size_t)nv;
    dim3 grid((unsigned)((N + t - 1) / t));
    imp_picard::scatter_and_residual_kernel_double<<<grid, t, 0, stream>>>(
        fz, solution, fn_tmp, nx, nv, res_out);
}

} // extern "C"
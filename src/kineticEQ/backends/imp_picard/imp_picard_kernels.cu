// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>
#include <cmath>

namespace cg = cooperative_groups;

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); asm("trap;"); } } while(0)
#endif

// ===== ユーティリティ =====
__device__ __forceinline__
double fast_abs(double x){ return x >= 0.0 ? x : -x; }

__device__ __forceinline__
double clamp_pos(double x){ return x > 0.0 ? x : 0.0; }

__device__ __forceinline__
double clamp_neg(double x){ return x < 0.0 ? -x : 0.0; }

// atomicMax for double（CAS 版）
__device__ __forceinline__
void atomicMax_double(double* addr, double val){
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ull, assumed;
    while (true){
        double old_val = __longlong_as_double(old);
        if (old_val >= val) break;
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
        if (assumed == old) break;
    }
}

// ===== モーメント計算: 各 x に1ブロック =====
extern "C" __global__
void moments_kernel_double(
    const double* __restrict__ f,     // (nx, nv)
    const double* __restrict__ v,     // (nv)
    int nx, int nv, double dv,
    double* __restrict__ n_out,       // (nx)
    double* __restrict__ u_out,       // (nx)
    double* __restrict__ T_out        // (nx)
){
    int i = blockIdx.x;
    if (i >= nx) return;

    extern __shared__ double sh[];  // 3*blockDim.x
    double* s0 = sh;
    double* s1 = s0 + blockDim.x;
    double* s2 = s1 + blockDim.x;

    double s0_local = 0.0;
    double s1_local = 0.0;
    double s2_local = 0.0;

    const double* fi = f + (size_t)i * nv;

    for (int j = threadIdx.x; j < nv; j += blockDim.x){
        double fj = fi[j];
        double vj = v[j];
        s0_local += fj;
        s1_local += fj * vj;
        s2_local += fj * vj * vj;
    }

    s0[threadIdx.x] = s0_local;
    s1[threadIdx.x] = s1_local;
    s2[threadIdx.x] = s2_local;
    __syncthreads();

    // block reduce (単純でOK: nv<=~1k)
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            s0[threadIdx.x] += s0[threadIdx.x + stride];
            s1[threadIdx.x] += s1[threadIdx.x + stride];
            s2[threadIdx.x] += s2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        double n  = s0[0] * dv;
        double s1v= s1[0] * dv;
        double s2v= s2[0] * dv;
        double u  = s1v / n;
        double T  = s2v / n - u * u;
        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = T;
    }
}

// ===== a,b,c,B を組み立て（内部セルのみ） =====
// 配列は (nv, n_inner)。書き込みコアレスになるよう i_inner を最速にする。
extern "C" __global__
void build_system_kernel_double(
    const double* __restrict__ f,           // (nx, nv)
    const double* __restrict__ v,           // (nv)
    const double* __restrict__ n_arr,       // (nx)
    const double* __restrict__ u_arr,       // (nx)
    const double* __restrict__ T_arr,       // (nx)
    int nx, int nv, double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    // out
    double* __restrict__ dl,                // (nv, n_inner)
    double* __restrict__ dd,                // (nv, n_inner)
    double* __restrict__ du,                // (nv, n_inner)
    double* __restrict__ B                  // (nv, n_inner)
){
    const int n_inner = nx - 2;
    if (n_inner <= 0) return;

    const int t   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tot = nv * n_inner;
    if (t >= tot) return;

    // i_inner 最速 → 連続書き込み
    const int i_inner = t % n_inner;
    const int j       = t / n_inner;
    const int i       = i_inner + 1;

    // 各 j について一度だけでよい値
    const double vj    = v[j];
    const double alpha = (dt/dx) * clamp_pos(vj);
    const double beta  = (dt/dx) * clamp_neg(vj);

    // モーメントと tau
    const double n  = n_arr[i];
    const double u  = u_arr[i];
    const double T  = T_arr[i];
    const double invT = 0.5 / T;
    const double tau  = tau_tilde / (n * sqrt(T));

    // f_M(i,j)
    const double diff = (vj - u);
    const double fMij = (n * inv_sqrt_2pi / sqrt(T)) * exp(-(diff*diff) * invT);

    // 三重対角
    dl[j * n_inner + i_inner] = beta;                                 // 上対角（右に進むので注意）
    dd[j * n_inner + i_inner] = 1.0 + alpha + beta + dt / tau;         // 主
    du[j * n_inner + i_inner] = alpha;                                 // 下

    // 右辺
    double Bij = f[i * (size_t)nv + j] + (dt / tau) * fMij;

    // 境界寄与（upwind）
    if (i == 1) {
        // 左境界: v>0 のとき流入
        Bij += alpha * f[0 * (size_t)nv + j];
    }
    if (i == nx - 2) {
        // 右境界: v<0 のとき流入（beta は -v>0）
        Bij += beta * f[(nx-1) * (size_t)nv + j];
    }
    B[j * n_inner + i_inner] = Bij;
}

// ===== 解の書き戻し + 残差最大値（ブロック縮約→1原子更新） =====
extern "C" __global__
void scatter_and_residual_kernel_double(
    const double* __restrict__ fz,          // (nx, nv)
    const double* __restrict__ sol,         // (nv, n_inner)
    double* __restrict__ fn_tmp,            // (nx, nv)
    int nx, int nv,
    double* __restrict__ res_out            // (1)
){
    const int n_inner = nx - 2;
    const int tot = (n_inner > 0) ? nv * n_inner : 0;
    double local_max = 0.0;

    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < tot; t += gridDim.x * blockDim.x){
        const int i_inner = t % n_inner;
        const int j       = t / n_inner;
        const int i       = i_inner + 1;

        const double newv = sol[j * n_inner + i_inner];
        const size_t idx  = (size_t)i * nv + j;
        const double oldv = fz[idx];

        fn_tmp[idx] = newv;

        const double r = fast_abs(newv - oldv);
        if (r > local_max) local_max = r;
    }

    // block reduce
    __shared__ double smax[256]; // blockDim.x <= 256 前提
    smax[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            double a = smax[threadIdx.x];
            double b = smax[threadIdx.x + stride];
            smax[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        atomicMax_double(res_out, smax[0]);
    }
}
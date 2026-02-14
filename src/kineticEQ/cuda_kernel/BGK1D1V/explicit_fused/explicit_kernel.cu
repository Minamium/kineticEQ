#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

// (nx, nv) row-major の f を想定（PyTorch contiguous）
// k0 は v>=0 が始まる列インデックス。
// tau = tau_tilde / (n * sqrt(T)) を使うが、カーネルでは inv_tau = 1/tau を直接作る。

template <typename T>
__global__ void explicit_step_kernel(const T* __restrict__ f,
                                     T* __restrict__ fn,
                                     const T* __restrict__ v,
                                     int nx, int nv,
                                     T dv, T dt, T dx,
                                     T tau_tilde, T inv_sqrt_2pi,
                                     int k0)
{
    const int i = blockIdx.x;                 // 1 block = 1 row (xセル)
    if (i >= nx) return;

    extern __shared__ unsigned char smem[];
    T* s0 = reinterpret_cast<T*>(smem);                       // 長さ blockDim.x
    T* s1 = reinterpret_cast<T*>(smem + sizeof(T)*blockDim.x);          // "
    T* s2 = reinterpret_cast<T*>(smem + sizeof(T)*blockDim.x*2);        // "

    // ===== 1) 行 i のモーメント s0,s1,s2 を縮約 =====
    T p0 = T(0), p1 = T(0), p2 = T(0);
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        const T fij = f[i * nv + j];
        const T vj  = v[j];
        p0 += fij;
        p1 += fij * vj;
        p2 += fij * vj * vj;
    }
    s0[threadIdx.x] = p0;
    s1[threadIdx.x] = p1;
    s2[threadIdx.x] = p2;
    __syncthreads();

    // 共有メモリで単純二分木リダクション
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s0[threadIdx.x] += s0[threadIdx.x + offset];
            s1[threadIdx.x] += s1[threadIdx.x + offset];
            s2[threadIdx.x] += s2[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // スカラーモーメント
    const T n   = s0[0] * dv;
    const T s1d = s1[0] * dv;
    const T s2d = s2[0] * dv;
    const T u   = s1d / n;
    T Tgas      = s2d / n - u * u;

    // 数値安定化（T<=0 の NaN 回避、極小クランプ）
    if (!(Tgas > T(0))) Tgas = T(1e-300);

    const T sqrtT      = sqrt(Tgas);
    const T inv_sqrtT  = T(1) / sqrtT;
    const T coeff      = n * inv_sqrt_2pi * inv_sqrtT;          // n / sqrt(2π T)
    const T inv_tau    = (n * sqrtT) / tau_tilde;               // = 1/tau

    // ===== 2) 行 i の各列 j で fn を更新（移流＋衝突） =====
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        const int idx = i * nv + j;
        const T fij   = f[idx];
        const T vj    = v[j];

        // まず元の値をベースに
        T out = fij;

        // --- streaming (Upwind 相当) ---
        if (j >= k0) { // v>=0
            if (i >= 1) {
                const T df = fij - f[(i - 1) * nv + j];   // f[i] - f[i-1]
                out += dt * ((-vj) / dx) * df;
            }
        } else {       // v<0
            if (i <= nx - 2) {
                const T df = f[(i + 1) * nv + j] - fij;   // f[i+1] - f[i]
                out += dt * ((-vj) / dx) * df;
            }
        }

        // --- collision: (f_M - f) * inv_tau ---
        const T diff = vj - u;
        const T expo = -T(0.5) * (diff * diff) / Tgas;    // -(v-u)^2 / (2T)
        const T fM   = coeff * exp(expo);
        out += dt * (fM - fij) * inv_tau;

        fn[idx] = out;
    }
}

// host-side launcher（外部リンケージ、明示実体を .so に出す）
template <typename T>
void launch_explicit_step(const T* f, T* fn, const T* v,
                          int nx, int nv,
                          T dv, T dt, T dx,
                          T tau_tilde, T inv_sqrt_2pi, int k0,
                          cudaStream_t stream)
{
    // 1行=1ブロック。nv が大きいときはスレッドでストライド。
    const int block = 256;
    const dim3 grid(nx);
    const size_t shmem = sizeof(T) * block * 3; // s0,s1,s2
    explicit_step_kernel<T><<<grid, block, shmem, stream>>>(
        f, fn, v, nx, nv, dv, dt, dx, tau_tilde, inv_sqrt_2pi, k0
    );
}

// ★ 明示的インスタンシエーション（FP64）★
template void launch_explicit_step<double>(const double*, double*, const double*,
                                           int, int,
                                           double, double, double,
                                           double, double, int,
                                           cudaStream_t);

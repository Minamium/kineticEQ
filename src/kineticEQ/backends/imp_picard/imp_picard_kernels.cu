// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) { return (x > eps) ? x : eps; }

template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tg, T vj, T inv_sqrt_2pi) {
    T inv_sqrtT = rsqrt(clamp_pos(Tg, T(1e-300)));
    T coeff = n * inv_sqrt_2pi * inv_sqrtT;
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / clamp_pos(Tg, T(1e-300));
    return coeff * exp(expo);
}

// atomicMax for double (positive domain)
__device__ inline double atomicMaxDouble(double* addr, double val) {
    unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old  = *ull, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur >= val) break;
        old = atomicCAS(ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// 1 block per velocity j
template <typename T>
__global__ void picard_coop_kernel(
    // in/out
    const T* __restrict__ f_in,   // (nx,nv) : 前ステップ（境界固定用）
    T* __restrict__ fA,           // (nx,nv) : ピンポン用 buffer A (初期: f^z)
    T* __restrict__ fB,           // (nx,nv) : ピンポン用 buffer B (書き込み)
    const T* __restrict__ v,      // (nv)
    // const
    int nx, int nv, T dv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    int picard_iter, T picard_tol,
    // scratch (全体共有)
    T* __restrict__ s0,           // (nx)  累積 ∑_j f
    T* __restrict__ s1,           // (nx)  累積 ∑_j f*v
    T* __restrict__ s2,           // (nx)  累積 ∑_j f*v^2
    T* __restrict__ n_arr,        // (nx)
    T* __restrict__ u_arr,        // (nx)
    T* __restrict__ T_arr,        // (nx)
    // outputs
    int* __restrict__ iters_out,
    T*   __restrict__ resid_out)
{
    cg::grid_group grid = cg::this_grid();

    const int j  = blockIdx.x;
    if (j >= nv) return;
    const T vj   = v[j];
    const int n_inner = nx - 2;
    const T alpha = (vj > T(0)) ? (dt/dx * vj)   : T(0);
    const T beta  = (vj < T(0)) ? (dt/dx * (-vj)) : T(0);

    // 共有メモリ: dl, dd, du, B （長さ n_inner）
    extern __shared__ unsigned char smem_raw[];
    T* dl = reinterpret_cast<T*>(smem_raw);
    T* dd = dl + n_inner;
    T* du = dd + n_inner;
    T*  B = du + n_inner;

    // ピンポン用ローカルポインタ
    T* fz = fA;
    T* fn = fB;

    // 反復ループ
    T last_res = 0;
    int iters = 0;

    for (int z = 0; z < picard_iter; ++z) {
        // --- 0) s0,s1,s2 を全体でゼロクリア（グリッド分割）
        for (int i = threadIdx.x + blockIdx.x; i < nx; i += blockDim.x * gridDim.x) {
            s0[i] = T(0); s1[i] = T(0); s2[i] = T(0);
        }
        grid.sync();

        // --- 1) fz から s0,s1,s2 を蓄積（各ブロックが自分の j で加算）
        for (int i = threadIdx.x; i < nx; i += blockDim.x) {
            T fij = fz[i*nv + j];
            atomicAdd(&s0[i], fij);
            atomicAdd(&s1[i], fij * vj);
            atomicAdd(&s2[i], fij * vj * vj);
        }
        grid.sync();

        // --- 2) n,u,T を全体で計算（i をグリッドで分割）
        for (int i = threadIdx.x + blockIdx.x; i < nx; i += blockDim.x * gridDim.x) {
            T n  = s0[i] * dv;
            T s1d= s1[i] * dv;
            T s2d= s2[i] * dv;
            T u  = s1d / clamp_pos(n, T(1e-300));
            T Tg = s2d / clamp_pos(n, T(1e-300)) - u*u;
            if (!(Tg > T(0))) Tg = T(1e-300);
            n_arr[i] = n;
            u_arr[i] = u;
            T_arr[i] = Tg;
        }
        grid.sync();

        // --- 3) j 固有の境界値（今回は「前状態の境界を維持」）
        const T fL = f_in[0*nv + j];
        const T fR = f_in[(nx-1)*nv + j];

        // --- 4) 帯行列・RHS 構成（各ブロックが自分の j を担当）
        for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
            const int i = k + 1; // 内部セル
            const T n   = n_arr[i];
            const T u   = u_arr[i];
            const T Tg  = T_arr[i];

            const T inv_tau = (n * sqrt(Tg)) / tau_tilde;   // = 1/τ
            dd[k] = T(1) + alpha + beta + dt * inv_tau;
            dl[k] = -alpha;                 // 下対角（i-1）
            du[k] = -beta;                  // 上対角（i+1）

            T fij = fz[i*nv + j];
            T fM  = maxwell_1v<T>(n, u, Tg, vj, inv_sqrt_2pi);
            T rhs = fij + dt * inv_tau * fM;

            if (k == 0)           rhs += alpha * fL;                  // 左境界寄与
            if (k == n_inner-1)   rhs += beta  * fR;                  // 右境界寄与

            B[k] = rhs;
        }
        __syncthreads();

        // --- 5) ブロック内 Thomas（逐次 TDMA）
        if (threadIdx.x == 0) {
            // 前進消去
            for (int k = 1; k < n_inner; ++k) {
                T m = dl[k] / dd[k-1];
                dd[k] -= m * du[k-1];
                B[k]  -= m * B[k-1];
            }
            // 後退代入
            B[n_inner-1] = B[n_inner-1] / dd[n_inner-1];
            for (int k = n_inner - 2; k >= 0; --k) {
                B[k] = (B[k] - du[k] * B[k+1]) / dd[k];
            }
        }
        __syncthreads();

        // --- 6) 書き戻し + 局所残差
        T local_max = 0;
        for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
            const int i = k + 1;
            const T oldv = fz[i*nv + j];
            const T newv = B[k];
            fn[i*nv + j] = newv;
            T diff = fabs(newv - oldv);
            if (diff > local_max) local_max = diff;
        }
        // 境界は前状態を維持
        if (threadIdx.x == 0) {
            fn[0*nv + j]       = f_in[0*nv + j];
            fn[(nx-1)*nv + j]  = f_in[(nx-1)*nv + j];
        }

        // ブロック内最大→グローバル最大
        // まずブロック内還元
        extern __shared__ double sres[]; // 先頭を別用途で使っているが、末尾を利用
        // 安全のため別バッファ: 共有メモリ総量は十分（dl,dd,du,B に続く領域として確保するのが簡単でないため、ここはレジスタ->atomic で OK）
        atomicMaxDouble(reinterpret_cast<double*>(resid_out), (double)local_max);
        __syncthreads();
        grid.sync();

        // 収束判定（grid 全体で一度だけ）
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            last_res = *resid_out;
            // 初期化（次の反復に向けたリセット）
            *resid_out = 0.0;
        }
        grid.sync();

        ++iters;

        // 収束ならブレーク
        bool done = false;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            done = (last_res <= picard_tol);
        }
        __shared__ int s_done;
        if (threadIdx.x == 0) s_done = done ? 1 : 0;
        __syncthreads();
        grid.sync();
        if (s_done) break;

        // 次反復へ：fz/fn をポインタスワップ
        T* tmp = fz; fz = fn; fn = tmp;
        grid.sync();
    }

    // 最終結果を f_out (= fn) に置く（偶数/奇数反復で場所が揺れるため）
    // ここでは「fn に最終解がある」前提にしている：
    //   収束で break した時点では fn に直近解が書かれている。
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *iters_out = iters;
    }
}

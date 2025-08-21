#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <algorithm>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace {

// double の atomicMax（L∞残差用）
__device__ inline void atomicMaxDouble(double* addr, double val) {
#if __CUDA_ARCH__ >= 800
    // Ampere 以降は atomicMax(double) が使える GPU もあるが、移植性のため CAS 実装に統一
#endif
    unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ull, assumed;
    double old_val;
    do {
        old_val = __longlong_as_double(old);
        if (old_val >= val) break;
        assumed = old;
        old = atomicCAS(ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Maxwellian（1 速度点）
__device__ inline double maxwell_1v(double n, double u, double T, double vj, double inv_sqrt_2pi) {
    T = fmax(T, 1e-300);
    const double inv_sqrtT = rsqrt(T);
    const double c = n * inv_sqrt_2pi * inv_sqrtT;
    const double diff = vj - u;
    const double expo = -0.5 * (diff*diff) / T;
    return c * exp(expo);
}

} // anon

// ==== 単一 cooperative カーネル ====
// gridDim.x は occupancy に合わせて「小さく」起動し、j は grid-stride で回す。
__global__ void picard_kernel_double(
    const double* __restrict__ f,  // (nx, nv) in
    double* __restrict__ fn,       // (nx, nv) out
    const double* __restrict__ v,  // (nv)
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    // workspaces
    double* __restrict__ dl, double* __restrict__ dd,
    double* __restrict__ du, double* __restrict__ B,   // (nv, nx-2)
    double* __restrict__ n_arr, double* __restrict__ u_arr, double* __restrict__ T_arr, // (nx)
    double* __restrict__ s0_arr, double* __restrict__ s1_arr, double* __restrict__ s2_arr, // (nx)
    double* __restrict__ res_dev, int* __restrict__ iters_dev)
{
    cg::grid_group grid = cg::this_grid();
    const int n_inner = max(0, nx - 2);

    // Picard 初期候補：fz = f（境界も含めて保持）
    // fn は毎イテレーションで「候補の書き込み先」として使う
    // まず f を fn にコピー（grid-stride）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx*nv; i += gridDim.x * blockDim.x) {
        fn[i] = f[i];
    }
    grid.sync();

    // 反復
    double residual = 0.0;
    int iters = 0;

    for (int z = 0; z < max_iters; ++z) {
        // --- moments s0/s1/s2 を 0 クリア（grid-stride over i）---
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx; i += gridDim.x * blockDim.x) {
            s0_arr[i] = 0.0;
            s1_arr[i] = 0.0;
            s2_arr[i] = 0.0;
        }
        grid.sync();

        // --- s0/s1/s2 の蓄積（grid-stride over j, 各 i は atomicAdd）---
        for (int j = blockIdx.x; j < nv; j += gridDim.x) {
            const double vj = v[j];
            const double vj2 = vj*vj;

            // i を thread でストライド
            for (int i_local = threadIdx.x; i_local < nx; i_local += blockDim.x) {
                const double fij = fn[i_local*nv + j];
                atomicAdd(&s0_arr[i_local], fij);
                atomicAdd(&s1_arr[i_local], fij * vj);
                atomicAdd(&s2_arr[i_local], fij * vj2);
            }
        }
        grid.sync();

        // --- n, u, T の完成（grid-stride over i）---
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx; i += gridDim.x * blockDim.x) {
            const double n = s0_arr[i] * dv;
            const double m1 = s1_arr[i] * dv;
            const double m2 = s2_arr[i] * dv;
            const double u  = m1 / n;
            double T = m2 / n - u*u;
            if (!(T > 0.0)) T = 1e-300;

            n_arr[i] = n; u_arr[i] = u; T_arr[i] = T;
        }
        grid.sync();

        // --- 三重対角(Bも) 構築 & Thomas で内部セル解く（grid-stride over j）---
        for (int j = blockIdx.x; j < nv; j += gridDim.x) {
            if (n_inner <= 0) continue;

            const double vj = v[j];
            const double alpha = (dt/dx) * fmax(vj, 0.0);
            const double beta  = (dt/dx) * fmax(-vj, 0.0);

            double* dl_j = dl + j * n_inner;
            double* dd_j = dd + j * n_inner;
            double* du_j = du + j * n_inner;
            double*  B_j =  B + j * n_inner;

            // 構築（並列化しにくいので 1 thread が k を回す）
            if (threadIdx.x == 0) {
                for (int k = 0; k < n_inner; ++k) {
                    const int i = k + 1;

                    const double inv_tau = (n_arr[i] * sqrt(T_arr[i])) / tau_tilde;
                    dl_j[k] = (k == 0         ? 0.0 : beta);  // 上対角の符号は式の定義に合わせる
                    du_j[k] = (k == n_inner-1 ? 0.0 : alpha);
                    dd_j[k] = 1.0 + alpha + beta + dt * inv_tau;

                    // 右辺 B： f^k + (dt/tau) fM  + 境界の upwind 寄与
                    const double fij_old = fn[i*nv + j];  // ここでは直前候補 fn を f^k として使う
                    const double fM = maxwell_1v(n_arr[i], u_arr[i], T_arr[i], vj, inv_sqrt_2pi);

                    double rhs = fij_old + dt * inv_tau * fM;

                    // 境界寄与（セル値は触らない方針。境界の「前状態」を参照）
                    const double fL = fn[0*nv + j];
                    const double fR = fn[(nx-1)*nv + j];
                    if (k == 0)          rhs += alpha * fL;
                    if (k == n_inner-1)  rhs += beta  * fR;

                    B_j[k] = rhs;
                }

                // Thomas forward/backward
                for (int k = 1; k < n_inner; ++k) {
                    const double m = dl_j[k] / dd_j[k-1];
                    dd_j[k] -= m * du_j[k-1];
                    B_j[k]  -= m * B_j[k-1];
                }
                B_j[n_inner-1] /= dd_j[n_inner-1];
                for (int k = n_inner-2; k >= 0; --k) {
                    B_j[k] = (B_j[k] - du_j[k]*B_j[k+1]) / dd_j[k];
                }

                // 書き戻しは後段でまとめて（残差評価のため）
            }
        }
        grid.sync();

        // --- 書き戻し & 残差（内部セルのみ L∞）---
        //   fn を新解、旧解は（この段では）fn にまだ残っていないので f_old を読みたい。
        //   直前反復の解は fn にあるため、まず差分を測ってから fn を上書きする。
        double local_max = 0.0;

        for (int j = blockIdx.x; j < nv; j += gridDim.x) {
            if (n_inner <= 0) continue;
            double*  B_j =  B + j * n_inner;

            for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
                const int i = k + 1;  // 内部セル
                const double oldv = fn[i*nv + j];
                const double newv = B_j[k];
                const double diff = fabs(newv - oldv);
                if (diff > local_max) local_max = diff;
            }
        }

        // CTA 内最大
        __shared__ double s_max;
        double x = local_max;
        // block 内 reduce
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            x = fmax(x, __shfl_down_sync(0xffffffff, x, offset));
        if ((threadIdx.x & (warpSize - 1)) == 0) atomicMaxDouble(&s_max, x);
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicMaxDouble(res_dev, s_max);
            s_max = 0.0;
        }
        grid.sync();

        // 収束チェック
        residual = *res_dev;
        if (residual <= tol) {
            // 解を fn に確定書き込み
            for (int j = blockIdx.x; j < nv; j += gridDim.x) {
                if (n_inner <= 0) continue;
                double*  B_j =  B + j * n_inner;
                for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
                    const int i = k + 1;
                    fn[i*nv + j] = B_j[k];
                }
            }
            grid.sync();
            iters = z + 1;
            break;
        }

        // 非収束：解を確定して次反復へ（fn ← B）
        for (int j = blockIdx.x; j < nv; j += gridDim.x) {
            if (n_inner <= 0) continue;
            double*  B_j =  B + j * n_inner;
            for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
                const int i = k + 1;
                fn[i*nv + j] = B_j[k];
            }
        }
        grid.sync();

        // 次反復のために残差を 0 初期化
        if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
            *res_dev = 0.0;
        }
        grid.sync();

        iters = z + 1;
    }

    // 最終：境界は「前状態を維持」の方針（fn の境界は f と同じ）
    // 何もする必要なし。

    // 反復回数と残差を保存
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        *iters_dev = iters;
        // res_dev はすでに最終値
    }
}

// occupancy を見て「載るだけ」起動する
} // namespace

namespace imp_picard {

void launch_picard_double(
    const double* f, double* fn, const double* v,
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    double* dl, double* dd, double* du, double* B,
    double* n_arr, double* u_arr, double* T_arr,
    double* s0_arr, double* s1_arr, double* s2_arr,
    double* res_dev, int* iters_dev,
    cudaStream_t stream)
{
    // block サイズは 256 をデフォルト
    const int block = 256;

    // cooperative occupancy を見積もる
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    int maxBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, picard_kernel_double, block, 0 /*shared*/);

    int maxCoopBlocks = std::max(1, sms * maxBlocksPerSm);

    // grid は「常駐可能な最大」に制限。j は grid-stride で処理するので nv と独立。
    dim3 grid(std::min(std::max(1, maxCoopBlocks),  /*保守的に*/ std::max(1, sms)));

    void* args[] = {
        (void*)&f, (void*)&fn, (void*)&v,
        (void*)&nx, (void*)&nv,
        (void*)&dv, (void*)&dt, (void*)&dx,
        (void*)&tau_tilde, (void*)&inv_sqrt_2pi,
        (void*)&max_iters, (void*)&tol,
        (void*)&dl, (void*)&dd, (void*)&du, (void*)&B,
        (void*)&n_arr, (void*)&u_arr, (void*)&T_arr,
        (void*)&s0_arr, (void*)&s1_arr, (void*)&s2_arr,
        (void*)&res_dev, (void*)&iters_dev
    };

    // cooperative 起動
    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)picard_kernel_double, grid, dim3(block), args, 0, stream);

    if (st == cudaErrorCooperativeLaunchTooLarge) {
        // さらに減らして再挑戦（1 CTA/SM の超保守）
        cudaGetLastError(); // clear
        dim3 grid2(std::max(1, sms));
        st = cudaLaunchCooperativeKernel(
            (void*)picard_kernel_double, grid2, dim3(block), args, 0, stream);
    }
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchCooperativeKernel failed: ") +
                                 cudaGetErrorString(st));
    }
}

} // namespace imp_picard

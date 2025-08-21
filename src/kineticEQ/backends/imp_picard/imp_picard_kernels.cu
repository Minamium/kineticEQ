// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" {

__device__ inline double atomicMaxDouble(double* addr, double val) {
    unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ull, assumed;
    while (true) {
        double old_d = __longlong_as_double(old);
        if (old_d >= val) break;
        assumed = old;
        old = atomicCAS(ull, assumed, __double_as_longlong(val));
        if (assumed == old) break;
    }
    return __longlong_as_double(old);
}

__device__ inline double maxwell(double n, double u, double T, double vj, double inv_sqrt_2pi){
    double coeff = (n * inv_sqrt_2pi) / sqrt(T);
    double t = vj - u;
    double ex = exp(-(t*t)/(2.0*T));
    return coeff * ex;
}

// Thomas in-place (length = n)
__device__ inline void thomas_inplace(double* dl, double* dd, double* du, double* B, int n){
    // forward
    for (int i=1; i<n; ++i){
        double m = dl[i] / dd[i-1];
        dd[i]    -= m * du[i-1];
        B[i]     -= m * B[i-1];
    }
    // back
    B[n-1] /= dd[n-1];
    for (int i=n-2; i>=0; --i){
        B[i] = (B[i] - du[i]*B[i+1]) / dd[i];
    }
}

__global__ void picard_kernel_double(
    const double* __restrict__ f, double* __restrict__ fn, const double* __restrict__ v,
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    double* __restrict__ dl, double* __restrict__ dd, double* __restrict__ du, double* __restrict__ B,
    double* __restrict__ n_arr, double* __restrict__ u_arr, double* __restrict__ T_arr,
    double* __restrict__ s0_arr, double* __restrict__ s1_arr, double* __restrict__ s2_arr,
    double* __restrict__ res_dev, int* __restrict__ iters_dev)
{
    cg::grid_group grid = cg::this_grid();
    const int tid   = threadIdx.x + blockIdx.x * blockDim.x;
    const int tdim  = blockDim.x * gridDim.x;

    const int n_inner = max(nx - 2, 0);
    if (nx <= 0 || nv <= 0 || n_inner <= 0){
        if (tid==0){ *res_dev = 0.0; *iters_dev = 0; }
        return;
    }

    // 初期候補: read_ptr=f, write_ptr=fn
    const double* read_ptr  = f;
    double*       write_ptr = fn;

    if (tid==0){ *res_dev = 0.0; *iters_dev = 0; }

    // ===== Picard loop =====
    for (int it=0; it<max_iters; ++it){
        // s0,s1,s2 をゼロ化
        for (int i = tid; i < nx; i += tdim){
            s0_arr[i] = 0.0; s1_arr[i] = 0.0; s2_arr[i] = 0.0;
        }
        grid.sync();

        // 瞬時モーメント集計（原始的: 原子加算）
        for (int j = tid; j < nv; j += tdim){
            const double vj = v[j];
            for (int i=0; i<nx; ++i){
                double fij = read_ptr[i*nv + j];
                atomicAdd(&s0_arr[i], fij*dv);
                atomicAdd(&s1_arr[i], fij*dv*vj);
                atomicAdd(&s2_arr[i], fij*dv*vj*vj);
            }
        }
        grid.sync();

        // n,u,T 作成
        for (int i = tid; i < nx; i += tdim){
            double n  = s0_arr[i];
            double u  = (n>0.0) ? (s1_arr[i]/n) : 0.0;
            double T  = (n>0.0) ? (s2_arr[i]/n - u*u) : 1.0;
            if (T < 1e-20) T = 1e-20;
            n_arr[i] = n; u_arr[i]=u; T_arr[i]=T;
        }
        grid.sync();

        // 境界の Maxwell
        // (固定： write_ptr の 0, nx-1 は前ステップ f を保持)
        // 右辺境界寄与用に fL,fR を生成（都度）
        // ここでは配列を持たず、必要な j で都度 maxwell を使う

        // 係数と RHS 構築（j を並列化、各 j で i=1..nx-2 をシリアル）
        for (int j = tid; j < nv; j += tdim){
            const double vj  = v[j];
            const double alpha = (dt/dx) * (vj>0.0 ? vj : 0.0);
            const double beta  = (dt/dx) * ((-vj)>0.0 ? (-vj) : 0.0);

            double* dlj = &dl[j*n_inner];
            double* ddj = &dd[j*n_inner];
            double* duj = &du[j*n_inner];
            double* Bj  = &B [j*n_inner];

            // 下上対角の端をゼロ
            if (n_inner >= 1){
                dlj[0]          = 0.0;
                duj[n_inner-1]  = 0.0;
            }
            // 各行
            for (int k=0; k<n_inner; ++k){
                const int i = k + 1; // 物理セル 1..nx-2

                double n = n_arr[i];
                double u = u_arr[i];
                double T = T_arr[i];
                double tau = tau_tilde / (n * sqrt(T));

                // 主対角
                ddj[k] = 1.0 + alpha + beta + (dt/tau);
                // 下・上対角（行インデックスずれに注意）
                if (k>=1)    dlj[k] = -alpha;
                if (k<=n_inner-2) duj[k] = -beta;

                // RHS: f^k + (dt/τ) fM
                double fM = maxwell(n, u, T, vj, inv_sqrt_2pi);
                double rhs = read_ptr[i*nv + j];   // ←時間レベル k の既知 f
                rhs += (dt/tau) * fM;

                // 境界からの流入
                if (k==0 && vj>0.0){
                    // 左境界 i=0 の Maxwell
                    double nL=n_arr[0], uL=u_arr[0], TL=T_arr[0];
                    double fL = maxwell(nL,uL,TL,vj,inv_sqrt_2pi);
                    rhs += (dt/dx) * vj * fL;
                }
                if (k==n_inner-1 && vj<0.0){
                    // 右境界 i=nx-1 の Maxwell
                    double nR=n_arr[nx-1], uR=u_arr[nx-1], TR=T_arr[nx-1];
                    double fR = maxwell(nR,uR,TR,vj,inv_sqrt_2pi);
                    rhs += (dt/dx) * (-vj) * fR;
                }
                Bj[k] = rhs;
            }
        }
        grid.sync();

        // 各 j を1スレッドでトーマス解法（競合なし）
        for (int j = tid; j < nv; j += tdim){
            thomas_inplace(&dl[j*n_inner], &dd[j*n_inner], &du[j*n_inner], &B[j*n_inner], n_inner);
        }
        grid.sync();

        // 書き戻し (内部セルのみ)
        for (int j = tid; j < nv; j += tdim){
            for (int k=0; k<n_inner; ++k){
                const int i = k + 1;
                write_ptr[i*nv + j] = B[j*n_inner + k];
            }
        }
        grid.sync();

        // 境界はそのまま（read_ptr == f か fn のいずれでも同じセル値を維持）
        for (int j = tid; j < nv; j += tdim){
            write_ptr[0*nv + j]      = read_ptr[0*nv + j];
            write_ptr[(nx-1)*nv + j] = read_ptr[(nx-1)*nv + j];
        }
        grid.sync();

        // 残差 L∞
        // まずブロック内で最大を取り、global に atomicMaxDouble
        __shared__ double smax[256];
        double local_max = 0.0;
        for (int idx = tid; idx < nx*nv; idx += tdim){
            double diff = fabs(write_ptr[idx] - read_ptr[idx]);
            if (diff > local_max) local_max = diff;
        }
        int lane = threadIdx.x;
        smax[lane] = local_max;
        __syncthreads();
        // block reduce
        for (int off = blockDim.x/2; off>0; off/=2){
            if (lane < off) smax[lane] = fmax(smax[lane], smax[lane+off]);
            __syncthreads();
        }
        if (lane==0) atomicMaxDouble(res_dev, smax[0]);
        grid.sync();

        if (tid==0){
            *iters_dev = it+1;
        }
        grid.sync();

        // 収束判定（全ブロック同一分岐）
        bool converged = false;
        if (tid==0) converged = (*res_dev <= tol);
        // 全グリッドへブロードキャスト相当
        converged = cg::sync(grid, converged);

        if (converged) break;

        // 次反復へ: ポインタスワップ
        if (tid==0){
            *res_dev = 0.0; // 次回に向けてクリア
        }
        grid.sync();

        const double* tmp_r = read_ptr;
        read_ptr  = write_ptr;
        write_ptr = (double*)tmp_r;

        grid.sync();
    } // picard loop
}

} // extern "C"
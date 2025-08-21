// imp_picard_kernels.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
namespace cg = cooperative_groups;

extern "C" {

// ---- atomicMax for double -------------------------------------------------
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

// ---- Maxwellian -----------------------------------------------------------
__device__ inline double maxwell(double n, double u, double T, double vj, double inv_sqrt_2pi){
    double coeff = (n * inv_sqrt_2pi) / sqrt(T);
    double t = vj - u;
    double ex = exp(-(t*t)/(2.0*T));
    return coeff * ex;
}

// ---- Thomas solver (in-place), size = n -----------------------------------
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

// ---- Single cooperative kernel --------------------------------------------
__global__ void picard_kernel_double(
    const double* __restrict__ f,  double* __restrict__ fn, const double* __restrict__ v,
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

    const int tid  = threadIdx.x + blockIdx.x * blockDim.x;
    const int tdim = blockDim.x * gridDim.x;

    const int n_inner = max(nx - 2, 0);
    if (nx <= 0 || nv <= 0 || n_inner <= 0){
        if (tid==0){ *res_dev = 0.0; *iters_dev = 0; }
        return; // 全スレッド同一条件なので安全
    }

    // ピカード反復で使う入力/出力ポインタ
    const double* read_ptr  = f;
    double*       write_ptr = fn;

    if (tid==0){ *res_dev = 0.0; *iters_dev = 0; }

    // ========================= Picard loop ==================================
    for (int it=0; it<max_iters; ++it){

        // s0,s1,s2 のクリア
        for (int i = tid; i < nx; i += tdim){
            s0_arr[i] = 0.0; s1_arr[i] = 0.0; s2_arr[i] = 0.0;
        }
        grid.sync();

        // モーメント（素直に atomicAdd）
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

        // n, u, T 作成
        for (int i = tid; i < nx; i += tdim){
            double n  = s0_arr[i];
            double u  = (n>0.0) ? (s1_arr[i]/n) : 0.0;
            double T  = (n>0.0) ? (s2_arr[i]/n - u*u) : 1.0;
            if (T < 1e-20) T = 1e-20;
            n_arr[i] = n; u_arr[i] = u; T_arr[i] = T;
        }
        grid.sync();

        // 係数と RHS 構築（各 j を並列）
        for (int j = tid; j < nv; j += tdim){
            const double vj = v[j];
            const double alpha = (dt/dx) * (vj>0.0 ? vj : 0.0);
            const double beta  = (dt/dx) * ((-vj)>0.0 ? (-vj) : 0.0);

            double* dlj = &dl[j*n_inner];
            double* ddj = &dd[j*n_inner];
            double* duj = &du[j*n_inner];
            double* Bj  = &B [j*n_inner];

            if (n_inner >= 1){
                dlj[0]         = 0.0;            // 未使用要素を明示ゼロ
                duj[n_inner-1] = 0.0;
            }
            for (int k=0; k<n_inner; ++k){
                const int i = k + 1; // 物理セル 1..nx-2

                double n  = n_arr[i];
                double u  = u_arr[i];
                double T  = T_arr[i];
                double tau = tau_tilde / (n * sqrt(T));

                // 主対角
                ddj[k] = 1.0 + alpha + beta + (dt/tau);

                // 下・上対角
                if (k>=1)            dlj[k] = -alpha;
                if (k<=n_inner-2)    duj[k] = -beta;

                // 右辺: f^k + (dt/τ) fM
                double fM  = maxwell(n, u, T, vj, inv_sqrt_2pi);
                double rhs = read_ptr[i*nv + j] + (dt/tau) * fM;

                // 境界の流入
                if (k==0 && vj>0.0){
                    double fL = maxwell(n_arr[0], u_arr[0], T_arr[0], vj, inv_sqrt_2pi);
                    rhs += (dt/dx) * vj * fL;
                }
                if (k==n_inner-1 && vj<0.0){
                    double fR = maxwell(n_arr[nx-1], u_arr[nx-1], T_arr[nx-1], vj, inv_sqrt_2pi);
                    rhs += (dt/dx) * (-vj) * fR;
                }
                Bj[k] = rhs;
            }
        }
        grid.sync();

        // トーマス（各 j を 1 スレッドで処理）
        for (int j = tid; j < nv; j += tdim){
            thomas_inplace(&dl[j*n_inner], &dd[j*n_inner],
                           &du[j*n_inner], &B[j*n_inner], n_inner);
        }
        grid.sync();

        // 書き戻し（内部セル）
        for (int j = tid; j < nv; j += tdim){
            for (int k=0; k<n_inner; ++k){
                const int i = k + 1;
                write_ptr[i*nv + j] = B[j*n_inner + k];
            }
        }
        grid.sync();

        // 境界は維持
        for (int j = tid; j < nv; j += tdim){
            write_ptr[0*nv + j]      = read_ptr[0*nv + j];
            write_ptr[(nx-1)*nv + j] = read_ptr[(nx-1)*nv + j];
        }
        grid.sync();

        // 残差 L∞ = max |write - read|
        __shared__ double smax[256];
        double local_max = 0.0;
        for (int idx = tid; idx < nx*nv; idx += tdim){
            double diff = fabs(write_ptr[idx] - read_ptr[idx]);
            if (diff > local_max) local_max = diff;
        }
        int lane = threadIdx.x;
        smax[lane] = local_max;
        __syncthreads();
        for (int off = blockDim.x/2; off>0; off/=2){
            if (lane < off) smax[lane] = fmax(smax[lane], smax[lane+off]);
            __syncthreads();
        }
        if (lane==0) atomicMaxDouble(res_dev, smax[0]);
        grid.sync();

        if (tid==0) *iters_dev = it + 1;
        grid.sync();

        // ★ 収束判定（全スレッドが同じ条件を独自に評価）
        bool converged = (*res_dev <= tol);
        grid.sync();
        if (converged) break;

        // 次反復準備
        if (tid==0) *res_dev = 0.0; // クリア
        grid.sync();

        const double* tmp = read_ptr;
        read_ptr  = write_ptr;
        write_ptr = (double*)tmp;

        grid.sync();
    } // end Picard loop
}

} // extern "C"
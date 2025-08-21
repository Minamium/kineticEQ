// implicit_kernels.cu
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <assert.h>

namespace implicit_fused {

// ========= helpers =========
template <typename T>
__device__ __forceinline__ T clamp_pos(T x, T eps) {
    return (x > eps) ? x : eps;
}

template <typename T>
__device__ __forceinline__ T maxwell_1v(T n, T u, T Tgas, T vj, T inv_sqrt_2pi) {
    // fM = n / sqrt(2π T) * exp(-(v-u)^2/(2T))
    T Tpos = clamp_pos(Tgas, T(1e-300));
    T coeff = n * inv_sqrt_2pi / sqrt(Tpos);
    T diff  = vj - u;
    T expo  = -T(0.5) * (diff*diff) / Tpos;
    return coeff * exp(expo);
}

// ========= moments =========
// 1 block / x-row, shared-memory reduction
template <typename T>
__global__ void moments_kernel(
    const T* __restrict__ f,  // (nx,nv)
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
        T u  = s1d / n;
        T Tg = s2d / n - u*u;
        if (!(Tg > T(0))) Tg = T(1e-300);
        n_out[i] = n;
        u_out[i] = u;
        T_out[i] = Tg;
    }
}

// ========= boundary Maxwell =========
// compute fL(j), fR(j) only for RHS contribution; **don't touch boundary cells**
template <typename T>
__global__ void boundary_maxwell_kernel(
    const T* __restrict__ v, int nv, T inv_sqrt_2pi,
    T nL, T uL, T TL, T nR, T uR, T TR,
    T* __restrict__ fL, T* __restrict__ fR)
{
    TL = clamp_pos(TL, T(1e-300));
    TR = clamp_pos(TR, T(1e-300));
    T coeffL = nL * inv_sqrt_2pi / sqrt(TL);
    T coeffR = nR * inv_sqrt_2pi / sqrt(TR);

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < nv; j += blockDim.x * gridDim.x) {
        T vj = v[j];
        T eL = exp(-T(0.5) * (vj-uL)*(vj-uL) / TL);
        T eR = exp(-T(0.5) * (vj-uR)*(vj-uR) / TR);
        fL[j] = coeffL * eL;
        fR[j] = coeffR * eR;
    }
}

// ========= build tri-diagonal & RHS =========
// 1 block / velocity j, write (nv, n_inner) arrays (row-major by j)
template <typename T>
__global__ void build_tridiag_rhs_kernel(
    const T* __restrict__ f,   // (nx,nv) current candidate fz
    const T* __restrict__ v,   // (nv)
    const T* __restrict__ n,   // (nx)
    const T* __restrict__ u,   // (nx)
    const T* __restrict__ Tg,  // (nx)
    const T* __restrict__ fL,  // (nv)
    const T* __restrict__ fR,  // (nv)
    int nx, int nv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    T* __restrict__ dl,        // (nv, n_inner)
    T* __restrict__ dd,        // (nv, n_inner)
    T* __restrict__ du,        // (nv, n_inner)
    T* __restrict__ B)         // (nv, n_inner)  (RHS)
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int n_inner = nx - 2;
    if (n_inner <= 0) return;

    const T vj = v[j];
    const T a  = -dt/dx * fmax(vj, T(0));   // <= 0
    const T c  = -dt/dx * fmax(-vj, T(0));  // <= 0
    const T minus_a = -a;                   // >= 0
    const T minus_c = -c;                   // >= 0

    T* dl_j = dl + j * n_inner;
    T* dd_j = dd + j * n_inner;
    T* du_j = du + j * n_inner;
    T*  B_j =  B + j * n_inner;

    const T fL_j = fL[j];
    const T fR_j = fR[j];

    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1; // interior row index in [1..nx-2]

        T inv_tau = (n[i] * sqrt(clamp_pos(Tg[i], T(1e-300)))) / tau_tilde;

        // diagonal & off-diagonals
        dd_j[k] = T(1) + minus_a + minus_c + dt * inv_tau;
        dl_j[k] = (k==0) ? T(0) : a;
        du_j[k] = (k==n_inner-1) ? T(0) : c;

        // RHS
        const T fij = f[i*nv + j];
        const T fM  = maxwell_1v<T>(n[i], u[i], Tg[i], vj, inv_sqrt_2pi);
        T rhs = fij + dt * inv_tau * fM;

        if (k == 0)          rhs += (dt/dx) * fmax(vj,  T(0)) * fL_j;
        if (k == n_inner-1)  rhs += (dt/dx) * fmax(-vj, T(0)) * fR_j;

        B_j[k] = rhs;
    }
}

// ========= writeback (interior only) + per-velocity residual =========
// 1 block / velocity j, parallel max-reduction
template <typename T>
__global__ void writeback_residual_kernel(
    const T* __restrict__ fz,   // (nx,nv) old candidate
    T* __restrict__ fn_tmp,     // (nx,nv) new candidate (only interior written)
    const T* __restrict__ B,    // (nv, n_inner) solution
    int nx, int nv,
    T* __restrict__ res_per_v)  // (nv)
{
    const int j = blockIdx.x;
    if (j >= nv) return;
    const int n_inner = nx - 2;
    if (n_inner <= 0) {
        if (threadIdx.x == 0) res_per_v[j] = T(0);
        return;
    }

    extern __shared__ unsigned char smem_raw[];
    T* smax = reinterpret_cast<T*>(smem_raw);

    T local_max = T(0);
    const T* Bj = B + j * n_inner;

    // scatter write & track max diff for this velocity
    for (int k = threadIdx.x; k < n_inner; k += blockDim.x) {
        const int i = k + 1;
        const T oldv = fz[i*nv + j];
        const T newv = Bj[k];
        fn_tmp[i*nv + j] = newv;  // interior only
        T diff = fabs(newv - oldv);
        if (diff > local_max) local_max = diff;
    }

    smax[threadIdx.x] = local_max;
    __syncthreads();

    for (int offset = blockDim.x>>1; offset>0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) res_per_v[j] = smax[0];
}

// reduce max over length-nv vector
template <typename T>
__global__ void reduce_max_kernel(const T* __restrict__ x, int n, T* __restrict__ out) {
    extern __shared__ unsigned char smem_raw[];
    T* s = reinterpret_cast<T*>(smem_raw);

    T vmax = T(0);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        vmax = fmax(vmax, x[idx]);
    }
    s[threadIdx.x] = vmax;
    __syncthreads();

    for (int offset = blockDim.x>>1; offset>0; offset >>= 1) {
        if (threadIdx.x < offset) s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(out),
                  __double_as_longlong(static_cast<double>(s[0])));
    }
}

// ========= launchers (double) =========
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

void launch_boundary_maxwell_double(
    const double* v, int nv, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    double* fL, double* fR,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (nv + block - 1) / block;
    boundary_maxwell_kernel<double><<<grid, block, 0, stream>>>(
        v, nv, inv_sqrt_2pi, nL, uL, TL, nR, uR, TR, fL, fR);
}

void launch_build_tridiag_rhs_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    const double* fL, const double* fR,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* d, double* du, double* B,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    build_tridiag_rhs_kernel<double><<<grid, block, 0, stream>>>(
        f, v, n, u, T, fL, fR, nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi, dl, d, du, B);
}

void launch_writeback_and_residual_double(
    const double* fz, double* fn_tmp,
    const double* B, int nx, int nv,
    double* res_per_v,
    cudaStream_t stream)
{
    const int block = 256;
    dim3 grid(nv);
    size_t shmem = sizeof(double) * block;
    writeback_residual_kernel<double><<<grid, block, shmem, stream>>>(
        fz, fn_tmp, B, nx, nv, res_per_v);
}

void launch_reduce_max_double(const double* x, int n, double* out, cudaStream_t stream) {
    int block = 256;
    int grid  = std::min(1024, (n + block - 1) / block);
    size_t shmem = sizeof(double) * block;
    // initialize out to 0 on device
    cudaMemsetAsync(out, 0, sizeof(double), stream);
    reduce_max_kernel<double><<<grid, block, shmem, stream>>>(x, n, out);
}

} // namespace implicit_fused

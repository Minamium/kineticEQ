#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ __forceinline__ T my_exp(T x);
template <> __device__ __forceinline__ float  my_exp<float >(float  x){ return expf(x); }
template <> __device__ __forceinline__ double my_exp<double>(double x){ return exp (x); }

template <typename T>
__global__ void explicit_step_kernel(
    const T* __restrict__ F,
    T* __restrict__ FN,
    const T* __restrict__ V,
    int nx, int nv,
    T dv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi, int k0)
{
    // interior 行のみ（i = 1..nx-2）
    int i = blockIdx.x + 1;
    if (i <= 0 || i >= nx-1) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const T* row_i   = F + (size_t)i * nv;
    const T* row_im1 = F + (size_t)(i-1) * nv;
    const T* row_ip1 = F + (size_t)(i+1) * nv;
    T*       out_i   = FN + (size_t)i * nv;

    // --- 1) モーメント（並列集約） ---
    T s0 = 0, s1 = 0, s2 = 0;
    for (int j = tid; j < nv; j += nthreads){
        T fi = row_i[j];
        T v  = V[j];
        s0 += fi;
        s1 += fi * v;
        s2 += fi * v * v;
    }

    extern __shared__ unsigned char smem_raw[];
    T* sh0 = reinterpret_cast<T*>(smem_raw);
    T* sh1 = sh0 + nthreads;
    T* sh2 = sh1 + nthreads;
    sh0[tid] = s0; sh1[tid] = s1; sh2[tid] = s2;
    __syncthreads();
    for (int stride = nthreads>>1; stride>0; stride>>=1){
        if (tid < stride){
            sh0[tid] += sh0[tid+stride];
            sh1[tid] += sh1[tid+stride];
            sh2[tid] += sh2[tid+stride];
        }
        __syncthreads();
    }

    __shared__ T u_s, inv_tau_s, coeff_s, invT_s;
    if (tid == 0){
        T n = sh0[0] * dv;
        T u = (sh1[0] * dv) / n;
        T Tm = (sh2[0] * dv) / n - u * u;
        // τ と各係数
        u_s = u;
        inv_tau_s = (n * sqrt(Tm)) / tau_tilde;     // 1/τ
        coeff_s   = (n * inv_sqrt_2pi) / sqrt(Tm); // n/sqrt(2πT)
        invT_s    = (T)0.5 / Tm;                   // 1/(2T)
    }
    __syncthreads();

    // --- 2) streaming + collision 合成 ---
    for (int j = tid; j < nv; j += nthreads){
        T v = V[j];
        T fi = row_i[j];
        T fim1 = row_im1[j];
        T fip1 = row_ip1[j];

        // upwind: v>=0 → (fi - fim1) / dx, v<0 → (fip1 - fi) / dx
        T flux_pos = (-(v / dx)) * (fi - fim1);
        T flux_neg = (-(v / dx)) * (fip1 - fi);
        T stream   = (j >= k0) ? flux_pos : flux_neg;

        // Maxwellian & collision
        T diff = v - u_s;
        T fm   = coeff_s * my_exp<T>(-(diff*diff) * invT_s);
        T coll = (fm - fi) * inv_tau_s;

        out_i[j] = fi + dt * (stream + coll);
    }
}

template <typename T>
void launch_explicit_step(const T* f, T* fn, const T* v,
                          int nx, int nv,
                          T dv, T dt, T dx,
                          T tau_tilde, T inv_sqrt_2pi,
                          int k0, cudaStream_t stream)
{
    if (nx <= 2) return;
    int grid = nx - 2;
    int block = (nv >= 512) ? 512 : (nv >= 256 ? 256 : 128);
    size_t shmem = 3 * block * sizeof(T); // s0,s1,s2

    explicit_step_kernel<T><<<grid, block, shmem, stream>>>(
        f, fn, v, nx, nv, dv, dt, dx, tau_tilde, inv_sqrt_2pi, k0);
}

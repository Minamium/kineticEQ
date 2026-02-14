// kineticEQ/src/kineticEQ/cuda_kernel/BGK1D1V/implicit_AA/implicit_AA_kernel.cu
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <mutex>
#include <vector>
#include <tuple>

#define CUDA_CHECK(err) TORCH_CHECK((err) == cudaSuccess, "CUDA error: ", cudaGetErrorString(err))
#define CUBLAS_CHECK(st) TORCH_CHECK((st) == CUBLAS_STATUS_SUCCESS, "cuBLAS error code: ", int(st))
#define CUBLAS_CHECK_MSG(st, msg) TORCH_CHECK((st) == CUBLAS_STATUS_SUCCESS, msg, " cuBLAS error code: ", int(st))
#define CUSOLVER_CHECK(st) TORCH_CHECK((st) == CUSOLVER_STATUS_SUCCESS, "cuSOLVER error code: ", int(st))

// ------------------------
// device-handle cache
// ------------------------
struct DeviceHandles {
    bool inited = false;
    int device = -1;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
};

static std::mutex g_handle_mtx;
static std::vector<DeviceHandles> g_handles;

static DeviceHandles& get_handles(int device) {
    std::lock_guard<std::mutex> lock(g_handle_mtx);
    if ((int)g_handles.size() <= device) g_handles.resize(device + 1);
    auto& h = g_handles[device];
    if (!h.inited) {
        h.device = device;
        CUDA_CHECK(cudaSetDevice(device));
        CUBLAS_CHECK(cublasCreate(&h.cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&h.cusolver));
        h.inited = true;
    }
    return h;
}

// ------------------------
// kernels
// ------------------------
template <typename scalar_t>
__global__ void pack_wk_wnew_r_kernel(
    const scalar_t* __restrict__ n,
    const scalar_t* __restrict__ nu,
    const scalar_t* __restrict__ T,
    const scalar_t* __restrict__ n_new,
    const scalar_t* __restrict__ nu_new,
    const scalar_t* __restrict__ T_new,
    scalar_t* __restrict__ wk,
    scalar_t* __restrict__ wnew,
    scalar_t* __restrict__ r,
    int n_inner
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inner) return;

    // interior index in original arrays is (i+1)
    scalar_t nk  = n[i + 1];
    scalar_t nuk = nu[i + 1];
    scalar_t Tk  = T[i + 1];

    scalar_t nn  = n_new[i + 1];
    scalar_t nun = nu_new[i + 1];
    scalar_t Tn  = T_new[i + 1];

    int o0 = i;
    int o1 = n_inner + i;
    int o2 = 2 * n_inner + i;

    wk[o0] = nk;   wk[o1] = nuk;  wk[o2] = Tk;
    wnew[o0] = nn; wnew[o1] = nun; wnew[o2] = Tn;

    r[o0] = wnew[o0] - wk[o0];
    r[o1] = wnew[o1] - wk[o1];
    r[o2] = wnew[o2] - wk[o2];
}

template <typename scalar_t>
__global__ void write_history_col_kernel(
    const scalar_t* __restrict__ wnew,
    const scalar_t* __restrict__ r,
    scalar_t* __restrict__ G, // (d, aa_cols) row-major
    scalar_t* __restrict__ R, // (d, aa_cols) row-major
    int d,
    int aa_cols,
    int head
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    // row-major: (i, head) => i*aa_cols + head
    int idx = i * aa_cols + head;
    G[idx] = wnew[i];
    R[idx] = r[i];
}

template <typename scalar_t>
__global__ void gather_ring_to_work_kernel(
    const scalar_t* __restrict__ G, // (d, aa_cols)
    const scalar_t* __restrict__ R, // (d, aa_cols)
    scalar_t* __restrict__ Gwork,   // (d, aa_cols)  (we fill columns 0..m-1)
    scalar_t* __restrict__ Rwork,   // (d, aa_cols)
    int d,
    int aa_cols,
    int head,     // head AFTER write (next insertion index)
    int m         // hist_len (<=aa_cols)
) {
    // linearize over d*m
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * m;
    if (tid >= total) return;

    int i = tid / m;   // row [0..d)
    int j = tid - i * m; // col [0..m)

    // ring: valid columns are head-m ... head-1
    int col = head - m + j;
    col %= aa_cols;
    if (col < 0) col += aa_cols;

    // read G(i,col) -> write Gwork(i,j)
    Gwork[i * aa_cols + j] = G[i * aa_cols + col];
    Rwork[i * aa_cols + j] = R[i * aa_cols + col];
}

template <typename scalar_t>
__global__ void add_reg_diag_kernel(scalar_t* __restrict__ A, int lda, int m, scalar_t reg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    A[i * lda + i] += reg;
}

template <typename scalar_t>
__global__ void init_alpha_ones_kernel(scalar_t* __restrict__ alpha, int aa_cols, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < aa_cols) {
        alpha[i] = (i < m) ? scalar_t(1) : scalar_t(0);
    }
}

// Normalize alpha[0:m] by sum, clamp to [-alpha_max, alpha_max], and set alpha[m:]=0.
// One block is enough because m <= ~16 typically.
template <typename scalar_t>
__global__ void normalize_alpha_kernel(scalar_t* __restrict__ alpha, int aa_cols, int m, scalar_t alpha_max) {
    __shared__ scalar_t ssum;
    if (threadIdx.x == 0) ssum = scalar_t(0);
    __syncthreads();

    scalar_t local = scalar_t(0);
    for (int i = threadIdx.x; i < m; i += blockDim.x) local += alpha[i];
    atomicAdd(&ssum, local);
    __syncthreads();

    scalar_t denom = ssum;
    if (denom < scalar_t(1e-30)) denom = scalar_t(1e-30);

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        scalar_t a = alpha[i] / denom;
        if (alpha_max > scalar_t(0)) {
            if (a >  alpha_max) a =  alpha_max;
            if (a < -alpha_max) a = -alpha_max;
        }
        alpha[i] = a;
    }
    for (int i = threadIdx.x + m; i < aa_cols; i += blockDim.x) {
        alpha[i] = scalar_t(0);
    }
}

template <typename scalar_t>
__global__ void blend_kernel(
    scalar_t* __restrict__ w,              // in/out (d)
    const scalar_t* __restrict__ wnew,     // (d)
    int d,
    scalar_t beta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    // w = beta*w + (1-beta)*wnew
    w[i] = beta * w[i] + (scalar_t(1) - beta) * wnew[i];
}

template <typename scalar_t>
__global__ void unpack_project_kernel(
    scalar_t* __restrict__ n,
    scalar_t* __restrict__ nu,
    scalar_t* __restrict__ T,
    const scalar_t* __restrict__ w,   // (d)
    int nx,
    int n_inner,
    scalar_t n_floor,
    scalar_t T_floor,
    scalar_t u_max
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_inner) return;

    int o0 = i;
    int o1 = n_inner + i;
    int o2 = 2 * n_inner + i;

    scalar_t nn = w[o0];
    scalar_t nun = w[o1];
    scalar_t Tn = w[o2];

    if (nn < n_floor) nn = n_floor;
    if (Tn < T_floor) Tn = T_floor;

    // project u
    scalar_t u = nun / nn;
    if (u >  u_max) u =  u_max;
    if (u < -u_max) u = -u_max;
    nun = nn * u;

    int ix = i + 1;
    n[ix]  = nn;
    nu[ix] = nun;
    T[ix]  = Tn;
}

template <typename scalar_t>
__global__ void copy_noaa_kernel(
    scalar_t* __restrict__ n,
    scalar_t* __restrict__ nu,
    scalar_t* __restrict__ T,
    const scalar_t* __restrict__ n_new,
    const scalar_t* __restrict__ nu_new,
    const scalar_t* __restrict__ T_new,
    int nx,
    scalar_t n_floor,
    scalar_t T_floor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    scalar_t nn = n_new[i];
    scalar_t Tn = T_new[i];
    if (nn < n_floor) nn = n_floor;
    if (Tn < T_floor) Tn = T_floor;

    n[i]  = nn;
    nu[i] = nu_new[i];
    T[i]  = Tn;
}

// set boundary from *_new (keeps consistent with your python path)
template <typename scalar_t>
__global__ void copy_boundary_kernel(
    scalar_t* __restrict__ n,
    scalar_t* __restrict__ nu,
    scalar_t* __restrict__ T,
    const scalar_t* __restrict__ n_new,
    const scalar_t* __restrict__ nu_new,
    const scalar_t* __restrict__ T_new,
    int nx,
    scalar_t n_floor,
    scalar_t T_floor
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // left
        scalar_t nl = n_new[0];
        scalar_t Tl = T_new[0];
        if (nl < n_floor) nl = n_floor;
        if (Tl < T_floor) Tl = T_floor;
        n[0] = nl; nu[0] = nu_new[0]; T[0] = Tl;

        // right
        scalar_t nr = n_new[nx - 1];
        scalar_t Tr = T_new[nx - 1];
        if (nr < n_floor) nr = n_floor;
        if (Tr < T_floor) Tr = T_floor;
        n[nx - 1] = nr; nu[nx - 1] = nu_new[nx - 1]; T[nx - 1] = Tr;
    }
}

// ------------------------
// cuBLAS/cuSOLVER wrappers
// ------------------------
template <typename scalar_t> struct Blas;

template <> struct Blas<float> {
    static cublasStatus_t gemm(cublasHandle_t h,
                              cublasOperation_t opA, cublasOperation_t opB,
                              int m, int n, int k,
                              const float* alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              const float* beta,
                              float* C, int ldc) {
        return cublasSgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static cublasStatus_t gemv(cublasHandle_t h,
                              cublasOperation_t opA,
                              int m, int n,
                              const float* alpha,
                              const float* A, int lda,
                              const float* x, int incx,
                              const float* beta,
                              float* y, int incy) {
        return cublasSgemv(h, opA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    static cusolverStatus_t potrf_buffersize(cusolverDnHandle_t s, int n, float* A, int lda, int* lwork) {
        return cusolverDnSpotrf_bufferSize(s, CUBLAS_FILL_MODE_LOWER, n, A, lda, lwork);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t s, int n, float* A, int lda, float* work, int lwork, int* info) {
        return cusolverDnSpotrf(s, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lwork, info);
    }
    static cusolverStatus_t potrs(cusolverDnHandle_t s, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* info) {
        return cusolverDnSpotrs(s, CUBLAS_FILL_MODE_LOWER, n, nrhs, A, lda, B, ldb, info);
    }
};

template <> struct Blas<double> {
    static cublasStatus_t gemm(cublasHandle_t h,
                              cublasOperation_t opA, cublasOperation_t opB,
                              int m, int n, int k,
                              const double* alpha,
                              const double* A, int lda,
                              const double* B, int ldb,
                              const double* beta,
                              double* C, int ldc) {
        return cublasDgemm(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static cublasStatus_t gemv(cublasHandle_t h,
                              cublasOperation_t opA,
                              int m, int n,
                              const double* alpha,
                              const double* A, int lda,
                              const double* x, int incx,
                              const double* beta,
                              double* y, int incy) {
        return cublasDgemv(h, opA, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    static cusolverStatus_t potrf_buffersize(cusolverDnHandle_t s, int n, double* A, int lda, int* lwork) {
        return cusolverDnDpotrf_bufferSize(s, CUBLAS_FILL_MODE_LOWER, n, A, lda, lwork);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t s, int n, double* A, int lda, double* work, int lwork, int* info) {
        return cusolverDnDpotrf(s, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lwork, info);
    }
    static cusolverStatus_t potrs(cusolverDnHandle_t s, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* info) {
        return cusolverDnDpotrs(s, CUBLAS_FILL_MODE_LOWER, n, nrhs, A, lda, B, ldb, info);
    }
};

// ------------------------
// lwork query
// ------------------------
int64_t implicit_aa_potrf_lwork_cuda(torch::Tensor aa_A, int64_t n) {
    c10::cuda::CUDAGuard device_guard(aa_A.device());
    int device = aa_A.get_device();
    auto& h = get_handles(device);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const cudaStream_t cuda_stream = stream.stream();
    CUBLAS_CHECK(cublasSetStream(h.cublas, cuda_stream));
    CUSOLVER_CHECK(cusolverDnSetStream(h.cusolver, cuda_stream));

    TORCH_CHECK(aa_A.is_cuda() && aa_A.is_contiguous(), "aa_A must be contiguous CUDA");
    TORCH_CHECK(aa_A.dim() == 2 && aa_A.size(0) == aa_A.size(1), "aa_A must be square");
    TORCH_CHECK(n >= 1 && n <= aa_A.size(0), "invalid n");
    int lda = (int)aa_A.size(0);

    int lwork = 0;
    if (aa_A.scalar_type() == torch::kFloat32) {
        float* A = (float*)aa_A.data_ptr<float>();
        CUSOLVER_CHECK(Blas<float>::potrf_buffersize(h.cusolver, (int)n, A, lda, &lwork));
    } else {
        double* A = (double*)aa_A.data_ptr<double>();
        CUSOLVER_CHECK(Blas<double>::potrf_buffersize(h.cusolver, (int)n, A, lda, &lwork));
    }
    return (int64_t)lwork;
}

// ------------------------
// main AA step
// ------------------------
std::tuple<int64_t, int64_t> implicit_aa_step_inplace_cuda(
    torch::Tensor n,
    torch::Tensor nu,
    torch::Tensor T,
    const torch::Tensor n_new,
    const torch::Tensor nu_new,
    const torch::Tensor T_new,
    torch::Tensor aa_G,
    torch::Tensor aa_R,
    torch::Tensor aa_A,
    torch::Tensor aa_alpha,
    torch::Tensor aa_wk,
    torch::Tensor aa_wnew,
    torch::Tensor aa_wtmp,
    int64_t hist_len,
    int64_t head,
    bool apply,
    double beta_,
    double reg_,
    double alpha_max_,
    double n_floor_,
    double T_floor_,
    double u_max_,
    torch::Tensor solver_work,
    torch::Tensor solver_info,
    torch::Tensor G_work,
    torch::Tensor R_work
) {
    c10::cuda::CUDAGuard device_guard(n.device());
    int device = n.get_device();

    const auto stream = at::cuda::getCurrentCUDAStream();
    const cudaStream_t cuda_stream = stream.stream();

    const int64_t nx = n.numel();
    TORCH_CHECK(nx >= 2, "nx must be >= 2");
    TORCH_CHECK(n_new.numel() == nx && nu_new.numel() == nx && T_new.numel() == nx, "new moments size mismatch");

    const int n_inner = (int)std::max<int64_t>(nx - 2, 0);
    const int d = 3 * n_inner;

    const int aa_cols = (int)aa_G.size(1);
    TORCH_CHECK(aa_G.dim() == 2 && aa_R.dim() == 2, "aa_G/aa_R must be 2D");
    TORCH_CHECK(aa_G.size(0) == d && aa_R.size(0) == d, "aa_G/aa_R first dim must be d=3*(nx-2)");
    TORCH_CHECK(aa_R.size(1) == aa_G.size(1), "aa_G/aa_R second dim mismatch");
    TORCH_CHECK(aa_A.dim() == 2 && aa_A.size(0) == aa_A.size(1) && aa_A.size(0) == aa_cols, "aa_A must be (aa_cols, aa_cols)");
    TORCH_CHECK(aa_alpha.numel() == aa_cols, "aa_alpha must be (aa_cols)");
    TORCH_CHECK(aa_wk.numel() == d && aa_wnew.numel() == d && aa_wtmp.numel() == d, "aa_wk/wnew/wtmp must be (d)");

    // update ring indices on CPU (no sync)
    int64_t hist_len_in = std::max<int64_t>(hist_len, 0);
    int64_t head_in = std::max<int64_t>(head, 0) % std::max<int>(aa_cols, 1);
    int64_t hist_len_out = std::min<int64_t>(hist_len_in + 1, aa_cols);
    int64_t head_out = (head_in + 1) % aa_cols;

    // If degenerate (no interior), just copy new moments with floors.
    if (n_inner <= 0 || d <= 0) {
        AT_DISPATCH_FLOATING_TYPES(n.scalar_type(), "copy_noaa_degenerate", [&](){
            const scalar_t n_floor = (scalar_t)n_floor_;
            const scalar_t T_floor = (scalar_t)T_floor_;
            int threads = 256;
            int blocks = (int)((nx + threads - 1) / threads);
            copy_noaa_kernel<scalar_t><<<blocks, threads, 0, cuda_stream>>>(
                (scalar_t*)n.data_ptr<scalar_t>(),
                (scalar_t*)nu.data_ptr<scalar_t>(),
                (scalar_t*)T.data_ptr<scalar_t>(),
                (const scalar_t*)n_new.data_ptr<scalar_t>(),
                (const scalar_t*)nu_new.data_ptr<scalar_t>(),
                (const scalar_t*)T_new.data_ptr<scalar_t>(),
                (int)nx, n_floor, T_floor
            );
        });
        return {hist_len_out, head_out};
    }

    // Reusable workspace is required from Python side; no hidden allocation here.
    TORCH_CHECK(G_work.defined() && R_work.defined(), "G_work/R_work must be defined");
    TORCH_CHECK(G_work.numel() > 0 && R_work.numel() > 0, "G_work/R_work must be preallocated");
    TORCH_CHECK(G_work.is_cuda() && R_work.is_cuda(), "G_work/R_work must be CUDA");
    TORCH_CHECK(G_work.scalar_type() == n.scalar_type() && R_work.scalar_type() == n.scalar_type(), "G_work/R_work dtype mismatch");
    TORCH_CHECK(G_work.is_contiguous() && R_work.is_contiguous(), "G_work/R_work must be contiguous");
    TORCH_CHECK(G_work.sizes() == aa_G.sizes() && R_work.sizes() == aa_R.sizes(), "G_work/R_work must match aa_G/aa_R shape");

    // Pack wk,wnew,r (=wtmp) from (n,nu,T) and (n_new,nu_new,T_new)
    AT_DISPATCH_FLOATING_TYPES(n.scalar_type(), "implicit_aa_pack_write", [&](){
        const int threads = 256;
        const int blocks = (n_inner + threads - 1) / threads;
        pack_wk_wnew_r_kernel<scalar_t><<<blocks, threads, 0, cuda_stream>>>(
            (const scalar_t*)n.data_ptr<scalar_t>(),
            (const scalar_t*)nu.data_ptr<scalar_t>(),
            (const scalar_t*)T.data_ptr<scalar_t>(),
            (const scalar_t*)n_new.data_ptr<scalar_t>(),
            (const scalar_t*)nu_new.data_ptr<scalar_t>(),
            (const scalar_t*)T_new.data_ptr<scalar_t>(),
            (scalar_t*)aa_wk.data_ptr<scalar_t>(),
            (scalar_t*)aa_wnew.data_ptr<scalar_t>(),
            (scalar_t*)aa_wtmp.data_ptr<scalar_t>(), // r
            n_inner
        );

        // Write new column into history at head_in
        {
            int blocks2 = (d + threads - 1) / threads;
            write_history_col_kernel<scalar_t><<<blocks2, threads, 0, cuda_stream>>>(
                (const scalar_t*)aa_wnew.data_ptr<scalar_t>(),
                (const scalar_t*)aa_wtmp.data_ptr<scalar_t>(), // r
                (scalar_t*)aa_G.data_ptr<scalar_t>(),
                (scalar_t*)aa_R.data_ptr<scalar_t>(),
                d, aa_cols, (int)head_in
            );
        }

        // Decide whether to apply AA
        const int m = (int)hist_len_out;
        const bool do_apply = apply && (m >= 2);

        if (!do_apply) {
            // Just copy new moments with floors
            int threads3 = 256;
            int blocks3 = (int)((nx + threads3 - 1) / threads3);
            copy_noaa_kernel<scalar_t><<<blocks3, threads3, 0, cuda_stream>>>(
                (scalar_t*)n.data_ptr<scalar_t>(),
                (scalar_t*)nu.data_ptr<scalar_t>(),
                (scalar_t*)T.data_ptr<scalar_t>(),
                (const scalar_t*)n_new.data_ptr<scalar_t>(),
                (const scalar_t*)nu_new.data_ptr<scalar_t>(),
                (const scalar_t*)T_new.data_ptr<scalar_t>(),
                (int)nx,
                (scalar_t)n_floor_, (scalar_t)T_floor_
            );
            return;
        }

        auto& h = get_handles(device);
        CUBLAS_CHECK_MSG(cublasSetStream(h.cublas, cuda_stream), "cublasSetStream(step_inplace)");
        CUSOLVER_CHECK(cusolverDnSetStream(h.cusolver, cuda_stream));

        // Gather ring history into columns 0..m-1 of G_work/R_work using head_out (after insertion)
        {
            int threads4 = 256;
            int total = d * m;
            int blocks4 = (total + threads4 - 1) / threads4;
            gather_ring_to_work_kernel<scalar_t><<<blocks4, threads4, 0, cuda_stream>>>(
                (const scalar_t*)aa_G.data_ptr<scalar_t>(),
                (const scalar_t*)aa_R.data_ptr<scalar_t>(),
                (scalar_t*)G_work.data_ptr<scalar_t>(),
                (scalar_t*)R_work.data_ptr<scalar_t>(),
                d, aa_cols,
                (int)head_out,
                m
            );
        }

        // Build A = R^T R (size m×m) in aa_A (top-left)
        // Interpretation:
        // - R_work is (d, aa_cols) row-major
        // - treat it as column-major matrix of size (aa_cols × d) with lda=aa_cols (rows=aa_cols, cols=d)
        // - use first m rows => (m × d), lda=aa_cols
        const scalar_t one = (scalar_t)1;
        const scalar_t zero = (scalar_t)0;
        scalar_t* A = (scalar_t*)aa_A.data_ptr<scalar_t>();
        const scalar_t* Rcol = (const scalar_t*)R_work.data_ptr<scalar_t>();

        // cublas gemm: A(m×m) = R(m×d) * R(m×d)^T
        // column-major: opN (m×d), opT (d×m)
        CUBLAS_CHECK_MSG(Blas<scalar_t>::gemm(
            h.cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m, m, d,
            &one,
            Rcol, aa_cols,
            Rcol, aa_cols,
            &zero,
            A, aa_cols
        ), "cublasGEMM(R*R^T)");

        // Add reg to diag
        {
            int threads5 = 128;
            int blocks5 = (m + threads5 - 1) / threads5;
            add_reg_diag_kernel<scalar_t><<<blocks5, threads5, 0, cuda_stream>>>(A, aa_cols, m, (scalar_t)reg_);
        }

        // Prepare alpha RHS = ones (first m), zeros elsewhere
        {
            int threads6 = 128;
            int blocks6 = (aa_cols + threads6 - 1) / threads6;
            init_alpha_ones_kernel<scalar_t><<<blocks6, threads6, 0, cuda_stream>>>(
                (scalar_t*)aa_alpha.data_ptr<scalar_t>(), aa_cols, m
            );
        }

        // Solver work/info are also required preallocated buffers.
        TORCH_CHECK(solver_info.defined(), "solver_info must be defined");
        TORCH_CHECK(solver_info.is_cuda() && solver_info.scalar_type() == torch::kInt32 && solver_info.numel() == 1,
                    "solver_info must be int32 CUDA scalar");
        TORCH_CHECK(solver_info.is_contiguous(), "solver_info must be contiguous");

        TORCH_CHECK(solver_work.defined() && solver_work.numel() > 0, "solver_work must be preallocated");
        TORCH_CHECK(solver_work.is_cuda() && solver_work.scalar_type() == n.scalar_type(), "solver_work dtype/device mismatch");
        TORCH_CHECK(solver_work.is_contiguous(), "solver_work must be contiguous");
        const int lwork = (int)solver_work.numel();

        // Cholesky factorization (in-place on A)
        CUSOLVER_CHECK(Blas<scalar_t>::potrf(
            h.cusolver, m, A, aa_cols,
            (scalar_t*)solver_work.data_ptr<scalar_t>(),
            lwork,
            (int*)solver_info.data_ptr<int>()
        ));

        // Solve A x = ones  (x stored in aa_alpha first m entries)
        CUSOLVER_CHECK(Blas<scalar_t>::potrs(
            h.cusolver, m, 1,
            A, aa_cols,
            (scalar_t*)aa_alpha.data_ptr<scalar_t>(), aa_cols,
            (int*)solver_info.data_ptr<int>()
        ));

        // Normalize alpha by sum, clamp
        {
            int threads7 = 32;
            normalize_alpha_kernel<scalar_t><<<1, threads7, 0, cuda_stream>>>(
                (scalar_t*)aa_alpha.data_ptr<scalar_t>(), aa_cols, m, (scalar_t)alpha_max_
            );
        }

        // Compute wAA = G^T * alpha  (G is m×d column-major view of G_work with lda=aa_cols)
        const scalar_t* Gcol = (const scalar_t*)G_work.data_ptr<scalar_t>();
        scalar_t* wAA = (scalar_t*)aa_wtmp.data_ptr<scalar_t>(); // overwrite r with wAA

        // y(d) = G(m×d)^T * alpha(m)
        CUBLAS_CHECK_MSG(Blas<scalar_t>::gemv(
            h.cublas,
            CUBLAS_OP_T,
            m, d,
            &one,
            Gcol, aa_cols,
            (const scalar_t*)aa_alpha.data_ptr<scalar_t>(), 1,
            &zero,
            wAA, 1
        ), "cublasGEMV(G^T*alpha)");

        // Blend with wnew (beta)
        {
            int threads8 = 256;
            int blocks8 = (d + threads8 - 1) / threads8;
            blend_kernel<scalar_t><<<blocks8, threads8, 0, cuda_stream>>>(
                wAA,
                (const scalar_t*)aa_wnew.data_ptr<scalar_t>(),
                d,
                (scalar_t)beta_
            );
        }

        // Unpack to n,nu,T interior + projection; boundaries from *_new
        copy_boundary_kernel<scalar_t><<<1, 1, 0, cuda_stream>>>(
            (scalar_t*)n.data_ptr<scalar_t>(),
            (scalar_t*)nu.data_ptr<scalar_t>(),
            (scalar_t*)T.data_ptr<scalar_t>(),
            (const scalar_t*)n_new.data_ptr<scalar_t>(),
            (const scalar_t*)nu_new.data_ptr<scalar_t>(),
            (const scalar_t*)T_new.data_ptr<scalar_t>(),
            (int)nx,
            (scalar_t)n_floor_,
            (scalar_t)T_floor_
        );

        {
            int threads9 = 256;
            int blocks9 = (n_inner + threads9 - 1) / threads9;
            unpack_project_kernel<scalar_t><<<blocks9, threads9, 0, cuda_stream>>>(
                (scalar_t*)n.data_ptr<scalar_t>(),
                (scalar_t*)nu.data_ptr<scalar_t>(),
                (scalar_t*)T.data_ptr<scalar_t>(),
                (const scalar_t*)aa_wtmp.data_ptr<scalar_t>(),
                (int)nx, n_inner,
                (scalar_t)n_floor_,
                (scalar_t)T_floor_,
                (scalar_t)u_max_
            );
        }
    });

    return {hist_len_out, head_out};
}
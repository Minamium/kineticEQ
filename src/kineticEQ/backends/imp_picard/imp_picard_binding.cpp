// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <string>

// .cu 側ラッパ（カーネル起動は .cu 内）
extern "C" {
void launch_moments_kernel_double(
    const double* f, const double* v, int nx, int nv, double dv,
    double* n_out, double* u_out, double* T_out, cudaStream_t stream);

void launch_build_system_kernel_double(
    const double* f_k, const double* v,
    const double* n_arr, const double* u_arr, const double* T_arr,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream);

void launch_scatter_and_residual_kernel_double(
    const double* fz, const double* solution, double* fn_tmp,
    int nx, int nv, double* res_out, cudaStream_t stream);
}

// ========== build_system: moments + band-system ==========
static void build_system_and_moments(
    at::Tensor f_k,   // (nx,nv) f^k
    at::Tensor fz,    // (nx,nv) Picard candidate
    at::Tensor v,     // (nv,)
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    // outputs
    at::Tensor dl, at::Tensor dd, at::Tensor du, at::Tensor B, // (nv, nx-2)
    at::Tensor n_arr, at::Tensor u_arr, at::Tensor T_arr       // (nx,)
){
    TORCH_CHECK(f_k.is_cuda() && fz.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f_k.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(f_k.is_contiguous() && fz.is_contiguous() && v.is_contiguous(),
                "inputs must be contiguous");

    const int64_t nx = f_k.size(0);
    const int64_t nv = f_k.size(1);
    TORCH_CHECK(fz.size(0) == nx && fz.size(1) == nv, "fz shape mismatch");
    TORCH_CHECK(v.dim() == 1 && v.size(0) == nv, "v shape mismatch");

    const int64_t n_inner = std::max<int64_t>(nx - 2, 0);
    if (n_inner <= 0) return;

    TORCH_CHECK(dl.size(0) == nv && dl.size(1) == n_inner, "dl shape (nv,nx-2)");
    TORCH_CHECK(dd.size(0) == nv && dd.size(1) == n_inner, "dd shape (nv,nx-2)");
    TORCH_CHECK(du.size(0) == nv && du.size(1) == n_inner, "du shape (nv,nx-2)");
    TORCH_CHECK(B .size(0) == nv && B .size(1) == n_inner, "B  shape (nv,nx-2)");

    TORCH_CHECK(n_arr.size(0) == nx && u_arr.size(0) == nx && T_arr.size(0) == nx,
                "moment arrays must be (nx,)");

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t s = stream.stream();

    // 1) fz → moments
    launch_moments_kernel_double(
        fz.data_ptr<double>(), v.data_ptr<double>(),
        (int)nx, (int)nv, dv,
        n_arr.data_ptr<double>(), u_arr.data_ptr<double>(), T_arr.data_ptr<double>(),
        s);
    AT_CUDA_CHECK(cudaGetLastError());

    // 2) (n,u,T) と f_k から (dl,dd,du,B)
    launch_build_system_kernel_double(
        f_k.data_ptr<double>(), v.data_ptr<double>(),
        n_arr.data_ptr<double>(), u_arr.data_ptr<double>(), T_arr.data_ptr<double>(),
        (int)nx, (int)nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
        s);
    AT_CUDA_CHECK(cudaGetLastError());
}

// ========== writeback + residual ==========
static void writeback_and_residual(
    at::Tensor fz,         // (nx,nv)
    at::Tensor solution,   // (nv,n_inner)
    at::Tensor fn_tmp,     // (nx,nv)
    at::Tensor res_buf     // (1,)
){
    TORCH_CHECK(fz.is_cuda() && solution.is_cuda() && fn_tmp.is_cuda() && res_buf.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(fz.scalar_type() == at::kDouble &&
                solution.scalar_type() == at::kDouble &&
                fn_tmp.scalar_type() == at::kDouble &&
                res_buf.scalar_type() == at::kDouble, "float64 only");

    const int nx = (int)fz.size(0);
    const int nv = (int)fz.size(1);
    const int n_inner = std::max(nx - 2, 0);

    TORCH_CHECK(solution.size(0) == nv && solution.size(1) == n_inner,
                "solution must be (nv,nx-2)");
    TORCH_CHECK(fn_tmp.size(0) == nx && fn_tmp.size(1) == nv,
                "fn_tmp must be (nx,nv)");
    TORCH_CHECK(res_buf.numel() == 1, "res_buf must be scalar tensor");

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t s = stream.stream();

    // 残差初期化
    AT_CUDA_CHECK(cudaMemsetAsync(res_buf.data_ptr(), 0, sizeof(double), s));

    launch_scatter_and_residual_kernel_double(
        fz.data_ptr<double>(),
        solution.data_ptr<double>(),
        fn_tmp.data_ptr<double>(),
        nx, nv,
        res_buf.data_ptr<double>(),
        s);
    AT_CUDA_CHECK(cudaGetLastError());
}

// ========= pybind =========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_system", &build_system_and_moments,
          "Build tridiagonal system and moments (double)");
    m.def("writeback_and_residual", &writeback_and_residual,
          "Scatter solution and compute L-inf residual (double)");
}
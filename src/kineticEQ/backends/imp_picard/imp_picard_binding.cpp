// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

using at::Tensor;
namespace cgpu = at::cuda;

namespace imp_picard {
    // kernels
    __global__ void moments_kernel_double(
        const double* fz, const double* v, int nx, int nv, double dv,
        double* n_out, double* u_out, double* T_out);

    __global__ void build_system_kernel_double(
        const double* fk, const double* v,
        const double* n_arr, const double* u_arr, const double* T_arr,
        int nx, int nv, int n_inner,
        double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
        double* dl, double* dd, double* du, double* B);

    __global__ void scatter_and_residual_kernel_double(
        const double* fz, const double* solution,
        int nx, int nv, int n_inner,
        double* fn_out, double* res_dev);
}

// ---- 行列構成（moments + bands/RHS）----
void build_system_and_moments(
    Tensor fk, Tensor fz, Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    Tensor dl, Tensor dd, Tensor du, Tensor B,
    Tensor n_out, Tensor u_out, Tensor T_out)
{
    TORCH_CHECK(fk.is_cuda() && fz.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(fk.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(fk.sizes() == fz.sizes(), "fk and fz must have same shape");
    TORCH_CHECK(fk.dim()==2 && v.dim()==1, "fk/fz:(nx,nv), v:(nv,)");

    const int nx = fk.size(0);
    const int nv = fk.size(1);
    const int n_inner = std::max(nx - 2, 0);

    TORCH_CHECK(dl.sizes() == at::IntArrayRef{nv, n_inner}, "dl shape (nv,nx-2)");
    TORCH_CHECK(dd.sizes() == at::IntArrayRef{nv, n_inner}, "dd shape (nv,nx-2)");
    TORCH_CHECK(du.sizes() == at::IntArrayRef{nv, n_inner}, "du shape (nv,nx-2)");
    TORCH_CHECK(B.sizes()  == at::IntArrayRef{nv, n_inner}, "B  shape (nv,nx-2)");
    TORCH_CHECK(n_out.numel()==nx && u_out.numel()==nx && T_out.numel()==nx, "moments size nx");

    auto stream = cgpu::getCurrentCUDAStream();

    // 1) moments
    const int tMom = 256;
    size_t shm = sizeof(double) * tMom * 3;
    imp_picard::moments_kernel_double<<<nx, tMom, shm, stream>>>(
        fz.data_ptr<double>(), v.data_ptr<double>(),
        nx, nv, dv,
        n_out.data_ptr<double>(), u_out.data_ptr<double>(), T_out.data_ptr<double>());

    if (n_inner == 0) return;

    // 2) bands+RHS
    const int tBand = 128;
    dim3 grid(nv, (n_inner + tBand - 1) / tBand);
    imp_picard::build_system_kernel_double<<<grid, tBand, 0, stream>>>(
        fk.data_ptr<double>(), v.data_ptr<double>(),
        n_out.data_ptr<double>(), u_out.data_ptr<double>(), T_out.data_ptr<double>(),
        nx, nv, n_inner, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl.data_ptr<double>(), dd.data_ptr<double>(),
        du.data_ptr<double>(), B.data_ptr<double>());
}

// ---- 書き戻し＋残差 ----
void writeback_and_residual(
    Tensor fz, Tensor solution, Tensor fn_out, Tensor res_dev)
{
    TORCH_CHECK(fz.is_cuda() && solution.is_cuda() && fn_out.is_cuda() && res_dev.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(fz.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(res_dev.numel() == 1, "res_dev must be scalar");

    const int nx = fz.size(0);
    const int nv = fz.size(1);
    const int n_inner = std::max(nx - 2, 0);
    TORCH_CHECK(solution.sizes() == at::IntArrayRef{nv, n_inner}, "solution shape (nv,nx-2)");
    TORCH_CHECK(fn_out.sizes() == fz.sizes(), "fn_out must match fz");

    auto stream = cgpu::getCurrentCUDAStream();
    // res_dev を 0 で初期化
    AT_CUDA_CHECK(cudaMemsetAsync(res_dev.data_ptr(), 0, sizeof(double), stream));

    if (n_inner == 0) {
        fn_out.copy_(fz);
        return;
    }
    const int t = 256;
    const int N = n_inner * nv;
    const int blocks = (N + t - 1) / t;

    imp_picard::scatter_and_residual_kernel_double<<<blocks, t, 0, stream>>>(
        fz.data_ptr<double>(), solution.data_ptr<double>(),
        nx, nv, n_inner,
        fn_out.data_ptr<double>(), res_dev.data_ptr<double>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_system", &build_system_and_moments,
          "Build tri-diagonal system and moments (double)");
    m.def("writeback_and_residual", &writeback_and_residual,
          "Scatter solution to grid and compute Linf residual (double)");
}
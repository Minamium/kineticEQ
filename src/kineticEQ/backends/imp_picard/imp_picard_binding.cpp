// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

extern "C" __global__
void picard_kernel_double(
    const double* f, double* fn, const double* v,
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    double* dl, double* dd, double* du, double* B,
    double* n_arr, double* u_arr, double* T_arr,
    double* s0_arr, double* s1_arr, double* s2_arr,
    double* res_dev, int* iters_dev
);

static std::tuple<int,double> imp_picard_step_double(
    at::Tensor f, at::Tensor fn, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol)
{
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(fn.sizes() == f.sizes(), "fn shape mismatch");
    TORCH_CHECK(v.dim()==1, "v must be 1D");

    const int nx = f.size(0);
    const int nv = f.size(1);
    const int n_inner = std::max(nx - 2, 0);

    auto optsD = f.options().dtype(at::kDouble);
    auto optsI = f.options().dtype(at::kInt);

    at::Tensor dl = at::empty({nv, n_inner}, optsD);
    at::Tensor dd = at::empty({nv, n_inner}, optsD);
    at::Tensor du = at::empty({nv, n_inner}, optsD);
    at::Tensor B  = at::empty({nv, n_inner}, optsD);

    at::Tensor n_arr  = at::empty({nx}, optsD);
    at::Tensor u_arr  = at::empty({nx}, optsD);
    at::Tensor T_arr  = at::empty({nx}, optsD);
    at::Tensor s0_arr = at::empty({nx}, optsD);
    at::Tensor s1_arr = at::empty({nx}, optsD);
    at::Tensor s2_arr = at::empty({nx}, optsD);

    at::Tensor res_dev   = at::empty({1}, optsD);
    at::Tensor iters_dev = at::empty({1}, optsI);

    auto stream = at::cuda::getCurrentCUDAStream();

    // 単一ブロックで安全に実行（まずは正しさ優先）
    const int threads = 256;
    const dim3 grid(1), block(threads);
    void* args[] = {
        (void*)f.data_ptr<double>(),  (void*)fn.data_ptr<double>(), (void*)v.data_ptr<double>(),
        (void*)&nx, (void*)&nv,
        (void*)&dv, (void*)&dt, (void*)&dx,
        (void*)&tau_tilde, (void*)&inv_sqrt_2pi,
        (void*)&max_iters, (void*)&tol,
        (void*)dl.data_ptr<double>(), (void*)dd.data_ptr<double>(),
        (void*)du.data_ptr<double>(), (void*)B.data_ptr<double>(),
        (void*)n_arr.data_ptr<double>(), (void*)u_arr.data_ptr<double>(), (void*)T_arr.data_ptr<double>(),
        (void*)s0_arr.data_ptr<double>(), (void*)s1_arr.data_ptr<double>(), (void*)s2_arr.data_ptr<double>(),
        (void*)res_dev.data_ptr<double>(), (void*)iters_dev.data_ptr<int>()
    };

    // 動的共有メモリ: threads * sizeof(double)
    AT_CUDA_CHECK(cudaLaunchKernel(
        (void*)picard_kernel_double, grid, block, args,
        threads * sizeof(double), stream.stream()));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

    const int iters = *iters_dev.cpu().data_ptr<int>();
    const double residual = *res_dev.cpu().data_ptr<double>();
    return {iters, residual};
}

std::tuple<int,double> picard_step_double(
    at::Tensor f, at::Tensor fn, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol)
{
    return imp_picard_step_double(
        f, fn, v, dv, dt, dx, tau_tilde, inv_sqrt_2pi, max_iters, tol);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("picard_step", &picard_step_double,
          "Implicit Picard (single-kernel, single-block) step (float64)");
}
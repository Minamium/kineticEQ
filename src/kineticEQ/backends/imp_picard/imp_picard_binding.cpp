// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) C10_CUDA_CHECK(expr)
#endif

// device kernel (defined in .cu)
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
    TORCH_CHECK(fn.sizes() == f.sizes(), "fn shape must equal f");
    TORCH_CHECK(v.dim()==1, "v must be 1D");

    // enforce contiguous (kernel assumes row-major)
    if (!f.is_contiguous())  f = f.contiguous();
    if (!fn.is_contiguous()) fn = fn.contiguous();
    if (!v.is_contiguous())  v = v.contiguous();

    c10::cuda::CUDAGuard guard(f.get_device());

    const int nx = (int)f.size(0);
    const int nv = (int)f.size(1);
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

    // stream & cooperative capability
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int dev = f.get_device();
    int coop_ok = 0, sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_ok, cudaDevAttrCooperativeLaunch, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    TORCH_CHECK(coop_ok, "Device does not support cooperative launch");

    const int threads = 256;
    int maxPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxPerSm, (const void*)picard_kernel_double, threads, 0));

    int maxCoopBlocks = std::max(1, sms * maxPerSm);
    int blocks = std::min(std::max(1, nv), maxCoopBlocks);

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

    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)picard_kernel_double,
        dim3(blocks), dim3(threads), args, 0, stream);

    if (st == cudaErrorCooperativeLaunchTooLarge) {
        blocks = std::max(1, sms); // even safer
        st = cudaLaunchCooperativeKernel(
            (void*)picard_kernel_double,
            dim3(blocks), dim3(threads), args, 0, stream);
    }
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchCooperativeKernel failed: ")
                                 + cudaGetErrorString(st));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int iters = *iters_dev.cpu().data_ptr<int>();
    const double residual = *res_dev.cpu().data_ptr<double>();
    return {iters, residual};
}

static std::tuple<int,double> picard_step_double(
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
          "Implicit Picard (single cooperative kernel) step (float64)");
}
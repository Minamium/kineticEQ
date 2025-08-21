// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>     // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAException.h>    // C10_CUDA_CHECK
#include <c10/cuda/CUDAGuard.h>        // c10::cuda::CUDAGuard
#include <cuda_runtime.h>

#include <stdexcept>
#include <algorithm>
#include <string>

// CUDA エラーチェック: 環境差異を吸収
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) C10_CUDA_CHECK(expr)
#endif

// デバイス側カーネル（.cu 内に定義）
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

// ランチャ（ホスト側）
static std::tuple<int,double> imp_picard_step_double(
    at::Tensor f, at::Tensor fn, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol)
{
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(f.scalar_type() == at::kDouble, "only float64 supported");
    TORCH_CHECK(fn.sizes() == f.sizes(), "fn shape must equal f");
    TORCH_CHECK(v.dim() == 1, "v must be 1D");

    // すべて同一デバイス上で動作させる
    c10::cuda::CUDAGuard dev_guard(f.get_device());

    const int nx = static_cast<int>(f.size(0));
    const int nv = static_cast<int>(f.size(1));
    const int n_inner = std::max(nx - 2, 0);

    auto optsD = f.options().dtype(at::kDouble);
    auto optsI = f.options().dtype(at::kInt);

    // 作業領域（1 step 内で再利用）
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

    // PyTorch の現在ストリームを取得
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // cooperative launch 上限に合わせた grid 設定
    int dev_idx = f.get_device();
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev_idx));

    const int threads = 256;
    int maxBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, (const void*)picard_kernel_double, threads, /*dynamic_smem=*/0));

    int maxCoopBlocks = std::max(1, sms * maxBlocksPerSm);
    int blocks        = std::min(nv, maxCoopBlocks);
    if (blocks < 1) blocks = 1;

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

    // cooperative kernel 起動（ブロック数が過大なら保険で縮小）
    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)picard_kernel_double,
        dim3(blocks), dim3(threads),
        args, /*sharedMem*/0, stream);

    if (st == cudaErrorCooperativeLaunchTooLarge) {
        blocks = std::max(1, sms); // さらに安全な下限へ
        st = cudaLaunchCooperativeKernel(
            (void*)picard_kernel_double,
            dim3(blocks), dim3(threads),
            args, 0, stream);
    }
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("cudaLaunchCooperativeKernel failed: ")
                                 + cudaGetErrorString(st));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 結果（iters / residual）を取得
    const int iters = *iters_dev.cpu().data_ptr<int>();
    const double residual = *res_dev.cpu().data_ptr<double>();
    return {iters, residual};
}

// バインディング
static std::tuple<int,double> picard_step_double(
    at::Tensor f, at::Tensor fn, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol)
{
    return imp_picard_step_double(
        f, fn, v, dv, dt, dx,
        tau_tilde, inv_sqrt_2pi,
        max_iters, tol
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("picard_step", &picard_step_double,
          "Implicit Picard (single cooperative kernel) step (float64)");
}

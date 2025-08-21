#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <tuple>
#include <sstream>

// kernel 本体（cu 側で定義）
namespace imp_picard {
void launch_picard_double(
    const double* f, double* fn, const double* v,
    int nx, int nv,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol,
    // workspaces (preallocated)
    double* dl, double* dd, double* du, double* B,
    double* n_arr, double* u_arr, double* T_arr,
    double* s0_arr, double* s1_arr, double* s2_arr,
    double* res_dev, int* iters_dev,
    cudaStream_t stream);
}

// ユーティリティ：CUDA エラーを例外化
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << msg << " : " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

// Python から呼ぶエントリ
std::tuple<int,double> picard_step_double(
    at::Tensor f,        // (nx, nv) in
    at::Tensor fn,       // (nx, nv) out
    at::Tensor v,        // (nv)
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int max_iters, double tol)
{
    TORCH_CHECK(f.dtype() == at::kDouble && fn.dtype() == at::kDouble && v.dtype() == at::kDouble,
                "imp_picard: only float64 supported");
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "imp_picard: tensors must be CUDA");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(), "imp_picard: tensors must be contiguous");

    const int nx = (int)f.size(0);
    const int nv = (int)f.size(1);
    TORCH_CHECK((int)v.numel() == nv, "imp_picard: v.size mismatch");

    // cooperative launch サポート確認
    int dev = 0, coop = 0, sms = 0;
    checkCuda(cudaGetDevice(&dev), "cudaGetDevice failed");
    checkCuda(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev), "get coop attr failed");
    TORCH_CHECK(coop, "Device does not support cooperative launch");

    checkCuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "get SM count failed");

    auto opts = f.options();

    // --- workspaces を 1 回分確保（グローバル） ---
    const int n_inner = std::max(0, nx - 2);
    at::Tensor dl = at::empty({nv, n_inner}, opts);
    at::Tensor dd = at::empty({nv, n_inner}, opts);
    at::Tensor du = at::empty({nv, n_inner}, opts);
    at::Tensor  B = at::empty({nv, n_inner}, opts);

    at::Tensor n_arr = at::empty({nx}, opts);
    at::Tensor u_arr = at::empty({nx}, opts);
    at::Tensor T_arr = at::empty({nx}, opts);

    // 一時累積（moments 用）
    at::Tensor s0_arr = at::zeros({nx}, opts);
    at::Tensor s1_arr = at::zeros({nx}, opts);
    at::Tensor s2_arr = at::zeros({nx}, opts);

    // 残差 / 反復回数（デバイス側）
    at::Tensor res_dev   = at::full({1}, 0.0, opts);
    at::Tensor iters_dev = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(f.device()));

    // ストリーム
    auto stream = c10::cuda::getCurrentCUDAStream();

    // カーネル起動（launch_picard_double 内で cooperative 制約に合わせた grid を組む）
    imp_picard::launch_picard_double(
        f.data_ptr<double>(), fn.data_ptr<double>(), v.data_ptr<double>(),
        nx, nv, dv, dt, dx, tau_tilde, inv_sqrt_2pi, max_iters, tol,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
        n_arr.data_ptr<double>(), u_arr.data_ptr<double>(), T_arr.data_ptr<double>(),
        s0_arr.data_ptr<double>(), s1_arr.data_ptr<double>(), s2_arr.data_ptr<double>(),
        res_dev.data_ptr<double>(), iters_dev.data_ptr<int>(),
        stream.stream());

    // 同期して結果を取得
    checkCuda(cudaStreamSynchronize(stream.stream()), "imp_picard kernel sync failed");

    const int    iters = iters_dev.cpu().item<int>();
    const double resid = res_dev.cpu().item<double>();
    return {iters, resid};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("picard_step", &picard_step_double, "Picard (single cooperative kernel, double)");
}

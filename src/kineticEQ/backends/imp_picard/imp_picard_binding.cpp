// imp_picard_binding.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <stdexcept>
#include <sstream>

template <typename T>
__global__ void picard_coop_kernel(
    const T* f_in, T* fA, T* fB, const T* v,
    int nx, int nv, T dv, T dt, T dx, T tau_tilde, T inv_sqrt_2pi,
    int picard_iter, T picard_tol,
    T* s0, T* s1, T* s2, T* n_arr, T* u_arr, T* T_arr,
    int* iters_out, T* resid_out);

static void check_coop_supported() {
    int dev=0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    if (!prop.cooperativeLaunch) {
        std::ostringstream oss;
        oss << "This GPU does not support cooperative launch.";
        throw std::runtime_error(oss.str());
    }
}

std::tuple<int,double> imp_picard_step_double(
    torch::Tensor f_in,     // (nx,nv) contiguous, cuda, float64
    torch::Tensor f_out,    // (nx,nv) contiguous, cuda, float64
    torch::Tensor v,        // (nv)
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int picard_iter, double picard_tol)
{
    TORCH_CHECK(f_in.is_cuda() && f_out.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f_in.dtype() == torch::kFloat64 && f_out.dtype() == torch::kFloat64 && v.dtype() == torch::kFloat64, "dtype must be float64");
    TORCH_CHECK(f_in.is_contiguous() && f_out.is_contiguous() && v.is_contiguous(), "tensors must be contiguous");

    const int nx = f_in.size(0);
    const int nv = f_in.size(1);
    TORCH_CHECK(v.size(0) == nv, "v shape mismatch");

    check_coop_supported();

    // ピンポン用バッファ：最初 fA = f_in（コピー）、fB = f_out を使用
    auto fA = torch::empty_like(f_in);
    fA.copy_(f_in);
    auto fB = f_out; // f_out を直接ピンポン片として使う

    // モーメント・作業配列
    auto s0   = torch::zeros({nx}, f_in.options());
    auto s1   = torch::zeros({nx}, f_in.options());
    auto s2   = torch::zeros({nx}, f_in.options());
    auto n_ar = torch::empty({nx}, f_in.options());
    auto u_ar = torch::empty({nx}, f_in.options());
    auto T_ar = torch::empty({nx}, f_in.options());

    // 出力格納先（デバイス）
    auto iters_d = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(f_in.device()));
    auto resid_d = torch::zeros({1}, f_in.options());

    // カーネル起動設定
    dim3 grid(nv);
    const int block = 256;
    size_t n_inner = (nx >= 2) ? (nx - 2) : 0;
    size_t shmem = sizeof(double) * (n_inner * 4); // dl,dd,du,B

    // cooperative launch: args の並びはテンプレートと一致させる
    void* args[] = {
        (void*)f_in.data_ptr<double>(),
        (void*)fA.data_ptr<double>(),
        (void*)fB.data_ptr<double>(),
        (void*)v.data_ptr<double>(),
        (void*)&nx, (void*)&nv,
        (void*)&dv, (void*)&dt, (void*)&dx,
        (void*)&tau_tilde, (void*)&inv_sqrt_2pi,
        (void*)&picard_iter, (void*)&picard_tol,
        (void*)s0.data_ptr<double>(), (void*)s1.data_ptr<double>(), (void*)s2.data_ptr<double>(),
        (void*)n_ar.data_ptr<double>(), (void*)u_ar.data_ptr<double>(), (void*)T_ar.data_ptr<double>(),
        (void*)iters_d.data_ptr<int>(),
        (void*)resid_d.data_ptr<double>()
    };

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)picard_coop_kernel<double>, grid, dim3(block), args, shmem, stream.stream());
    TORCH_CHECK(st == cudaSuccess, "cudaLaunchCooperativeKernel failed: ", cudaGetErrorString(st));
    st = cudaGetLastError();
    TORCH_CHECK(st == cudaSuccess, "kernel error: ", cudaGetErrorString(st));

    // ホストへ読み出し
    int iters_h = iters_d.cpu().item<int>();
    double resid_h = resid_d.cpu().item<double>();
    return {iters_h, resid_h};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("picard_step", &imp_picard_step_double,
          "Implicit BGK Picard fused step (cooperative, double)");
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// .cu 側のテンプレ宣言（定義は .cu 内）
template <typename T>
void launch_explicit_step(const T* f, T* fn, const T* v,
                          int nx, int nv,
                          T dv, T dt, T dx,
                          T tau_tilde, T inv_sqrt_2pi, int k0,
                          cudaStream_t stream);

// Python から呼ばれるエントリ（FP64固定）
void explicit_step(at::Tensor f,
                   at::Tensor fn,
                   at::Tensor v,
                   double dv, double dt, double dx,
                   double tau_tilde, double inv_sqrt_2pi,
                   int k0)
{
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.dtype() == at::kDouble && fn.dtype() == at::kDouble && v.dtype() == at::kDouble,
                "dtype must be float64");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(f.dim() == 2, "f must be (nx, nv)");
    TORCH_CHECK(fn.sizes() == f.sizes(), "fn must have same shape as f");
    TORCH_CHECK(v.dim() == 1 && v.size(0) == f.size(1), "v must be (nv,)");

    const int nx = static_cast<int>(f.size(0));
    const int nv = static_cast<int>(f.size(1));

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_explicit_step<double>(
        f.data_ptr<double>(),
        fn.data_ptr<double>(),
        v.data_ptr<double>(),
        nx, nv,
        static_cast<double>(dv),
        static_cast<double>(dt),
        static_cast<double>(dx),
        static_cast<double>(tau_tilde),
        static_cast<double>(inv_sqrt_2pi),
        k0,
        stream.stream()
    );

    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("explicit_step", &explicit_step,
          "Fused explicit BGK step (double, CUDA)");
}

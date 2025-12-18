#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// .cu 側のテンプレ宣言（定義は .cu 内）
template <typename T>
void launch_explicit_step_2d2v(
    const T* f, T* fn,
    const T* vx, const T* vy,
    int nx, int ny, int nvx, int nvy,
    T dvx, T dvy,
    T dt, T dx, T dy,
    T tau_tilde,
    int scheme,   // 0: upwind, 1: MUSCL2 (この実装は両方対応)
    int bc_x,     // 0: periodic, 1: neumann(clamp)
    int bc_y,     // 0: periodic, 1: neumann(clamp)
    cudaStream_t stream
);

// Python から呼ばれるエントリ（FP64固定）
void explicit_step(
    at::Tensor f,
    at::Tensor fn,
    at::Tensor vx,
    at::Tensor vy,
    double dvx, double dvy,
    double dt, double dx, double dy,
    double tau_tilde,
    int scheme,
    int bc_x,
    int bc_y
) {
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && vx.is_cuda() && vy.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(f.dtype() == at::kDouble && fn.dtype() == at::kDouble &&
                vx.dtype() == at::kDouble && vy.dtype() == at::kDouble,
                "dtype must be float64 (torch.float64)");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() &&
                vx.is_contiguous() && vy.is_contiguous(),
                "tensors must be contiguous");

    TORCH_CHECK(f.dim() == 4, "f must be (nx, ny, nv_x, nv_y)");
    TORCH_CHECK(fn.sizes() == f.sizes(), "fn must have same shape as f");

    const int nx_  = static_cast<int>(f.size(0));
    const int ny_  = static_cast<int>(f.size(1));
    const int nvx_ = static_cast<int>(f.size(2));
    const int nvy_ = static_cast<int>(f.size(3));

    TORCH_CHECK(vx.dim() == 1 && vx.size(0) == nvx_, "vx must be (nv_x,)");
    TORCH_CHECK(vy.dim() == 1 && vy.size(0) == nvy_, "vy must be (nv_y,)");

    TORCH_CHECK((scheme == 0 || scheme == 1), "scheme must be 0(upwind) or 1(MUSCL2)");
    TORCH_CHECK((bc_x == 0 || bc_x == 1), "bc_x must be 0(periodic) or 1(neumann)");
    TORCH_CHECK((bc_y == 0 || bc_y == 1), "bc_y must be 0(periodic) or 1(neumann)");

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_explicit_step_2d2v<double>(
        f.data_ptr<double>(),
        fn.data_ptr<double>(),
        vx.data_ptr<double>(),
        vy.data_ptr<double>(),
        nx_, ny_, nvx_, nvy_,
        static_cast<double>(dvx),
        static_cast<double>(dvy),
        static_cast<double>(dt),
        static_cast<double>(dx),
        static_cast<double>(dy),
        static_cast<double>(tau_tilde),
        scheme,
        bc_x,
        bc_y,
        stream.stream()
    );

    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("explicit_step", &explicit_step,
          "Fused explicit BGK 2D2V step (double, CUDA)");
}

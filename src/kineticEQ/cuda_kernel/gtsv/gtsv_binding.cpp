// kineticEQ/src/kineticEQ/cuda_kernel/gtsv/gtsv_binding.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>

// ─── Forward declaration ────────────────────────────────────────────────
template <typename scalar_t>
size_t gtsv_strided_ws_size(int n,
                            int batch,
                            const scalar_t* dl,
                            const scalar_t* d,
                            const scalar_t* du,
                            scalar_t* B);

template <typename scalar_t>
void gtsv_strided_impl(int n,
                       int batch,
                       const scalar_t* dl,
                       const scalar_t* d,
                       const scalar_t* du,
                       scalar_t* B,
                       void* workspace);
// ────────────────────────────────────────────────────────────────────────

static void check_core_inputs(
    const torch::Tensor& dl,
    const torch::Tensor& d,
    const torch::Tensor& du,
    const torch::Tensor& B)
{
    TORCH_CHECK(dl.is_cuda() && d.is_cuda() && du.is_cuda() && B.is_cuda(),
                "All tensors must reside on CUDA.");
    TORCH_CHECK(dl.is_contiguous() && d.is_contiguous() &&
                du.is_contiguous() && B.is_contiguous(),
                "All tensors must be contiguous.");
    TORCH_CHECK(dl.scalar_type() == d.scalar_type() &&
                d.scalar_type()  == du.scalar_type() &&
                du.scalar_type() == B.scalar_type(),
                "All tensors must share the same dtype.");
    TORCH_CHECK(d.dim() == 2, "d must be 2D: (batch, n)");
    TORCH_CHECK(dl.sizes() == d.sizes() && du.sizes() == d.sizes() && B.sizes() == d.sizes(),
                "dl, d, du, B must have the same shape: (batch, n)");
}


int64_t gtsv_ws_bytes(torch::Tensor dl,
                      torch::Tensor d,
                      torch::Tensor du,
                      torch::Tensor B)
{
    check_core_inputs(dl, d, du, B);

    const int batch = static_cast<int>(d.size(0));
    const int n     = static_cast<int>(d.size(1));

    size_t ws_bytes = 0;
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(), "gtsv_batch_ws_size", [&]{
        ws_bytes = gtsv_strided_ws_size<scalar_t>(
            n, batch,
            dl.data_ptr<scalar_t>(),
            d .data_ptr<scalar_t>(),
            du.data_ptr<scalar_t>(),
            B .data_ptr<scalar_t>());
    });

    return static_cast<int64_t>(ws_bytes);
}


torch::Tensor gtsv_strided_inplace(torch::Tensor dl,
                                   torch::Tensor d,
                                   torch::Tensor du,
                                   torch::Tensor B,
                                   torch::Tensor workspace)
{
    check_core_inputs(dl, d, du, B);
    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be uint8 tensor");
    TORCH_CHECK(workspace.device() == dl.device(), "workspace must be on same device as system tensors");

    TORCH_CHECK(workspace.numel() > 0,
                "workspace must be preallocated with positive size");

    const int batch = static_cast<int>(d.size(0));
    const int n     = static_cast<int>(d.size(1));

    void* workspace_ptr = workspace.data_ptr();

    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(), "gtsv_batch_solve", [&]{
        gtsv_strided_impl<scalar_t>(
            n, batch,
            dl.data_ptr<scalar_t>(),
            d .data_ptr<scalar_t>(),
            du.data_ptr<scalar_t>(),
            B .data_ptr<scalar_t>(),
            workspace_ptr);
    });

    return B;
}

torch::Tensor gtsv_strided(torch::Tensor dl,
                           torch::Tensor d,
                           torch::Tensor du,
                           torch::Tensor B)
{
    const int64_t ws_bytes = gtsv_ws_bytes(dl, d, du, B);

    // 互換API: ここで確保して in-place solver を呼ぶ
    auto workspace_tensor = at::empty(
        {ws_bytes},
        dl.options().dtype(at::kByte));
    return gtsv_strided_inplace(dl, d, du, B, workspace_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gtsv_ws_bytes",
          &gtsv_ws_bytes,
          "Return required workspace size in bytes for cuSPARSE gtsv2StridedBatch");
    m.def("gtsv_strided_inplace",
          &gtsv_strided_inplace,
          "cuSPARSE batched tridiagonal solver with preallocated workspace (B overwritten with solution)");
    m.def("gtsv_strided",
          &gtsv_strided,
          "cuSPARSE batched tridiagonal solver (StridedBatch, workspace-aware)");
}

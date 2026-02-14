// kineticEQ/src/kineticEQ/cuda_kernel/gtsv/gtsv_binding.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

torch::Tensor gtsv_strided(torch::Tensor dl,
                           torch::Tensor d,
                           torch::Tensor du,
                           torch::Tensor B)
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

    const int batch = d.size(0);
    const int n     = d.size(1);

    // ── ワークスペースサイズを取得 ───────────────────────────────────────
    size_t ws_bytes = 0;

    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(), "gtsv_batch_ws_size", [&]{
        ws_bytes = gtsv_strided_ws_size<scalar_t>(
            n, batch,
            dl.data_ptr<scalar_t>(),
            d .data_ptr<scalar_t>(),
            du.data_ptr<scalar_t>(),
            B .data_ptr<scalar_t>());
    });

    // ── PyTorch のキャッシュアロケータを用いてワークスペース確保 ────────
    auto workspace_tensor = at::empty(
        {static_cast<long long>(ws_bytes)},
        dl.options().dtype(at::kByte));

    void* workspace_ptr = workspace_tensor.data_ptr();

    // ── 解く ────────────────────────────────────────────────────────────
    AT_DISPATCH_FLOATING_TYPES(d.scalar_type(), "gtsv_batch_solve", [&]{
        gtsv_strided_impl<scalar_t>(
            n, batch,
            dl.data_ptr<scalar_t>(),
            d .data_ptr<scalar_t>(),
            du.data_ptr<scalar_t>(),
            B .data_ptr<scalar_t>(),
            workspace_ptr);
    });

    /* B はインプレースで解 X に置き換わっている */
    return B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gtsv_strided",
          &gtsv_strided,
          "cuSPARSE batched tridiagonal solver (StridedBatch, workspace-aware)");
}

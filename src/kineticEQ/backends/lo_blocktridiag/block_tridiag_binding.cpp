// src/kineticEQ/backends/lo_blocktridiag/block_tridiag_binding.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// nvcc 側で定義されるランチャ関数（テンプレート宣言）
template <typename scalar_t>
void launch_block_tridiag_kernel(
    const scalar_t* A,
    scalar_t* B,
    scalar_t* C,
    scalar_t* D,
    scalar_t* X,
    int n,
    int batch,
    cudaStream_t stream
);

torch::Tensor block_tridiag_solve(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA tensor");
    TORCH_CHECK(D.is_cuda(), "D must be CUDA tensor");
    TORCH_CHECK(
        A.scalar_type() == B.scalar_type() &&
        A.scalar_type() == C.scalar_type() &&
        A.scalar_type() == D.scalar_type(),
        "dtype mismatch"
    );

    const bool was_3d = (A.dim() == 3);  // (n,3,3) or (n,3) だったか

    if (was_3d) {
        A = A.unsqueeze(0);
        B = B.unsqueeze(0);
        C = C.unsqueeze(0);
        D = D.unsqueeze(0);
    }

    TORCH_CHECK(A.dim() == 4, "A must be (batch, n, 3, 3)");
    TORCH_CHECK(B.sizes() == A.sizes(), "B shape mismatch");
    TORCH_CHECK(C.sizes() == A.sizes(), "C shape mismatch");
    TORCH_CHECK(D.dim() == 3, "D must be (batch, n, 3)");
    TORCH_CHECK(
        D.size(0) == A.size(0) &&
        D.size(1) == A.size(1) &&
        D.size(2) == 3,
        "D shape mismatch"
    );

    const int64_t batch = A.size(0);
    const int64_t n     = A.size(1);

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C_c = C.contiguous();
    auto D_c = D.contiguous();

    auto options = D_c.options();
    auto X       = torch::empty_like(D_c, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "block_tridiag_solve_cuda", [&] {
        launch_block_tridiag_kernel<scalar_t>(
            A_c.data_ptr<scalar_t>(),
            B_c.data_ptr<scalar_t>(),
            C_c.data_ptr<scalar_t>(),
            D_c.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            static_cast<int>(n),
            static_cast<int>(batch),
            stream
        );
    });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "block_tridiag_kernel launch failed");

    if (was_3d) {
        return X.squeeze(0);  // (n,3) に戻す
    }
    return X;  // (batch,n,3)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_tridiag_solve", &block_tridiag_solve,
          "3x3 block tridiagonal Thomas solver (CUDA)");
}

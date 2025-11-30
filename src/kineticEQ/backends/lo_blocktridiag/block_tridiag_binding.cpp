// src/kineticEQ/backends/lo_blocktridiag/block_tridiag_binding.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// nvcc 側で定義されるランチャ関数（テンプレート宣言）
// ここでは 2 セットのバッファ（A0/B0/C0/D0 と A1/B1/C1/D1）を
// ping-pong 用として渡す設計にしている。
template <typename scalar_t>
void launch_block_tridiag_kernel(
    scalar_t* A0,
    scalar_t* B0,
    scalar_t* C0,
    scalar_t* D0,
    scalar_t* A1,
    scalar_t* B1,
    scalar_t* C1,
    scalar_t* D1,
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

    const bool was_3d = (A.dim() == 3);  // (n,3,3) or (n,3) かどうか

    // ユーザが (n,3,3) / (n,3) を渡してきた場合は内部的に batch=1 とする
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

    // contiguous を取っておく（このテンソルはデバイス側で上書きして良いコピー）
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C_c = C.contiguous();
    auto D_c = D.contiguous();

    // ping-pong 用のワークバッファ
    auto A_buf = torch::empty_like(A_c);
    auto B_buf = torch::empty_like(B_c);
    auto C_buf = torch::empty_like(C_c);
    auto D_buf = torch::empty_like(D_c);

    auto options = D_c.options();
    auto X       = torch::empty_like(D_c, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "block_tridiag_solve_cuda", [&] {
        launch_block_tridiag_kernel<scalar_t>(
            A_c.data_ptr<scalar_t>(),  // A0  (初期係数)
            B_c.data_ptr<scalar_t>(),  // B0
            C_c.data_ptr<scalar_t>(),  // C0
            D_c.data_ptr<scalar_t>(),  // D0
            A_buf.data_ptr<scalar_t>(),// A1  (ワーク)
            B_buf.data_ptr<scalar_t>(),// B1
            C_buf.data_ptr<scalar_t>(),// C1
            D_buf.data_ptr<scalar_t>(),// D1
            X.data_ptr<scalar_t>(),    // 解
            static_cast<int>(n),
            static_cast<int>(batch),
            stream
        );
    });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "block_tridiag_kernel (PCR) launch failed");

    if (was_3d) {
        return X.squeeze(0);  // (n,3) に戻す
    }
    return X;  // (batch,n,3)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_tridiag_solve", &block_tridiag_solve,
          "3x3 block tridiagonal solver (CUDA, PCR-based)");
}

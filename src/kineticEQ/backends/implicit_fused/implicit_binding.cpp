// implicit_binding.cpp  (cuSPARSE GTSV 版)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <tuple>
#include <cmath>

// ===== 既存 implicit_fused の .cu 側ランチャ (①②＋境界) =====
namespace implicit_fused {
void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream);

void launch_build_tridiag_rhs_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    const double* fL, const double* fR,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* d, double* du, double* B,
    cudaStream_t stream);

void launch_boundary_maxwell_double(
    const double* v, int nv, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    double* fL, double* fR,
    cudaStream_t stream);
}

// ===== gtsv_batch.cu のコア関数（実体は .cu、ここでは宣言だけ） =====
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

// ===== 入口関数：implicit_step（Python から呼ばれる） =====
std::tuple<int,double> implicit_step(
    at::Tensor f,        // (nx,nv)
    at::Tensor fn,       // (nx,nv) out
    at::Tensor v,        // (nv,)
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int /*k0*/,          // API整合のため未使用
    int picard_max, double picard_tol,
    double nL, double uL, double TL,
    double nR, double uR, double TR
){
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.dtype()==at::kDouble && fn.dtype()==at::kDouble && v.dtype()==at::kDouble, "float64 only");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(), "must be contiguous");
    TORCH_CHECK(f.dim()==2 && fn.sizes()==f.sizes(), "f,fn shapes");
    TORCH_CHECK(v.dim()==1 && v.size(0)==f.size(1), "v shape");

    const int nx = static_cast<int>(f.size(0));
    const int nv = static_cast<int>(f.size(1));
    TORCH_CHECK(nx>=3, "nx must be >=3");
    const int n_inner = nx - 2;

    auto opts   = f.options();
    auto stream = at::cuda::getCurrentCUDAStream();

    // work buffers
    at::Tensor n  = at::empty({nx}, opts);
    at::Tensor u  = at::empty({nx}, opts);
    at::Tensor T  = at::empty({nx}, opts);
    at::Tensor fL = at::empty({nv}, opts);
    at::Tensor fR = at::empty({nv}, opts);

    // tri-diagonal & RHS : shape (nv, n_inner)  ※各速度で 1 本の線形系
    at::Tensor dl = at::empty({nv, n_inner}, opts);
    at::Tensor d  = at::empty({nv, n_inner}, opts);
    at::Tensor du = at::empty({nv, n_inner}, opts);
    at::Tensor B  = at::empty({nv, n_inner}, opts);

    // まず境界 Maxwell（固定モーメント）を一度だけ作る
    implicit_fused::launch_boundary_maxwell_double(
        v.data_ptr<double>(), nv, inv_sqrt_2pi,
        nL, uL, TL, nR, uR, TR,
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());

    // GTSV ワークスペース（最大サイズで一度だけ確保）
    size_t ws_bytes = gtsv_strided_ws_size<double>(
        n_inner, nv,
        dl.data_ptr<double>(), d.data_ptr<double>(), du.data_ptr<double>(),
        B.data_ptr<double>());
    at::Tensor ws = at::empty({static_cast<long long>(ws_bytes)}, f.options().dtype(at::kByte));
    void* ws_ptr = ws.data_ptr();

    int iters = 0;
    double residual = 0.0;

    for (iters = 0; iters < picard_max; ++iters) {
        // ① moments
        implicit_fused::launch_moments_double(
            f.data_ptr<double>(), v.data_ptr<double>(),
            nx, nv, dv,
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            stream.stream());
        C10_CUDA_CHECK(cudaGetLastError());

        // ② tri-diagonal & RHS 構築（境界寄与込み）
        implicit_fused::launch_build_tridiag_rhs_double(
            f.data_ptr<double>(), v.data_ptr<double>(),
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            fL.data_ptr<double>(), fR.data_ptr<double>(),
            nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
            dl.data_ptr<double>(), d.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
            stream.stream());
        C10_CUDA_CHECK(cudaGetLastError());

        // ③ cuSPARSE GTSV（B がインプレースで解 X に）
        gtsv_strided_impl<double>(
            n_inner, nv,
            dl.data_ptr<double>(), d.data_ptr<double>(), du.data_ptr<double>(),
            B.data_ptr<double>(),
            ws_ptr);
        C10_CUDA_CHECK(cudaGetLastError());

        // 解を fn の内部へ書き戻し、境界は Maxwell 固定
        // interior: fn[1:-1, :].T  ==  shape (nv, n_inner)
        fn.index_put_({at::indexing::Slice(1, -1), at::indexing::Slice()},
                      B.transpose(0,1));
        fn.index_put_({0,  at::indexing::Slice()}, fL);
        fn.index_put_({-1, at::indexing::Slice()}, fR);

        // 残差 = max |fn - f| で十分（Velocity毎の max を取る実装を簡略化）
        residual = at::amax((fn - f).abs()).item<double>();
        if (residual < picard_tol) break;

        // 次の反復へ
        f.copy_(fn);
    }

    return {iters + 1, residual};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_step", &implicit_step,
          "Implicit BGK step (cuSPARSE GTSV, internal Picard, double)");
}

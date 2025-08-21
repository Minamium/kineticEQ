#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <tuple>
#include <vector>
#include <cmath>
#include <cassert>

namespace implicit_fused {
// ---- kernels (defined in implicit_kernels.cu) ----
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

void launch_boundary_from_current_f_double(
    const double* f, int nx, int nv,
    double* fL, double* fR,
    cudaStream_t stream);

void launch_residual_interior_double(
    const double* f_old, const double* f_new,
    int nx, int nv, double* res_per_v,
    cudaStream_t stream);

// ---- helpers ----
static inline void cudaCheck(cudaError_t e, const char* msg){
    TORCH_CHECK(e == cudaSuccess, msg, " : ", cudaGetErrorString(e));
}
static inline void cusparseCheck(cusparseStatus_t s, const char* msg){
    TORCH_CHECK(s == CUSPARSE_STATUS_SUCCESS, "cuSPARSE error: ", (int)s, " : ", msg);
}

// Build & solve one implicit step (Picard) entirely on GPU (boundaries untouched)
std::pair<int,double> implicit_step(
    torch::Tensor f,      // (nx, nv) double, cuda, contiguous
    torch::Tensor fn,     // (nx, nv) double, cuda, contiguous (output)
    torch::Tensor v,      // (nv)     double, cuda, contiguous
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int /*k0_unused*/,    // for API compatibility only
    int picard_iter, double picard_tol
){
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.dtype() == torch::kFloat64 && fn.dtype() == torch::kFloat64 && v.dtype() == torch::kFloat64,
                "dtype must be float64");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(),
                "tensors must be contiguous");

    const int nx = f.size(0);
    const int nv = f.size(1);
    TORCH_CHECK(v.size(0) == nv, "v.size(0) must equal nv");
    TORCH_CHECK(nx >= 3, "nx must be >= 3 for interior update");

    const int m = nx - 2;         // system size per velocity
    const int batch = nv;         // number of tri-diagonal systems
    auto stream = at::cuda::getCurrentCUDAStream();

    // Work buffers
    auto opts1 = torch::TensorOptions().dtype(torch::kFloat64).device(f.device());
    auto n  = torch::empty({nx}, opts1);
    auto u  = torch::empty({nx}, opts1);
    auto Tg = torch::empty({nx}, opts1);

    // boundary values from current f (do not construct Maxwell here)
    auto fL = torch::empty({nv}, opts1);
    auto fR = torch::empty({nv}, opts1);

    // tri-diagonal bands and RHS for all velocities: (nv, m) row-major => stride = m
    auto dl = torch::empty({nv, m}, opts1);
    auto dd = torch::empty({nv, m}, opts1);
    auto du = torch::empty({nv, m}, opts1);
    auto  B = torch::empty({nv, m}, opts1);

    // per-velocity residual (interior only)
    auto res_v = torch::empty({nv}, opts1);
    double residual = 0.0;
    int iters = 0;

    // constant device pointers
    const double* f_ptr  = f.data_ptr<double>();
    const double* v_ptr  = v.data_ptr<double>();

    // kernels use stream
    // Picard iteration
    for (int z = 0; z < picard_iter; ++z) {
        // 1) moments from current f
        launch_moments_double(
            f_ptr, v_ptr, nx, nv, dv,
            n.data_ptr<double>(), u.data_ptr<double>(), Tg.data_ptr<double>(),
            stream.stream());

        // 2) boundary vectors from current f (not Maxwell)
        launch_boundary_from_current_f_double(
            f_ptr, nx, nv, fL.data_ptr<double>(), fR.data_ptr<double>(),
            stream.stream());

        // 3) build tri-diagonal and RHS for interior cells only
        launch_build_tridiag_rhs_double(
            f_ptr, v_ptr,
            n.data_ptr<double>(), u.data_ptr<double>(), Tg.data_ptr<double>(),
            fL.data_ptr<double>(), fR.data_ptr<double>(),
            nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
            dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
            stream.stream());

        // 4) cuSPARSE batched gtsv2 (in-place on B)
        {
            cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();

            size_t ws_bytes = 0;
            cusparseCheck(
                cusparseDgtsv2StridedBatch_bufferSizeExt(
                    handle, m,
                    dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(),
                    B.data_ptr<double>(), batch, /*ldb=strideB*/ m, &ws_bytes),
                "bufferSizeExt");

            // at least 1 byte
            if (ws_bytes == 0) ws_bytes = 4;
            auto workspace = torch::empty({static_cast<long long>(ws_bytes)}, torch::TensorOptions()
                                          .dtype(torch::kUInt8).device(f.device()));

            cusparseCheck(
                cusparseDgtsv2StridedBatch(
                    handle, m,
                    dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(),
                    B.data_ptr<double>(), batch, /*ldb=strideB*/ m,
                    workspace.data_ptr<uint8_t>()),
                "gtsv2StridedBatch");
        }

        // 5) write back interior only to fn (boundaries untouched)
        {
            // copy old f to fn first, then overwrite interior with solution
            fn.copy_(f);
            // interior view
            auto fn_in = fn.index({torch::indexing::Slice(1, nx-1), torch::indexing::Slice()});
            // B is (nv,m) row-major => need transpose to (m,nv) to match (i,j)
            fn_in.copy_(B.transpose(0,1));
        }

        // 6) residual on interior only, and swap f <- fn for next Picard iter
        launch_residual_interior_double(
            f.data_ptr<double>(), fn.data_ptr<double>(),
            nx, nv, res_v.data_ptr<double>(),
            stream.stream());
        // reduce on CPU (one scalar per v)
        residual = res_v.max().item<double>();
        iters = z + 1;

        // next iterate: f <- fn
        f.copy_(fn);

        if (residual < picard_tol) break;
    }

    // 手元のポリシー通り、このバックエンドは境界を触らない。
    // Python 側で最後に  f_new[0,:]=f[0,:], f_new[-1,:]=f[-1,:] を実行してください。

    return {iters, residual};
}

} // namespace implicit_fused

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_step", &implicit_fused::implicit_step, "Implicit BGK step (cuSPARSE GTSV2, boundaries untouched)");
}

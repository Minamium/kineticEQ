#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace {

// ---- cuSPARSE エラーチェック ----
static inline const char* cusparseGetErrorString_(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        default: return "CUSPARSE_STATUS_UNKNOWN_ERROR";
    }
}
#define CUSPARSE_CHECK(expr)                                                     \
    do {                                                                         \
        cusparseStatus_t _st = (expr);                                           \
        if (_st != CUSPARSE_STATUS_SUCCESS) {                                    \
            std::ostringstream _oss;                                             \
            _oss << "cuSPARSE error: " << (int)_st << " ("                       \
                 << cusparseGetErrorString_(_st) << ") at "                      \
                 << __FILE__ << ":" << __LINE__;                                 \
            throw std::runtime_error(_oss.str());                                \
        }                                                                        \
    } while (0)

// ---- CUDA エラーチェック（念のため）----
#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _ce = (expr);                                                \
        if (_ce != cudaSuccess) {                                                \
            std::ostringstream _oss;                                             \
            _oss << "CUDA error: " << (int)_ce << " ("                           \
                 << cudaGetErrorString(_ce) << ") at "                           \
                 << __FILE__ << ":" << __LINE__;                                 \
            throw std::runtime_error(_oss.str());                                \
        }                                                                        \
    } while (0)

} // namespace

// ====== kernels (implicit_kernels.cu) のランチャ宣言 ======
namespace implicit_fused {
void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream);

void launch_boundary_maxwell_double(
    const double* v, int nv, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    double* fL, double* fR,
    cudaStream_t stream);

void launch_build_tridiag_rhs_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    const double* fL, const double* fR,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* d, double* du, double* B,
    cudaStream_t stream);

double launch_writeback_and_residual_double(
    const double* fz, double* fn_tmp, const double* B,
    int nx, int nv,
    cudaStream_t stream);
} // namespace implicit_fused

// ====== Python から呼ぶラッパ ======
static void moments_binding(
    torch::Tensor f,  // (nx, nv) float64, cuda
    torch::Tensor v,  // (nv)     float64, cuda
    double dv,
    torch::Tensor n,  // (nx)     float64, cuda
    torch::Tensor u,  // (nx)     float64, cuda
    torch::Tensor T   // (nx)     float64, cuda
) {
    TORCH_CHECK(f.is_cuda() && v.is_cuda() && n.is_cuda() && u.is_cuda() && T.is_cuda(),
                "moments: all tensors must be CUDA");
    TORCH_CHECK(f.scalar_type() == torch::kFloat64 &&
                v.scalar_type() == torch::kFloat64 &&
                n.scalar_type() == torch::kFloat64 &&
                u.scalar_type() == torch::kFloat64 &&
                T.scalar_type() == torch::kFloat64,
                "moments: dtype must be float64");
    TORCH_CHECK(f.is_contiguous() && v.is_contiguous() &&
                n.is_contiguous() && u.is_contiguous() && T.is_contiguous(),
                "moments: tensors must be contiguous");

    int64_t nx = f.size(0);
    int64_t nv = f.size(1);
    TORCH_CHECK(v.numel() == nv, "moments: v length must equal nv");

    auto stream = at::cuda::getCurrentCUDAStream();

    implicit_fused::launch_moments_double(
        f.data_ptr<double>(), v.data_ptr<double>(),
        static_cast<int>(nx), static_cast<int>(nv), dv,
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        stream.stream());
}

static void boundary_maxwell_binding(
    torch::Tensor v,  // (nv)
    double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    torch::Tensor fL, // (nv)
    torch::Tensor fR  // (nv)
) {
    TORCH_CHECK(v.is_cuda() && fL.is_cuda() && fR.is_cuda(), "boundary_maxwell: CUDA tensors required");
    TORCH_CHECK(v.scalar_type() == torch::kFloat64 &&
                fL.scalar_type() == torch::kFloat64 &&
                fR.scalar_type() == torch::kFloat64,
                "boundary_maxwell: dtype must be float64");
    TORCH_CHECK(v.is_contiguous() && fL.is_contiguous() && fR.is_contiguous(),
                "boundary_maxwell: tensors must be contiguous");

    int64_t nv = v.size(0);
    TORCH_CHECK(fL.numel() == nv && fR.numel() == nv, "boundary_maxwell: size mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    implicit_fused::launch_boundary_maxwell_double(
        v.data_ptr<double>(), static_cast<int>(nv), inv_sqrt_2pi,
        nL, uL, TL, nR, uR, TR,
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        stream.stream());
}

static void build_system_binding(
    torch::Tensor fz, // (nx, nv)
    torch::Tensor v,  // (nv)
    torch::Tensor n,  // (nx)
    torch::Tensor u,  // (nx)
    torch::Tensor T,  // (nx)
    double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    torch::Tensor fL, // (nv)
    torch::Tensor fR, // (nv)
    torch::Tensor dl, // (nv, n_inner)
    torch::Tensor dd, // (nv, n_inner)
    torch::Tensor du, // (nv, n_inner)
    torch::Tensor B   // (nv, n_inner)
) {
    TORCH_CHECK(fz.is_cuda() && v.is_cuda() && n.is_cuda() && u.is_cuda() && T.is_cuda() &&
                fL.is_cuda() && fR.is_cuda() && dl.is_cuda() && dd.is_cuda() &&
                du.is_cuda() && B.is_cuda(),
                "build_system: all tensors must be CUDA");
    TORCH_CHECK(fz.scalar_type() == torch::kFloat64 && v.scalar_type() == torch::kFloat64 &&
                n.scalar_type()  == torch::kFloat64 && u.scalar_type() == torch::kFloat64 &&
                T.scalar_type()  == torch::kFloat64 && fL.scalar_type() == torch::kFloat64 &&
                fR.scalar_type() == torch::kFloat64 && dl.scalar_type()== torch::kFloat64 &&
                dd.scalar_type() == torch::kFloat64 && du.scalar_type()== torch::kFloat64 &&
                B.scalar_type()  == torch::kFloat64,
                "build_system: dtype must be float64");
    TORCH_CHECK(fz.is_contiguous() && v.is_contiguous() && n.is_contiguous() &&
                u.is_contiguous() && T.is_contiguous() && fL.is_contiguous() &&
                fR.is_contiguous() && dl.is_contiguous() && dd.is_contiguous() &&
                du.is_contiguous() && B.is_contiguous(),
                "build_system: tensors must be contiguous");

    int64_t nx = fz.size(0);
    int64_t nv = fz.size(1);
    int64_t n_inner = nx - 2;
    TORCH_CHECK(n_inner >= 1, "build_system: nx must be >= 3");

    TORCH_CHECK(v.numel() == nv, "build_system: v length must equal nv");
    TORCH_CHECK(n.numel() == nx && u.numel() == nx && T.numel() == nx,
                "build_system: n,u,T must have length nx");
    TORCH_CHECK(dl.size(0) == nv && dd.size(0) == nv && du.size(0) == nv && B.size(0) == nv,
                "build_system: (dl,dd,du,B) first dim must be nv");
    TORCH_CHECK(dl.size(1) == n_inner && dd.size(1) == n_inner &&
                du.size(1) == n_inner && B.size(1) == n_inner,
                "build_system: second dim must be nx-2");

    auto stream = at::cuda::getCurrentCUDAStream();
    implicit_fused::launch_build_tridiag_rhs_double(
        fz.data_ptr<double>(), v.data_ptr<double>(),
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        static_cast<int>(nx), static_cast<int>(nv),
        dt, dx, tau_tilde, inv_sqrt_2pi,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
        stream.stream());
}

static void gtsv_solve_inplace_binding(
    torch::Tensor dl, // (nv, n_inner)
    torch::Tensor dd, // (nv, n_inner)
    torch::Tensor du, // (nv, n_inner)
    torch::Tensor B,  // (nv, n_inner) -> 解で上書き
    int nx,           // フル格子点数
    int nv            // 速度格子点数
) {
    TORCH_CHECK(dl.is_cuda() && dd.is_cuda() && du.is_cuda() && B.is_cuda(),
                "gtsv_solve_inplace: tensors must be CUDA");
    TORCH_CHECK(dl.scalar_type() == torch::kFloat64 &&
                dd.scalar_type() == torch::kFloat64 &&
                du.scalar_type() == torch::kFloat64 &&
                B.scalar_type()  == torch::kFloat64,
                "gtsv_solve_inplace: dtype must be float64");
    TORCH_CHECK(dl.is_contiguous() && dd.is_contiguous() &&
                du.is_contiguous() && B.is_contiguous(),
                "gtsv_solve_inplace: tensors must be contiguous");

    const int m = nx - 2;     // 内部セル数
    const int batch = nv;     // バッチ数（= 速度数）
    TORCH_CHECK(m >= 1, "gtsv_solve_inplace: nx must be >= 3");

    TORCH_CHECK(dl.size(0) == batch && dd.size(0) == batch &&
                du.size(0) == batch && B.size(0)  == batch,
                "gtsv_solve_inplace: first dimension must be nv");
    TORCH_CHECK(dl.size(1) == m && dd.size(1) == m &&
                du.size(1) == m && B.size(1)  == m,
                "gtsv_solve_inplace: second dimension must be nx-2");

    // cuSPARSE ハンドルとストリーム
    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    auto stream = at::cuda::getCurrentCUDAStream();
    CUSPARSE_CHECK(cusparseSetStream(handle, stream.stream()));

    const int64_t stride = static_cast<int64_t>(m);  // 各バッチ間ストライド
    const int ldb = m;                                // 右辺の leading dimension

    // ワークスペースサイズ
#if CUDART_VERSION >= 10000
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt(
        handle, m,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(),
        B.data_ptr<double>(), ldb, batch, &bufferSize));
    // Torch の CUDA キャッシュアロケータで確保
    auto workspace = torch::empty(
        {static_cast<long>(bufferSize)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    void* pBuffer = workspace.data_ptr<uint8_t>();
    // 実行（B が解で上書き）
    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch(
        handle, m,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(),
        B.data_ptr<double>(), ldb, batch, pBuffer));
#else
#   error "This source requires CUDA 10.0+ (cuSPARSE gtsv2 API)."
#endif

    CUSPARSE_CHECK(cusparseDestroy(handle));
}

static double writeback_and_residual_binding(
    torch::Tensor fz,     // (nx, nv)  旧候補
    torch::Tensor fn_tmp, // (nx, nv)  新候補(出力先)
    torch::Tensor B,      // (nv, n_inner)  解
    int nx,
    int nv
) {
    TORCH_CHECK(fz.is_cuda() && fn_tmp.is_cuda() && B.is_cuda(),
                "writeback_and_residual: CUDA tensors required");
    TORCH_CHECK(fz.scalar_type() == torch::kFloat64 &&
                fn_tmp.scalar_type() == torch::kFloat64 &&
                B.scalar_type() == torch::kFloat64,
                "writeback_and_residual: dtype must be float64");
    TORCH_CHECK(fz.is_contiguous() && fn_tmp.is_contiguous() && B.is_contiguous(),
                "writeback_and_residual: tensors must be contiguous");

    const int m = nx - 2;
    TORCH_CHECK(m >= 1, "writeback_and_residual: nx must be >= 3");
    TORCH_CHECK(fz.size(0) == nx && fz.size(1) == nv, "writeback_and_residual: fz shape mismatch");
    TORCH_CHECK(fn_tmp.size(0) == nx && fn_tmp.size(1) == nv, "writeback_and_residual: fn_tmp shape mismatch");
    TORCH_CHECK(B.size(0) == nv && B.size(1) == m, "writeback_and_residual: B shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    double res = implicit_fused::launch_writeback_and_residual_double(
        fz.data_ptr<double>(), fn_tmp.data_ptr<double>(), B.data_ptr<double>(),
        nx, nv, stream.stream());
    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moments", &moments_binding, "Compute (n,u,T) from f");
    m.def("boundary_maxwell", &boundary_maxwell_binding, "Compute Maxwellians at boundaries");
    m.def("build_system", &build_system_binding, "Build tri-diagonal (dl,dd,du) and RHS B");
    m.def("gtsv_solve_inplace", &gtsv_solve_inplace_binding, "Solve batched tridiagonal in-place with cuSPARSE");
    m.def("writeback_and_residual", &writeback_and_residual_binding, "Write back interior cells and return L∞ residual");
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

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
        double* dl, double* dd, double* du, double* B,
        cudaStream_t stream);
}

static inline void check_cuda(const char* msg){
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess){
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(e));
    }
}

static void require_cuda_fp64(const at::Tensor& t, const char* name){
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.scalar_type() == at::kDouble, name, " must be float64 (Double)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void require_shape(const at::Tensor& t, int dim0, int dim1, const char* name){
    TORCH_CHECK(t.dim()==2 && t.size(0)==dim0 && t.size(1)==dim1,
        name, " must have shape (", dim0, ", ", dim1, "), got (", t.size(0), ", ", t.size(1), ")");
}

// f: (nx,nv), v:(nv)
// dl,dd,du,B: (nv, nx-2)  ※右辺Bは解法の右辺（gtsv_stridedに渡す形）
// 戻り値なし（出力テンソルに書き込み）
void build_system_fused(
    at::Tensor f, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    at::Tensor dl, at::Tensor dd, at::Tensor du, at::Tensor B)
{
    require_cuda_fp64(f, "f");
    require_cuda_fp64(v, "v");
    require_cuda_fp64(dl, "dl");
    require_cuda_fp64(dd, "dd");
    require_cuda_fp64(du, "du");
    require_cuda_fp64(B,  "B");

    TORCH_CHECK(v.dim()==1, "v must be 1-D (nv,)");

    const int64_t nx = f.size(0);
    const int64_t nv_ = f.size(1);
    TORCH_CHECK(v.size(0) == nv_, "v and f nv mismatch");
    const int64_t n_inner = std::max<int64_t>(nx - 2, 0);

    if (n_inner > 0){
        require_shape(dl, nv_, n_inner, "dl");
        require_shape(dd, nv_, n_inner, "dd");
        require_shape(du, nv_, n_inner, "du");
        require_shape(B,  nv_, n_inner, "B");
    } else {
        TORCH_CHECK(dl.numel()==0 && dd.numel()==0 && du.numel()==0 && B.numel()==0,
            "nx<3 の場合、dl/dd/du/B は空テンソルである必要があります");
        return;
    }

    auto opts1d = f.options();
    at::Tensor n = at::empty({nx}, opts1d);
    at::Tensor u = at::empty({nx}, opts1d);
    at::Tensor T = at::empty({nx}, opts1d);
    at::Tensor fL = at::empty({nv_}, opts1d);
    at::Tensor fR = at::empty({nv_}, opts1d);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // (1) moments
    implicit_fused::launch_moments_double(
        f.data_ptr<double>(), v.data_ptr<double>(),
        (int)nx, (int)nv_, dv,
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        stream);
    check_cuda("moments kernel failed");

    // (2) boundary Maxwell
    implicit_fused::launch_boundary_maxwell_double(
        v.data_ptr<double>(), (int)nv_, inv_sqrt_2pi,
        nL, uL, TL, nR, uR, TR,
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        stream);
    check_cuda("boundary maxwell kernel failed");

    // (3) tri-diagonal & RHS
    implicit_fused::launch_build_tridiag_rhs_double(
        f.data_ptr<double>(), v.data_ptr<double>(),
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        (int)nx, (int)nv_, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
        stream);
    check_cuda("build tridiag RHS kernel failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_system_fused", &build_system_fused,
          "Build batched tri-diagonal system (dl,dd,du,B) from f and v (double/CUDA)");
}

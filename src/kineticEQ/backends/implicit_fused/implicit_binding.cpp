// implicit_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <sstream>

using torch::Tensor;

#define CUDA_CHECK(err) do { \
  cudaError_t e = (err); \
  if (e != cudaSuccess) { \
    std::ostringstream oss; \
    oss << "CUDA error: " << cudaGetErrorString(e) \
        << " at " << __FILE__ << ":" << __LINE__; \
    throw std::runtime_error(oss.str()); \
  } \
} while(0)

#define CUSPARSE_CHECK(stat) do { \
  cusparseStatus_t s = (stat); \
  if (s != CUSPARSE_STATUS_SUCCESS) { \
    std::ostringstream oss; \
    oss << "cuSPARSE error: " << int(s) \
        << " at " << __FILE__ << ":" << __LINE__; \
    throw std::runtime_error(oss.str()); \
  } \
} while(0)

namespace implicit_fused {
// from .cu
void launch_moments_double(const double*, const double*, int, int, double, double*, double*, double*, cudaStream_t);
void launch_boundary_maxwell_double(const double*, int, double, double,double,double, double,double,double, double*, double*, cudaStream_t);
void launch_build_tridiag_rhs_double(const double*, const double*, const double*, const double*, const double*, const double*, const double*, int, int, double, double, double, double, double*, double*, double*, double*, cudaStream_t);
void launch_writeback_and_residual_double(const double*, double*, const double*, int, int, double*, cudaStream_t);
void launch_reduce_max_double(const double*, int, double*, cudaStream_t);
}

// ========== guards ==========
static inline void check_cuda_double_contig(const Tensor& t, const char* name) {
    if (!t.is_cuda())  throw std::invalid_argument(std::string(name) + " must be CUDA tensor");
    if (t.scalar_type() != torch::kFloat64) throw std::invalid_argument(std::string(name) + " must be float64");
    if (!t.is_contiguous()) throw std::invalid_argument(std::string(name) + " must be contiguous");
}

// ========== API: moments ==========
void moments(Tensor f, Tensor v, double dv, Tensor n, Tensor u, Tensor T) {
    check_cuda_double_contig(f, "f");
    check_cuda_double_contig(v, "v");
    check_cuda_double_contig(n, "n");
    check_cuda_double_contig(u, "u");
    check_cuda_double_contig(T, "T");

    const int64_t nx = f.size(0);
    const int64_t nv = f.size(1);
    TORCH_CHECK(v.numel() == nv, "v length mismatch");
    TORCH_CHECK(n.numel() == nx && u.numel() == nx && T.numel() == nx, "n,u,T size mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    implicit_fused::launch_moments_double(
        f.data_ptr<double>(), v.data_ptr<double>(),
        (int)nx, (int)nv, dv,
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        stream.stream());
    CUDA_CHECK(cudaGetLastError());
}

// ========== API: boundary Maxwell ==========
void boundary_maxwell(Tensor v, double inv_sqrt_2pi,
                      double nL, double uL, double TL,
                      double nR, double uR, double TR,
                      Tensor fL, Tensor fR)
{
    check_cuda_double_contig(v, "v");
    check_cuda_double_contig(fL, "fL");
    check_cuda_double_contig(fR, "fR");
    const int64_t nv = v.numel();
    TORCH_CHECK(fL.numel() == nv && fR.numel() == nv, "fL/fR length mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    implicit_fused::launch_boundary_maxwell_double(
        v.data_ptr<double>(), (int)nv, inv_sqrt_2pi,
        nL, uL, TL, nR, uR, TR,
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        stream.stream());
    CUDA_CHECK(cudaGetLastError());
}

// ========== API: build tri-diagonal + RHS ==========
void build_system(Tensor fz, Tensor v, Tensor n, Tensor u, Tensor T,
                  double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
                  Tensor fL, Tensor fR,
                  Tensor dl, Tensor dd, Tensor du, Tensor B)
{
    check_cuda_double_contig(fz, "fz");
    check_cuda_double_contig(v,  "v");
    check_cuda_double_contig(n,  "n");
    check_cuda_double_contig(u,  "u");
    check_cuda_double_contig(T,  "T");
    check_cuda_double_contig(fL, "fL");
    check_cuda_double_contig(fR, "fR");
    check_cuda_double_contig(dl, "dl");
    check_cuda_double_contig(dd, "dd");
    check_cuda_double_contig(du, "du");
    check_cuda_double_contig(B,  "B");

    const int64_t nx = fz.size(0);
    const int64_t nv = fz.size(1);
    TORCH_CHECK(v.numel() == nv, "v length mismatch");
    TORCH_CHECK(n.numel() == nx && u.numel() == nx && T.numel() == nx, "n,u,T size mismatch");

    const int64_t n_inner = nx - 2;
    TORCH_CHECK(n_inner >= 0, "nx must be >= 2");
    if (n_inner == 0) return;

    TORCH_CHECK(dl.size(0) == nv && dl.size(1) == n_inner, "dl size mismatch");
    TORCH_CHECK(dd.size(0) == nv && dd.size(1) == n_inner, "dd size mismatch");
    TORCH_CHECK(du.size(0) == nv && du.size(1) == n_inner, "du size mismatch");
    TORCH_CHECK(B.size(0)  == nv && B.size(1)  == n_inner, "B size mismatch");
    TORCH_CHECK(fL.numel() == nv && fR.numel() == nv, "fL/fR length mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    implicit_fused::launch_build_tridiag_rhs_double(
        fz.data_ptr<double>(), v.data_ptr<double>(),
        n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        (int)nx, (int)nv, dt, dx, tau_tilde, inv_sqrt_2pi,
        dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
        stream.stream());
    CUDA_CHECK(cudaGetLastError());
}

// ========== API: batched gtsv (cuSPARSE) ==========
void gtsv_solve_inplace(Tensor dl, Tensor dd, Tensor du, Tensor B, int nx, int nv)
{
    check_cuda_double_contig(dl, "dl");
    check_cuda_double_contig(dd, "dd");
    check_cuda_double_contig(du, "du");
    check_cuda_double_contig(B,  "B");

    const int64_t n_inner = nx - 2;
    if (n_inner <= 0) return;

    TORCH_CHECK(dl.size(0)==nv && dl.size(1)==n_inner, "dl size mismatch");
    TORCH_CHECK(dd.size(0)==nv && dd.size(1)==n_inner, "dd size mismatch");
    TORCH_CHECK(du.size(0)==nv && du.size(1)==n_inner, "du size mismatch");
    TORCH_CHECK(B.size(0)==nv  && B.size(1)==n_inner,  "B size mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();

    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream.stream()));

    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt(
        handle,
        (int)n_inner,
        dl.data_ptr<double>(),
        dd.data_ptr<double>(),
        du.data_ptr<double>(),
        B.data_ptr<double>(),
        (int)n_inner,   // batch stride
        (int)nv,
        &bufferSize
    ));

    // workspace
    Tensor workspace = torch::empty({(long long)bufferSize}, torch::dtype(torch::kUInt8).device(B.device()));
    void* pBuffer = workspace.data_ptr<uint8_t>();

    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch(
        handle,
        (int)n_inner,
        dl.data_ptr<double>(),
        dd.data_ptr<double>(),
        du.data_ptr<double>(),
        B.data_ptr<double>(),
        (int)n_inner,   // batch stride
        (int)nv,
        pBuffer
    ));

    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaGetLastError());
}

// ========== API: writeback + residual (interior only) ==========
double writeback_and_residual(Tensor fz, Tensor fn_tmp, Tensor B, int nx, int nv)
{
    check_cuda_double_contig(fz,     "fz");
    check_cuda_double_contig(fn_tmp, "fn_tmp");
    check_cuda_double_contig(B,      "B");

    const int64_t n_inner = nx - 2;
    if (n_inner <= 0) return 0.0;

    TORCH_CHECK(fz.size(0)==nx && fz.size(1)==nv, "fz size mismatch");
    TORCH_CHECK(fn_tmp.size(0)==nx && fn_tmp.size(1)==nv, "fn_tmp size mismatch");
    TORCH_CHECK(B.size(0)==nv && B.size(1)==n_inner, "B size mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();

    // per-velocity maxima
    Tensor res_per_v = torch::empty({nv}, torch::dtype(torch::kFloat64).device(fz.device()));
    implicit_fused::launch_writeback_and_residual_double(
        fz.data_ptr<double>(), fn_tmp.data_ptr<double>(),
        B.data_ptr<double>(), nx, nv,
        res_per_v.data_ptr<double>(),
        stream.stream());

    // reduce to scalar on device, then copy to host
    Tensor d_max = torch::empty({1}, torch::dtype(torch::kFloat64).device(fz.device()));
    implicit_fused::launch_reduce_max_double(
        res_per_v.data_ptr<double>(), nv, d_max.data_ptr<double>(), stream.stream());

    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

    double host_res = d_max.item<double>();
    return host_res;
}

// ========== pybind ==========
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moments", &moments, "Compute (n,u,T) from f");
    m.def("boundary_maxwell", &boundary_maxwell, "Compute Maxwell at boundaries (fL,fR)");
    m.def("build_system", &build_system, "Build tri-diagonal (dl,dd,du) and RHS (B) for interior cells");
    m.def("gtsv_solve_inplace", &gtsv_solve_inplace, "Solve tri-diagonal systems in-place via cuSPARSE gtsv2StridedBatch");
    m.def("writeback_and_residual", &writeback_and_residual, "Write interior solution and return global residual");
}

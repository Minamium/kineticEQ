// kineticEQ/src/kineticEQ/cuda_kernel/implicit_fused/implicit_binding.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace implicit_fused {

// ================= device launchers (cu) ================
void launch_moments_double(
    const double* fz, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream);

void launch_moments_n_nu_T_double(
    const double* fz, const double* v,
    int nx, int nv, double dv,
    double* n, double* nu, double* T,
    cudaStream_t stream);

void launch_build_system_fused_double(
    const double* f_prev, const double* fz, const double* v,
    const double* n, const double* u, const double* T,
    int nx, int nv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream);

void launch_build_system_from_moments_double(
    const double* B0, const double* v,
    const double* n, const double* nu, const double* T,
    const double* f_bc_l, const double* f_bc_r,
    int nx, int nv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream);

} // namespace implicit_fused


// =================== helpers =====================
static void check_inputs(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), std::string(name) + " must be CUDA tensor");
  TORCH_CHECK(t.dtype() == torch::kFloat64, std::string(name) + " must be float64");
  TORCH_CHECK(t.is_contiguous(), std::string(name) + " must be contiguous");
}


void moments_n_nu_T(
    torch::Tensor fz,       // (nx, nv)
    torch::Tensor v,        // (nv,)
    double dv,
    torch::Tensor n,        // (nx,)
    torch::Tensor nu,       // (nx,)
    torch::Tensor T         // (nx,)
) {
  check_inputs(fz, "fz");
  check_inputs(v, "v");
  check_inputs(n, "n");
  check_inputs(nu, "nu");
  check_inputs(T, "T");

  TORCH_CHECK(fz.dim() == 2, "fz must be (nx,nv)");
  TORCH_CHECK(v.dim() == 1, "v must be (nv,)");
  TORCH_CHECK(n.dim() == 1 && nu.dim() == 1 && T.dim() == 1,
              "n,nu,T must be 1D tensors");

  const int64_t nx = fz.size(0);
  const int64_t nv = fz.size(1);

  TORCH_CHECK(v.size(0) == nv, "v size must match fz.nv");
  TORCH_CHECK(n.size(0) == nx && nu.size(0) == nx && T.size(0) == nx,
              "n,nu,T sizes must match fz.nx");

  auto stream = at::cuda::getCurrentCUDAStream();
  implicit_fused::launch_moments_n_nu_T_double(
      fz.data_ptr<double>(), v.data_ptr<double>(),
      (int)nx, (int)nv, dv,
      n.data_ptr<double>(), nu.data_ptr<double>(), T.data_ptr<double>(),
      stream.stream());
}


void build_system_from_moments(
    torch::Tensor B0,       // (nv, nx-2)
    torch::Tensor v,        // (nv,)
    double dt,
    double dx,
    double tau_tilde,
    double inv_sqrt_2pi,
    torch::Tensor n,        // (nx,)
    torch::Tensor nu,       // (nx,)
    torch::Tensor T,        // (nx,)
    torch::Tensor f_bc_l,   // (nv,)
    torch::Tensor f_bc_r,   // (nv,)
    torch::Tensor dl,       // (nv, nx-2)
    torch::Tensor dd,       // (nv, nx-2)
    torch::Tensor du,       // (nv, nx-2)
    torch::Tensor B         // (nv, nx-2)
) {
  check_inputs(B0, "B0");
  check_inputs(v, "v");
  check_inputs(n, "n");
  check_inputs(nu, "nu");
  check_inputs(T, "T");
  check_inputs(f_bc_l, "f_bc_l");
  check_inputs(f_bc_r, "f_bc_r");
  check_inputs(dl, "dl");
  check_inputs(dd, "dd");
  check_inputs(du, "du");
  check_inputs(B, "B");

  TORCH_CHECK(v.dim() == 1, "v must be (nv,)");
  TORCH_CHECK(n.dim() == 1 && nu.dim() == 1 && T.dim() == 1,
              "n,nu,T must be 1D tensors");
  TORCH_CHECK(f_bc_l.dim() == 1 && f_bc_r.dim() == 1,
              "f_bc_l and f_bc_r must be 1D tensors");

  const int64_t nx = n.size(0);
  TORCH_CHECK(nx >= 3, "n must have nx>=3");
  TORCH_CHECK(nu.size(0) == nx && T.size(0) == nx, "nu and T size must match n");

  const int64_t nv = v.size(0);
  const int64_t n_inner = nx - 2;

  TORCH_CHECK(f_bc_l.size(0) == nv && f_bc_r.size(0) == nv,
              "f_bc_l/f_bc_r size must match nv");

  TORCH_CHECK(B0.dim() == 2 && B0.size(0) == nv && B0.size(1) == n_inner,
              "B0 must be (nv, nx-2)");
  TORCH_CHECK(dl.dim() == 2 && dl.size(0) == nv && dl.size(1) == n_inner,
              "dl must be (nv, nx-2)");
  TORCH_CHECK(dd.dim() == 2 && dd.size(0) == nv && dd.size(1) == n_inner,
              "dd must be (nv, nx-2)");
  TORCH_CHECK(du.dim() == 2 && du.size(0) == nv && du.size(1) == n_inner,
              "du must be (nv, nx-2)");
  TORCH_CHECK(B.dim() == 2 && B.size(0) == nv && B.size(1) == n_inner,
              "B must be (nv, nx-2)");

  if (n_inner <= 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  implicit_fused::launch_build_system_from_moments_double(
      B0.data_ptr<double>(), v.data_ptr<double>(),
      n.data_ptr<double>(), nu.data_ptr<double>(), T.data_ptr<double>(),
      f_bc_l.data_ptr<double>(), f_bc_r.data_ptr<double>(),
      (int)nx, (int)nv, dt, dx, tau_tilde, inv_sqrt_2pi,
      dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
      stream.stream());
}


// =================== API: build_system_fused =====================
// 参照式に完全準拠：
//  - モーメント (n,u,T) と f_M, τ は Picard 候補 fz から計算
//  - 右辺の f は “前ステップ” f_prev を使用（固定）
//  - 境界フラックスは fz の i=0, i=nx-1 のモーメントから得た f_M を使用
//  - 内部未知数は i=1..nx-2 → a,b,c,B の shape は (nv, nx-2)
void build_system_fused(
    torch::Tensor f_prev,   // (nx, nv)
    torch::Tensor fz,       // (nx, nv)
    torch::Tensor v,        // (nv,)
    double dv,
    double dt,
    double dx,
    double tau_tilde,
    double inv_sqrt_2pi,
    torch::Tensor dl,       // (nv, nx-2)  subdiag  (= -alpha) except k=0 -> 0
    torch::Tensor dd,       // (nv, nx-2)  diag     (= 1+alpha+beta+dt/tau)
    torch::Tensor du,       // (nv, nx-2)  superdiag(= -beta ) except k=last -> 0
    torch::Tensor B         // (nv, nx-2)  RHS (= f_prev + dt/tau * fM + boundary)
) {
  check_inputs(f_prev, "f_prev");
  check_inputs(fz,     "fz");
  check_inputs(v,      "v");
  check_inputs(dl,     "dl");
  check_inputs(dd,     "dd");
  check_inputs(du,     "du");
  check_inputs(B,      "B");

  TORCH_CHECK(f_prev.sizes() == fz.sizes(), "f_prev and fz must have same shape (nx,nv)");
  TORCH_CHECK(f_prev.dim()==2 && f_prev.size(0)>=3,
              "f_prev must be (nx,nv) with nx>=3 because interior unknowns are i=1..nx-2");
  TORCH_CHECK(v.dim()==1 && v.size(0)==f_prev.size(1), "v must be (nv,) and match f_prev.nv");

  const int64_t nx = f_prev.size(0);
  const int64_t nv = f_prev.size(1);
  const int64_t n_inner = nx - 2;

  TORCH_CHECK(dl.size(0)==nv && dl.size(1)==n_inner, "dl must be (nv, nx-2)");
  TORCH_CHECK(dd.size(0)==nv && dd.size(1)==n_inner, "dd must be (nv, nx-2)");
  TORCH_CHECK(du.size(0)==nv && du.size(1)==n_inner, "du must be (nv, nx-2)");
  TORCH_CHECK(B .size(0)==nv && B .size(1)==n_inner, "B  must be (nv, nx-2)");

  if (n_inner <= 0) {
    // 何もすることがない
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  // 一時ワーク: (n,u,T) from fz
  auto opts_1d = torch::TensorOptions().dtype(torch::kFloat64).device(fz.device());
  torch::Tensor n = torch::empty({nx}, opts_1d);
  torch::Tensor u = torch::empty({nx}, opts_1d);
  torch::Tensor T = torch::empty({nx}, opts_1d);

  // 1) moments from fz (必ず dv を重みとして使う)
  implicit_fused::launch_moments_double(
      fz.data_ptr<double>(), v.data_ptr<double>(),
      (int)nx, (int)nv, dv,
      n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
      stream.stream());

  // 2) tri-diagonal と RHS を構築（RHS の f は f_prev で固定）
  implicit_fused::launch_build_system_fused_double(
      f_prev.data_ptr<double>(), fz.data_ptr<double>(), v.data_ptr<double>(),
      n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
      (int)nx, (int)nv, dt, dx, tau_tilde, inv_sqrt_2pi,
      dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
      stream.stream());
}


// =================== pybind =====================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moments_n_nu_T", &moments_n_nu_T,
        "Compute BGK moments (n,nu,T) into preallocated buffers");
  m.def("build_system_from_moments", &build_system_from_moments,
        "Build (dl,dd,du,B) from precomputed moments and fixed boundary f");
  m.def("build_system_fused", &build_system_fused,
        "Build (dl,dd,du,B) for implicit BGK with upwind flux (RHS uses f_prev; boundary Maxwell from fz)");
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>
#include <tuple>
#include <stdexcept>

// ===================== Kernel launchers (defined in .cu) =====================
namespace implicit_fused {

// Moments: compute n,u,T for each x-row
void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream);

// Build tri-diagonal (dl,d,du) and RHS (B) per velocity, with boundary contributions
void launch_build_tridiag_rhs_double(
    const double* f, const double* v,
    const double* n, const double* u, const double* T,
    const double* fL, const double* fR,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* d, double* du, double* B,
    cudaStream_t stream);

// Batched TDMA solve in-place for each velocity j (1 block per j).
// Then write back fn (including boundaries), and compute per-velocity residual maxima.
void launch_tdma_solve_and_writeback_double(
    const double* f, double* fn,
    const double* fL, const double* fR,
    const double* dl, const double* d, const double* du, double* B,
    int nx, int nv,
    double* res_per_v,  // length nv
    cudaStream_t stream);

// Compute Maxwellians at left/right boundaries for all velocities
void launch_boundary_maxwell_double(
    const double* v, int nv, double inv_sqrt_2pi,
    double nL, double uL, double TL,
    double nR, double uR, double TR,
    double* fL, double* fR,
    cudaStream_t stream);

} // namespace implicit_fused

// ===================== Entry point: implicit_step =====================
std::tuple<int,double> implicit_step(
    at::Tensor f,        // (nx,nv) in, updated each Picard iteration
    at::Tensor fn,       // (nx,nv) out buffer (will hold final result)
    at::Tensor v,        // (nv,)
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int k0,              // not used, kept for API parity
    int picard_max, double picard_tol,
    double nL, double uL, double TL,
    double nR, double uR, double TR
) {
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "All tensors must be CUDA.");
    TORCH_CHECK(f.dtype() == at::kDouble && fn.dtype() == at::kDouble && v.dtype() == at::kDouble,
                "dtype must be float64.");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(),
                "All tensors must be contiguous.");
    TORCH_CHECK(f.dim()==2, "f must be (nx,nv).");
    TORCH_CHECK(fn.sizes()==f.sizes(), "fn must have same shape as f.");
    TORCH_CHECK(v.dim()==1 && v.size(0)==f.size(1), "v must be (nv,).");

    const int64_t nx64 = f.size(0);
    const int64_t nv64 = f.size(1);
    TORCH_CHECK(nx64 >= 3, "nx must be >=3 (need interior cells).");
    const int nx = static_cast<int>(nx64);
    const int nv = static_cast<int>(nv64);
    const int n_inner = nx - 2;

    auto opts = f.options();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Work buffers (persist across Picard iters)
    at::Tensor n   = at::empty({nx}, opts);
    at::Tensor u   = at::empty({nx}, opts);
    at::Tensor T   = at::empty({nx}, opts);
    at::Tensor fL  = at::empty({nv}, opts);
    at::Tensor fR  = at::empty({nv}, opts);

    // Tri-diagonal & RHS storage (nv systems, each length n_inner)
    at::Tensor dl  = at::empty({nv, n_inner}, opts);
    at::Tensor d   = at::empty({nv, n_inner}, opts);
    at::Tensor du  = at::empty({nv, n_inner}, opts);
    at::Tensor B   = at::empty({nv, n_inner}, opts);
    at::Tensor res_per_v = at::empty({nv}, opts);

    // Precompute boundary Maxwellians once since boundaries are fixed-moment
    implicit_fused::launch_boundary_maxwell_double(
        v.data_ptr<double>(), nv, inv_sqrt_2pi,
        nL, uL, TL, nR, uR, TR,
        fL.data_ptr<double>(), fR.data_ptr<double>(),
        stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());

    int iters = 0;
    double residual = 0.0;

    for (iters = 0; iters < picard_max; ++iters) {
        // 1) moments n,u,T from current f
        implicit_fused::launch_moments_double(
            f.data_ptr<double>(), v.data_ptr<double>(),
            nx, nv, dv,
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            stream.stream());
        C10_CUDA_CHECK(cudaGetLastError());

        // 2) build tri-diagonal and RHS
        implicit_fused::launch_build_tridiag_rhs_double(
            f.data_ptr<double>(), v.data_ptr<double>(),
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            fL.data_ptr<double>(), fR.data_ptr<double>(),
            nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
            dl.data_ptr<double>(), d.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
            stream.stream());
        C10_CUDA_CHECK(cudaGetLastError());

        // 3) solve & writeback fn (also compute per-velocity residual maxima)
        implicit_fused::launch_tdma_solve_and_writeback_double(
            f.data_ptr<double>(), fn.data_ptr<double>(),
            fL.data_ptr<double>(), fR.data_ptr<double>(),
            dl.data_ptr<double>(), d.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
            nx, nv,
            res_per_v.data_ptr<double>(),
            stream.stream());
        C10_CUDA_CHECK(cudaGetLastError());

        // Reduce residual on GPU → scalar
        residual = at::amax(res_per_v).item<double>();

        if (residual < picard_tol) {
            break;
        }

        // Copy fn → f for next Picard iteration (keeps python API simple)
        f.copy_(fn);
    }

    return {iters+1, residual};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_step", &implicit_step,
          "Fused implicit BGK step with internal Picard (double, CUDA)");
}

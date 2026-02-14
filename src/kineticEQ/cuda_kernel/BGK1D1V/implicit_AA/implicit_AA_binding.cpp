// kineticEQ/src/kineticEQ/cuda_kernel/BGK1D1V/implicit_AA/implicit_AA_binding.cpp
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;

std::tuple<int64_t, int64_t> implicit_aa_step_inplace_cuda(
    torch::Tensor n,
    torch::Tensor nu,
    torch::Tensor T,
    const torch::Tensor n_new,
    const torch::Tensor nu_new,
    const torch::Tensor T_new,
    torch::Tensor aa_G,     // (d, aa_cols)
    torch::Tensor aa_R,     // (d, aa_cols)
    torch::Tensor aa_A,     // (aa_cols, aa_cols)
    torch::Tensor aa_alpha, // (aa_cols)
    torch::Tensor aa_wk,    // (d)
    torch::Tensor aa_wnew,  // (d)
    torch::Tensor aa_wtmp,  // (d)  (work/output)
    int64_t hist_len,
    int64_t head,
    bool apply,
    double beta,
    double reg,
    double alpha_max,
    double n_floor,
    double T_floor,
    double u_max,
    torch::Tensor solver_work, // (>=lwork) dtype=float/double, CUDA. empty ok.
    torch::Tensor solver_info, // int32 scalar CUDA. empty ok.
    torch::Tensor G_work,      // (d, aa_cols) CUDA, same dtype, empty ok.
    torch::Tensor R_work       // (d, aa_cols) CUDA, same dtype, empty ok.
);

int64_t implicit_aa_potrf_lwork_cuda(torch::Tensor aa_A, int64_t n); // returns work elements

static void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.defined(), name, " is not defined");
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

static void check_dtype(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(
        t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kFloat64,
        name, " must be float32 or float64"
    );
}

std::tuple<int64_t, int64_t> implicit_aa_step_inplace(
    torch::Tensor n,
    torch::Tensor nu,
    torch::Tensor T,
    torch::Tensor n_new,
    torch::Tensor nu_new,
    torch::Tensor T_new,
    torch::Tensor aa_G,
    torch::Tensor aa_R,
    torch::Tensor aa_A,
    torch::Tensor aa_alpha,
    torch::Tensor aa_wk,
    torch::Tensor aa_wnew,
    torch::Tensor aa_wtmp,
    int64_t hist_len,
    int64_t head,
    bool apply,
    double beta,
    double reg,
    double alpha_max,
    double n_floor,
    double T_floor,
    double u_max,
    torch::Tensor solver_work,
    torch::Tensor solver_info,
    torch::Tensor G_work,
    torch::Tensor R_work
) {
    check_cuda_tensor(n, "n");
    check_cuda_tensor(nu, "nu");
    check_cuda_tensor(T, "T");
    check_cuda_tensor(n_new, "n_new");
    check_cuda_tensor(nu_new, "nu_new");
    check_cuda_tensor(T_new, "T_new");
    check_cuda_tensor(aa_G, "aa_G");
    check_cuda_tensor(aa_R, "aa_R");
    check_cuda_tensor(aa_A, "aa_A");
    check_cuda_tensor(aa_alpha, "aa_alpha");
    check_cuda_tensor(aa_wk, "aa_wk");
    check_cuda_tensor(aa_wnew, "aa_wnew");
    check_cuda_tensor(aa_wtmp, "aa_wtmp");

    check_dtype(n, "n");
    TORCH_CHECK(n.scalar_type() == nu.scalar_type() && n.scalar_type() == T.scalar_type(), "n/nu/T dtype mismatch");
    TORCH_CHECK(n.scalar_type() == n_new.scalar_type() && n.scalar_type() == nu_new.scalar_type() && n.scalar_type() == T_new.scalar_type(),
                "n_new/nu_new/T_new dtype mismatch");
    TORCH_CHECK(n.scalar_type() == aa_G.scalar_type() && n.scalar_type() == aa_R.scalar_type(), "AA history dtype mismatch");
    TORCH_CHECK(n.scalar_type() == aa_A.scalar_type() && n.scalar_type() == aa_alpha.scalar_type(), "AA A/alpha dtype mismatch");
    TORCH_CHECK(n.scalar_type() == aa_wk.scalar_type() && n.scalar_type() == aa_wnew.scalar_type() && n.scalar_type() == aa_wtmp.scalar_type(),
                "AA wk/wnew/wtmp dtype mismatch");

    TORCH_CHECK(n.is_contiguous(), "n must be contiguous");
    TORCH_CHECK(nu.is_contiguous(), "nu must be contiguous");
    TORCH_CHECK(T.is_contiguous(), "T must be contiguous");
    TORCH_CHECK(n_new.is_contiguous(), "n_new must be contiguous");
    TORCH_CHECK(nu_new.is_contiguous(), "nu_new must be contiguous");
    TORCH_CHECK(T_new.is_contiguous(), "T_new must be contiguous");
    TORCH_CHECK(aa_G.is_contiguous(), "aa_G must be contiguous");
    TORCH_CHECK(aa_R.is_contiguous(), "aa_R must be contiguous");
    TORCH_CHECK(aa_A.is_contiguous(), "aa_A must be contiguous");
    TORCH_CHECK(aa_alpha.is_contiguous(), "aa_alpha must be contiguous");
    TORCH_CHECK(aa_wk.is_contiguous(), "aa_wk must be contiguous");
    TORCH_CHECK(aa_wnew.is_contiguous(), "aa_wnew must be contiguous");
    TORCH_CHECK(aa_wtmp.is_contiguous(), "aa_wtmp must be contiguous");

    return implicit_aa_step_inplace_cuda(
        n, nu, T,
        n_new, nu_new, T_new,
        aa_G, aa_R, aa_A, aa_alpha,
        aa_wk, aa_wnew, aa_wtmp,
        hist_len, head,
        apply, beta, reg, alpha_max,
        n_floor, T_floor, u_max,
        solver_work, solver_info,
        G_work, R_work
    );
}

int64_t implicit_aa_potrf_lwork(torch::Tensor aa_A, int64_t n) {
    check_cuda_tensor(aa_A, "aa_A");
    check_dtype(aa_A, "aa_A");
    TORCH_CHECK(aa_A.is_contiguous(), "aa_A must be contiguous");
    TORCH_CHECK(aa_A.dim() == 2 && aa_A.size(0) == aa_A.size(1), "aa_A must be square");
    TORCH_CHECK(n >= 1 && n <= aa_A.size(0), "invalid n for aa_A");
    return implicit_aa_potrf_lwork_cuda(aa_A, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "step_inplace",
        &implicit_aa_step_inplace,
        "Implicit Anderson Acceleration step (in-place, ring-buffer history)",
        py::arg("n"),
        py::arg("nu"),
        py::arg("T"),
        py::arg("n_new"),
        py::arg("nu_new"),
        py::arg("T_new"),
        py::arg("aa_G"),
        py::arg("aa_R"),
        py::arg("aa_A"),
        py::arg("aa_alpha"),
        py::arg("aa_wk"),
        py::arg("aa_wnew"),
        py::arg("aa_wtmp"),
        py::arg("hist_len"),
        py::arg("head"),
        py::arg("apply"),
        py::arg("beta"),
        py::arg("reg"),
        py::arg("alpha_max"),
        py::arg("n_floor"),
        py::arg("T_floor"),
        py::arg("u_max"),
        py::arg("solver_work"),
        py::arg("solver_info"),
        py::arg("G_work"),
        py::arg("R_work")
    );

    m.def(
        "potrf_lwork",
        &implicit_aa_potrf_lwork,
        "cuSOLVER potrf required work elements for given n (dtype/device from aa_A)",
        py::arg("aa_A"),
        py::arg("n")
    );
}

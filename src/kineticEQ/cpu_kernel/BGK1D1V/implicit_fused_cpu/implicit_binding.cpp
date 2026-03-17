#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr double kTiny = 1.0e-300;

inline void check_inputs(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(!t.is_cuda(), std::string(name) + " must be CPU tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat64, std::string(name) + " must be float64");
    TORCH_CHECK(t.is_contiguous(), std::string(name) + " must be contiguous");
}

inline double clamp_pos(double x) {
    return x > kTiny ? x : kTiny;
}

inline double maxwell_1v(double n, double u, double Tgas, double vj, double inv_sqrt_2pi) {
    const double Tg = clamp_pos(Tgas);
    const double diff = vj - u;
    return n * inv_sqrt_2pi / std::sqrt(Tg) * std::exp(-0.5 * diff * diff / Tg);
}

}  // namespace

void moments_n_nu_T(
    torch::Tensor fz,
    torch::Tensor v,
    double dv,
    torch::Tensor n,
    torch::Tensor nu,
    torch::Tensor T
) {
    check_inputs(fz, "fz");
    check_inputs(v, "v");
    check_inputs(n, "n");
    check_inputs(nu, "nu");
    check_inputs(T, "T");

    TORCH_CHECK(fz.dim() == 2, "fz must be (nx, nv)");
    TORCH_CHECK(v.dim() == 1, "v must be (nv,)");

    const auto nx = fz.size(0);
    const auto nv = fz.size(1);

    TORCH_CHECK(v.size(0) == nv, "v size must match fz.nv");
    TORCH_CHECK(n.size(0) == nx && nu.size(0) == nx && T.size(0) == nx,
                "n/nu/T size must match fz.nx");

    const double* fz_ptr = fz.data_ptr<double>();
    const double* v_ptr = v.data_ptr<double>();
    double* n_ptr = n.data_ptr<double>();
    double* nu_ptr = nu.data_ptr<double>();
    double* T_ptr = T.data_ptr<double>();

    at::parallel_for(0, nx, 0, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const double* row = fz_ptr + i * nv;
            double p0 = 0.0;
            double p1 = 0.0;
            double p2 = 0.0;

            for (int64_t j = 0; j < nv; ++j) {
                const double fij = row[j];
                const double vj = v_ptr[j];
                p0 += fij;
                p1 += fij * vj;
                p2 += fij * vj * vj;
            }

            const double n_raw = p0 * dv;
            const double nu_raw = p1 * dv;
            const double s2_raw = p2 * dv;
            const double n_val = clamp_pos(n_raw);
            const double u_val = nu_raw / n_val;
            double T_val = s2_raw / n_val - u_val * u_val;
            if (!(T_val > 0.0)) {
                T_val = kTiny;
            }

            n_ptr[i] = n_val;
            nu_ptr[i] = nu_raw;
            T_ptr[i] = T_val;
        }
    });
}

void build_system_from_moments(
    torch::Tensor B0,
    torch::Tensor v,
    double dt,
    double dx,
    double tau_tilde,
    double inv_sqrt_2pi,
    torch::Tensor n,
    torch::Tensor nu,
    torch::Tensor T,
    torch::Tensor f_bc_l,
    torch::Tensor f_bc_r,
    torch::Tensor dl,
    torch::Tensor dd,
    torch::Tensor du,
    torch::Tensor B
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
    TORCH_CHECK(n.dim() == 1 && nu.dim() == 1 && T.dim() == 1, "n/nu/T must be 1D");

    const auto nx = n.size(0);
    const auto nv = v.size(0);
    const auto n_inner = nx - 2;
    TORCH_CHECK(nx >= 3, "n must have nx >= 3");

    TORCH_CHECK(B0.dim() == 2 && B0.size(0) == nv && B0.size(1) == n_inner, "B0 must be (nv, nx-2)");
    TORCH_CHECK(dl.dim() == 2 && dl.sizes() == B0.sizes(), "dl must match B0 shape");
    TORCH_CHECK(dd.dim() == 2 && dd.sizes() == B0.sizes(), "dd must match B0 shape");
    TORCH_CHECK(du.dim() == 2 && du.sizes() == B0.sizes(), "du must match B0 shape");
    TORCH_CHECK(B.dim() == 2 && B.sizes() == B0.sizes(), "B must match B0 shape");

    if (n_inner <= 0) {
        return;
    }

    const double* B0_ptr = B0.data_ptr<double>();
    const double* v_ptr = v.data_ptr<double>();
    const double* n_ptr = n.data_ptr<double>();
    const double* nu_ptr = nu.data_ptr<double>();
    const double* T_ptr = T.data_ptr<double>();
    const double* f_bc_l_ptr = f_bc_l.data_ptr<double>();
    const double* f_bc_r_ptr = f_bc_r.data_ptr<double>();
    double* dl_ptr = dl.data_ptr<double>();
    double* dd_ptr = dd.data_ptr<double>();
    double* du_ptr = du.data_ptr<double>();
    double* B_ptr = B.data_ptr<double>();

    at::parallel_for(0, nv, 0, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
            const double vj = v_ptr[j];
            const double alpha = vj > 0.0 ? (dt / dx) * vj : 0.0;
            const double beta = vj < 0.0 ? (dt / dx) * (-vj) : 0.0;

            const int64_t offset = j * n_inner;
            const double* B0_j = B0_ptr + offset;
            double* dl_j = dl_ptr + offset;
            double* dd_j = dd_ptr + offset;
            double* du_j = du_ptr + offset;
            double* B_j = B_ptr + offset;

            for (int64_t k = 0; k < n_inner; ++k) {
                const int64_t i = k + 1;
                const double n_i = clamp_pos(n_ptr[i]);
                const double T_i = clamp_pos(T_ptr[i]);
                const double u_i = nu_ptr[i] / n_i;
                const double inv_tau = (n_i * std::sqrt(T_i)) / tau_tilde;
                const double fM_ij = maxwell_1v(n_i, u_i, T_i, vj, inv_sqrt_2pi);

                dd_j[k] = 1.0 + alpha + beta + dt * inv_tau;
                dl_j[k] = k == 0 ? 0.0 : -alpha;
                du_j[k] = k == n_inner - 1 ? 0.0 : -beta;

                double rhs = B0_j[k] + dt * inv_tau * fM_ij;
                if (k == 0) {
                    rhs += alpha * f_bc_l_ptr[j];
                }
                if (k == n_inner - 1) {
                    rhs += beta * f_bc_r_ptr[j];
                }
                B_j[k] = rhs;
            }
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moments_n_nu_T", &moments_n_nu_T, "Compute BGK moments on CPU into preallocated buffers");
    m.def("build_system_from_moments", &build_system_from_moments, "Build implicit BGK tridiagonal systems on CPU");
}

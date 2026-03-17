#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr double kPivotFloor = 1.0e-300;

inline void check_core_inputs(
    const torch::Tensor& dl,
    const torch::Tensor& d,
    const torch::Tensor& du,
    const torch::Tensor& B
) {
    TORCH_CHECK(!dl.is_cuda() && !d.is_cuda() && !du.is_cuda() && !B.is_cuda(),
                "All tensors must reside on CPU.");
    TORCH_CHECK(dl.is_contiguous() && d.is_contiguous() && du.is_contiguous() && B.is_contiguous(),
                "All tensors must be contiguous.");
    TORCH_CHECK(dl.scalar_type() == torch::kFloat64 &&
                d.scalar_type() == torch::kFloat64 &&
                du.scalar_type() == torch::kFloat64 &&
                B.scalar_type() == torch::kFloat64,
                "All tensors must be float64.");
    TORCH_CHECK(dl.sizes() == d.sizes() && du.sizes() == d.sizes() && B.sizes() == d.sizes(),
                "dl, d, du, B must share shape (batch, n).");
    TORCH_CHECK(d.dim() == 2, "d must be 2D: (batch, n)");
}

inline double safe_pivot(double x) {
    if (std::abs(x) >= kPivotFloor) {
        return x;
    }
    return x < 0.0 ? -kPivotFloor : kPivotFloor;
}

}  // namespace

int64_t gtsv_ws_bytes(
    torch::Tensor dl,
    torch::Tensor d,
    torch::Tensor du,
    torch::Tensor B
) {
    check_core_inputs(dl, d, du, B);
    return 1;
}

torch::Tensor gtsv_strided_inplace(
    torch::Tensor dl,
    torch::Tensor d,
    torch::Tensor du,
    torch::Tensor B,
    torch::Tensor workspace
) {
    check_core_inputs(dl, d, du, B);
    TORCH_CHECK(!workspace.is_cuda(), "workspace must be CPU tensor");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8, "workspace must be uint8 tensor");
    TORCH_CHECK(workspace.numel() > 0, "workspace must have positive size");

    const auto batch = d.size(0);
    const auto n = d.size(1);
    if (batch == 0 || n == 0) {
        return B;
    }

    double* dl_ptr = dl.data_ptr<double>();
    double* d_ptr = d.data_ptr<double>();
    double* du_ptr = du.data_ptr<double>();
    double* B_ptr = B.data_ptr<double>();

    at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
            double* dl_row = dl_ptr + b * n;
            double* d_row = d_ptr + b * n;
            double* du_row = du_ptr + b * n;
            double* B_row = B_ptr + b * n;

            for (int64_t i = 1; i < n; ++i) {
                const double pivot = safe_pivot(d_row[i - 1]);
                const double w = dl_row[i] / pivot;
                d_row[i] -= w * du_row[i - 1];
                B_row[i] -= w * B_row[i - 1];
            }

            B_row[n - 1] /= safe_pivot(d_row[n - 1]);
            for (int64_t i = n - 1; i-- > 0;) {
                B_row[i] = (B_row[i] - du_row[i] * B_row[i + 1]) / safe_pivot(d_row[i]);
            }
        }
    });

    return B;
}

torch::Tensor gtsv_strided(
    torch::Tensor dl,
    torch::Tensor d,
    torch::Tensor du,
    torch::Tensor B
) {
    auto workspace = torch::empty({1}, dl.options().dtype(torch::kUInt8));
    return gtsv_strided_inplace(dl, d, du, B, workspace);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gtsv_ws_bytes", &gtsv_ws_bytes, "Return dummy workspace size for CPU gtsv");
    m.def("gtsv_strided_inplace", &gtsv_strided_inplace, "Solve batched tridiagonal systems on CPU in-place");
    m.def("gtsv_strided", &gtsv_strided, "Solve batched tridiagonal systems on CPU");
}

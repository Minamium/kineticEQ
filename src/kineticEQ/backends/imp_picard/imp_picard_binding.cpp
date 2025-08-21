// imp_picard_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// ---- device kernels (from .cu) ----
extern "C" __global__
void moments_kernel_double(const double*, const double*, int, int, double,
                           double*, double*, double*);

extern "C" __global__
void build_system_kernel_double(const double*, const double*, const double*,
                                const double*, const double*, int, int,
                                double, double, double, double, double,
                                double*, double*, double*, double*);

extern "C" __global__
void scatter_and_residual_kernel_double(const double*, const double*,
                                        double*, int, int, double*);

// ---- Host launchers ----
static void build_system_and_moments(
    at::Tensor f, at::Tensor fz, at::Tensor v,
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    at::Tensor dl, at::Tensor dd, at::Tensor du, at::Tensor B,
    at::Tensor n_arr, at::Tensor u_arr, at::Tensor T_arr)
{
    TORCH_CHECK(f.is_cuda() && fz.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(fz.scalar_type() == at::kDouble, "float64 only");
    TORCH_CHECK(v.scalar_type() == at::kDouble, "float64 only");

    const int64_t nx = f.size(0);
    const int64_t nv = f.size(1);
    TORCH_CHECK(fz.sizes() == f.sizes(), "fz shape must equal f");

    const int64_t n_inner = std::max<int64_t>(nx - 2, 0);

    // shape checks using temporary vector (workaround for IntArrayRef literal)
    {
        std::vector<int64_t> expect2{nv, n_inner};
        if (n_inner > 0){
            TORCH_CHECK(dl.sizes() == at::IntArrayRef(expect2), "dl shape (nv, nx-2)");
            TORCH_CHECK(dd.sizes() == at::IntArrayRef(expect2), "dd shape (nv, nx-2)");
            TORCH_CHECK(du.sizes() == at::IntArrayRef(expect2), "du shape (nv, nx-2)");
            TORCH_CHECK(B .sizes() == at::IntArrayRef(expect2), "B  shape (nv, nx-2)");
        }
        std::vector<int64_t> expect1{nx};
        TORCH_CHECK(n_arr.sizes() == at::IntArrayRef(expect1), "n_arr shape (nx)");
        TORCH_CHECK(u_arr.sizes() == at::IntArrayRef(expect1), "u_arr shape (nx)");
        TORCH_CHECK(T_arr.sizes() == at::IntArrayRef(expect1), "T_arr shape (nx)");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // 1) moments
    const int tMom = (nv >= 256) ? 256 : (nv >= 128 ? 128 : 64);
    const size_t shm = sizeof(double) * tMom * 3;
    void* args_mom[] = {
        (void*)fz.data_ptr<double>(),
        (void*)v.data_ptr<double>(),
        (void*)&nx, (void*)&nv, (void*)&dv,
        (void*)n_arr.data_ptr<double>(),
        (void*)u_arr.data_ptr<double>(),
        (void*)T_arr.data_ptr<double>()
    };
    dim3 grid_m(nx);
    dim3 block_m(tMom);
    AT_CUDA_CHECK(cudaLaunchKernel(
        (void*)moments_kernel_double, grid_m, block_m, args_mom, shm, stream.stream()));

    if (n_inner == 0) return;

    // 2) build (coalesced writes)
    const int t = 256;
    const int total = (int)(nv * n_inner);
    dim3 grid_b((total + t - 1) / t);
    void* args_build[] = {
        (void*)fz.data_ptr<double>(),
        (void*)v.data_ptr<double>(),
        (void*)n_arr.data_ptr<double>(),
        (void*)u_arr.data_ptr<double>(),
        (void*)T_arr.data_ptr<double>(),
        (void*)&nx, (void*)&nv,
        (void*)&dv, (void*)&dt, (void*)&dx,
        (void*)&tau_tilde, (void*)&inv_sqrt_2pi,
        (void*)dl.data_ptr<double>(),
        (void*)dd.data_ptr<double>(),
        (void*)du.data_ptr<double>(),
        (void*)B .data_ptr<double>()
    };
    AT_CUDA_CHECK(cudaLaunchKernel(
        (void*)build_system_kernel_double, grid_b, dim3(t), args_build, 0, stream.stream()));
}

static void writeback_and_residual(
    at::Tensor fz, at::Tensor solution, at::Tensor fn_tmp, at::Tensor res_buf)
{
    TORCH_CHECK(fz.is_cuda() && solution.is_cuda() && fn_tmp.is_cuda() && res_buf.is_cuda(),
                "tensors must be CUDA");
    const int64_t nx = fz.size(0);
    const int64_t nv = fz.size(1);
    const int64_t n_inner = std::max<int64_t>(nx - 2, 0);

    if (n_inner == 0){
        res_buf.zero_();
        fn_tmp.copy_(fz);
        return;
    }

    // shape check
    {
        std::vector<int64_t> expect2{nv, n_inner};
        TORCH_CHECK(solution.sizes() == at::IntArrayRef(expect2), "solution shape (nv, nx-2)");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    // residual = 0
    AT_CUDA_CHECK(cudaMemsetAsync(res_buf.data_ptr(), 0, sizeof(double), stream.stream()));

    const int t = 256;
    const int total = (int)(nv * n_inner);
    dim3 grid((total + t - 1) / t);
    void* args[] = {
        (void*)fz.data_ptr<double>(),
        (void*)solution.data_ptr<double>(),
        (void*)fn_tmp.data_ptr<double>(),
        (void*)&nx, (void*)&nv,
        (void*)res_buf.data_ptr<double>()
    };
    AT_CUDA_CHECK(cudaLaunchKernel(
        (void*)scatter_and_residual_kernel_double, grid, dim3(t), args, 0, stream.stream()));
}

// ---- pybind ----
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("build_system",
          &build_system_and_moments,
          "build (a,b,c,B) and moments (double)");
    m.def("writeback_and_residual",
          &writeback_and_residual,
          "writeback solution and compute L-inf residual on GPU (double)");
}
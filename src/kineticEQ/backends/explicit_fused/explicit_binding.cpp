#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
void launch_explicit_step(const scalar_t* f, scalar_t* fn, const scalar_t* v,
                          int nx, int nv,
                          scalar_t dv, scalar_t dt, scalar_t dx,
                          scalar_t tau_tilde, scalar_t inv_sqrt_2pi,
                          int k0, cudaStream_t stream);

torch::Tensor explicit_step(torch::Tensor f, torch::Tensor fn, torch::Tensor v,
                            double dv, double dt, double dx,
                            double tau_tilde, double inv_sqrt_2pi, int64_t k0) {
  TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(f.dtype() == fn.dtype() && (f.dtype()==torch::kFloat64 || f.dtype()==torch::kFloat32),
              "dtype must be float64 or float32 (same for f & fn)");
  TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(), "must be contiguous");
  TORCH_CHECK(f.dim()==2 && fn.sizes()==f.sizes(), "f and fn must be (nx,nv)");
  TORCH_CHECK(v.dim()==1 && v.size(0)==f.size(1), "v must be (nv,)");

  int nx = (int)f.size(0);
  int nv = (int)f.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "explicit_step", [&](){
    using scalar_t_ = scalar_t;
    launch_explicit_step<scalar_t_>(
      f.data_ptr<scalar_t_>(),
      fn.data_ptr<scalar_t_>(),
      v.data_ptr<scalar_t_>(),
      nx, nv,
      static_cast<scalar_t_>(dv),
      static_cast<scalar_t_>(dt),
      static_cast<scalar_t_>(dx),
      static_cast<scalar_t_>(tau_tilde),
      static_cast<scalar_t_>(inv_sqrt_2pi),
      (int)k0, stream.stream());
  });

  return fn;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("explicit_step", &explicit_step, "Fused explicit BGK step (CUDA)");
}

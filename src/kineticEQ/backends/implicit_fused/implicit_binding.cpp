// implicit_binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <tuple>
#include <stdexcept>
#include <string>

namespace implicit_fused {
void launch_moments_double(
    const double* f, const double* v,
    int nx, int nv, double dv,
    double* n, double* u, double* T,
    cudaStream_t stream);

void launch_build_tridiag_rhs_double(
    const double* f_prev, const double* v,
    const double* n, const double* u, const double* T,
    int nx, int nv, double dt, double dx, double tau_tilde, double inv_sqrt_2pi,
    double* dl, double* dd, double* du, double* B,
    cudaStream_t stream);

void launch_writeback_and_residual_double(
    const double* f_iter, const double* B,
    int nx, int nv,
    double* fn, double* res_per_v,
    cudaStream_t stream);
} // namespace implicit_fused

// ====================== cuSPARSE helpers ======================
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t _status = (call); \
        if (_status != CUSPARSE_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuSPARSE error: ") + std::to_string(_status) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

static void gtsv_strided_batch_double(
    cusparseHandle_t handle,
    int n,                  // system size (n_inner)
    int batchCount,         // nv
    double* dl,             // (batchCount, n)
    double* d,              // (batchCount, n)
    double* du,             // (batchCount, n)
    double* B               // (batchCount, n), in/out
){
    // API: cusparseDgtsv2StridedBatch requires a temporary buffer
    size_t pBufferSizeInBytes = 0;
    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt(
        handle, n, dl, d, du, B, n, batchCount, &pBufferSizeInBytes));

    auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
    at::Tensor buffer = at::empty({static_cast<long long>(pBufferSizeInBytes)}, options);
    void* pBuffer = buffer.data_ptr();

    CUSPARSE_CHECK(cusparseDgtsv2StridedBatch(
        handle, n, dl, d, du, B, n, batchCount, pBuffer));
}

// ====================== main entry ======================
std::tuple<int,double> implicit_step(
    at::Tensor f,             // (nx,nv), double, cuda, contiguous
    at::Tensor fn,            // (nx,nv), double, cuda, contiguous (出力)
    at::Tensor v,             // (nv),    double, cuda, contiguous
    double dv, double dt, double dx,
    double tau_tilde, double inv_sqrt_2pi,
    int /*k0_unused*/,
    int picard_iter, double picard_tol
    // 境界は「触らない」仕様のため、ここでは追加の境界パラメータは受け取らない
){
    TORCH_CHECK(f.is_cuda() && fn.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(f.dtype() == at::kDouble && fn.dtype() == at::kDouble && v.dtype() == at::kDouble,
                "dtype must be float64");
    TORCH_CHECK(f.is_contiguous() && fn.is_contiguous() && v.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(f.dim()==2, "f must be (nx,nv)");
    TORCH_CHECK(fn.sizes()==f.sizes(), "fn must have same shape as f");
    TORCH_CHECK(v.dim()==1 && v.size(0)==f.size(1), "v must be (nv,)");

    const int64_t nx64 = f.size(0), nv64 = f.size(1);
    TORCH_CHECK(nx64 >= 3, "nx must be >= 3 (need interior cells)");
    const int nx = static_cast<int>(nx64);
    const int nv = static_cast<int>(nv64);
    const int n_inner = nx - 2;

    auto optsD = f.options();
    auto opts1 = f.options().dtype(at::kDouble);

    // work arrays
    at::Tensor n   = at::empty({nx},     opts1);
    at::Tensor u   = at::empty({nx},     opts1);
    at::Tensor T   = at::empty({nx},     opts1);
    at::Tensor dl  = at::empty({nv, n_inner}, opts1); // 下対角
    at::Tensor dd  = at::empty({nv, n_inner}, opts1); // 主対角
    at::Tensor du  = at::empty({nv, n_inner}, opts1); // 上対角
    at::Tensor B   = at::empty({nv, n_inner}, opts1); // RHS / 解
    at::Tensor res = at::empty({nv}, opts1);          // 列毎最大残差

    // 交互バッファ（境界は常に f をコピーしておく）
    at::Tensor work = at::empty_like(f);

    // 境界を含めて、両バッファを f に初期化（内部更新のみを行うので境界は保持される）
    fn.copy_(f);
    work.copy_(f);

    // CUDA stream & cuSPARSE handle
    auto stream = at::cuda::getCurrentCUDAStream();
    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream.stream()));

    // device pointers
    const double* f_prev = f.data_ptr<double>();   // RHS 用に固定（タイムレベル n）
    const double* v_ptr  = v.data_ptr<double>();

    double* fn_ptr   = fn.data_ptr<double>();
    double* work_ptr = work.data_ptr<double>();

    // Picard 反復バッファ: p_iter = 現在反復の参照配列, p_out = 次の解を書き込む先
    const double* p_iter = f_prev; // 初回は f を基準（境界一定）
    double*       p_out  = fn_ptr; // 初回は fn に書く

    int iters_done = 0;
    double final_residual = 0.0;

    // 反復ループ
    for (int it = 0; it < picard_iter; ++it) {
        // 1) moments from p_iter
        implicit_fused::launch_moments_double(
            p_iter, v_ptr, nx, nv, dv,
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            stream.stream());

        // 2) build tri-diagonal and RHS from f_prev (境界流入も f_prev の境界を使用)
        implicit_fused::launch_build_tridiag_rhs_double(
            f_prev, v_ptr,
            n.data_ptr<double>(), u.data_ptr<double>(), T.data_ptr<double>(),
            nx, nv, dt, dx, tau_tilde, inv_sqrt_2pi,
            dl.data_ptr<double>(), dd.data_ptr<double>(), du.data_ptr<double>(), B.data_ptr<double>(),
            stream.stream());

        // 3) batched gtsv solve (in-place on B)
        gtsv_strided_batch_double(
            handle, n_inner, nv,
            dl.data_ptr<double>(),
            dd.data_ptr<double>(),
            du.data_ptr<double>(),
            B.data_ptr<double>()); // overwritten with solution

        // 4) writeback interior & residual (境界は一切触らない)
        implicit_fused::launch_writeback_and_residual_double(
            p_iter, B.data_ptr<double>(), nx, nv, p_out, res.data_ptr<double>(),
            stream.stream());

        // 5) 残差（列最大のさらに最大）
        //    res: (nv,) → amax
        auto max_res = std::get<0>(res.max(/*dim=*/0));
        final_residual = max_res.item<double>();
        iters_done     = it + 1;

        // 6) 収束判定
        if (final_residual < picard_tol) {
            // 結果は p_out 内部セルに入っている。境界は既に f_prev と同じ（初期化済み）。
            // もし p_out != fn_ptr なら最終結果を fn にコピー（境界も含め全体コピーでOK）
            if (p_out != fn_ptr) {
                fn.copy_(work); // p_out==work の場合
            }
            break;
        }

        // 7) 交互（次反復へ）
        //    p_iter <- p_out, p_out <- もう一方
        if (p_out == fn_ptr) {
            p_iter = fn_ptr;
            p_out  = work_ptr;
            // 境界を f_prev にしておく（明示的に安全側）
            // 内部は writeback が上書き、境界は保持。
            cudaMemcpyAsync(work_ptr, f_prev, sizeof(double)*nx*nv,
                            cudaMemcpyDeviceToDevice, stream.stream());
        } else {
            p_iter = work_ptr;
            p_out  = fn_ptr;
            cudaMemcpyAsync(fn_ptr, f_prev, sizeof(double)*nx*nv,
                            cudaMemcpyDeviceToDevice, stream.stream());
        }
    }

    // 反復を使い切って未収束の場合、最終反復結果が p_out にあるとは限らないので整える
    if (iters_done >= picard_iter && final_residual >= picard_tol) {
        if (p_iter != fn_ptr) {
            fn.copy_(work); // p_iter==work の場合の保険
        }
    }

    // cuSPARSE 終了
    CUSPARSE_CHECK(cusparseDestroy(handle));

    // 戻り値: (反復回数, 残差)
    return std::make_tuple(iters_done, final_residual);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_step", &implicit_step,
          "Fused implicit BGK step (double, CUDA, boundaries untouched; RHS uses f-prev boundaries)");
}

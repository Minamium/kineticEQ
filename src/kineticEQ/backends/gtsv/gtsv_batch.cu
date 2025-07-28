
#include <cusparse.h>
#include <cuda_runtime_api.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

/* ----------------------------------------------------------------------
   cuSPARSE GTSV2 Strided-Batch  テンプレート特化
   -------------------------------------------------------------------- */
template <typename T> struct CuSparseGtsv;

/* ----- float --------------------------------------------------------- */
template <>
struct CuSparseGtsv<float> {
    static inline cusparseStatus_t buffer(cusparseHandle_t h,
                                          int n,
                                          const float* dl,
                                          const float* d,
                                          const float* du,
                                          float* B,
                                          int batch,
                                          size_t* bytes)
    {
        return cusparseSgtsv2StridedBatch_bufferSizeExt(
            h, n, dl, d, du, B, batch, n, bytes);
    }

    static inline cusparseStatus_t solve(cusparseHandle_t h,
                                         int n,
                                         const float* dl,
                                         const float* d,
                                         const float* du,
                                         float* B,
                                         int batch,
                                         void* buf)
    {
        return cusparseSgtsv2StridedBatch(
            h, n, dl, d, du, B, batch, n, buf);
    }
};

/* ----- double -------------------------------------------------------- */
template <>
struct CuSparseGtsv<double> {
    static inline cusparseStatus_t buffer(cusparseHandle_t h,
                                          int n,
                                          const double* dl,
                                          const double* d,
                                          const double* du,
                                          double* B,
                                          int batch,
                                          size_t* bytes)
    {
        return cusparseDgtsv2StridedBatch_bufferSizeExt(
            h, n, dl, d, du, B, batch, n, bytes);
    }

    static inline cusparseStatus_t solve(cusparseHandle_t h,
                                         int n,
                                         const double* dl,
                                         const double* d,
                                         const double* du,
                                         double* B,
                                         int batch,
                                         void* buf)
    {
        return cusparseDgtsv2StridedBatch(
            h, n, dl, d, du, B, batch, n, buf);
    }
};

/* ----------------------------------------------------------------------
   ユーティリティ関数群
   -------------------------------------------------------------------- */
template <typename scalar_t>
size_t gtsv_strided_ws_size(int n,
                            int batch,
                            const scalar_t* dl,
                            const scalar_t* d,
                            const scalar_t* du,
                            scalar_t* B)
{
    auto handle = at::cuda::getCurrentCUDASparseHandle();

    size_t ws_bytes = 0;
    TORCH_CHECK(
        CuSparseGtsv<scalar_t>::buffer(
            handle, n, dl, d, du, B, batch, &ws_bytes) ==
        CUSPARSE_STATUS_SUCCESS,
        "cuSPARSE bufferSizeExt failed");

    return ws_bytes;
}

/*! \brief 事前に確保済みワークスペースを利用して batched 三重対角行列を解く
 *
 *  \param[in]  n      行列サイズ
 *  \param[in]  batch  バッチ数
 *  \param[in]  dl     下対角要素 (device)
 *  \param[in]  d      対角要素   (device)
 *  \param[in]  du     上対角要素 (device)
 *  \param[in]  B      右辺 / 解ベクトル (in-place, device)
 *  \param[in]  workspace  ワークスペース (device)
 */
template <typename scalar_t>
void gtsv_strided_impl(int n,
                       int batch,
                       const scalar_t* dl,
                       const scalar_t* d,
                       const scalar_t* du,
                       scalar_t* B,
                       void* workspace)
{
    TORCH_CHECK(workspace != nullptr,
                "workspace pointer must not be nullptr");

    auto handle = at::cuda::getCurrentCUDASparseHandle();

    TORCH_CHECK(
        CuSparseGtsv<scalar_t>::solve(
            handle, n, dl, d, du, B, batch, workspace) ==
        CUSPARSE_STATUS_SUCCESS,
        "cuSPARSE gtsv2StridedBatch failed");
}

/* 明示的インスタンス化 ------------------------------------------------- */
template size_t gtsv_strided_ws_size<float >(int, int,
                                             const float* , const float* ,
                                             const float* , float* );
template size_t gtsv_strided_ws_size<double>(int, int,
                                             const double*, const double*,
                                             const double*, double*);

template void gtsv_strided_impl<float >(int, int,
                                        const float* , const float* ,
                                        const float* , float* , void*);
template void gtsv_strided_impl<double>(int, int,
                                        const double*, const double*,
                                        const double*, double*, void*);

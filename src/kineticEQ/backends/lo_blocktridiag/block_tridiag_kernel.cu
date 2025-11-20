// src/kineticEQ/backends/lo_blocktridiag/block_tridiag_kernel.cu

#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__device__ inline void mat3_mul(const T* A, const T* B, T* C) {
    // C = A * B, row-major 3x3
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        T a0 = A[3*i + 0];
        T a1 = A[3*i + 1];
        T a2 = A[3*i + 2];
        C[3*i + 0] = a0 * B[0]  + a1 * B[3]  + a2 * B[6];
        C[3*i + 1] = a0 * B[1]  + a1 * B[4]  + a2 * B[7];
        C[3*i + 2] = a0 * B[2]  + a1 * B[5]  + a2 * B[8];
    }
}

template <typename T>
__device__ inline void mat3_vec_mul(const T* A, const T* x, T* y) {
    // y = A * x, A: 3x3, x:3
    y[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    y[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    y[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

template <typename T>
__device__ inline bool invert3x3(const T* M, T* Minv) {
    const T a00 = M[0], a01 = M[1], a02 = M[2];
    const T a10 = M[3], a11 = M[4], a12 = M[5];
    const T a20 = M[6], a21 = M[7], a22 = M[8];

    const T c00 =  (a11 * a22 - a12 * a21);
    const T c01 = -(a10 * a22 - a12 * a20);
    const T c02 =  (a10 * a21 - a11 * a20);

    const T c10 = -(a01 * a22 - a02 * a21);
    const T c11 =  (a00 * a22 - a02 * a20);
    const T c12 = -(a00 * a21 - a01 * a20);

    const T c20 =  (a01 * a12 - a02 * a11);
    const T c21 = -(a00 * a12 - a02 * a10);
    const T c22 =  (a00 * a11 - a01 * a10);

    const T det = a00 * c00 + a01 * c01 + a02 * c02;
    if (fabs((double)det) < 1e-30) {
        return false;
    }
    const T inv_det = 1.0 / det;

    // adjugate^T * (1/det)
    Minv[0] = c00 * inv_det;
    Minv[1] = c10 * inv_det;
    Minv[2] = c20 * inv_det;

    Minv[3] = c01 * inv_det;
    Minv[4] = c11 * inv_det;
    Minv[5] = c21 * inv_det;

    Minv[6] = c02 * inv_det;
    Minv[7] = c12 * inv_det;
    Minv[8] = c22 * inv_det;
    return true;
}

template <typename T>
__global__ void block_tridiag_kernel(
    const T* __restrict__ A,   // (batch, n, 3, 3)
    T* __restrict__       B,   // (batch, n, 3, 3) ← in-placeで更新
    T* __restrict__       C,   // (batch, n, 3, 3) ← in-placeで C'
    T* __restrict__       D,   // (batch, n, 3)    ← in-placeで D'
    T* __restrict__       X,   // (batch, n, 3)    ← 解
    int n,
    int batch
) {
    const int b = blockIdx.x;
    if (b >= batch) return;

    // 各系統の先頭ポインタ
    const int mat_stride = n * 9;  // 3x3=9
    const int vec_stride = n * 3;

    const T* Ab = A + (size_t)b * mat_stride;
          T* Bb = B + (size_t)b * mat_stride;
          T* Cb = C + (size_t)b * mat_stride;
          T* Db = D + (size_t)b * vec_stride;
          T* Xb = X + (size_t)b * vec_stride;

    if (n <= 0) return;

    // ---- Forward elimination ----
    T denom[9];
    T denom_inv[9];
    T tmp3[3];
    T mat_tmp[9];

    // k = 0
    // C'_0 = inv(B_0) * C_0
    if (!invert3x3(&Bb[0], denom_inv)) {
        // 特異な場合はそのまま返してしまう（要検討）
        return;
    }
    mat3_mul(denom_inv, &Cb[0], &Cb[0]);       // Cb[0] ← C'_0
    mat3_vec_mul(denom_inv, &Db[0], &Db[0]);   // Db[0] ← D'_0

    // k = 1..n-1
    for (int k = 1; k < n; ++k) {
        const int idx_k  = k * 9;
        const int idx_km = (k - 1) * 9;
        const int idv_k  = k * 3;
        const int idv_km = (k - 1) * 3;

        // rhs_tmp = D_k - A_k * D'_{k-1}
        mat3_vec_mul(&Ab[idx_k], &Db[idv_km], tmp3);
        tmp3[0] = Db[idv_k + 0] - tmp3[0];
        tmp3[1] = Db[idv_k + 1] - tmp3[1];
        tmp3[2] = Db[idv_k + 2] - tmp3[2];

        // denom = B_k - A_k * C'_{k-1}
        mat3_mul(&Ab[idx_k], &Cb[idx_km], mat_tmp);
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            denom[i] = Bb[idx_k + i] - mat_tmp[i];
        }

        if (!invert3x3(denom, denom_inv)) {
            return;
        }

        // C'_k = inv(denom) * C_k （最後の行では使わないが、一応計算）
        if (k < n - 1) {
            mat3_mul(denom_inv, &Cb[idx_k], &Cb[idx_k]);
        }

        // D'_k = inv(denom) * rhs_tmp
        mat3_vec_mul(denom_inv, tmp3, &Db[idv_k]);
    }

    // ---- Back substitution ----
    // X_{n-1} = D'_{n-1}
    Xb[(n - 1)*3 + 0] = Db[(n - 1)*3 + 0];
    Xb[(n - 1)*3 + 1] = Db[(n - 1)*3 + 1];
    Xb[(n - 1)*3 + 2] = Db[(n - 1)*3 + 2];

    for (int k = n - 2; k >= 0; --k) {
        const int idx_k  = k * 9;
        const int idv_k  = k * 3;
        const int idv_kp = (k + 1) * 3;

        // y = C'_k * X_{k+1}
        mat3_vec_mul(&Cb[idx_k], &Xb[idv_kp], tmp3);

        Xb[idv_k + 0] = Db[idv_k + 0] - tmp3[0];
        Xb[idv_k + 1] = Db[idv_k + 1] - tmp3[1];
        Xb[idv_k + 2] = Db[idv_k + 2] - tmp3[2];
    }
}

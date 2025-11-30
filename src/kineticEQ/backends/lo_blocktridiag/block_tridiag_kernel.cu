// src/kineticEQ/backends/lo_blocktridiag/block_tridiag_kernel.cu

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

// 3x3 行列積: C = A * B
template <typename T>
__device__ __forceinline__ void mat3_mul(const T* A, const T* B, T* C) {
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const T a0 = A[3 * i + 0];
        const T a1 = A[3 * i + 1];
        const T a2 = A[3 * i + 2];
        C[3 * i + 0] = a0 * B[0] + a1 * B[3] + a2 * B[6];
        C[3 * i + 1] = a0 * B[1] + a1 * B[4] + a2 * B[7];
        C[3 * i + 2] = a0 * B[2] + a1 * B[5] + a2 * B[8];
    }
}

// 3x3 行列と 3 ベクトルの積: y = A * x
template <typename T>
__device__ __forceinline__ void mat3_vec_mul(const T* A, const T* x, T* y) {
    y[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    y[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    y[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

// 3x3 行列の逆行列（単純な adjugate / det 実装）
template <typename T>
__device__ __forceinline__ bool invert3x3(const T* M, T* Minv) {
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
    if (::fabs((double)det) < 1e-30) {
        return false;
    }
    const T inv_det = T(1.0) / det;

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

/**
 * Parallel Cyclic Reduction (PCR) ベースの 3x3 ブロック三重対角ソルバー
 *
 * A0,B0,C0,D0 : ステージ 0 の係数 (batch, n, 3x3 or 3)
 * A1,B1,C1,D1 : ping-pong 用ワークバッファ (batch, n, 3x3 or 3)
 * X           : 解 (batch, n, 3)
 *
 * 1 block = 1 系列 (バッチの 1 系列)、block 内で i (= 0..n-1) を並列処理。
 */
template <typename T>
__global__ void block_tridiag_pcr_kernel(
    T* __restrict__ A0,
    T* __restrict__ B0,
    T* __restrict__ C0,
    T* __restrict__ D0,
    T* __restrict__ A1,
    T* __restrict__ B1,
    T* __restrict__ C1,
    T* __restrict__ D1,
    T* __restrict__ X,
    int n,
    int batch,
    int max_stage
) {
    const int b = blockIdx.x;
    if (b >= batch || n <= 0) {
        return;
    }

    const size_t mat_stride = static_cast<size_t>(n) * 9;  // 3x3 = 9
    const size_t vec_stride = static_cast<size_t>(n) * 3;

    // このバッチの先頭ポインタ
    T* A_curr = A0 + b * mat_stride;
    T* B_curr = B0 + b * mat_stride;
    T* C_curr = C0 + b * mat_stride;
    T* D_curr = D0 + b * vec_stride;

    T* A_next = A1 + b * mat_stride;
    T* B_next = B1 + b * mat_stride;
    T* C_next = C1 + b * mat_stride;
    T* D_next = D1 + b * vec_stride;

    T* Xb     = X  + b * vec_stride;

    const int tid  = threadIdx.x;
    const int step = blockDim.x;

    // ステージループ：s = 0..max_stage-1, stride = 2^s
    for (int s = 0; s < max_stage; ++s) {
        const int m = 1 << s;
        if (m >= n) {
            break;
        }

        // 各 i を block 内で並列処理
        for (int i = tid; i < n; i += step) {
            const int left  = i - m;
            const int right = i + m;

            const T* Ai = &A_curr[9 * i];
            const T* Bi = &B_curr[9 * i];
            const T* Ci = &C_curr[9 * i];
            const T* Di = &D_curr[3 * i];

            T alpha[9] = {T(0)};
            T gamma[9] = {T(0)};

            // alpha = A_i * (B_left)^{-1} (left が存在する場合)
            if (left >= 0) {
                const T* B_left = &B_curr[9 * left];
                T B_left_inv[9];
                if (invert3x3(B_left, B_left_inv)) {
                    mat3_mul(Ai, B_left_inv, alpha);
                }
            }

            // gamma = C_i * (B_right)^{-1} (right が存在する場合)
            if (right < n) {
                const T* B_right = &B_curr[9 * right];
                T B_right_inv[9];
                if (invert3x3(B_right, B_right_inv)) {
                    mat3_mul(Ci, B_right_inv, gamma);
                }
            }

            // A_next[i] = - alpha * A_left
            if (left >= 0) {
                const T* A_left = &A_curr[9 * left];
                T tmpA[9];
                mat3_mul(alpha, A_left, tmpA);
                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    A_next[9 * i + k] = -tmpA[k];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    A_next[9 * i + k] = T(0);
                }
            }

            // C_next[i] = - gamma * C_right
            if (right < n) {
                const T* C_right = &C_curr[9 * right];
                T tmpC[9];
                mat3_mul(gamma, C_right, tmpC);
                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    C_next[9 * i + k] = -tmpC[k];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    C_next[9 * i + k] = T(0);
                }
            }

            // B_next[i] = B_i - alpha * C_left - gamma * A_right
            {
                T tmp1[9] = {T(0)};
                T tmp2[9] = {T(0)};

                if (left >= 0) {
                    const T* C_left = &C_curr[9 * left];
                    mat3_mul(alpha, C_left, tmp1);
                }
                if (right < n) {
                    const T* A_right = &A_curr[9 * right];
                    mat3_mul(gamma, A_right, tmp2);
                }

                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    B_next[9 * i + k] = Bi[k] - tmp1[k] - tmp2[k];
                }
            }

            // D_next[i] = D_i - alpha * D_left - gamma * D_right
            {
                T tmpL[3] = {T(0)};
                T tmpR[3] = {T(0)};

                if (left >= 0) {
                    const T* D_left = &D_curr[3 * left];
                    mat3_vec_mul(alpha, D_left, tmpL);
                }
                if (right < n) {
                    const T* D_right = &D_curr[3 * right];
                    mat3_vec_mul(gamma, D_right, tmpR);
                }

                D_next[3 * i + 0] = Di[0] - tmpL[0] - tmpR[0];
                D_next[3 * i + 1] = Di[1] - tmpL[1] - tmpR[1];
                D_next[3 * i + 2] = Di[2] - tmpL[2] - tmpR[2];
            }
        }

        __syncthreads();

        // ping-pong: curr と next のポインタを入れ替える
        T* tmp;
        tmp = A_curr; A_curr = A_next; A_next = tmp;
        tmp = B_curr; B_curr = B_next; B_next = tmp;
        tmp = C_curr; C_curr = C_next; C_next = tmp;
        tmp = D_curr; D_curr = D_next; D_next = tmp;

        __syncthreads();
    }

    // 最終ステージ後: B_curr[i] x_i = D_curr[i] を各 i で独立に解く
    for (int i = tid; i < n; i += step) {
        const T* Bi = &B_curr[9 * i];
        const T* Di = &D_curr[3 * i];

        T Bi_inv[9];
        T xi[3];

        if (!invert3x3(Bi, Bi_inv)) {
            xi[0] = xi[1] = xi[2] = T(0);
        } else {
            mat3_vec_mul(Bi_inv, Di, xi);
        }

        Xb[3 * i + 0] = xi[0];
        Xb[3 * i + 1] = xi[1];
        Xb[3 * i + 2] = xi[2];
    }
}

// nvcc 側でカーネルを起動するランチャ関数
template <typename scalar_t>
void launch_block_tridiag_kernel(
    scalar_t* A0,
    scalar_t* B0,
    scalar_t* C0,
    scalar_t* D0,
    scalar_t* A1,
    scalar_t* B1,
    scalar_t* C1,
    scalar_t* D1,
    scalar_t* X,
    int n,
    int batch,
    cudaStream_t stream
) {
    if (batch <= 0 || n <= 0) {
        return;
    }

    // log2(n) 程度のステージ数
    int max_stage = 0;
    while ((1 << max_stage) < n) {
        ++max_stage;
    }

    const int threads = (n < 128) ? n : 128;  // 1 系列あたりのスレッド数
    const dim3 blocks(batch);                 // 1 block = 1 系列

    block_tridiag_pcr_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        A0, B0, C0, D0,
        A1, B1, C1, D1,
        X,
        n,
        batch,
        max_stage
    );
}

// 明示的インスタンス化
template void launch_block_tridiag_kernel<float>(
    float*, float*, float*, float*,
    float*, float*, float*, float*,
    float*, int, int, cudaStream_t);

template void launch_block_tridiag_kernel<double>(
    double*, double*, double*, double*,
    double*, double*, double*, double*,
    double*, int, int, cudaStream_t);

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------
// small device utilities
// ---------------------------
template <typename T>
__device__ __forceinline__ T dabs(T x) { return (x < T(0)) ? -x : x; }

template <typename T>
__device__ __forceinline__ T dmin(T a, T b) { return (a < b) ? a : b; }

// minmod limiter
template <typename T>
__device__ __forceinline__ T minmod(T a, T b) {
    // if opposite signs or either is zero -> 0
    if (a * b <= T(0)) return T(0);
    // same sign
    T s = (a > T(0)) ? T(1) : T(-1);
    return s * dmin(dabs(a), dabs(b));
}

// boundary index mapping
// bc_type: 0 periodic, 1 neumann(clamp)
__device__ __forceinline__ int bc_index(int idx, int offset, int size, int bc_type) {
    int r = idx + offset;
    if (bc_type == 0) { // periodic
        r %= size;
        if (r < 0) r += size;
        return r;
    } else { // neumann(clamp)
        if (r < 0) r = 0;
        if (r >= size) r = size - 1;
        return r;
    }
}

// ---------------------------
// main fused kernel
// one block per (i,j) cell
// threads loop over velocity indices
// ---------------------------
template <typename T>
__global__ void explicit_step_2d2v_kernel(
    const T* __restrict__ f,
    T* __restrict__ fn,
    const T* __restrict__ vx,
    const T* __restrict__ vy,
    int nx, int ny, int nvx, int nvy,
    T dvx, T dvy,
    T dt, T dx, T dy,
    T tau_tilde,
    int scheme,  // 0 upwind, 1 MUSCL2
    int bc_x,    // 0 periodic, 1 neumann
    int bc_y     // 0 periodic, 1 neumann
) {
    // dynamic shared memory:
    // [s_n, s_nux, s_nuy, s_U] each blockDim.x
    // + scalars
    extern __shared__ unsigned char smem_raw[];
    T* s = reinterpret_cast<T*>(smem_raw);

    T* s_n   = s; s += blockDim.x;
    T* s_nux = s; s += blockDim.x;
    T* s_nuy = s; s += blockDim.x;
    T* s_U   = s; s += blockDim.x;

    // scalar pack (at least 5)
    T* scal = s; // [ux, uy, Tgas, inv_tau, coeff]

    // cell mapping
    const int cell = static_cast<int>(blockIdx.x);
    const int i = cell / ny;
    const int j = cell - i * ny;
    if (i >= nx) return;

    const int nv = nvx * nvy;
    const int base = cell * nv;

    // ---------------------------
    // 1) moments reduction
    // ---------------------------
    T sum_n   = T(0);
    T sum_nux = T(0);
    T sum_nuy = T(0);
    T sum_U   = T(0);

    for (int k = static_cast<int>(threadIdx.x); k < nv; k += static_cast<int>(blockDim.x)) {
        const int p = k / nvy;
        const int q = k - p * nvy;

        const T fij = f[base + k];
        const T vxp = vx[p];
        const T vyq = vy[q];

        sum_n   += fij;
        sum_nux += vxp * fij;
        sum_nuy += vyq * fij;
        sum_U   += T(0.5) * (vxp * vxp + vyq * vyq) * fij;
    }

    s_n[threadIdx.x]   = sum_n;
    s_nux[threadIdx.x] = sum_nux;
    s_nuy[threadIdx.x] = sum_nuy;
    s_U[threadIdx.x]   = sum_U;
    __syncthreads();

    // reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_n[threadIdx.x]   += s_n[threadIdx.x + stride];
            s_nux[threadIdx.x] += s_nux[threadIdx.x + stride];
            s_nuy[threadIdx.x] += s_nuy[threadIdx.x + stride];
            s_U[threadIdx.x]   += s_U[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const T dv = dvx * dvy;

        const T n   = s_n[0]   * dv;
        const T nux = s_nux[0] * dv;
        const T nuy = s_nuy[0] * dv;
        const T U   = s_U[0]   * dv;

        const T n_safe = n + T(1e-30);
        const T ux = nux / n_safe;
        const T uy = nuy / n_safe;

        T Tgas = U / n_safe - T(0.5) * (ux * ux + uy * uy);
        if (!(Tgas > T(0))) Tgas = T(1e-30);
        if (Tgas < T(1e-30)) Tgas = T(1e-30);

        // tau = tau_tilde / (n*sqrt(T))
        T tau = tau_tilde / (n_safe * sqrt(Tgas));
        if (!(tau > T(0))) tau = T(1e-30);
        if (tau < T(1e-30)) tau = T(1e-30);
        const T inv_tau = T(1.0) / tau;

        // Maxwellian coeff: n/(2*pi*T)
        const T PI = T(3.14159265358979323846);
        const T coeff = n / (T(2.0) * PI * Tgas);

        scal[0] = ux;
        scal[1] = uy;
        scal[2] = Tgas;
        scal[3] = inv_tau;
        scal[4] = coeff;
    }
    __syncthreads();

    const T ux     = scal[0];
    const T uy     = scal[1];
    const T Tgas   = scal[2];
    const T inv_tau= scal[3];
    const T coeff  = scal[4];

    // ---------------------------
    // 2) fused update per velocity
    // ---------------------------
    for (int k = static_cast<int>(threadIdx.x); k < nv; k += static_cast<int>(blockDim.x)) {
        const int p = k / nvy;
        const int q = k - p * nvy;

        const T vxp = vx[p];
        const T vyq = vy[q];

        const T fij = f[base + k];

        // ---- advection term ----
        T adv = T(0);

        if (scheme == 0) {
            // ===== upwind =====
            const int i_m1 = bc_index(i, -1, nx, bc_x);
            const int i_p1 = bc_index(i,  1, nx, bc_x);
            const int j_m1 = bc_index(j, -1, ny, bc_y);
            const int j_p1 = bc_index(j,  1, ny, bc_y);

            const int cell_im1 = i_m1 * ny + j;
            const int cell_ip1 = i_p1 * ny + j;
            const int cell_jm1 = i * ny + j_m1;
            const int cell_jp1 = i * ny + j_p1;

            const T fim1 = f[cell_im1 * nv + k];
            const T fip1 = f[cell_ip1 * nv + k];
            const T fjm1 = f[cell_jm1 * nv + k];
            const T fjp1 = f[cell_jp1 * nv + k];

            const T dfdx = (vxp > T(0)) ? (fij - fim1) / dx : (fip1 - fij) / dx;
            const T dfdy = (vyq > T(0)) ? (fij - fjm1) / dy : (fjp1 - fij) / dy;

            adv = vxp * dfdx + vyq * dfdy;

        } else {
            // ===== MUSCL2 =====
            // x stencil indices
            const int i_m2 = bc_index(i, -2, nx, bc_x);
            const int i_m1 = bc_index(i, -1, nx, bc_x);
            const int i_p1 = bc_index(i,  1, nx, bc_x);
            const int i_p2 = bc_index(i,  2, nx, bc_x);

            const int cell_im2 = i_m2 * ny + j;
            const int cell_im1 = i_m1 * ny + j;
            const int cell_ip1 = i_p1 * ny + j;
            const int cell_ip2 = i_p2 * ny + j;

            const T fim2 = f[cell_im2 * nv + k];
            const T fim1 = f[cell_im1 * nv + k];
            const T fip1 = f[cell_ip1 * nv + k];
            const T fip2 = f[cell_ip2 * nv + k];

            const T slope_i   = minmod(fij  - fim1, fip1 - fij);
            const T slope_ip1 = minmod(fip1 - fij,  fip2 - fip1);
            const T slope_im1 = minmod(fim1 - fim2, fij  - fim1);

            // F_{i+1/2}
            const T fL_ip = fij  + T(0.5) * slope_i;
            const T fR_ip = fip1 - T(0.5) * slope_ip1;
            const T fup_ip= (vxp > T(0)) ? fL_ip : fR_ip;
            const T F_ip  = vxp * fup_ip;

            // F_{i-1/2}
            const T fL_im = fim1 + T(0.5) * slope_im1;
            const T fR_im = fij  - T(0.5) * slope_i;
            const T fup_im= (vxp > T(0)) ? fL_im : fR_im;
            const T F_im  = vxp * fup_im;

            const T adv_x = (F_ip - F_im) / dx;

            // y stencil indices
            const int j_m2 = bc_index(j, -2, ny, bc_y);
            const int j_m1 = bc_index(j, -1, ny, bc_y);
            const int j_p1 = bc_index(j,  1, ny, bc_y);
            const int j_p2 = bc_index(j,  2, ny, bc_y);

            const int cell_jm2 = i * ny + j_m2;
            const int cell_jm1 = i * ny + j_m1;
            const int cell_jp1 = i * ny + j_p1;
            const int cell_jp2 = i * ny + j_p2;

            const T fjm2 = f[cell_jm2 * nv + k];
            const T fjm1 = f[cell_jm1 * nv + k];
            const T fjp1 = f[cell_jp1 * nv + k];
            const T fjp2 = f[cell_jp2 * nv + k];

            const T slope_j   = minmod(fij  - fjm1, fjp1 - fij);
            const T slope_jp1 = minmod(fjp1 - fij,  fjp2 - fjp1);
            const T slope_jm1 = minmod(fjm1 - fjm2, fij  - fjm1);

            // G_{j+1/2}
            const T fD_jp = fij  + T(0.5) * slope_j;
            const T fU_jp = fjp1 - T(0.5) * slope_jp1;
            const T fup_jp= (vyq > T(0)) ? fD_jp : fU_jp;
            const T G_jp  = vyq * fup_jp;

            // G_{j-1/2}
            const T fD_jm = fjm1 + T(0.5) * slope_jm1;
            const T fU_jm = fij  - T(0.5) * slope_j;
            const T fup_jm= (vyq > T(0)) ? fD_jm : fU_jm;
            const T G_jm  = vyq * fup_jm;

            const T adv_y = (G_jp - G_jm) / dy;

            adv = adv_x + adv_y;
        }

        // ---- collision term ----
        const T dvx_ = vxp - ux;
        const T dvy_ = vyq - uy;
        const T expo = -(dvx_ * dvx_ + dvy_ * dvy_) / (T(2.0) * Tgas);
        const T fM   = coeff * exp(expo);

        const T out = fij + dt * ( -adv + (fM - fij) * inv_tau );
        fn[base + k] = out;
    }
}

// host-side launcher
template <typename T>
void launch_explicit_step_2d2v(
    const T* f, T* fn,
    const T* vx, const T* vy,
    int nx, int ny, int nvx, int nvy,
    T dvx, T dvy,
    T dt, T dx, T dy,
    T tau_tilde,
    int scheme,
    int bc_x,
    int bc_y,
    cudaStream_t stream
) {
    const int blocks = nx * ny;
    const int threads = 256;

    // shared mem: 4 arrays * threads + scalars
    const size_t shmem = sizeof(T) * (threads * 4 + 8);

    explicit_step_2d2v_kernel<T><<<blocks, threads, shmem, stream>>>(
        f, fn, vx, vy,
        nx, ny, nvx, nvy,
        dvx, dvy,
        dt, dx, dy,
        tau_tilde,
        scheme, bc_x, bc_y
    );
}

// explicit instantiation (FP64)
template void launch_explicit_step_2d2v<double>(
    const double*, double*,
    const double*, const double*,
    int, int, int, int,
    double, double,
    double, double, double,
    double,
    int, int, int,
    cudaStream_t
);

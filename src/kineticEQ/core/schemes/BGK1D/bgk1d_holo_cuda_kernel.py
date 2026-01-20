# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_holo_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch
import logging

from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V

from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_streaming import _compute_streaming_upwind
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_maxwellian import maxwellian

from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_holo_ws import HoloWorkspace, allocate_holo_workspace

from kineticEQ.cuda_kernel.compile import load_gtsv
# 重要: compile.py 側の関数名が違う場合はここを合わせてください
from kineticEQ.cuda_kernel.compile import load_lo_blocktridiag

logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]


# ----------------------------
# HO-side helpers (ported)
# ----------------------------
@torch.no_grad()
def _HO_calculate_moments_face(
    state: State1D1V,
    f_z: torch.Tensor,
    *,
    SVdown: bool,
    S1_out: torch.Tensor,
    S2_out: torch.Tensor,
    S3_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    元クラスの _HO_calculate_moments を移植:
      upwind により界面 i+1/2 のフラックスモーメント S1,S2,S3 (shape nx-1) を計算。
    SVdown=False の場合:
      (w1,w2,w3) = (v, v^2, 0.5 v^3) で、LO側の F^{HO}_{i+1/2}=[S1,S2,S3+2Q] と整合。
    """
    nx, nv = f_z.shape
    dv = float(state.dv)
    v = state.v  # (nv,)

    fL = f_z[:-1, :]
    fR = f_z[1:, :]

    # f_up: (nx-1, nv)
    f_up = torch.empty((nx - 1, nv), device=f_z.device, dtype=f_z.dtype)
    f_up[:, state.pos_mask] = fL[:, state.pos_mask]
    f_up[:, state.neg_mask] = fR[:, state.neg_mask]

    if SVdown:
        w1 = torch.ones_like(v)
        w2 = v
        w3 = 0.5 * v * v
    else:
        w1 = v
        w2 = v * v
        w3 = 0.5 * v * v * v

    # out buffers are (nx-1,)
    torch.sum(f_up * w1[None, :], dim=1, out=S1_out)
    S1_out.mul_(dv)

    torch.sum(f_up * w2[None, :], dim=1, out=S2_out)
    S2_out.mul_(dv)

    torch.sum(f_up * w3[None, :], dim=1, out=S3_out)
    S3_out.mul_(dv)

    return S1_out, S2_out, S3_out


@torch.no_grad()
def _HO_calculate_fluxes_Q(
    state: State1D1V,
    f_z: torch.Tensor,
    Q_out: torch.Tensor,
) -> torch.Tensor:
    """
    元クラスの _HO_calculate_fluxes を移植:
      Q_i = 0.5 * ∫ (v - u_i)^3 f_i(v) dv
    """
    dv = float(state.dv)
    # u from f_z moments
    _, u, _ = calculate_moments(state, f_z)

    diff = state.v[None, :] - u[:, None]
    diff3 = diff * diff * diff

    torch.sum(diff3 * f_z, dim=1, out=Q_out)
    Q_out.mul_(0.5 * dv)
    return Q_out


@torch.no_grad()
def _compute_explicit_term(
    state: State1D1V,
    cfg: Config,
    theta: float,
    out: torch.Tensor,
    fM_buf: torch.Tensor,
) -> torch.Tensor:
    """
    元クラスの _compute_Explicit_term を移植:
      Explicit_term = f^k + (1-theta)*dt * (streaming(f^k) + collision(f^k))
    """
    dt = float(cfg.model_cfg.time.dt)

    # moments & tau(f^k)
    n, u, T = calculate_moments(state, state.f)
    # τ = tau_tilde / (n * sqrt(T))
    tau = cfg.model_cfg.params.tau_tilde / (n * torch.sqrt(T))

    # Maxwellian at time k
    state.n, state.u, state.T = n, u, T
    fM_buf.copy_(maxwellian(state))

    streaming = _compute_streaming_upwind(state, state.f)
    collision = (fM_buf - state.f) / tau[:, None]
    rhs = streaming + collision

    out.copy_(state.f)
    out.add_(rhs, alpha=(1.0 - theta) * dt)
    return out


@torch.no_grad()
def _compute_Y_I_terms(
    state: State1D1V,
    cfg: Config,
    ws: HoloWorkspace,
    f_z: torch.Tensor,
    n_HO: torch.Tensor,
    u_HO: torch.Tensor,
    T_HO: torch.Tensor,
    Q_HO: torch.Tensor,
    theta: float,
    *,
    flux_consistency_do: bool,
    SVdown: bool,
) -> torch.Tensor:
    """
    右辺整合項ベクトル Y_I_terms (= dt*(y_i + I_i)) を元実装の形で移植。
    """
    device = f_z.device
    dtype = f_z.dtype

    nx, nv = f_z.shape
    dv = float(state.dv)
    dx = float(state.dx)
    dt = float(cfg.model_cfg.time.dt)

    Y = ws.Y_I_terms
    Y.zero_()

    # ---- 1) flux consistency y_i via gamma_face ----
    S1, S2, S3 = _HO_calculate_moments_face(
        state, f_z, SVdown=SVdown, S1_out=ws.S1_face, S2_out=ws.S2_face, S3_out=ws.S3_face
    )

    # new HO moments from f_z
    n_new, u_new, T_new = calculate_moments(state, f_z)
    ws.n_new.copy_(n_new)
    ws.u_new.copy_(u_new)
    ws.T_new.copy_(T_new)

    P_new = ws.n_new * ws.T_new
    nu_new = ws.n_new * ws.u_new
    U_new = 0.5 * ws.n_new * (ws.u_new * ws.u_new + ws.T_new)

    # LO-form flux constructed from "new" HO moments
    nu_half = 0.5 * (nu_new[:-1] + nu_new[1:])
    mom_flux_center = nu_new * nu_new / (ws.n_new + 1e-30) + P_new
    mom_flux_half = 0.5 * (mom_flux_center[:-1] + mom_flux_center[1:])

    energy_flux_center = (U_new + P_new) * ws.u_new + Q_HO
    energy_flux_half = 0.5 * (energy_flux_center[:-1] + energy_flux_center[1:])

    F_LO_half = ws.F_HO_half  # reuse buffer shape (nx-1,3)
    F_LO_half[:, 0] = nu_half
    F_LO_half[:, 1] = mom_flux_half
    F_LO_half[:, 2] = energy_flux_half

    # S_face vector
    S_face = torch.stack((S1, S2, S3), dim=1)  # (nx-1,3)
    gamma_face = -S_face + F_LO_half

    if nx > 2 and flux_consistency_do:
        y_internal = (gamma_face[1:, :] - gamma_face[:-1, :]) / dx  # (nx-2,3)
        Y[1:-1, :].add_(y_internal, alpha=dt)

    # ---- 2) collision consistency I_i via moment residual of C_mix ----
    # C_new from f_z
    tau_new = cfg.model_cfg.params.tau_tilde / (ws.n_new * torch.sqrt(ws.T_new))
    state.n, state.u, state.T = ws.n_new, ws.u_new, ws.T_new
    ws.fM_new.copy_(maxwellian(state))
    C_new = (ws.fM_new - f_z) / tau_new[:, None]

    # C_old from f^k (HO old moments are given)
    tau_old = cfg.model_cfg.params.tau_tilde / (n_HO * torch.sqrt(T_HO))
    state.n, state.u, state.T = n_HO, u_HO, T_HO
    ws.fM_old.copy_(maxwellian(state))
    C_old = (ws.fM_old - state.f) / tau_old[:, None]

    C_mix = theta * C_new + (1.0 - theta) * C_old

    v = state.v
    w1 = torch.ones_like(v)
    w2 = v
    w3 = 0.5 * v * v

    M_n = torch.sum(C_mix * w1[None, :], dim=1) * dv
    M_nu = torch.sum(C_mix * w2[None, :], dim=1) * dv
    M_U = torch.sum(C_mix * w3[None, :], dim=1) * dv

    Y[:, 0].add_(M_n, alpha=dt)
    Y[:, 1].add_(M_nu, alpha=dt)
    Y[:, 2].add_(M_U, alpha=dt)

    return Y


# ----------------------------
# LO moment solve (ported)
# ----------------------------
@torch.no_grad()
def _LO_calculate_moments_picard(
    state: State1D1V,
    cfg: Config,
    ws: HoloWorkspace,
    lo_block_module,
    n_HO: torch.Tensor,
    u_HO: torch.Tensor,
    T_HO: torch.Tensor,
    Q_HO: torch.Tensor,
    S1_HO: torch.Tensor,
    S2_HO: torch.Tensor,
    S3_HO: torch.Tensor,
    Y_I_terms: torch.Tensor,
    *,
    lo_iter: int,
    lo_tol: float,
    lo_abs_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int, float]:
    """
    元クラスの _LO_calculate_moments を、workspace 利用前提で移植。
    3x3 block tridiagonal は CUDA 拡張 lo_blocktridiag に投げる。
    """
    device = state.f.device
    dtype = state.f.dtype

    nx = state.f.shape[0]
    dt = float(cfg.model_cfg.time.dt)
    dx = float(state.dx)
    tau_tilde = float(cfg.model_cfg.params.tau_tilde)

    n_inner = max(nx - 2, 0)

    dt_over_4dx = dt / (4.0 * dx)
    dt_over_2dx = dt / (2.0 * dx)

    # W_HO = [n, nu, U]
    W_HO = ws.W_HO
    W_HO[:, 0] = n_HO
    W_HO[:, 1] = n_HO * u_HO
    W_HO[:, 2] = 0.5 * n_HO * (u_HO * u_HO + T_HO)

    W_m = ws.W_m
    W_m.copy_(W_HO)

    # Q_half and F_HO_half = [S1,S2,S3+2Q_half]
    Q_half = ws.Q_half
    Q_half.copy_(0.5 * (Q_HO[:-1] + Q_HO[1:]))

    F_HO_half = ws.F_HO_half
    F_HO_half[:, 0] = S1_HO
    F_HO_half[:, 1] = S2_HO
    F_HO_half[:, 2] = S3_HO + 2.0 * Q_half

    I3 = torch.eye(3, device=device, dtype=dtype)

    lo_residual_val: float = float("inf")
    lo_iter_done: int = 0

    # buffers
    A_int = ws.A_int
    b_int = ws.b_int

    AA = ws.AA
    BB = ws.BB
    CC = ws.CC
    DD = ws.DD

    for m in range(lo_iter):
        lo_iter_done = m + 1

        n_m = W_m[:, 0]
        nu_m = W_m[:, 1]
        U_m = W_m[:, 2]

        denom = n_m + 1e-30
        u_star = nu_m / denom
        T_star = 2.0 * (U_m / denom - 0.5 * u_star * u_star)
        T_star = torch.clamp(T_star, min=1e-30)
        P_star = n_m * T_star

        u_half_p = 0.5 * (u_star[:-1] + u_star[1:])
        P_half_p = 0.5 * (P_star[:-1] + P_star[1:])

        # A_int, b_int build
        A_int.zero_()
        b_int.zero_()

        A_int[:, 0, 1] = 1.0
        A_int[:, 1, 0] = u_half_p * u_half_p
        A_int[:, 2, 2] = u_half_p

        b_int[:, 1] = P_half_p
        b_int[:, 2] = u_half_p * P_half_p

        if n_inner <= 0:
            ws.W_full.copy_(W_HO)
        else:
            # internal slices
            A_L = A_int[:-1]   # (n_inner,3,3)
            A_R = A_int[1:]    # (n_inner,3,3)
            b_L = b_int[:-1]   # (n_inner,3)
            b_R = b_int[1:]    # (n_inner,3)

            F_L = F_HO_half[:-1]  # (n_inner,3)
            F_R = F_HO_half[1:]   # (n_inner,3)
            F_diff = F_R - F_L
            b_diff = b_R - b_L

            coef = dt_over_4dx

            AA.zero_()
            CC.zero_()

            # AA[1:] = -coef * A_L[1:]
            if n_inner > 1:
                AA[1:].copy_(A_L[1:])
                AA[1:].mul_(-coef)

            # CC[:-1] = coef * A_R[:-1]
            if n_inner > 1:
                CC[:-1].copy_(A_R[:-1])
                CC[:-1].mul_(coef)

            # BB = I3 + coef*(A_R - A_L)
            BB.copy_(A_R)
            BB.sub_(A_L)
            BB.mul_(coef)
            BB.add_(I3[None, :, :])

            # DD = W_HO[1:-1] - dt/2dx * F_diff - dt/2dx * b_diff + Y_I_terms[1:-1]
            DD.copy_(W_HO[1:-1])
            DD.add_(F_diff, alpha=-dt_over_2dx)
            DD.add_(b_diff, alpha=-dt_over_2dx)
            DD.add_(Y_I_terms[1:-1])

            # solve block tridiag
            X_inner = lo_block_module.block_tridiag_solve(
                AA.contiguous(),
                BB.contiguous(),
                CC.contiguous(),
                DD.contiguous(),
            )
            ws.X_inner.copy_(X_inner)

            ws.W_full.copy_(W_HO)
            ws.W_full[1:-1, :].copy_(ws.X_inner)

        # residual
        diff = torch.abs(ws.W_full - W_m)
        ref  = torch.maximum(torch.abs(ws.W_full), torch.abs(W_m))
        den  = lo_abs_tol + lo_tol * ref
        lo_residual_val = float(torch.max(diff).item())
        std_lo_residual_val = float(torch.max(diff / den).item())
        W_m.copy_(ws.W_full)

        if std_lo_residual_val < 1.0:
            break

    # back to (n,u,T,tau)
    n_lo = ws.W_m[:, 0]
    nu_lo = ws.W_m[:, 1]
    U_lo = ws.W_m[:, 2]

    denom = n_lo + 1e-30
    u_lo = nu_lo / denom
    T_lo = 2.0 * (U_lo / denom - 0.5 * u_lo * u_lo)
    T_lo = torch.clamp(T_lo, min=1e-30)

    tau_lo = tau_tilde / (n_lo * torch.sqrt(T_lo))

    ws.n_lo.copy_(n_lo)
    ws.u_lo.copy_(u_lo)
    ws.T_lo.copy_(T_lo)
    ws.tau_lo.copy_(tau_lo)

    return ws.n_lo, ws.u_lo, ws.T_lo, ws.tau_lo, lo_residual_val, lo_iter_done, std_lo_residual_val


# ----------------------------
# HOLO step (ported)
# ----------------------------
@torch.no_grad()
def step(
    state: State1D1V,
    cfg: Config,
    ws: HoloWorkspace,
    gtsv_module,
    lo_block_module,
    num_steps: int,
) -> tuple[State1D1V, dict]:
    """
    元クラス _implicit_update_holo の移植版。
    - HOLO outer loop: ho_iter / ho_tol
    - LO Picard inner: lo_iter / lo_tol
    - distribution update: batched gtsv_strided
    - consistency terms: Y_I_terms (optional)
    """
    sp = cfg.model_cfg.scheme_params

    ho_iter = int(getattr(sp, "ho_iter", 8))
    ho_tol = float(getattr(sp, "ho_tol", 1e-4))
    ho_abs_tol = float(getattr(sp, "ho_abs_tol", 1e-12))
    lo_iter = int(getattr(sp, "lo_iter", 16))
    lo_tol = float(getattr(sp, "lo_tol", 1e-4))
    lo_abs_tol = float(getattr(sp, "lo_abs_tol", 1e-12))

    # flags (元クラスの挙動に合わせるための互換フラグ)
    Con_Terms_do = bool(getattr(sp, "Con_Terms_do", True))
    flux_consistency_do = bool(getattr(sp, "flux_consistency_do", True))
    SVdown = bool(getattr(sp, "SVdown", False))

    # theta (暫定: 強制 Crank–Nicolson)
    theta = 0.5

    # init
    ws.fz.copy_(state.f)
    ws.fn_tmp.copy_(state.f)

    # HO old moments (k)
    n_HO, u_HO, T_HO = calculate_moments(state, state.f)
    ws.n_HO.copy_(n_HO)
    ws.u_HO.copy_(u_HO)
    ws.T_HO.copy_(T_HO)

    # HO higher moments on f^k (界面)
    _HO_calculate_moments_face(
        state, state.f, SVdown=SVdown,
        S1_out=ws.S1_face, S2_out=ws.S2_face, S3_out=ws.S3_face
    )

    # Explicit_term on f^k
    _compute_explicit_term(state, cfg, theta, ws.explicit_term, ws.fM_old)

    dt = float(cfg.model_cfg.time.dt)
    dx = float(state.dx)

    # alpha, beta depend only on v, dt, dx (reuse tensors on-the-fly)
    v = state.v
    alpha = (dt / (2.0 * dx)) * torch.clamp(v, min=0.0)     # (nv,)
    beta = (dt / (2.0 * dx)) * torch.clamp(-v, min=0.0)     # (nv,)

    ho_residual_val = float("inf")
    lo_residual_val = float("inf")
    lo_iter_done = 0
    max_YI = 0.0

    latest = ws.fz  # will be overwritten

    for z in range(ho_iter):
        # Q_HO from current fz
        _HO_calculate_fluxes_Q(state, ws.fz, ws.Q_HO)

        # Y_I_terms
        if Con_Terms_do:
            _compute_Y_I_terms(
                state, cfg, ws, ws.fz,
                ws.n_HO, ws.u_HO, ws.T_HO,
                ws.Q_HO, theta,
                flux_consistency_do=flux_consistency_do,
                SVdown=SVdown,
            )
        else:
            ws.Y_I_terms.zero_()

        max_YI = float(torch.max(torch.abs(ws.Y_I_terms)).item())

        # LO moment solve
        n_lo, u_lo, T_lo, tau_lo, lo_residual_val, lo_iter_done, std_lo_residual_val = _LO_calculate_moments_picard(
            state, cfg, ws, lo_block_module,
            ws.n_HO, ws.u_HO, ws.T_HO,
            ws.Q_HO,
            ws.S1_face, ws.S2_face, ws.S3_face,
            ws.Y_I_terms,
            lo_iter=lo_iter,
            lo_tol=lo_tol,
            lo_abs_tol=lo_abs_tol,
        )

        # Maxwellian from LO moments
        state.n, state.u, state.T = n_lo, u_lo, T_lo
        ws.fM_lo.copy_(maxwellian(state))

        # build tridiag for distribution update (matches 元クラス式)
        n_inner = max(state.f.shape[0] - 2, 0)
        if n_inner <= 0:
            ws.fn_tmp.copy_(ws.fz)
            latest = ws.fn_tmp
            ho_residual_val = 0.0
            break

        # dl, du constant across x except boundary zeroing (matches 元実装)
        ws.du[:, :].copy_(-beta[:, None])
        ws.du[:, -1] = 0.0

        ws.dl[:, :].copy_(-alpha[:, None])
        ws.dl[:, 0] = 0.0

        # dd = 1 + alpha + beta + theta*(dt/tau_lo[1:-1])
        ws.dd[:, :].copy_(1.0 + alpha[:, None] + beta[:, None])
        ws.dd.add_((theta * dt) / tau_lo[1:-1][None, :])

        # B = (Explicit_term + theta*(dt/tau_lo)*fM_lo) on inner, then transpose to (nv,n_inner)
        # careful broadcasting: tau_lo[1:-1] is (n_inner,)
        B_inner = ws.explicit_term[1:-1, :] + (theta * dt) * (ws.fM_lo[1:-1, :] / tau_lo[1:-1, None])
        ws.B.copy_(B_inner.T)

        # boundary contributions
        ws.B[:, 0].add_(alpha * state.f[0, :])
        ws.B[:, -1].add_(beta * state.f[-1, :])

        # solve
        solution = gtsv_module.gtsv_strided(
            ws.dl.contiguous(),
            ws.dd.contiguous(),
            ws.du.contiguous(),
            ws.B.contiguous(),
        )  # (nv, nx-2)

        # writeback
        ws.fn_tmp.copy_(ws.fz)
        ws.fn_tmp[1:-1, :].copy_(solution.T)

        # residual & convergence
        df  = torch.abs(ws.fn_tmp - ws.fz)
        ref = torch.maximum(torch.abs(ws.fn_tmp), torch.abs(ws.fz))
        den = ho_abs_tol + ho_tol * ref

        
        ho_res = torch.max(df / den)
        ho_residual_val = float((torch.max(df) / torch.max(ref)).item())
        latest = ws.fn_tmp
        ho_std_residual_val = float(ho_res.item())

        if ho_res <= 1.0:
            break

        # next iteration
        ws.fz, ws.fn_tmp = ws.fn_tmp, ws.fz

    # boundary fixed
    state.f_tmp.copy_(latest)
    state.f_tmp[0, :].copy_(state.f[0, :])
    state.f_tmp[-1, :].copy_(state.f[-1, :])

    # NaN/Inf check (同一方針)
    if num_steps % 100 == 0:
        logger.debug(f"NaN/Inf check executed at step: {num_steps}")
        if not torch.isfinite(state.f_tmp).all():
            raise ValueError("NaN/Inf detected in f_tmp")

    # swap
    state.f, state.f_tmp = state.f_tmp, state.f

    benchlog = {
        "ho_iter": z + 1,
        "ho_residual": ho_residual_val,
        "std_ho_residual": ho_std_residual_val,
        "lo_iter": lo_iter_done,
        "lo_residual": lo_residual_val,
        "std_lo_residual":std_lo_residual_val,
        "max_YI": max_YI,
    }
    return state, benchlog


def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # CFL check (同一方針)
    bgk1d_check_CFL(cfg)

    # modules
    gtsv_module = load_gtsv()
    lo_block_module = load_lo_blocktridiag()

    # workspace
    nx, nv = state.f.shape
    ws = allocate_holo_workspace(nx, nv, state.f.device, state.f.dtype)

    # initial condition
    set_initial_condition(state, cfg)

    def _stepper(num_steps: int) -> None:
        _, benchlog = step(state, cfg, ws, gtsv_module, lo_block_module, num_steps)
        _stepper.benchlog = benchlog

    _stepper.benchlog = None
    return _stepper

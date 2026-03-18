from __future__ import annotations

from typing import Callable
import math

import torch

from kineticEQ.CNN.BGK1D1V.util.models import MomentCNN1D
from kineticEQ.api.config import Config
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.general.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.implicit import bgk1d_aa_cpu
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.implicit.bgk1d_conv_util import check_convergence
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.implicit.bgk1d_implicit_ws import ImplicitWorkspace, allocate_implicit_workspace
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.implicit.bgk1d_momentCNN_util import (
    load_moments_cnn_model,
    predict_next_moments_delta,
)
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.cpu_kernel.compile import load_gtsv_cpu, load_implicit_fused_cpu

import logging

logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]


@torch.no_grad()
def step(
    state: State1D1V,
    cfg: Config,
    ws: ImplicitWorkspace,
    cpu_module,
    gtsv_module,
    num_steps: int,
    inv_sqrt_2pi: float,
    model: MomentCNN1D | None = None,
    model_meta: dict | None = None,
    warmstart_enabled: bool = False,
) -> tuple[State1D1V, dict]:
    dt = float(cfg.model_cfg.time.dt)
    dx = float(state.dx)
    dv = float(state.dv)
    tau_tilde = float(cfg.model_cfg.params.tau_tilde)
    n_floor = 1e-12
    T_floor = 1e-12

    picard_iter = cfg.model_cfg.scheme_params.picard_iter
    picard_tol = cfg.model_cfg.scheme_params.picard_tol
    abs_tol = cfg.model_cfg.scheme_params.abs_tol
    conv_type = str(getattr(cfg.model_cfg.scheme_params, "conv_type", "f")).lower()
    if conv_type not in ("f", "w"):
        raise ValueError(f"invalid conv_type={conv_type!r}, expected 'f' or 'w'")

    aa_enable_cfg = bool(getattr(cfg.model_cfg.scheme_params, "aa_enable", False))
    aa_m = max(int(getattr(cfg.model_cfg.scheme_params, "aa_m", 0)), 0)
    aa_beta = min(max(float(getattr(cfg.model_cfg.scheme_params, "aa_beta", 1.0)), 0.0), 1.0)
    aa_stride = max(int(getattr(cfg.model_cfg.scheme_params, "aa_stride", 1)), 1)
    aa_start_iter = max(int(getattr(cfg.model_cfg.scheme_params, "aa_start_iter", 2)), 1)
    aa_reg = max(float(getattr(cfg.model_cfg.scheme_params, "aa_reg", 1e-10)), 1e-14)
    aa_alpha_max = max(float(getattr(cfg.model_cfg.scheme_params, "aa_alpha_max", 50.0)), 0.0)

    n_inner = max(int(ws.B.shape[1]), 0)
    aa_cols = int(ws.aa_G.shape[1]) if ws.aa_G.dim() == 2 else 0
    aa_enable = aa_enable_cfg and n_inner > 0 and aa_m >= 1 and aa_cols >= 2
    aa_hist_len = 0
    aa_head = 0
    u_max = 0.95 * float(torch.max(torch.abs(state.v)).item()) if n_inner > 0 else 0.0

    f_bc_l = state.f[0, :].contiguous()
    f_bc_r = state.f[-1, :].contiguous()

    if ws.B0.numel() > 0:
        ws.B0.copy_(state.f[1:-1, :].T)

    ws.fz.copy_(state.f)
    cpu_module.moments_n_nu_T(ws.fz, state.v, dv, ws.n, ws.nu, ws.T)
    n0_hist = ws.n.clone()
    n0_hist_safe = torch.clamp(n0_hist, min=n_floor)
    u0_hist = ws.nu / n0_hist_safe
    T0_hist = ws.T.clone()
    prev_hist = getattr(ws, "_warm_prev_W", None)
    if isinstance(prev_hist, tuple) and len(prev_hist) == 3:
        prev_n0, prev_u0, prev_T0 = prev_hist
        has_prev_hist = True
    else:
        prev_n0 = None
        prev_u0 = None
        prev_T0 = None
        has_prev_hist = False
    ws._warm_prev_W = (n0_hist, u0_hist.clone(), T0_hist)

    external_W_injected = False
    init_W = getattr(ws, "_init_W", None)
    if init_W is not None:
        n_init, nu_init, T_init = init_W
        n_safe = torch.clamp(n_init, min=n_floor)
        T_safe = torch.clamp(T_init, min=T_floor)
        n_den = torch.where(
            torch.abs(n_init) >= n_floor,
            n_init,
            torch.full_like(n_init, n_floor),
        )
        u_init = torch.nan_to_num(nu_init / n_den, nan=0.0, posinf=0.0, neginf=0.0)

        ws.n.copy_(n_safe)
        ws.nu.copy_(n_safe * u_init)
        ws.T.copy_(T_safe)
        ws._init_W = None
        external_W_injected = True

    if warmstart_enabled and (not external_W_injected):
        if model is None:
            raise RuntimeError("moments_cnn_modelpath is set but model is None")

        n0 = ws.n
        n0_safe = torch.clamp(n0, min=n_floor)
        u0 = ws.nu / n0_safe
        T0 = ws.T

        n1p = n0.clone()
        u1p = u0.clone()
        T1p = T0.clone()

        dtp = (model_meta or {}).get("delta_type", "dnu")
        input_state_type = (model_meta or {}).get("input_state_type", "nut")
        input_temporal_mode = (model_meta or {}).get("input_temporal_mode", "none")
        n1_int, u1_int, T1_int, _, _, _ = predict_next_moments_delta(
            model,
            n0,
            u0,
            T0,
            math.log10(dt),
            math.log10(tau_tilde),
            delta_type=dtp,
            input_state_type=input_state_type,
            input_temporal_mode=input_temporal_mode,
            prev_n=prev_n0,
            prev_u=prev_u0,
            prev_T=prev_T0,
            has_prev=has_prev_hist,
        )

        n1p[1:-1].copy_(torch.clamp(n1_int, min=n_floor))
        u1p[1:-1].copy_(u1_int)
        T1p[1:-1].copy_(torch.clamp(T1_int, min=T_floor))

        ws.n.copy_(n1p)
        ws.nu.copy_(n1p * u1p)
        ws.T.copy_(T1p)

    residual_val = float("inf")
    std_residual_val = float("inf")
    latest = ws.fz

    for z in range(picard_iter):
        cpu_module.build_system_from_moments(
            ws.B0,
            state.v,
            dt,
            dx,
            tau_tilde,
            inv_sqrt_2pi,
            ws.n,
            ws.nu,
            ws.T,
            f_bc_l,
            f_bc_r,
            ws.dl,
            ws.dd,
            ws.du,
            ws.B,
        )

        gtsv_module.gtsv_strided_inplace(
            ws.dl.contiguous(),
            ws.dd.contiguous(),
            ws.du.contiguous(),
            ws.B.contiguous(),
            ws.gtsv_ws,
        )

        ws.fn_tmp[1:-1, :].copy_(ws.B.T)
        ws.fn_tmp[0, :].copy_(f_bc_l)
        ws.fn_tmp[-1, :].copy_(f_bc_r)

        cpu_module.moments_n_nu_T(ws.fn_tmp, state.v, dv, ws.n_new, ws.nu_new, ws.T_new)
        residual, residual_val, std_residual_val = check_convergence(
            conv_type=conv_type,
            f_new=ws.fn_tmp,
            f_old=ws.fz,
            n_new=ws.n_new,
            n_old=ws.n,
            nu_new=ws.nu_new,
            nu_old=ws.nu,
            T_new=ws.T_new,
            T_old=ws.T,
            abs_tol=abs_tol,
            picard_tol=picard_tol,
        )

        latest = ws.fn_tmp
        if residual <= 1.0:
            break

        if z + 1 < picard_iter:
            if aa_enable:
                aa_should_apply = (z + 1) >= aa_start_iter and ((z + 1 - aa_start_iter) % aa_stride == 0)
                aa_hist_len, aa_head = bgk1d_aa_cpu.step_inplace(
                    ws.n,
                    ws.nu,
                    ws.T,
                    ws.n_new,
                    ws.nu_new,
                    ws.T_new,
                    ws.aa_G,
                    ws.aa_R,
                    ws.aa_A,
                    ws.aa_alpha,
                    ws.aa_wk,
                    ws.aa_wnew,
                    ws.aa_wtmp,
                    int(aa_hist_len),
                    int(aa_head),
                    bool(aa_should_apply),
                    float(aa_beta),
                    float(aa_reg),
                    float(aa_alpha_max),
                    float(n_floor),
                    float(T_floor),
                    float(u_max),
                    ws.aa_solver_work,
                    ws.aa_solver_info,
                    ws.aa_G_work,
                    ws.aa_R_work,
                )
            else:
                ws.n.copy_(torch.clamp(ws.n_new, min=n_floor))
                ws.nu.copy_(ws.nu_new)
                ws.T.copy_(torch.clamp(ws.T_new, min=T_floor))

        ws.fz, ws.fn_tmp = ws.fn_tmp, ws.fz

    state.f_tmp.copy_(latest)
    state.f_tmp[0, :].copy_(state.f[0, :])
    state.f_tmp[-1, :].copy_(state.f[-1, :])

    if num_steps % 100 == 0 and not torch.isfinite(state.f_tmp).all():
        raise ValueError("NaN/Inf detected in f_tmp")

    state.f, state.f_tmp = state.f_tmp, state.f
    benchlog = {
        "picard_iter": z + 1,
        "picard_residual": residual_val,
        "std_picard_residual": std_residual_val,
    }
    return state, benchlog


def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    if state.f.device.type != "cpu":
        raise RuntimeError("backend='cpu_kernel' requires device='cpu'")

    cpu_module = load_implicit_fused_cpu()
    gtsv_module = load_gtsv_cpu()

    aa_enable_cfg = bool(getattr(cfg.model_cfg.scheme_params, "aa_enable", False))
    aa_m = max(int(getattr(cfg.model_cfg.scheme_params, "aa_m", 0)), 0)

    nx, nv = state.f.shape
    ws = allocate_implicit_workspace(
        nx,
        nv,
        state.f.device,
        state.f.dtype,
        aa_m=(aa_m if aa_enable_cfg else 0),
    )

    if aa_enable_cfg and ws.aa_A.numel() > 0:
        ws.aa_G_work = torch.empty_like(ws.aa_G)
        ws.aa_R_work = torch.empty_like(ws.aa_R)
        ws.aa_solver_work = torch.empty((1,), device=state.f.device, dtype=state.f.dtype)
        ws.aa_solver_info = torch.zeros((1,), device=state.f.device, dtype=torch.int32)

    if ws.B.shape[1] > 0:
        ws_bytes = int(
            gtsv_module.gtsv_ws_bytes(
                ws.dl.contiguous(),
                ws.dd.contiguous(),
                ws.du.contiguous(),
                ws.B.contiguous(),
            )
        )
        ws.gtsv_ws = torch.empty((max(ws_bytes, 1),), device=state.f.device, dtype=torch.uint8)
    else:
        ws.gtsv_ws = torch.empty((1,), device=state.f.device, dtype=torch.uint8)

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    scheme_params = cfg.model_cfg.scheme_params
    model_path = getattr(scheme_params, "moments_cnn_modelpath", None)
    has_model_path = (model_path is not None) and (str(model_path) != "")

    warm_enable_cfg = getattr(scheme_params, "warm_enable", None)
    if warm_enable_cfg is None:
        warm_enable_cfg = getattr(scheme_params, "Warm_enable", None)

    if warm_enable_cfg is None:
        warmstart_enabled = has_model_path
    else:
        warmstart_enabled = bool(warm_enable_cfg) and has_model_path
        if bool(warm_enable_cfg) and (not has_model_path):
            logger.warning("warm_enable=True but moments_cnn_modelpath is not set; warm-start is disabled")

    if warmstart_enabled:
        model, model_meta = load_moments_cnn_model(str(model_path), device=state.f.device)
    else:
        model = None
        model_meta = {}

    set_initial_condition(state, cfg)

    def _stepper(num_steps: int) -> None:
        _, benchlog = step(
            state,
            cfg,
            ws,
            cpu_module,
            gtsv_module,
            num_steps,
            inv_sqrt_2pi,
            model,
            model_meta,
            warmstart_enabled,
        )
        _stepper.benchlog = benchlog

    _stepper.benchlog = None
    _stepper.ws = ws
    return _stepper

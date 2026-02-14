# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_implicit_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch, math
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_implicit_ws import ImplicitWorkspace, allocate_implicit_workspace
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.cuda_kernel.compile import load_implicit_fused
from kineticEQ.cuda_kernel.compile import load_gtsv
from kineticEQ.CNN.BGK1D1V.models import MomentCNN1D

# CNN用ユーティリティ
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_momentCNN_util import (
    load_moments_cnn_model, predict_next_moments_delta
)

import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]


def _pack_W_interior(n: torch.Tensor, nu: torch.Tensor, T: torch.Tensor, out: torch.Tensor) -> None:
    n_inner = max(int(n.numel()) - 2, 0)
    if n_inner <= 0:
        return
    out[:n_inner].copy_(n[1:-1])
    out[n_inner:2 * n_inner].copy_(nu[1:-1])
    out[2 * n_inner:3 * n_inner].copy_(T[1:-1])


def _unpack_W_interior(w: torch.Tensor, n: torch.Tensor, nu: torch.Tensor, T: torch.Tensor) -> None:
    n_inner = max(int(n.numel()) - 2, 0)
    if n_inner <= 0:
        return
    n[1:-1].copy_(w[:n_inner])
    nu[1:-1].copy_(w[n_inner:2 * n_inner])
    T[1:-1].copy_(w[2 * n_inner:3 * n_inner])


def _project_W_interior(
    n: torch.Tensor,
    nu: torch.Tensor,
    T: torch.Tensor,
    *,
    n_floor: float,
    T_floor: float,
    u_max: float,
) -> None:
    n_int = n[1:-1]
    nu_int = nu[1:-1]
    T_int = T[1:-1]

    n_int.clamp_(min=n_floor)
    T_int.clamp_(min=T_floor)

    u_int = nu_int / torch.clamp(n_int, min=n_floor)
    u_int.clamp_(min=-u_max, max=u_max)
    nu_int.copy_(n_int * u_int)

# implicit stepper
@torch.no_grad()
def step(
    state: State1D1V, 
    cfg: Config, 
    ws: ImplicitWorkspace, 
    cuda_module, 
    gtsv_module, 
    num_steps: int,
    inv_sqrt_2pi: float,
    model: MomentCNN1D | None = None,
    model_meta: dict | None = None,
) -> tuple[State1D1V, dict]:
    dt = float(cfg.model_cfg.time.dt)
    dx = float(state.dx)
    dv = float(state.dv)
    tau_tilde = float(cfg.model_cfg.params.tau_tilde)
    n_floor = 1e-12
    T_floor = 1e-12

    # scheme_params から取得
    picard_iter = cfg.model_cfg.scheme_params.picard_iter
    picard_tol = cfg.model_cfg.scheme_params.picard_tol
    abs_tol = cfg.model_cfg.scheme_params.abs_tol

    aa_enable_cfg = bool(getattr(cfg.model_cfg.scheme_params, "aa_enable", False))
    aa_m = max(int(getattr(cfg.model_cfg.scheme_params, "aa_m", 0)), 0)
    aa_beta = float(getattr(cfg.model_cfg.scheme_params, "aa_beta", 1.0))
    aa_beta = min(max(aa_beta, 0.0), 1.0)
    aa_reg = float(getattr(cfg.model_cfg.scheme_params, "aa_reg", 1e-10))
    aa_reg = max(aa_reg, 0.0)
    aa_alpha_max = float(getattr(cfg.model_cfg.scheme_params, "aa_alpha_max", 50.0))
    aa_alpha_max = max(aa_alpha_max, 1.0)

    n_inner = max(int(ws.B.shape[1]), 0)
    aa_cols = max(min(aa_m + 1, int(ws.aa_G.shape[1])), 1)
    aa_enable = aa_enable_cfg and (n_inner > 0) and (aa_m >= 1)
    aa_hist_len = 0
    aa_hist_ptr = 0
    aa_applied = 0
    aa_restarted = 0
    w_residual_val = float("inf")
    u_max = 0.95 * float(torch.max(torch.abs(state.v)).item()) if n_inner > 0 else 0.0

    # 初期候補 fz（外部フックがあればそれを優先）
    init_fz = getattr(ws, "_init_fz", None)
    if init_fz is None:
        ws.fz.copy_(state.f)
    else:
        ws.fz.copy_(init_fz)
        ws._init_fz = None

    # 固定境界 f（本step中は不変）
    f_bc_l = state.f[0, :].contiguous()
    f_bc_r = state.f[-1, :].contiguous()

    # f_prev interior を RHS 基底として1回だけ転置キャッシュ
    if ws.B0.numel() > 0:
        ws.B0.copy_(state.f[1:-1, :].T)

    # 初期 moments from fz
    cuda_module.moments_n_nu_T(ws.fz, state.v, dv, ws.n, ws.nu, ws.T)

    # 外部フック: W=(n,nu,T) を直接注入（teacher-forcing/eval用）
    external_W_injected = False
    init_W = getattr(ws, "_init_W", None)
    if init_W is not None:
        n_init, nu_init, T_init = init_W
        ws.n.copy_(torch.clamp(n_init, min=n_floor))
        ws.nu.copy_(nu_init)
        ws.T.copy_(torch.clamp(T_init, min=T_floor))
        ws._init_W = None
        external_W_injected = True

    # CNN warm-start: W=(n,nu,T) を直接更新
    if (cfg.model_cfg.scheme_params.moments_cnn_modelpath is not None) and (not external_W_injected):
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
        n1_int, u1_int, T1_int, _, _, _ = predict_next_moments_delta(
            model,
            n0,
            u0,
            T0,
            math.log10(dt),
            math.log10(tau_tilde),
            delta_type=dtp,
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
        # moments から (dl,dd,du,B) 構築（境界寄与は固定境界 f）
        cuda_module.build_system_from_moments(
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

        # cuSPARSE batched tridiagonal solve（workspace再利用）
        gtsv_module.gtsv_strided_inplace(
            ws.dl.contiguous(),
            ws.dd.contiguous(),
            ws.du.contiguous(),
            ws.B.contiguous(),
            ws.gtsv_ws,
        )

        # 内部セルのみ書き戻し。境界は固定境界 f を維持
        ws.fn_tmp[1:-1, :].copy_(ws.B.T)
        ws.fn_tmp[0, :].copy_(f_bc_l)
        ws.fn_tmp[-1, :].copy_(f_bc_r)

        # 正規化誤差
        df  = torch.abs(ws.fn_tmp - ws.fz)
        ref = torch.maximum(torch.abs(ws.fn_tmp), torch.abs(ws.fz))
        den = abs_tol + picard_tol * ref

        residual = torch.max(df / den)
        residual_val = float((torch.max(df) / torch.max(ref)).item())
        std_residual_val = float(residual.item())

        latest = ws.fn_tmp

        if residual <= 1.0:
            break

        # 次反復用 moments
        if z + 1 < picard_iter:
            cuda_module.moments_n_nu_T(ws.fn_tmp, state.v, dv, ws.n_new, ws.nu_new, ws.T_new)

            if aa_enable:
                _pack_W_interior(ws.n, ws.nu, ws.T, ws.aa_wk)
                _pack_W_interior(ws.n_new, ws.nu_new, ws.T_new, ws.aa_wnew)

                ws.aa_wtmp.copy_(ws.aa_wnew)
                ws.aa_wtmp.sub_(ws.aa_wk)
                w_residual_val = float(torch.max(torch.abs(ws.aa_wtmp)).item())

                ws.aa_G[:, aa_hist_ptr].copy_(ws.aa_wnew)
                ws.aa_R[:, aa_hist_ptr].copy_(ws.aa_wtmp)
                aa_hist_ptr = (aa_hist_ptr + 1) % aa_cols
                aa_hist_len = min(aa_hist_len + 1, aa_cols)

                aa_ok = False
                if aa_hist_len >= 2:
                    R_use = ws.aa_R[:, :aa_hist_len]
                    G_use = ws.aa_G[:, :aa_hist_len]
                    A = ws.aa_A[:aa_hist_len, :aa_hist_len]
                    alpha = ws.aa_alpha[:aa_hist_len]
                    ones = ws.aa_ones[:aa_hist_len, :]

                    A.copy_(R_use.T @ R_use)
                    A.diagonal().add_(aa_reg)

                    try:
                        L, info = torch.linalg.cholesky_ex(A)
                        if bool((info == 0).all().item()):
                            x = torch.cholesky_solve(ones, L)
                            denom = ones.T @ x
                            denom_v = float(denom[0, 0].item())
                            if math.isfinite(denom_v) and abs(denom_v) > 1e-30:
                                alpha.copy_((x / denom).squeeze(1))
                                if bool(torch.isfinite(alpha).all().item()):
                                    alpha_abs_max = float(torch.max(torch.abs(alpha)).item())
                                    if alpha_abs_max <= aa_alpha_max:
                                        ws.aa_wtmp.copy_(G_use @ alpha)
                                        ws.aa_wtmp.mul_(aa_beta)
                                        ws.aa_wtmp.add_(ws.aa_wnew, alpha=(1.0 - aa_beta))
                                        if bool(torch.isfinite(ws.aa_wtmp).all().item()):
                                            aa_ok = True
                    except RuntimeError:
                        aa_ok = False

                if aa_ok:
                    aa_applied += 1
                    ws.n.copy_(torch.clamp(ws.n_new, min=n_floor))
                    ws.nu.copy_(ws.nu_new)
                    ws.T.copy_(torch.clamp(ws.T_new, min=T_floor))
                    _unpack_W_interior(ws.aa_wtmp, ws.n, ws.nu, ws.T)
                    _project_W_interior(ws.n, ws.nu, ws.T, n_floor=n_floor, T_floor=T_floor, u_max=u_max)
                else:
                    if aa_hist_len >= 2:
                        aa_hist_len = 0
                        aa_hist_ptr = 0
                        aa_restarted += 1
                    ws.n.copy_(torch.clamp(ws.n_new, min=n_floor))
                    ws.nu.copy_(ws.nu_new)
                    ws.T.copy_(torch.clamp(ws.T_new, min=T_floor))
            else:
                ws.n.copy_(torch.clamp(ws.n_new, min=n_floor))
                ws.nu.copy_(ws.nu_new)
                ws.T.copy_(torch.clamp(ws.T_new, min=T_floor))

        # 次反復へ
        ws.fz, ws.fn_tmp = ws.fn_tmp, ws.fz
    
    state.f_tmp.copy_(latest)
    state.f_tmp[0, :].copy_(state.f[0, :])
    state.f_tmp[-1, :].copy_(state.f[-1, :])

    # NaN/Infチェック
    if num_steps % 100 == 0:
        logger.debug(f"NaN/Inf check executed at step: {num_steps}")
        if not torch.isfinite(state.f_tmp).all():
            raise ValueError("NaN/Inf detected in f_tmp")

    # swap
    state.f, state.f_tmp = state.f_tmp, state.f

    # benchlog
    benchlog = {
        "picard_iter": z + 1,
        "picard_residual": residual_val,
        "std_picard_residual": std_residual_val,
        "aa_enable": aa_enable,
        "aa_applied": aa_applied,
        "aa_restart": aa_restarted,
        "w_residual": w_residual_val,
    }

    return state, benchlog

def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # CFL条件チェック
    #bgk1d_check_CFL(cfg)

    # JITコンパイル
    cuda_module = load_implicit_fused()
    gtsv_module = load_gtsv()

    # implicit 専用ワークスペース確保
    nx, nv = state.f.shape
    aa_m = max(int(getattr(cfg.model_cfg.scheme_params, "aa_m", 0)), 0)
    ws = allocate_implicit_workspace(nx, nv, state.f.device, state.f.dtype, aa_m=aa_m)

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

    # モデルパスがあればロード
    if cfg.model_cfg.scheme_params.moments_cnn_modelpath is not None:
        model, model_meta = load_moments_cnn_model(
            cfg.model_cfg.scheme_params.moments_cnn_modelpath,
            device=state.f.device
        )
    else:
        model = None
        model_meta = {}

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        _, benchlog = step(
            state,
            cfg,
            ws,
            cuda_module,
            gtsv_module,
            num_steps,
            inv_sqrt_2pi,
            model,
            model_meta,
        )
        _stepper.benchlog = benchlog  # bench-logを属性として載せる

    _stepper.benchlog = None  # 初期値
    _stepper.ws = ws 
    return _stepper

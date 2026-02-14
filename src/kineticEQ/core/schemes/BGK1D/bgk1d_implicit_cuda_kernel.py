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
            cuda_module.moments_n_nu_T(ws.fn_tmp, state.v, dv, ws.n, ws.nu, ws.T)

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
    ws = allocate_implicit_workspace(nx, nv, state.f.device, state.f.dtype)

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

# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_implicit_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_implicit_ws import ImplicitWorkspace, allocate_implicit_workspace
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
from kineticEQ.cuda_kernel.compile import load_implicit_fused
from kineticEQ.cuda_kernel.compile import load_gtsv
import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]

@torch.no_grad()
def step(state: State1D1V, cfg: Config, ws: ImplicitWorkspace, cuda_module, gtsv_module, num_steps: int) -> tuple[State1D1V, dict]:
    # 初期候補：前ステップを参照, 外部フックがあればそちらを優先
    init_fz = getattr(ws, "_init_fz", None)
    if init_fz is None:
        ws.fz.copy_(state.f)
    else:
        ws.fz.copy_(init_fz)   # shape (nx, nv)
        ws._init_fz = None

    residual_val = float('inf')

    # scheme_params から取得
    picard_iter = cfg.model_cfg.scheme_params.picard_iter
    picard_tol = cfg.model_cfg.scheme_params.picard_tol
    abs_tol = cfg.model_cfg.scheme_params.abs_tol

    latest = ws.fz

    for z in range(picard_iter):
        # (a,b,c,B) を一括構築（Maxwellの境界寄与も旧実装と同等）
        cuda_module.build_system_fused(
            state.f, ws.fz, state.v,
            float(state.dv), float(cfg.model_cfg.time.dt), float(state.dx),
            float(cfg.model_cfg.params.tau_tilde), float(state.inv_sqrt_2pi.item()),
            ws.dl, ws.dd, ws.du, ws.B
        )

        # 既存 cuSOLVER バインダで一括解法（戻り値 shape: (nv, nx-2)）
        solution = gtsv_module.gtsv_strided(
                ws.dl.contiguous(),
                ws.dd.contiguous(),
                ws.du.contiguous(),
                ws.B.contiguous()
            )

        # 内部セルのみ書き戻し。境界は前状態を維持
        ws.fn_tmp.copy_(ws.fz)
        ws.fn_tmp[1:-1, :].copy_(solution.T)

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

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        _, benchlog = step(state, cfg, ws, cuda_module, gtsv_module, num_steps)
        _stepper.benchlog = benchlog  # bench-logを属性として載せる

    _stepper.benchlog = None  # 初期値
    _stepper.ws = ws 
    return _stepper

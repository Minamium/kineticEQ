# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_implicit_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_implicit_ws import ImplicitWorkspace, allocate_implicit_workspace
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.cuda_kernel.compile import load_implicit_fused
from kineticEQ.cuda_kernel.compile import load_gtsv
import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]

@torch.no_grad()
def step(state: State1D1V, cfg: Config, ws: ImplicitWorkspace, cuda_module, gtsv_module, num_steps: int) -> State1D1V:
    # 初期候補：前ステップ
    ws.fz.copy_(state.f)
    swapped_last = False
    residual_val = float('inf')

    # scheme_params から取得
    picard_iter = cfg.model_cfg.scheme_params.picard_iter
    picard_tol = cfg.model_cfg.scheme_params.picard_tol

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

        # 残差
        residual = torch.max(torch.abs(ws.fn_tmp - ws.fz))
        residual_val = float(residual)

        if residual <= picard_tol:
            swapped_last = False
            break

        # 次反復へ
        ws.fz, ws.fn_tmp = ws.fn_tmp, ws.fz
        swapped_last = True

        # 直近で swap したかで最新候補の位置が変わる
        latest = ws.fz if swapped_last else ws.fn_tmp
        state.f_tmp.copy_(latest)
        # 念のため境界は前状態を維持（latest の境界は _fz と同じだが、方針の明確化）
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
    self.benchlog = {
        "picard_iter": z + 1,
        "picard_tol": residual_val,
    }

    return state

def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # CFL条件チェック
    bgk1d_check_CFL(cfg)

    # JITコンパイル
    cuda_module = load_implicit_fused()
    gtsv_module = load_gtsv()

    # implicit 専用ワークスペース確保
    nx, nv = state.f.shape
    ws = allocate_implicit_workspace(nx, nv, state.f.device, state.f.dtype)

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        step(state, cfg, ws, cuda_module, gtsv_module, num_steps)
    return _stepper

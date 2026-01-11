# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_explicit_cuda_kernel.py
from __future__ import annotations
from typing import Callable
import torch
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_check_CFL import bgk1d_check_CFL
from kineticEQ.cuda_kernel.compile import load_explicit_fused
import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]

@torch.no_grad()
def step(state: State1D1V, cfg: Config, cuda_module, num_steps: int) -> State1D1V:
    # CUDAカーネル呼び出し
    cuda_module.explicit_step(
        state.f, state.f_tmp, state.v,
        float(state.dv), float(cfg.model_cfg.time.dt), float(state.dx),
        float(cfg.model_cfg.params.tau_tilde), float(state.inv_sqrt_2pi.item()), int(state.k0)
    )

    # 境界固定
    state.f_tmp[0, :] = state.f[0, :]   
    state.f_tmp[-1, :] = state.f[-1, :]

    if num_steps % 100 == 0:
        logger.debug(f"NaN/Inf check executed at step: {num_steps}")
        if not torch.isfinite(state.f_tmp).all():
            raise ValueError("NaN/Inf detected in f_tmp")

    # swap
    state.f, state.f_tmp = state.f_tmp, state.f

    return state

def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # CFL条件チェック
    bgk1d_check_CFL(cfg)

    # JITコンパイル
    cuda_module = load_explicit_fused()

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        step(state, cfg, cuda_module, num_steps)
    return _stepper

# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_explicit_torch.py
from __future__ import annotations
from typing import Callable
import torch
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_compute_moments import calculate_moments
from kineticEQ.core.schemes.BGK1D.bgk1d_maxwellian import maxwellian
from kineticEQ.core.schemes.BGK1D.bgk1d_compute_streaming import _compute_streaming_upwind
from kineticEQ.core.schemes.BGK1D.bgk1d_set_initial_condition import set_initial_condition
from kineticEQ.core.schemes.BGK1D.bgk1d_check_CFL import bgk1d_check_CFL
import logging
logger = logging.getLogger(__name__)
Stepper = Callable[[int], None]

@torch.no_grad()
def step(state: State1D1V, cfg: Config, num_steps: int) -> State1D1V:
    dt = cfg.model_cfg.time.dt

    # モーメントの計算
    state.n, state.u, state.T = calculate_moments(state, state.f)
    
    # マクスウェル分布の計算
    state.f_m = maxwellian(state)

    # 緩和時間の計算
    tau = cfg.model_cfg.params.tau_tilde / (state.n * torch.sqrt(state.T))

    # 移流項計算（Upwind）
    streaming = _compute_streaming_upwind(state, state.f)

    # 衝突項計算
    collision = (state.f_m - state.f) / tau[:, None]

    # 時間発展
    state.f_tmp[1:-1, :] = state.f[1:-1, :] + dt * (streaming[1:-1, :] + collision[1:-1, :])

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

    # 初期条件設定
    set_initial_condition(state, cfg)
    def _stepper(num_steps: int) -> None:
        step(state, cfg, num_steps)
    return _stepper

# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_explicit_torch.py
from __future__ import annotations
from typing import Callable
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V

Stepper = Callable[[], None]

def step(state: State1D1V, cfg: Config) -> None:
    # TODO: ここに本体（今はダミーでよい）
    # state.f_new[...] = ...
    # state.f, state.f_new = state.f_new, state.f
    return

def build_stepper(cfg: Config, state: State1D1V) -> Stepper:
    # closure: Engine からは stepper() だけ呼べばよい
    def _stepper() -> None:
        step(state, cfg)
    return _stepper

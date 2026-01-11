# kineticEQ/src/kineticEQ/core/schemes/BGK2D2V/bgk2d2v_explicit_torch.py
from __future__ import annotations
from typing import Callable
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_2d2v import State2D2V

Stepper = Callable[[int], None]

def step(state: State2D2V, cfg: Config, num_steps: int) -> None:
    # TODO: ここに explicit 2D2V 本体を実装
    # いまはダミーでOK（とにかく動かす）
    return

def build_stepper(cfg: Config, state: State2D2V) -> Stepper:
    def _stepper(num_steps: int) -> None:
        step(state, cfg, num_steps)
    return _stepper

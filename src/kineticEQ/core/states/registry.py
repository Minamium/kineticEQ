# kineticEQ/core/states/registry.py
from __future__ import annotations
from typing import Any, Callable

from kineticEQ.api.config import Config, Model
from . import state_1d, state_2d2v

Factory = Callable[[Config], Any]

_FACTORIES: dict[Model, Factory] = {
    Model.BGK1D1V: state_1d.build_state,
    Model.BGK2D2V: state_2d2v.build_state,
}

def build_state(cfg: Config) -> Any:
    try:
        f = _FACTORIES[cfg.model]
    except KeyError as e:
        raise NotImplementedError(f"unknown model: {cfg.model}") from e
    return f(cfg)

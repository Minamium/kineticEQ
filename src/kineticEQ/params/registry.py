# kineticEQ/params/registry.py
from __future__ import annotations
from typing import Any, Callable

from kineticEQ.api.config import Model
from . import BGK1D1V_params, BGK2D2V_params

_FACTORIES: dict[Model, Callable[[], Any]] = {
    Model.BGK1D1V: BGK1D1V_params.ModelConfig,
    Model.BGK2D2V: BGK2D2V_params.ModelConfig,
}

_TYPES: dict[Model, type] = {
    Model.BGK1D1V: BGK1D1V_params.ModelConfig,
    Model.BGK2D2V: BGK2D2V_params.ModelConfig,
}

def default_model_cfg(model: Model) -> Any:
    return _FACTORIES[model]()

def expected_model_cfg_type(model: Model) -> type:
    return _TYPES[model]

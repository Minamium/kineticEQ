# kineticEQ/params/registry.py
from __future__ import annotations
from typing import Any, Callable

from kineticEQ.api.config import Model, Scheme
from .BGK1D import BGK1D1V_params, explicit, implicit, holo
from .BGK2D2V import BGK2D2V_params

_FACTORIES: dict[Model, Callable[[], Any]] = {
    Model.BGK1D1V: BGK1D1V_params.ModelConfig,
    Model.BGK2D2V: BGK2D2V_params.ModelConfig,
}

_TYPES: dict[Model, type] = {
    Model.BGK1D1V: BGK1D1V_params.ModelConfig,
    Model.BGK2D2V: BGK2D2V_params.ModelConfig,
}

# Scheme 別 scheme_params デフォルト (Model, Scheme) -> Params
_SCHEME_PARAMS_FACTORIES: dict[tuple[Model, Scheme], Callable[[], Any]] = {
    (Model.BGK1D1V, Scheme.EXPLICIT): explicit.Params,
    (Model.BGK1D1V, Scheme.IMPLICIT): implicit.Params,
    (Model.BGK1D1V, Scheme.HOLO): holo.Params,
    (Model.BGK2D2V, Scheme.EXPLICIT): None,
}

def default_model_cfg(model: Model) -> Any:
    return _FACTORIES[model]()

def expected_model_cfg_type(model: Model) -> type:
    return _TYPES[model]

def default_scheme_params(model: Model, scheme: Scheme) -> Any:
    factory = _SCHEME_PARAMS_FACTORIES.get((model, scheme))
    if factory is None:
        return None
    return factory()
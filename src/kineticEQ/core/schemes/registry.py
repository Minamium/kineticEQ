# kineticEQ/src/kineticEQ/core/schemes/registry.py
from __future__ import annotations
from typing import Any, Callable

from kineticEQ.api.config import Config, Model, Scheme, Backend

# stepperは「引数なしで1step進める callable」とする
Stepper = Callable[[int], None]
StepperBuilder = Callable[[Config, Any], Stepper]  # (cfg, state) -> stepper

from .BGK1D import bgk1d_explicit_torch
from .BGK2D2V import bgk2d2v_explicit_torch

_FACTORIES: dict[tuple[Model, Scheme, Backend], StepperBuilder] = {
    (Model.BGK1D1V, Scheme.EXPLICIT, Backend.TORCH): bgk1d_explicit_torch.build_stepper,
    (Model.BGK2D2V, Scheme.EXPLICIT, Backend.TORCH): bgk2d2v_explicit_torch.build_stepper,
    # 例：CUDAカーネルを後で追加
    # (Model.BGK2D2V, Scheme.EXPLICIT, Backend.CUDA_KERNEL): bgk2d2v_explicit_cuda.build_stepper,
}

def build_stepper(cfg: Config, state: Any) -> Stepper:
    key = (cfg.model, cfg.scheme, cfg.backend)
    try:
        builder = _FACTORIES[key]
    except KeyError as e:
        raise NotImplementedError(f"scheme not registered: {key}") from e
    return builder(cfg, state)

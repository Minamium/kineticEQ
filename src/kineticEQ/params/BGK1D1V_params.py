# kineticEQ/params/BGK1D1V_params.py
from dataclasses import dataclass, field
from typing import Any
import math

@dataclass(frozen=True)
class Grid1D1V:
    nx: int = 124
    nv: int = 64
    Lx: float = 1.0
    v_max: float = 10.0

@dataclass(frozen=True)
class TimeConfig:
    dt: float = 5e-4
    T_total: float = 0.05
    @property
    def n_steps(self) -> int:
        return int(max(1, math.ceil(self.T_total / self.dt)))

@dataclass(frozen=True)
class BGK1D1VParams:
    tau_tilde: float = 5e-1

@dataclass(frozen=True)
class InitialRegion1D:
    x_range: tuple[float, float]
    n: float
    u: float
    T: float

@dataclass(frozen=True)
class InitialCondition1D:
    initial_regions: tuple[Any, ...] = (
        {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    )

@dataclass(frozen=True)
class ModelConfig:
    # 表示順の辞書
    __pretty_order__ = (
        "grid.nx",
        "grid.Lx",
        "grid.nv",
        "grid.v_max",
        "params.tau_tilde",
        "time.dt",
        "time.T_total",
        "initial.", 
    )

    grid: Grid1D1V = field(default_factory=Grid1D1V)
    time: TimeConfig = field(default_factory=TimeConfig)
    params: BGK1D1VParams = field(default_factory=BGK1D1VParams)
    initial: InitialCondition1D = field(default_factory=InitialCondition1D)

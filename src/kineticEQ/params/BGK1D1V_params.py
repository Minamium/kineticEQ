# kineticEQ/params/BGK1D1V_params.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Grid1D1V:
    nx: int = 124
    nv: int = 64
    Lx: float = 1.0
    v_max: float = 10.0

@dataclass(frozen=True)
class TimeConfig:
    dt: float = 5e-3
    T_total: float = 0.05
    @property
    def n_steps(self) -> int:
        return int(self.T_total / self.dt)

@dataclass(frozen=True)
class BGK1D1VParams:
    tau_tilde: float = 5e-1

@dataclass(frozen=True)
class ModelConfig:
    grid: Grid1D1V = field(default_factory=Grid1D1V)
    time: TimeConfig = field(default_factory=TimeConfig)
    params: BGK1D1VParams = field(default_factory=BGK1D1VParams)

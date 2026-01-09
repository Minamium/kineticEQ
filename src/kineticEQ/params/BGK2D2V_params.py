# kineticEQ/params/BGK2D2V_params.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Grid2D2V:
    nx: int = 124
    ny: int = 124
    nx_v: int = 16
    ny_v: int = 16
    Lx: float = 1.0
    Ly: float = 1.0
    v_max: float = 10.0

@dataclass(frozen=True)
class TimeConfig:
    dt: float = 5e-3
    T_total: float = 0.05
    @property
    def n_steps(self) -> int:
        return int(self.T_total / self.dt)

@dataclass(frozen=True)
class BGK2D2VParams:
    tau_tilde: float = 5e-1

@dataclass(frozen=True)
class ModelConfig:
    grid: Grid2D2V = field(default_factory=Grid2D2V)
    time: TimeConfig = field(default_factory=TimeConfig)
    params: BGK2D2VParams = field(default_factory=BGK2D2VParams)

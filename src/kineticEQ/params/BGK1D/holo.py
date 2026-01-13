# kineticEQ/params/BGK1D/holo.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    ho_iter: int = 8
    ho_tol: float = 1e-4
    lo_iter: int = 16
    lo_tol: float = 1e-4
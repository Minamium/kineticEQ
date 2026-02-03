# kineticEQ/params/BGK1D/implicit.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    picard_iter: int = 16
    picard_tol: float = 1e-4
    abs_tol: float = 1e-16
    
    # --- CNN warmstart ---
    moments_cnn_modelpath: str | None = None
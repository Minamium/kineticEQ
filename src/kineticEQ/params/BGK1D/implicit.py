# kineticEQ/params/BGK1D/implicit.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    picard_iter: int = 16
    picard_tol: float = 1e-4
    abs_tol: float = 1e-16

    # --- Anderson Acceleration on W=(n,nu,T) ---
    aa_enable: bool = False
    aa_m: int = 6
    aa_beta: float = 1.0
    aa_stride: int = 1
    aa_start_iter: int = 2
    aa_reg: float = 1e-10
    aa_alpha_max: float = 50.0
    
    # --- CNN warmstart ---
    moments_cnn_modelpath: str | None = None
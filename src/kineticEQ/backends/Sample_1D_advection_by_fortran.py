"""CPU OpenMP backend (Fortran) – 1-D upwind advection"""
from __future__ import annotations
import numpy as np
from . import advection1d as _f90              # ← ビルド後の .so

def step(q: np.ndarray, dx: float, dt: float, u: float) -> np.ndarray:
    """一次風上差分 1 ステップ"""
    q = np.ascontiguousarray(q, dtype=np.float64)
    q_new = np.empty_like(q)
    _f90.advec_upwind_step(q.size, dt, dx, u, q, q_new)
    return q_new

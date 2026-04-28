# kineticEQ/core/schemes/BGK1D1V/bgk1d_utils/general/bgk1d_apply_BC.py
from __future__ import annotations

import torch

from kineticEQ.core.states.state_1d import State1D1V

# BGK1D1Vの境界条件適用関数
@torch.no_grad()
def boundary_rows(
    state: State1D1V,
    bc_type: str,
    *,
    source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source is None:
        source = state.f

    bc = str(bc_type).strip().lower()
    if bc in ("fixed_maxwellian", "fixed", "dirichlet"):
        L_bc = source[0, :].contiguous()
        R_bc = source[-1, :].contiguous()
    elif bc == "reflective":
        raise NotImplementedError("reflective boundary condition is not implemented yet")
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    return L_bc, R_bc

@torch.no_grad()
def apply_bc(
    state: State1D1V,
    bc_type: str,
    *,
    target: torch.Tensor | None = None,
    source: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target is None:
        target = state.f_tmp

    L_bc, R_bc = boundary_rows(state, bc_type, source=source)
    target[0, :].copy_(L_bc)
    target[-1, :].copy_(R_bc)
    return L_bc, R_bc

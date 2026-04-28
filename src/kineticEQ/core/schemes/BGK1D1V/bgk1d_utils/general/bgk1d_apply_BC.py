# kineticEQ/core/schemes/BGK1D1V/bgk1d_utils/general/bgk1d_apply_BC.py
from __future__ import annotations

import math
from typing import Any

import torch

from kineticEQ.core.states.state_1d import State1D1V

def _boundary_options(
    bc_type: str | Any,
    *,
    Lwall_temperature: float | None = None,
    Rwall_temperature: float | None = None,
    wall_temperature: float | None = None,
) -> tuple[str, float, float]:
    if not isinstance(bc_type, str) and hasattr(bc_type, "bc_type"):
        boundary = bc_type
        bc_type = getattr(boundary, "bc_type")
        if Lwall_temperature is None:
            Lwall_temperature = getattr(boundary, "Lwall_temperature", None)
        if Rwall_temperature is None:
            Rwall_temperature = getattr(boundary, "Rwall_temperature", None)
        if wall_temperature is None:
            wall_temperature = getattr(boundary, "wall_temperature", None)

    if wall_temperature is not None:
        if Lwall_temperature is None:
            Lwall_temperature = wall_temperature
        if Rwall_temperature is None:
            Rwall_temperature = wall_temperature

    L_T = 1.0 if Lwall_temperature is None else float(Lwall_temperature)
    R_T = 1.0 if Rwall_temperature is None else float(Rwall_temperature)

    return str(bc_type).strip().lower(), L_T, R_T

def _unit_wall_maxwellian(v: torch.Tensor, wall_temperature: float) -> torch.Tensor:
    T = torch.as_tensor(wall_temperature, dtype=v.dtype, device=v.device)
    two_pi = torch.as_tensor(2.0 * torch.pi, dtype=v.dtype, device=v.device)
    return torch.exp(-(v * v) / (2.0 * T)) / torch.sqrt(two_pi * T)

def _check_positive_denominator(value: torch.Tensor, name: str) -> None:
    value_float = float(value.detach().cpu())
    if not math.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be positive for diffuse boundary condition")

# BGK1D1Vの境界条件適用関数
@torch.no_grad()
def boundary_rows(
    state: State1D1V,
    bc_type: str | Any,
    *,
    source: torch.Tensor | None = None,
    Lwall_temperature: float | None = None,
    Rwall_temperature: float | None = None,
    wall_temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source is None:
        source = state.f

    bc, L_T, R_T = _boundary_options(
        bc_type,
        Lwall_temperature=Lwall_temperature,
        Rwall_temperature=Rwall_temperature,
        wall_temperature=wall_temperature,
    )
    if bc in ("fixed_maxwellian", "fixed", "dirichlet"):
        L_bc = source[0, :].contiguous()
        R_bc = source[-1, :].contiguous()
    elif bc in ("reflective", "specular", "specular_reflective"):
        if source.shape[0] < 3:
            raise ValueError("reflective boundary condition requires boundary cells and at least one interior cell")

        pos = state.v > 0
        neg = state.v < 0

        L_inside = source[1, :]
        R_inside = source[-2, :]
        L_mirror = torch.flip(L_inside, dims=(0,))
        R_mirror = torch.flip(R_inside, dims=(0,))

        L_bc = L_inside.clone()
        R_bc = R_inside.clone()
        L_bc[pos] = L_mirror[pos]
        R_bc[neg] = R_mirror[neg]
        L_bc = L_bc.contiguous()
        R_bc = R_bc.contiguous()
    elif bc in ("diffuse", "diffuse_reflective", "diffuse_reflection"):
        if L_T <= 0.0 or R_T <= 0.0:
            raise ValueError("diffuse boundary wall temperatures must be positive")
        if source.shape[0] < 3:
            raise ValueError("diffuse boundary condition requires boundary cells and at least one interior cell")

        v = state.v.to(device=source.device, dtype=source.dtype)
        pos = v > 0
        neg = v < 0
        if not bool(pos.any()) or not bool(neg.any()):
            raise ValueError("diffuse boundary condition requires both positive and negative velocity nodes")

        dv = torch.as_tensor(float(state.dv), dtype=source.dtype, device=source.device)
        L_inside = source[1, :]
        R_inside = source[-2, :]
        M_L = _unit_wall_maxwellian(v, L_T)
        M_R = _unit_wall_maxwellian(v, R_T)

        J_L_out = dv * torch.sum((-v[neg]) * L_inside[neg])
        D_L = dv * torch.sum(v[pos] * M_L[pos])
        J_R_out = dv * torch.sum(v[pos] * R_inside[pos])
        D_R = dv * torch.sum((-v[neg]) * M_R[neg])
        _check_positive_denominator(D_L, "left diffuse wall Maxwellian flux denominator")
        _check_positive_denominator(D_R, "right diffuse wall Maxwellian flux denominator")

        rho_L = J_L_out / D_L
        rho_R = J_R_out / D_R

        L_bc = L_inside.clone()
        R_bc = R_inside.clone()
        L_bc[pos] = rho_L * M_L[pos]
        R_bc[neg] = rho_R * M_R[neg]
        L_bc = L_bc.contiguous()
        R_bc = R_bc.contiguous()
    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
    return L_bc, R_bc

@torch.no_grad()
def apply_bc(
    state: State1D1V,
    bc_type: str | Any,
    *,
    target: torch.Tensor | None = None,
    source: torch.Tensor | None = None,
    Lwall_temperature: float | None = None,
    Rwall_temperature: float | None = None,
    wall_temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target is None:
        target = state.f_tmp

    L_bc, R_bc = boundary_rows(
        state,
        bc_type,
        source=source,
        Lwall_temperature=Lwall_temperature,
        Rwall_temperature=Rwall_temperature,
        wall_temperature=wall_temperature,
    )
    target[0, :].copy_(L_bc)
    target[-1, :].copy_(R_bc)
    return L_bc, R_bc

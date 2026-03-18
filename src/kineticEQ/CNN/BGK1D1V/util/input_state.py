from __future__ import annotations

import torch


def normalize_input_state_type(value: str | None, *, default: str = "nut") -> str:
    s = str(default if value is None else value).strip().lower()
    if s in ("nut", "primitive", "n_u_t"):
        return "nut"
    if s in ("nnut", "n_nu_t", "conservative"):
        return "nnuT"
    raise ValueError(f"unknown input_state_type={value!r}")


def input_state_type_from_delta_type(delta_type: str) -> str:
    dtp = str(delta_type).strip().lower()
    if dtp == "dw":
        return "nut"
    if dtp == "dnu":
        return "nnuT"
    raise ValueError(f"unknown delta_type={delta_type!r}")


def split_input_state(
    x: torch.Tensor,
    *,
    input_state_type: str,
    n_floor: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stype = normalize_input_state_type(input_state_type)
    n = x[:, 0:1, :].float()
    second = x[:, 1:2, :].float()
    T = x[:, 2:3, :].float()

    if stype == "nut":
        u = second
        nu = n * u
    else:
        nu = second
        u = nu / torch.clamp(n, min=float(n_floor))
    return n, u, nu, T


def build_model_input(
    n: torch.Tensor,
    u: torch.Tensor,
    T: torch.Tensor,
    logdt: float,
    logtau: float,
    *,
    input_state_type: str,
) -> torch.Tensor:
    stype = normalize_input_state_type(input_state_type)
    nx = int(n.numel())
    x = torch.empty((1, 5, nx), device=n.device, dtype=torch.float32)
    x[0, 0] = n.to(torch.float32)
    x[0, 1] = u.to(torch.float32) if stype == "nut" else (n * u).to(torch.float32)
    x[0, 2] = T.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))
    return x

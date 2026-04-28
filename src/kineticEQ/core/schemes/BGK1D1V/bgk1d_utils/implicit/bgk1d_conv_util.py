# kineticEQ/src/kineticEQ/core/schemes/BGK1D1V/bgk1d_utils/implicit/bgk1d_conv_util.py
# BGK1Dのimplicitスキーム用のconvユーティリティ
from __future__ import annotations

import torch


@torch.no_grad()
def check_convergence(
    *,
    conv_type: str,
    f_new: torch.Tensor,
    f_old: torch.Tensor,
    n_new: torch.Tensor,
    n_old: torch.Tensor,
    nu_new: torch.Tensor,
    nu_old: torch.Tensor,
    T_new: torch.Tensor,
    T_old: torch.Tensor,
    abs_tol: float,
    picard_tol: float,
) -> tuple[torch.Tensor, float, float]:
    conv_type = str(conv_type).lower()

    if conv_type == "f":
        df = torch.abs(f_new - f_old)
        ref = torch.maximum(torch.abs(f_new), torch.abs(f_old))
        den = abs_tol + picard_tol * ref

        residual = torch.max(df / den)
        residual_val = float(torch.max(df / torch.clamp(ref, min=abs_tol)).item())
        std_residual_val = float(residual.item())
        return residual, residual_val, std_residual_val

    if conv_type == "w":
        df_n = torch.abs(n_new - n_old)
        df_nu = torch.abs(nu_new - nu_old)
        df_T = torch.abs(T_new - T_old)

        ref_n = torch.maximum(torch.abs(n_new), torch.abs(n_old))
        ref_nu = torch.maximum(torch.abs(nu_new), torch.abs(nu_old))
        ref_T = torch.maximum(torch.abs(T_new), torch.abs(T_old))

        den_n = abs_tol + picard_tol * ref_n
        den_nu = abs_tol + picard_tol * ref_nu
        den_T = abs_tol + picard_tol * ref_T

        r_n = torch.max(df_n / den_n)
        r_nu = torch.max(df_nu / den_nu)
        r_T = torch.max(df_T / den_T)
        residual = torch.maximum(r_n, torch.maximum(r_nu, r_T))

        rel_n = torch.max(df_n / torch.clamp(ref_n, min=abs_tol))
        rel_nu = torch.max(df_nu / torch.clamp(ref_nu, min=abs_tol))
        rel_T = torch.max(df_T / torch.clamp(ref_T, min=abs_tol))
        residual_val = float(torch.maximum(rel_n, torch.maximum(rel_nu, rel_T)).item())
        std_residual_val = float(residual.item())
        return residual, residual_val, std_residual_val

    raise ValueError(f"invalid conv_type={conv_type!r}, expected 'f' or 'w'")

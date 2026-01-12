# kineticEQ/core/schemes/BGK1D/bgk1d_set_initial_condition.py
from __future__ import annotations
import torch
from kineticEQ.api.config import Config
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_maxwellian import maxwellian

@torch.no_grad()
def set_initial_condition(state: State1D1V, cfg: Config) -> None:
    x = state.x
    nx = x.numel()
    device = state.f.device
    dtype = state.f.dtype

    n = torch.full((nx,), 1.0, device=device, dtype=dtype)
    u = torch.full((nx,), 0.0, device=device, dtype=dtype)
    T = torch.full((nx,), 1.0, device=device, dtype=dtype)

    ic = cfg.model_cfg.initial
    regions = getattr(ic, "initial_regions", ())

    for r in regions:
        if isinstance(r, dict):
            a, b = r["x_range"]
            rn, ru, rT = r["n"], r["u"], r["T"]
        else:
            # InitialRegion1D 等を想定
            a, b = r.x_range
            rn, ru, rT = r.n, r.u, r.T

        if r == regions[-1]:
            mask = (x >= a) & (x <= b)  # 最後の領域は右閉
        else:
            mask = (x >= a) & (x < b)
        
        n[mask] = float(rn)
        u[mask] = float(ru)
        T[mask] = float(rT)

    state.n = n
    state.u = u
    state.T = T

    state.f.copy_(maxwellian(state))

# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_utils/bgk1d_holo_ws.py
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class HoloWorkspace:
    # HOLO outer iteration buffers (distribution)
    fz: torch.Tensor        # (nx, nv) current HO iterate
    fn_tmp: torch.Tensor    # (nx, nv) candidate next HO iterate

    # batched tridiag for distribution update (gtsv_strided expects (nv, nx-2))
    dl: torch.Tensor        # (nv, nx-2)
    dd: torch.Tensor        # (nv, nx-2)
    du: torch.Tensor        # (nv, nx-2)
    B: torch.Tensor         # (nv, nx-2)

    # explicit term & Maxwellians
    explicit_term: torch.Tensor  # (nx, nv)
    fM_old: torch.Tensor         # (nx, nv)
    fM_new: torch.Tensor         # (nx, nv)
    fM_lo: torch.Tensor          # (nx, nv)

    # moments (HO/LO)
    n_HO: torch.Tensor       # (nx,)
    u_HO: torch.Tensor       # (nx,)
    T_HO: torch.Tensor       # (nx,)

    n_new: torch.Tensor      # (nx,)
    u_new: torch.Tensor      # (nx,)
    T_new: torch.Tensor      # (nx,)

    n_lo: torch.Tensor       # (nx,)
    u_lo: torch.Tensor       # (nx,)
    T_lo: torch.Tensor       # (nx,)
    tau_lo: torch.Tensor     # (nx,)

    # HO fluxes / higher moments
    Q_HO: torch.Tensor       # (nx,)
    S1_face: torch.Tensor    # (nx-1,)
    S2_face: torch.Tensor    # (nx-1,)
    S3_face: torch.Tensor    # (nx-1,)

    # consistency terms
    Y_I_terms: torch.Tensor  # (nx, 3)

    # LO system buffers (block tridiagonal, inner cells only)
    W_HO: torch.Tensor       # (nx, 3)
    W_m: torch.Tensor        # (nx, 3)
    W_full: torch.Tensor     # (nx, 3)

    Q_half: torch.Tensor     # (nx-1,)
    F_HO_half: torch.Tensor  # (nx-1, 3)

    A_int: torch.Tensor      # (nx-1, 3, 3)
    b_int: torch.Tensor      # (nx-1, 3)

    AA: torch.Tensor         # (nx-2, 3, 3)
    BB: torch.Tensor         # (nx-2, 3, 3)
    CC: torch.Tensor         # (nx-2, 3, 3)
    DD: torch.Tensor         # (nx-2, 3)
    X_inner: torch.Tensor    # (nx-2, 3)


def allocate_holo_workspace(nx: int, nv: int, device, dtype) -> HoloWorkspace:
    n_inner = max(nx - 2, 0)

    z2 = torch.zeros  # shorthand
    return HoloWorkspace(
        fz=z2((nx, nv), device=device, dtype=dtype),
        fn_tmp=z2((nx, nv), device=device, dtype=dtype),

        dl=z2((nv, n_inner), device=device, dtype=dtype),
        dd=z2((nv, n_inner), device=device, dtype=dtype),
        du=z2((nv, n_inner), device=device, dtype=dtype),
        B=z2((nv, n_inner), device=device, dtype=dtype),

        explicit_term=z2((nx, nv), device=device, dtype=dtype),
        fM_old=z2((nx, nv), device=device, dtype=dtype),
        fM_new=z2((nx, nv), device=device, dtype=dtype),
        fM_lo=z2((nx, nv), device=device, dtype=dtype),

        n_HO=z2((nx,), device=device, dtype=dtype),
        u_HO=z2((nx,), device=device, dtype=dtype),
        T_HO=z2((nx,), device=device, dtype=dtype),

        n_new=z2((nx,), device=device, dtype=dtype),
        u_new=z2((nx,), device=device, dtype=dtype),
        T_new=z2((nx,), device=device, dtype=dtype),

        n_lo=z2((nx,), device=device, dtype=dtype),
        u_lo=z2((nx,), device=device, dtype=dtype),
        T_lo=z2((nx,), device=device, dtype=dtype),
        tau_lo=z2((nx,), device=device, dtype=dtype),

        Q_HO=z2((nx,), device=device, dtype=dtype),
        S1_face=z2((max(nx - 1, 0),), device=device, dtype=dtype),
        S2_face=z2((max(nx - 1, 0),), device=device, dtype=dtype),
        S3_face=z2((max(nx - 1, 0),), device=device, dtype=dtype),

        Y_I_terms=z2((nx, 3), device=device, dtype=dtype),

        W_HO=z2((nx, 3), device=device, dtype=dtype),
        W_m=z2((nx, 3), device=device, dtype=dtype),
        W_full=z2((nx, 3), device=device, dtype=dtype),

        Q_half=z2((max(nx - 1, 0),), device=device, dtype=dtype),
        F_HO_half=z2((max(nx - 1, 0), 3), device=device, dtype=dtype),

        A_int=z2((max(nx - 1, 0), 3, 3), device=device, dtype=dtype),
        b_int=z2((max(nx - 1, 0), 3), device=device, dtype=dtype),

        AA=z2((n_inner, 3, 3), device=device, dtype=dtype),
        BB=z2((n_inner, 3, 3), device=device, dtype=dtype),
        CC=z2((n_inner, 3, 3), device=device, dtype=dtype),
        DD=z2((n_inner, 3), device=device, dtype=dtype),
        X_inner=z2((n_inner, 3), device=device, dtype=dtype),
    )

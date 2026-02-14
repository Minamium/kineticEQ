# kineticEQ/core/schemes/BGK1D/bgk1d_utils/bgk1d_implicit_ws.py
from dataclasses import dataclass
import torch

@dataclass
class ImplicitWorkspace:
    fz: torch.Tensor
    fn_tmp: torch.Tensor
    dl: torch.Tensor
    dd: torch.Tensor
    du: torch.Tensor
    B: torch.Tensor
    n: torch.Tensor
    nu: torch.Tensor
    T: torch.Tensor
    n_new: torch.Tensor
    nu_new: torch.Tensor
    T_new: torch.Tensor
    B0: torch.Tensor
    aa_G: torch.Tensor
    aa_R: torch.Tensor
    aa_A: torch.Tensor
    aa_alpha: torch.Tensor
    aa_ones: torch.Tensor
    aa_wk: torch.Tensor
    aa_wnew: torch.Tensor
    aa_wtmp: torch.Tensor
    gtsv_ws: torch.Tensor


def allocate_implicit_workspace(nx: int, nv: int, device, dtype, aa_m: int = 0) -> ImplicitWorkspace:
    n_inner = max(nx - 2, 0)
    d = 3 * n_inner
    aa_cols = max(int(aa_m) + 1, 1)
    return ImplicitWorkspace(
        fz=torch.zeros((nx, nv), device=device, dtype=dtype),
        fn_tmp=torch.zeros((nx, nv), device=device, dtype=dtype),
        dl=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        dd=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        du=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        B=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        n=torch.zeros((nx,), device=device, dtype=dtype),
        nu=torch.zeros((nx,), device=device, dtype=dtype),
        T=torch.zeros((nx,), device=device, dtype=dtype),
        n_new=torch.zeros((nx,), device=device, dtype=dtype),
        nu_new=torch.zeros((nx,), device=device, dtype=dtype),
        T_new=torch.zeros((nx,), device=device, dtype=dtype),
        B0=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        aa_G=torch.zeros((d, aa_cols), device=device, dtype=dtype),
        aa_R=torch.zeros((d, aa_cols), device=device, dtype=dtype),
        aa_A=torch.zeros((aa_cols, aa_cols), device=device, dtype=dtype),
        aa_alpha=torch.zeros((aa_cols,), device=device, dtype=dtype),
        aa_ones=torch.ones((aa_cols, 1), device=device, dtype=dtype),
        aa_wk=torch.zeros((d,), device=device, dtype=dtype),
        aa_wnew=torch.zeros((d,), device=device, dtype=dtype),
        aa_wtmp=torch.zeros((d,), device=device, dtype=dtype),
        gtsv_ws=torch.empty((0,), device=device, dtype=torch.uint8),
    )
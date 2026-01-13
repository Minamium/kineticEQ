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


def allocate_implicit_workspace(nx: int, nv: int, device, dtype) -> ImplicitWorkspace:
    n_inner = max(nx - 2, 0)
    return ImplicitWorkspace(
        fz=torch.zeros((nx, nv), device=device, dtype=dtype),
        fn_tmp=torch.zeros((nx, nv), device=device, dtype=dtype),
        dl=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        dd=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        du=torch.zeros((nv, n_inner), device=device, dtype=dtype),
        B=torch.zeros((nv, n_inner), device=device, dtype=dtype),
    )
# kineticEQ/core/states/state_1d.py
from dataclasses import dataclass
import torch
import math

# BGK1D1Vの状態空間のデータクラス
@dataclass
class State1D1V:
    f: torch.Tensor
    f_new: torch.Tensor
    x: torch.Tensor
    v: torch.Tensor

    # caches (scheme-shared)
    v_col: torch.Tensor
    inv_sqrt_2pi: torch.Tensor
    pos_mask: torch.Tensor
    neg_mask: torch.Tensor
    v_coeff: torch.Tensor
    k0: int

# BGK1D1Vの状態空間のメモリ確保関数
def allocate_state_1d1v(nx, nv, Lx, v_max, device, dtype) -> State1D1V:
    dx = Lx / (nx - 1)
    f = torch.zeros((nx, nv), device=device, dtype=dtype)
    f_new = torch.zeros_like(f)
    x = torch.linspace(0, Lx, nx, device=device, dtype=dtype)
    v = torch.linspace(-v_max, v_max, nv, device=device, dtype=dtype)

    pos = v > 0
    inv = torch.tensor(1.0 / math.sqrt(2.0 * math.pi), device=device, dtype=dtype)

    k0 = int(torch.searchsorted(v, torch.tensor(0.0, device=device, dtype=dtype)))

    return State1D1V(
        f=f, f_new=f_new, x=x, v=v,
        v_col=v[None, :],
        inv_sqrt_2pi=inv,
        pos_mask=pos,
        neg_mask=~pos,
        v_coeff=(-v / dx),
        k0=k0,
    )
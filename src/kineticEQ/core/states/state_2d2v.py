# kineticEQ/src/kineticEQ/core/states/state_2d2v.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import math
from kineticEQ.api.config import Config, DType

# BGK2D2Vの状態空間のデータクラス
@dataclass
class State2D2V:
    f: torch.Tensor
    f_new: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor

    # caches scheme-shared
    vx_row: torch.Tensor
    vy_col: torch.Tensor
    inv_sqrt_2pi: torch.Tensor

# BGK2D2Vの状態空間のメモリ確保関数
def allocate_state_2d2v(nx, ny, nx_v, ny_v, Lx, Ly, v_max, device, dtype) -> State2D2V:
    f = torch.zeros((nx, ny, nx_v, ny_v), device=device, dtype=dtype)
    f_new = torch.zeros_like(f)

    x = torch.linspace(0, Lx, nx, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, ny, device=device, dtype=dtype)
    vx = torch.linspace(-v_max, v_max, nx_v, device=device, dtype=dtype)
    vy = torch.linspace(-v_max, v_max, ny_v, device=device, dtype=dtype)

    inv = torch.tensor(1.0 / math.sqrt(2.0 * math.pi), device=device, dtype=dtype)

    return State2D2V(
        f=f, f_new=f_new,
        x=x, y=y, vx=vx, vy=vy,
        vx_row=vx[None, :],
        vy_col=vy[:, None],
        inv_sqrt_2pi=inv,
    )

def build_state(cfg: Config) -> State2D2V:
    g = cfg.model_cfg.grid
    dtype = torch.float64 if cfg.dtype == DType.FLOAT64 else torch.float32
    return allocate_state_2d2v(
        nx=g.nx, ny=g.ny, nx_v=g.nx_v, ny_v=g.ny_v,
        Lx=g.Lx, Ly=g.Ly, v_max=g.v_max,
        device=cfg.device, dtype=dtype
    )


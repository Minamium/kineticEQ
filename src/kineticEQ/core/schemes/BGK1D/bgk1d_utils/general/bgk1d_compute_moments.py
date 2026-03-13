# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_compute_moments.py
import torch
from kineticEQ.core.states.state_1d import State1D1V

# モーメント計算メソッド
@torch.no_grad()
def calculate_moments(state: State1D1V, f: torch.Tensor):
    dv = state.dv
    v = state.v

    n = f.sum(dim=1) * dv
    s1 = (f * v[None, :]).sum(dim=1) * dv
    s2 = (f * (v[None, :] ** 2)).sum(dim=1) * dv

    u = s1 / n
    T = s2 / n - u * u
    return n, u, T
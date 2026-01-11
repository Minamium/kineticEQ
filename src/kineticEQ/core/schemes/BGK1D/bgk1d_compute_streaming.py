# kineticEQ/core/schemes/BGK1D/bgk1d_compute_streaming.py
import torch
from kineticEQ.core.states.state_1d import State1D1V

@torch.no_grad()
def _compute_streaming_upwind(state: State1D1V, f: torch.Tensor) -> torch.Tensor:
    # df: (nx-1, nv)
    df = torch.diff(f, dim=0)
    # flux: (nx-1, nv)   flux = (-v/dx) * df
    flux = df * state.v_coeff  # v_coeff: (nv,) broadcast

    streaming = torch.zeros_like(f)
    streaming[1:,  state.pos_mask] = flux[:, state.pos_mask]
    streaming[:-1, state.neg_mask] = flux[:, state.neg_mask]
    return streaming
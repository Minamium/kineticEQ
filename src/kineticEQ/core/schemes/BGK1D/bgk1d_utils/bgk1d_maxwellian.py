# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_maxwellian.py
import torch
from kineticEQ.core.states.state_1d import State1D1V

# bbgk1dのモーメントよりマクスウェル分布を返す
@torch.no_grad()
def maxwellian(state: State1D1V):
    """マックスウェル分布 f_M を高速計算 (FP64 前提)

    Parameters
    ----------
    state : State1D1V
        密度, 流速, 温度

    Returns
    -------
    f_M : (nx, nv) torch.Tensor
        Maxwellian 分布
    """

    # 係数部  n / sqrt(2π T)
    coeff = (state.n * state.inv_sqrt_2pi) / torch.sqrt(state.T)      # (nx,)

    # 指数部  exp( -(v-u)^2 / (2T) )
    invT  = 0.5 / state.T                                       # (nx,)
    diff  = state.v_col - state.u[:, None]                      # (nx, nv), view+broadcast

    exponent = diff.mul(diff)                             # (nx, nv): (v-u)^2
    exponent.mul_(-invT[:, None])                         # -(v-u)^2 / (2T)
    torch.exp(exponent, out=exponent)                     # exp(·) in-place

    exponent.mul_(coeff[:, None])                         # f = coeff * exp
    return exponent
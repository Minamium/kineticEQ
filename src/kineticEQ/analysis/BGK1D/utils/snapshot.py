import numpy as np
import torch
from kineticEQ import Engine
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments

def snapshot_from_engine(engine: Engine) -> dict:
    st = engine.state
    # st.x, st.v, st.f の名前は要確認。違うならここを直す。
    x = st.x.detach().cpu().numpy().copy()
    v = st.v.detach().cpu().numpy().copy()
    f = st.f.detach().cpu().numpy().copy()

    # moments を state が保持していない前提で計算
    n, u, T = calculate_moments(st, st.f)
    n = n.detach().cpu().numpy().copy()
    u = u.detach().cpu().numpy().copy()
    T = T.detach().cpu().numpy().copy()

    dx = float(st.dx) if hasattr(st, "dx") else None
    dv = float(st.dv) if hasattr(st, "dv") else None

    return {"x": x, "v": v, "f": f, "n": n, "u": u, "T": T, "dx": dx, "dv": dv}

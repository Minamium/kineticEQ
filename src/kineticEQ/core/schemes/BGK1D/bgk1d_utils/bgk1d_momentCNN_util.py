# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_momentCNN_util.py
from __future__ import annotations

# BGK1DのモーメントCNN用のユーティリティ

import logging
logger = logging.getLogger(__name__)

import torch
from kineticEQ.core.states.state_1d import State1D1V
from kineticEQ.CNN.BGK1D1V.models import MomentCNN1D

# CNNモデルのロード
def _load_ckpt_state(ckpt_path: str) -> tuple[dict, dict]:
    """
    Returns:
      (state_dict, meta)
        meta may include:
          - "delta_type": "dw" or "dnu" (if present in checkpoint)
          - other fields (optional)
    Supports:
      - full ckpt dict with keys: {"model": state_dict, "args": {...}, ...}
      - {"model_state": ...}, {"state_dict": ...}, {"model_state_dict": ...}
      - raw state_dict
      - DDP/DataParallel "module." prefix
    """
    obj = torch.load(ckpt_path, map_location="cpu")

    meta: dict = {}
    state: dict | None = None

    if isinstance(obj, dict):
        # try to extract delta_type from args/config if present
        args = obj.get("args", None)
        if isinstance(args, dict):
            dtp = args.get("delta_type", None) or args.get("target", None)
            if isinstance(dtp, str):
                meta["delta_type"] = dtp

        # some runs may save target in config-like field
        cfg = obj.get("meta", None)
        if isinstance(cfg, dict):
            dtp = cfg.get("delta_type", None) or cfg.get("target", None)
            if isinstance(dtp, str) and "delta_type" not in meta:
                meta["delta_type"] = dtp

        # extract model state dict
        for k in ("model", "model_state", "state_dict", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break

        # if looks like a raw state_dict already
        if state is None and all(isinstance(k, str) for k in obj.keys()) and any(k.endswith(".weight") for k in obj.keys()):
            state = obj

    # if not found, error
    if state is None or not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format or missing state_dict: {type(obj)}")

    # strip DDP/DataParallel prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    # normalize delta_type
    dtp = meta.get("delta_type", None)
    if isinstance(dtp, str):
        dtp = dtp.strip().lower()
        if dtp not in ("dw", "dnu"):
            # unknown -> drop (fallback later)
            meta.pop("delta_type", None)
        else:
            meta["delta_type"] = dtp

    return state, meta


def _infer_arch_from_state(state: dict) -> tuple[int, int, int, int]:
    # stem.0.weight: (hidden, in_ch, kernel)
    w = state.get("stem.0.weight", None)
    if w is None or w.ndim != 3:
        raise KeyError("Cannot infer arch: missing stem.0.weight")
    hidden = int(w.shape[0])
    in_ch  = int(w.shape[1])
    kernel = int(w.shape[2])

    idx = set()
    for k in state.keys():
        if k.startswith("blocks."):
            p = k.split(".")
            if len(p) >= 2 and p[1].isdigit():
                idx.add(int(p[1]))
    if not idx:
        raise ValueError("Cannot infer n_blocks from blocks.{i}.* keys")
    n_blocks = max(idx) + 1
    return in_ch, hidden, kernel, n_blocks


def load_moments_cnn_model(model_path: str, device: torch.device) -> tuple[MomentCNN1D, dict]:
    """
    Returns:
      (model, meta) where meta includes "delta_type" if present.
    """
    sd, meta = _load_ckpt_state(model_path)
    in_ch, hidden, kernel, n_blocks = _infer_arch_from_state(sd)

    model = MomentCNN1D(in_ch=in_ch, hidden=hidden, out_ch=3, kernel=kernel, n_blocks=n_blocks)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def _maxwellian_from_nuT(state: State1D1V, n: torch.Tensor, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Non-destructive Maxwellian builder (eval_warmstart_debug と同等).
    Requires: state.v_col (nx, nv), state.inv_sqrt_2pi (scalar tensor)
    Returns: (nx, nv)
    """
    coeff = (n * state.inv_sqrt_2pi) / torch.sqrt(T)          # (nx,)
    invT  = 0.5 / T                                           # (nx,)
    diff  = state.v_col - u[:, None]                          # (nx, nv)
    expo  = diff.mul(diff)
    expo.mul_(-invT[:, None])
    torch.exp(expo, out=expo)
    expo.mul_(coeff[:, None])
    return expo


@torch.no_grad()
def _build_fz_from_moments(
    state: State1D1V,
    n1: torch.Tensor, u1: torch.Tensor, T1: torch.Tensor,
    n_floor: float = 1e-12, T_floor: float = 1e-12,
) -> torch.Tensor:
    """
    eval_warmstart_debug.build_fz_from_moments と同等:
      - n,T を floor
      - Maxwellian を作る
      - 境界は state.f の境界を保持
    """
    if (not torch.isfinite(n1).all()) or (not torch.isfinite(u1).all()) or (not torch.isfinite(T1).all()):
        return state.f.clone()

    n1 = torch.clamp(n1, min=float(n_floor))
    T1 = torch.clamp(T1, min=float(T_floor))

    fz = _maxwellian_from_nuT(state, n1, u1, T1)

    # keep boundary from current distribution
    fz[0, :].copy_(state.f[0, :])
    fz[-1, :].copy_(state.f[-1, :])
    return fz


# モデルによる推論
@torch.no_grad()
def predict_next_moments_delta(
    model: MomentCNN1D,
    n0: torch.Tensor, u0: torch.Tensor, T0: torch.Tensor,
    logdt: float, logtau: float,
    *,
    delta_type: str = "dw",
    n_floor: float = 1e-12,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    delta_type:
      - "dw":  dy=[dn, du, dT], u1 = u0 + du
      - "dnu": dy=[dn, d(nu), dT], u1 = (n0*u0 + d(nu)) / max(n0+dn, n_floor)
    Returns:
      (n1_interior, u1_interior, T1_interior, dn_full, dmid_full, dT_full)
    Where dmid is:
      - du for dw
      - d(nu) for dnu
    """
    nx = n0.numel()
    x = torch.empty((1, 5, nx), device=n0.device, dtype=torch.float32)
    x[0, 0] = n0.to(torch.float32)
    x[0, 1] = u0.to(torch.float32)
    x[0, 2] = T0.to(torch.float32)
    x[0, 3].fill_(float(logdt))
    x[0, 4].fill_(float(logtau))

    dy = model(x)[0]  # (3, nx) float32
    dn = dy[0].to(n0.dtype)
    dT = dy[2].to(T0.dtype)
    dmid = dy[1].to(n0.dtype)

    n1 = n0 + dn
    T1 = T0 + dT

    dtp = str(delta_type).strip().lower()
    if dtp == "dw":
        du = dmid
        u1 = u0 + du
        return n1[1:-1], u1[1:-1], T1[1:-1], dn, du, dT
    elif dtp == "dnu":
        dm = dmid
        n1_safe = torch.clamp(n1, min=float(n_floor))
        m0 = n0 * u0
        m1 = m0 + dm
        u1 = m1 / n1_safe
        return n1[1:-1], u1[1:-1], T1[1:-1], dn, dm, dT
    else:
        raise ValueError(f"unknown delta_type={delta_type!r}")


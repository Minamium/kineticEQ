# kineticEQ/src/kineticEQ/core/schemes/BGK1D/bgk1d_momentCNN_util.py
from __future__ import annotations

# BGK1DのモーメントCNN用のユーティリティ

import logging
logger = logging.getLogger(__name__)

from typing import Any, Mapping

import torch
from kineticEQ.CNN.BGK1D1V.util.models import MomentCNN1D


def _to_dict(x: Any) -> dict | None:
    if isinstance(x, Mapping):
        return dict(x)
    if hasattr(x, "__dict__"):
        try:
            return dict(vars(x))
        except Exception:
            return None
    return None


def _to_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _to_int_tuple(x: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if x is None:
        return tuple(default)

    try:
        if isinstance(x, (list, tuple)):
            vals = [int(v) for v in x]
        else:
            s = str(x).strip().strip("[]()")
            toks = [t for t in s.replace(",", " ").split() if t]
            vals = [int(t) for t in toks]
        if not vals:
            return tuple(default)
        return tuple(vals)
    except Exception:
        return tuple(default)

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
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    meta: dict = {}
    state: dict | None = None

    if isinstance(obj, dict):
        model_kwargs = _to_dict(obj.get("model_kwargs", None))
        if isinstance(model_kwargs, dict):
            meta["model_kwargs"] = model_kwargs

        # try to extract delta_type from args/config if present
        args = _to_dict(obj.get("args", None))
        if isinstance(args, dict):
            meta["train_args"] = args
            dtp = args.get("delta_type", None) or args.get("target", None)
            if isinstance(dtp, str):
                meta["delta_type"] = dtp

        # some runs may save target in config-like field
        cfg = _to_dict(obj.get("meta", None))
        if isinstance(cfg, dict):
            dtp = cfg.get("delta_type", None) or cfg.get("target", None)
            if isinstance(dtp, str) and "delta_type" not in meta:
                meta["delta_type"] = dtp

            if "model_kwargs" not in meta:
                cfg_model_kwargs = _to_dict(cfg.get("model_kwargs", None))
                if isinstance(cfg_model_kwargs, dict):
                    meta["model_kwargs"] = cfg_model_kwargs

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


def _infer_arch_from_state(state: dict) -> dict[str, Any]:
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

    head_w = state.get("head_base.weight", None)
    if head_w is None or head_w.ndim != 3:
        raise KeyError("Cannot infer arch: missing head_base.weight")
    out_ch = int(head_w.shape[0])

    use_gate_head = any(k.startswith("gate.") for k in state.keys()) or any(k.startswith("head_tail.") for k in state.keys())

    gate_per_channel = False
    if use_gate_head:
        gate_w = state.get("gate.0.weight", None)
        if gate_w is not None and gate_w.ndim == 3:
            gate_per_channel = int(gate_w.shape[0]) == out_ch

    bottleneck = 0.5
    pre_w = state.get("blocks.0.pre.2.weight", None)
    if pre_w is not None and pre_w.ndim == 3 and int(pre_w.shape[1]) == hidden:
        bottleneck = float(pre_w.shape[0]) / float(max(hidden, 1))

    return {
        "in_ch": in_ch,
        "hidden": hidden,
        "out_ch": out_ch,
        "kernel": kernel,
        "n_blocks": n_blocks,
        "use_gate_head": bool(use_gate_head),
        "gate_per_channel": bool(gate_per_channel),
        "bottleneck": float(bottleneck),
    }


def _restore_model_kwargs(state: dict, meta: Mapping[str, Any]) -> dict[str, Any]:
    inf = _infer_arch_from_state(state)

    kwargs: dict[str, Any] = {
        "in_ch": int(inf["in_ch"]),
        "hidden": int(inf["hidden"]),
        "out_ch": int(inf["out_ch"]),
        "kernel": int(inf["kernel"]),
        "n_blocks": int(inf["n_blocks"]),
        "gn_groups": 32,
        "bottleneck": float(inf["bottleneck"]),
        "dilation_cycle": (1, 2),
        "use_gate_head": bool(inf["use_gate_head"]),
        "gate_bias_init": -4.0,
        "gate_scale": 1.0,
        "gate_per_channel": bool(inf["gate_per_channel"]),
    }

    saw_gn_groups = False
    saw_dilation_cycle = False
    saw_gate_scale = False

    def _apply(d: Mapping[str, Any]) -> None:
        nonlocal saw_gn_groups, saw_dilation_cycle, saw_gate_scale

        if "in_ch" in d:
            kwargs["in_ch"] = int(d["in_ch"])
        if "hidden" in d:
            kwargs["hidden"] = int(d["hidden"])
        if "out_ch" in d:
            kwargs["out_ch"] = int(d["out_ch"])
        if "kernel" in d:
            kwargs["kernel"] = int(d["kernel"])
        if "n_blocks" in d:
            kwargs["n_blocks"] = int(d["n_blocks"])

        if "gn_groups" in d:
            kwargs["gn_groups"] = int(d["gn_groups"])
            saw_gn_groups = True

        if "bottleneck" in d:
            kwargs["bottleneck"] = float(d["bottleneck"])

        if "dilation_cycle" in d:
            kwargs["dilation_cycle"] = _to_int_tuple(d["dilation_cycle"], default=kwargs["dilation_cycle"])
            saw_dilation_cycle = True

        if "use_gate_head" in d:
            kwargs["use_gate_head"] = _to_bool(d["use_gate_head"], default=kwargs["use_gate_head"])

        if "gate_bias_init" in d:
            kwargs["gate_bias_init"] = float(d["gate_bias_init"])

        if "gate_scale" in d:
            kwargs["gate_scale"] = float(d["gate_scale"])
            saw_gate_scale = True

        if "gate_per_channel" in d:
            kwargs["gate_per_channel"] = _to_bool(d["gate_per_channel"], default=kwargs["gate_per_channel"])

    model_kwargs = _to_dict(meta.get("model_kwargs", None))
    if isinstance(model_kwargs, dict):
        _apply(model_kwargs)

    train_args = _to_dict(meta.get("train_args", None))
    if isinstance(train_args, dict):
        _apply(train_args)

    # 形状整合に関わる値は state_dict から強制する
    for k in ("in_ch", "hidden", "out_ch", "kernel", "n_blocks"):
        if int(kwargs[k]) != int(inf[k]):
            logger.warning("%s mismatch between checkpoint args and state_dict; use state_dict value %s", k, inf[k])
            kwargs[k] = int(inf[k])

    if bool(kwargs["use_gate_head"]) != bool(inf["use_gate_head"]):
        logger.warning("use_gate_head mismatch between checkpoint args and state_dict; use state_dict value %s", inf["use_gate_head"])
        kwargs["use_gate_head"] = bool(inf["use_gate_head"])

    if bool(kwargs["use_gate_head"]):
        if bool(kwargs["gate_per_channel"]) != bool(inf["gate_per_channel"]):
            logger.warning(
                "gate_per_channel mismatch between checkpoint args and state_dict; use state_dict value %s",
                inf["gate_per_channel"],
            )
            kwargs["gate_per_channel"] = bool(inf["gate_per_channel"])
    else:
        kwargs["gate_per_channel"] = False

    kwargs["gn_groups"] = max(int(kwargs["gn_groups"]), 1)
    kwargs["bottleneck"] = max(float(kwargs["bottleneck"]), 1e-3)
    kwargs["dilation_cycle"] = _to_int_tuple(kwargs["dilation_cycle"], default=(1, 2))
    kwargs["gate_scale"] = float(kwargs["gate_scale"])

    if not saw_gn_groups:
        logger.warning("checkpoint metadata has no gn_groups; fallback=%d may cause behavior drift", int(kwargs["gn_groups"]))
    if not saw_dilation_cycle:
        logger.warning("checkpoint metadata has no dilation_cycle; fallback=%s may cause behavior drift", tuple(kwargs["dilation_cycle"]))
    if not saw_gate_scale:
        logger.warning("checkpoint metadata has no gate_scale; fallback=%g may cause behavior drift", float(kwargs["gate_scale"]))

    return kwargs


def load_moments_cnn_model(model_path: str, device: torch.device) -> tuple[MomentCNN1D, dict]:
    """
    Returns:
      (model, meta) where meta includes "delta_type" if present.
    """
    sd, meta = _load_ckpt_state(model_path)
    model_kwargs = _restore_model_kwargs(sd, meta)

    model = MomentCNN1D(**model_kwargs)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    out_meta = dict(meta)
    out_meta["model_kwargs_restored"] = dict(model_kwargs)
    return model, out_meta

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


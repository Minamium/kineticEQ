from __future__ import annotations

import torch


def normalize_input_state_type(value: str | None, *, default: str = "nut") -> str:
    s = str(default if value is None else value).strip().lower()
    if s in ("nut", "primitive", "n_u_t"):
        return "nut"
    if s in ("nnut", "n_nu_t", "conservative"):
        return "nnuT"
    raise ValueError(f"unknown input_state_type={value!r}")


def input_state_type_from_delta_type(delta_type: str) -> str:
    dtp = str(delta_type).strip().lower()
    if dtp == "dw":
        return "nut"
    if dtp == "dnu":
        return "nnuT"
    raise ValueError(f"unknown delta_type={delta_type!r}")


def normalize_input_temporal_mode(value: str | None, *, default: str = "none") -> str:
    s = str(default if value is None else value).strip().lower()
    if s in ("none", "off", "current"):
        return "none"
    if s in ("prev_delta", "history", "prev"):
        return "prev_delta"
    raise ValueError(f"unknown input_temporal_mode={value!r}")


def input_channel_count(*, input_temporal_mode: str) -> int:
    tmode = normalize_input_temporal_mode(input_temporal_mode)
    if tmode == "none":
        return 5
    if tmode == "prev_delta":
        return 12
    raise ValueError(f"unknown input_temporal_mode={input_temporal_mode!r}")


def input_feature_names(
    *,
    input_state_type: str,
    input_temporal_mode: str,
) -> list[str]:
    stype = normalize_input_state_type(input_state_type)
    tmode = normalize_input_temporal_mode(input_temporal_mode)
    second = "u" if stype == "nut" else "nu"
    if tmode == "none":
        return ["n", second, "T", "logdt", "logtau"]
    if tmode == "prev_delta":
        dsecond = "du_prev" if stype == "nut" else "dnu_prev"
        return [
            "n",
            second,
            "T",
            "prev_n",
            f"prev_{second}",
            "prev_T",
            "dn_prev",
            dsecond,
            "dT_prev",
            "logdt",
            "logtau",
            "has_prev",
        ]
    raise ValueError(f"unknown input_temporal_mode={input_temporal_mode!r}")


def _second_state_channel(n: torch.Tensor, u: torch.Tensor, *, input_state_type: str) -> torch.Tensor:
    stype = normalize_input_state_type(input_state_type)
    return u if stype == "nut" else (n * u)


def split_input_state(
    x: torch.Tensor,
    *,
    input_state_type: str,
    n_floor: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stype = normalize_input_state_type(input_state_type)
    n = x[:, 0:1, :].float()
    second = x[:, 1:2, :].float()
    T = x[:, 2:3, :].float()

    if stype == "nut":
        u = second
        nu = n * u
    else:
        nu = second
        u = nu / torch.clamp(n, min=float(n_floor))
    return n, u, nu, T


def build_model_input(
    n: torch.Tensor,
    u: torch.Tensor,
    T: torch.Tensor,
    logdt: float,
    logtau: float,
    *,
    input_state_type: str,
    input_temporal_mode: str = "none",
    prev_n: torch.Tensor | None = None,
    prev_u: torch.Tensor | None = None,
    prev_T: torch.Tensor | None = None,
    has_prev: bool = False,
) -> torch.Tensor:
    stype = normalize_input_state_type(input_state_type)
    tmode = normalize_input_temporal_mode(input_temporal_mode)
    nx = int(n.numel())
    x = torch.empty((1, input_channel_count(input_temporal_mode=tmode), nx), device=n.device, dtype=torch.float32)

    n_f = n.to(torch.float32)
    u_f = u.to(torch.float32)
    T_f = T.to(torch.float32)
    second = _second_state_channel(n_f, u_f, input_state_type=stype).to(torch.float32)

    x[0, 0] = n_f
    x[0, 1] = second
    x[0, 2] = T_f

    if tmode == "none":
        x[0, 3].fill_(float(logdt))
        x[0, 4].fill_(float(logtau))
        return x

    if tmode != "prev_delta":
        raise ValueError(f"unknown input_temporal_mode={input_temporal_mode!r}")

    if has_prev and (prev_n is None or prev_u is None or prev_T is None):
        raise ValueError("prev_delta mode with has_prev=True requires prev_n/prev_u/prev_T")

    if has_prev:
        prev_n_f = prev_n.to(torch.float32)
        prev_u_f = prev_u.to(torch.float32)
        prev_T_f = prev_T.to(torch.float32)
    else:
        prev_n_f = n_f
        prev_u_f = u_f
        prev_T_f = T_f

    prev_second = _second_state_channel(prev_n_f, prev_u_f, input_state_type=stype).to(torch.float32)

    x[0, 3] = prev_n_f
    x[0, 4] = prev_second
    x[0, 5] = prev_T_f
    x[0, 6] = n_f - prev_n_f
    x[0, 7] = second - prev_second
    x[0, 8] = T_f - prev_T_f
    x[0, 9].fill_(float(logdt))
    x[0, 10].fill_(float(logtau))
    x[0, 11].fill_(1.0 if has_prev else 0.0)
    return x

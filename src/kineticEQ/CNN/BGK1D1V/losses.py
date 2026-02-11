# kineticEQ/CNN/BGK1D1V/losses.py
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------- normalized residuals on next moments ----------------
def compute_stdW_residuals(
    pred: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    nb: int = 10,
    n_floor: float = 1e-12,
    T_floor: float = 1e-12,
    eps: float = 1e-12,
    delta_type: str = "dnu",
):
    """
    Return channel-wise normalized residuals r_n, r_u, r_T with boundary masked.
    Shapes: (B,1,nx) each.
    """
    pred = pred.float()
    y = y.float()
    x = x.float()

    n0 = x[:, 0:1, :]
    u0 = x[:, 1:2, :]
    T0 = x[:, 2:3, :]

    dn_p, dm_p, dT_p = pred[:, 0:1, :], pred[:, 1:2, :], pred[:, 2:3, :]
    dn_t, dm_t, dT_t = y[:, 0:1, :],    y[:, 1:2, :],    y[:, 2:3, :]

    n1_p = n0 + dn_p
    n1_t = n0 + dn_t

    n1_p_safe = torch.clamp(n1_p, min=float(n_floor))
    n1_t_safe = torch.clamp(n1_t, min=float(n_floor))

    if delta_type == "dw":
        u1_p = (u0 + dm_p)
        u1_t = (u0 + dm_t)
    elif delta_type == "dnu":
        u1_p = (n0 * u0 + dm_p) / n1_p_safe
        u1_t = (n0 * u0 + dm_t) / n1_t_safe
    else:
        raise ValueError(f"unknown delta_type={delta_type}")

    T1_p = torch.clamp(T0 + dT_p, min=float(T_floor))
    T1_t = torch.clamp(T0 + dT_t, min=float(T_floor))

    # rn
    rn = (n1_p - n1_t) / (n1_t.abs() + float(eps))

    # ru
    den = torch.stack([u1_t.abs(), torch.sqrt(T1_t)], dim=0).max(dim=0).values
    ru = (u1_p - u1_t) / (den + float(eps))

    # rT
    rT = (T1_p - T1_t) / (T1_t.abs() + float(eps))

    nx = rn.shape[-1]
    if nb > 0 and 2 * nb < nx:
        bmask = rn.new_ones((1, 1, nx))
        bmask[..., :nb] = 0.0
        bmask[..., -nb:] = 0.0
        rn = rn * bmask
        ru = ru * bmask
        rT = rT * bmask
        valid = bmask.sum().clamp_min(1.0)  # scalar
    else:
        valid = torch.tensor(float(rn.numel() // rn.shape[0]), device=rn.device, dtype=rn.dtype)  # ~nx
    return rn, ru, rT, valid


def build_shock_mask_from_x(
    x: torch.Tensor,
    *,
    nb: int = 10,
    shock_q: float = 0.90,   # 0.90 -> top10% as shock
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    x: (B,5,nx) のうち n,u,T を使って shock 指標 s=|Δn|+|Δu|+|ΔT| を作り、
    上位(1-shock_q)を1とする mask を返す。境界 nb は 0。
    returns: mask (B,1,nx) float (0/1)
    """
    B, _, nx = x.shape
    n = x[:, 0:1, :].float()
    u = x[:, 1:2, :].float()
    T = x[:, 2:3, :].float()

    dn = (n[..., 1:] - n[..., :-1]).abs()
    du = (u[..., 1:] - u[..., :-1]).abs()
    dT = (T[..., 1:] - T[..., :-1]).abs()

    z = torch.zeros((B, 1, 1), device=x.device, dtype=torch.float32)
    g = torch.cat([dn + du + dT, z], dim=-1)  # (B,1,nx)

    bmask = None
    if nb > 0 and 2 * nb < nx:
        bmask = g.new_ones((1, 1, nx))
        bmask[..., :nb] = 0.0
        bmask[..., -nb:] = 0.0
        g = g * bmask

    g_flat = g.reshape(B, -1)  # (B,nx)

    # all-zero safety: quantile==0 -> mask becomes all-ones; we want all-zeros in that case
    g_max = torch.amax(g_flat, dim=1, keepdim=True)
    thr = torch.quantile(g_flat, q=float(shock_q), dim=1, keepdim=True)
    mask = (g_flat >= (thr - eps)).float()
    mask = torch.where(g_max > 0.0, mask, torch.zeros_like(mask))

    mask = mask.reshape(B, 1, nx)
    if bmask is not None:
        mask = mask * bmask
    return mask


def std_w_loss_from_residuals_shock(
    rn: torch.Tensor,
    ru: torch.Tensor,
    rT: torch.Tensor,
    valid_count: torch.Tensor,
    shock_mask: torch.Tensor,
    *,
    kind: str = "smoothl1",
    shock_ratio: float = 0.8,
    # --- for softmax-max approx ---
    softmax_beta: float = 20.0,
    softmax_eps: float = 1e-12,
    softmax_use_abs: bool = True,   # True推奨（max|r| を狙う）
):
    """
    base: 全点 SmoothL1 / MSE / L1 / softmax-max
    + shock: shock_mask==1 の点のみ MSE を追加
    rn,ru,rT は境界マスク済み（マスク部は0）を想定。
    shock_mask: (B,1,nx) 0/1（境界0）
    """
    r = torch.cat([rn, ru, rT], dim=1)  # (B,3,nx)
    B, C, nx = r.shape

    if kind == "smoothl1":
        base = F.smooth_l1_loss(r, torch.zeros_like(r), reduction="sum")
    elif kind == "mse":
        base = (r * r).sum()
    elif kind == "l1":
        base = r.abs().sum()
    elif kind == "softmax":
        # smooth max approx via log-sum-exp over all elements (B,C,nx)
        a = r.abs() if bool(softmax_use_abs) else r
        v = a.reshape(B, -1)  # (B, N)
        m = torch.amax(v, dim=1, keepdim=True)  # (B,1)
        z = float(softmax_beta) * (v - m)
        lse = m + (torch.log(torch.sum(torch.exp(z), dim=1, keepdim=True) + float(softmax_eps))
                   / float(softmax_beta))  # (B,1)
        base = lse.sum()  # batch sum（他のkindの "sum" に合わせる）
    else:
        raise ValueError(f"unknown kind={kind}")

    sm = shock_mask.float().expand(B, 3, nx)
    shock_mse_sum = ((r * sm) ** 2).sum()

    e = base + float(shock_ratio) * shock_mse_sum

    denom = (valid_count * B * 3.0).clamp_min(1.0)
    return (e / denom), (base / denom), (shock_mse_sum / denom)
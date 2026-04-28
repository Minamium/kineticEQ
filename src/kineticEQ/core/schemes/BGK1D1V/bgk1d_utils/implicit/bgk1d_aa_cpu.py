from __future__ import annotations

import torch


@torch.no_grad()
def step_inplace(
    n: torch.Tensor,
    nu: torch.Tensor,
    T: torch.Tensor,
    n_new: torch.Tensor,
    nu_new: torch.Tensor,
    T_new: torch.Tensor,
    aa_G: torch.Tensor,
    aa_R: torch.Tensor,
    aa_A: torch.Tensor,
    aa_alpha: torch.Tensor,
    aa_wk: torch.Tensor,
    aa_wnew: torch.Tensor,
    aa_wtmp: torch.Tensor,
    hist_len: int,
    head: int,
    apply: bool,
    beta: float,
    reg: float,
    alpha_max: float,
    n_floor: float,
    T_floor: float,
    u_max: float,
    solver_work: torch.Tensor,
    solver_info: torch.Tensor,
    G_work: torch.Tensor,
    R_work: torch.Tensor,
) -> tuple[int, int]:
    del solver_work, solver_info

    nx = int(n.numel())
    n_inner = max(nx - 2, 0)
    d = 3 * n_inner
    aa_cols = int(aa_G.shape[1]) if aa_G.dim() == 2 else 0

    hist_len_in = max(int(hist_len), 0)
    head_in = max(int(head), 0) % max(aa_cols, 1)
    hist_len_out = min(hist_len_in + 1, aa_cols)
    head_out = (head_in + 1) % max(aa_cols, 1)

    if n_inner <= 0 or d <= 0:
        n.copy_(torch.clamp(n_new, min=n_floor))
        nu.copy_(nu_new)
        T.copy_(torch.clamp(T_new, min=T_floor))
        return hist_len_out, head_out

    wk = aa_wk[:d]
    wnew = aa_wnew[:d]
    wtmp = aa_wtmp[:d]

    wk[:n_inner].copy_(n[1:-1])
    wk[n_inner:2 * n_inner].copy_(nu[1:-1])
    wk[2 * n_inner:].copy_(T[1:-1])

    wnew[:n_inner].copy_(n_new[1:-1])
    wnew[n_inner:2 * n_inner].copy_(nu_new[1:-1])
    wnew[2 * n_inner:].copy_(T_new[1:-1])

    wtmp.copy_(wnew)
    wtmp.sub_(wk)

    aa_G[:d, head_in].copy_(wnew)
    aa_R[:d, head_in].copy_(wtmp)

    do_apply = bool(apply) and hist_len_out >= 2
    if not do_apply:
        n.copy_(torch.clamp(n_new, min=n_floor))
        nu.copy_(nu_new)
        T.copy_(torch.clamp(T_new, min=T_floor))
        return hist_len_out, head_out

    m = hist_len_out
    cols = [(head_out - m + j) % aa_cols for j in range(m)]

    Gm = G_work[:d, :m]
    Rm = R_work[:d, :m]
    Gm.copy_(aa_G[:d, cols])
    Rm.copy_(aa_R[:d, cols])

    A = aa_A[:m, :m]
    A.copy_(Rm.transpose(0, 1).matmul(Rm))
    A.diagonal().add_(float(reg))

    rhs = aa_alpha[:m]
    rhs.fill_(1.0)

    try:
        chol = torch.linalg.cholesky(A)
        alpha = torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
    except RuntimeError:
        alpha = torch.linalg.lstsq(A, rhs[:, None]).solution.squeeze(1)

    alpha_sum = float(alpha.sum().item())
    if abs(alpha_sum) < 1.0e-30:
        alpha_sum = 1.0e-30

    alpha.div_(alpha_sum)
    if alpha_max > 0.0:
        alpha.clamp_(min=-alpha_max, max=alpha_max)

    aa_alpha.zero_()
    aa_alpha[:m].copy_(alpha)

    wtmp.copy_(Gm.matmul(alpha))
    wtmp.mul_(float(beta))
    wtmp.add_(wnew, alpha=(1.0 - float(beta)))

    n[0] = max(float(n_new[0].item()), n_floor)
    nu[0] = float(nu_new[0].item())
    T[0] = max(float(T_new[0].item()), T_floor)
    n[-1] = max(float(n_new[-1].item()), n_floor)
    nu[-1] = float(nu_new[-1].item())
    T[-1] = max(float(T_new[-1].item()), T_floor)

    n_mid = torch.clamp(wtmp[:n_inner], min=n_floor)
    T_mid = torch.clamp(wtmp[2 * n_inner:], min=T_floor)
    u_mid = torch.nan_to_num(wtmp[n_inner:2 * n_inner] / n_mid, nan=0.0, posinf=0.0, neginf=0.0)
    u_mid.clamp_(min=-u_max, max=u_max)

    n[1:-1].copy_(n_mid)
    nu[1:-1].copy_(n_mid * u_mid)
    T[1:-1].copy_(T_mid)
    return hist_len_out, head_out

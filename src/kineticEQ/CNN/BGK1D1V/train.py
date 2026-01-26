# kineticEQ/CNN/BGK1D1V/train.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import MomentCNN1D
from .dataloader_npz import BGK1D1VNPZDeltaDataset


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def make_loader(manifest: str, split: str, batch: int, workers: int, pin: bool, prefetch_factor: int):
    ds = BGK1D1VNPZDeltaDataset(manifest_path=manifest, split=split, mmap=True, cache_npz=True, target="dnu")
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        drop_last=(split == "train"),
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    return ds, dl


# ---------------- loss (scaled) ----------------
def loss_weighted_dnu_with_u(
    pred: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float = 1e-6,
    nb: int = 1,
    u_eps: float = 5e-2,
    s_min: float = 1e-3,
    # ---- weighting knobs ----
    w_mode: str = "dm",          # "dm" or "du" or "none"
    w_lambda: float = 5.0,       # weight strength
    w_scale: float = 1.0,        # normalization scale for |dm| or |du| (roughly typical magnitude)
    w_power: float = 1.0,        # emphasize tails: 1=linear, 2=quadratic
    w_max: float = 30.0,         # cap weight to avoid exploding grads
    # ---- u auxiliary loss ----
    beta_u: float = 0.3,         # 0 disables u-loss
    u_loss_kind: str = "du",     # "du" or "u1"
    n_floor: float = 1e-12,
) -> torch.Tensor:
    """
    Weighted scaled SmoothL1 for target="dnu" with optional u/du auxiliary loss.

    pred,y: (B,3,nx) = [dn, dm, dT]
    x    : (B,5,nx) = [n0, u0, T0, logdt, logtau]

    Core loss: SmoothL1( pred/s , y/s ) with s based on (n0,u0,T0) as before.
    Weighting: w = 1 + w_lambda * clamp( (|dm_true|/w_scale)^w_power, 0, w_max-1 )
               or same with |du_true|
    Aux: L_u = SmoothL1(du_pred, du_true) or SmoothL1(u1_pred,u1_true)
    """
    pred = pred.float(); y = y.float(); x = x.float()

    # --- unpack ---
    n0 = x[:, 0:1, :]
    u0 = x[:, 1:2, :]
    T0 = x[:, 2:3, :]

    dn_p = pred[:, 0:1, :]
    dm_p = pred[:, 1:2, :]
    dT_p = pred[:, 2:3, :]

    dn_t = y[:, 0:1, :]
    dm_t = y[:, 1:2, :]
    dT_t = y[:, 2:3, :]

    # --- scaling (same spirit as your loss_scaled_dnu) ---
    n0a = n0.abs()
    u0a = u0.abs()
    T0a = T0.abs()

    s_n = n0a + eps
    s_m = (n0a * (u0a + float(u_eps))) + eps
    s_T = T0a + eps
    s = torch.cat([s_n, s_m, s_T], dim=1)
    s = torch.clamp(s, min=float(s_min))

    # --- elementwise SmoothL1 (no reduction) ---
    e = F.smooth_l1_loss(pred / s, y / s, reduction="none")  # (B,3,nx)

    # --- boundary mask ---
    nx = e.shape[-1]
    if nb > 0 and 2 * nb < nx:
        bmask = e.new_ones((1, 1, nx))
        bmask[..., :nb] = 0.0
        bmask[..., -nb:] = 0.0
    else:
        bmask = None

    # --- build weights w(x,y): focus on hard updates ---
    if w_mode == "none":
        w = e.new_ones((e.shape[0], 1, nx))
    elif w_mode == "dm":
        mag = dm_t.abs()
        w = 1.0 + float(w_lambda) * torch.clamp((mag / float(w_scale)).pow(float(w_power)), 0.0, float(w_max) - 1.0)
    elif w_mode == "du":
        # du_true computed from (dn_t, dm_t)
        n1_t = n0 + dn_t
        n1_t_safe = torch.clamp(n1_t, min=float(n_floor))
        u1_t = (n0 * u0 + dm_t) / n1_t_safe
        du_t = u1_t - u0
        mag = du_t.abs()
        w = 1.0 + float(w_lambda) * torch.clamp((mag / float(w_scale)).pow(float(w_power)), 0.0, float(w_max) - 1.0)
    else:
        raise ValueError(f"unknown w_mode={w_mode}")

    # apply boundary mask to weights too (avoid bias from boundaries)
    if bmask is not None:
        w = w * bmask

    # weighted core loss
    e_core = e * w  # (B,3,nx)

    if bmask is not None:
        denom = (w.sum() * e.shape[1]).clamp_min(1.0)  # sum over B*nx, then *3ch
        L_core = e_core.sum() / denom
    else:
        L_core = e_core.mean()

    # --- auxiliary u/du loss (prevents "dm/dn ok but u broken") ---
    if float(beta_u) > 0.0:
        # pred u1, true u1 from (dn,dm)
        n1_p = n0 + dn_p
        n1_t = n0 + dn_t
        n1_p_safe = torch.clamp(n1_p, min=float(n_floor))
        n1_t_safe = torch.clamp(n1_t, min=float(n_floor))

        u1_p = (n0 * u0 + dm_p) / n1_p_safe
        u1_t = (n0 * u0 + dm_t) / n1_t_safe

        if u_loss_kind == "u1":
            eu = F.smooth_l1_loss(u1_p, u1_t, reduction="none")  # (B,1,nx)
        elif u_loss_kind == "du":
            du_p = u1_p - u0
            du_t = u1_t - u0
            eu = F.smooth_l1_loss(du_p, du_t, reduction="none")
        else:
            raise ValueError(f"unknown u_loss_kind={u_loss_kind}")

        if bmask is not None:
            eu = eu * bmask
            denom_u = bmask.sum() * eu.shape[0] * eu.shape[1]
            L_u = eu.sum() / denom_u.clamp_min(1.0)
        else:
            L_u = eu.mean()

        return L_core + float(beta_u) * L_u

    return L_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_dir", type=str, default="cnn_runs/bgk1d1v")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # ---- knobs for scaled loss ----
    ap.add_argument("--u_eps", type=float, default=5e-2, help="dm scale uses |u0|+u_eps (prevents collapse at u0~0)")
    ap.add_argument("--s_min", type=float, default=1e-2, help="minimum scale clamp to avoid exploding normalized residuals")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="clip grad-norm (0 disables)")

    ap.add_argument("--sched_plateau", action="store_true")
    ap.add_argument("--sched_patience", type=int, default=3)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)
    
    # ---- weighted / u-aux loss knobs ----
    ap.add_argument("--w_mode", type=str, default="dm", choices=["dm", "du", "none"])
    ap.add_argument("--w_lambda", type=float, default=5.0)
    ap.add_argument("--w_scale", type=float, default=1e-3)
    ap.add_argument("--w_power", type=float, default=1.0)
    ap.add_argument("--w_max", type=float, default=30.0)

    ap.add_argument("--beta_u", type=float, default=0.3)
    ap.add_argument("--u_loss_kind", type=str, default="du", choices=["du", "u1"])
    ap.add_argument("--n_floor", type=float, default=1e-12)

    args = ap.parse_args()

    # ---- reproducibility ----
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")

    if device.type == "cuda":
        print(f"Device_name: {torch.cuda.get_device_name(0)}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save config
    save_json(save_dir / "config.json", vars(args))

    # ---- model/optim ----
    model = MomentCNN1D(in_ch=5, hidden=256, out_ch=3, kernel=11, n_blocks=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if bool(args.sched_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(args.sched_factor),
            patience=int(args.sched_patience),
            min_lr=float(args.sched_min_lr),
        )

    # AMP (new API; warning-free)
    use_amp = (args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- data ----
    train_ds, train_dl = make_loader(args.manifest, "train", args.batch, args.workers, pin=pin, prefetch_factor=args.prefetch_factor)
    val_ds, val_dl     = make_loader(args.manifest, "val",   args.batch, args.workers, pin=pin, prefetch_factor=args.prefetch_factor)

    best_val = float("inf")
    log_path = save_dir / "log.jsonl"

    # tqdm設定（sbatchログでも崩れにくい）
    disable_bar = False  # 必要なら: not sys.stdout.isatty()

    for ep in range(1, args.epochs + 1):
        # ---------------- train ----------------
        model.train()
        tr_loss_sum = 0.0
        tr_n = 0

        t_ep0 = time.time()
        steps = len(train_dl)

        pbar = tqdm(
            enumerate(train_dl, start=1),
            total=steps,
            desc=f"train ep{ep:03d}",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
            disable=disable_bar,
        )

        for it, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = loss_weighted_dnu_with_u(
                    pred, y, x,
                    eps=1e-6,
                    nb=1,
                    u_eps=float(args.u_eps),
                    s_min=float(args.s_min),
                    w_mode=str(args.w_mode),
                    w_lambda=float(args.w_lambda),
                    w_scale=float(args.w_scale),
                    w_power=float(args.w_power),
                    w_max=float(args.w_max),
                    beta_u=float(args.beta_u),
                    u_loss_kind=str(args.u_loss_kind),
                    n_floor=float(args.n_floor),
                )

            scaler.scale(loss).backward()

            # (recommended) clip after unscale
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tr_loss_sum += float(loss.item()) * bs
            tr_n += bs

            # ---- progress ----
            if (it % args.log_interval == 0) or (it == 1) or (it == steps):
                elapsed = time.time() - t_ep0
                sec_per_it = elapsed / max(it, 1)
                eta_ep = sec_per_it * (steps - it)
                lr = opt.param_groups[0]["lr"]
                loss_mean = tr_loss_sum / max(tr_n, 1)
                sps = tr_n / max(elapsed, 1e-9)

                postfix = {
                    "loss": f"{loss_mean:.3e}",
                    "lr": f"{lr:.1e}",
                    "s/it": f"{sec_per_it:.2f}",
                    "ETA_ep(min)": f"{eta_ep/60:.1f}",
                    "samples/s": f"{sps:.1f}",
                    "u_eps": f"{args.u_eps:.2g}",
                    "s_min": f"{args.s_min:.2g}",
                }
                if device.type == "cuda":
                    postfix["max_mem(GB)"] = f"{torch.cuda.max_memory_allocated()/1e9:.2f}"
                pbar.set_postfix(postfix)

        train_loss = tr_loss_sum / max(tr_n, 1)
        train_time = time.time() - t_ep0

        # ---------------- val ----------------
        model.eval()
        va_loss_sum = 0.0
        va_n = 0

        val_pred_dn_over_n0_linf = 0.0
        val_pred_dT_over_T0_linf = 0.0
        val_pred_dm_abs_linf = 0.0
        val_pred_du_abs_linf = 0.0
        val_true_dn_over_n0_linf = 0.0
        val_true_dT_over_T0_linf = 0.0
        val_true_dm_abs_linf = 0.0
        val_true_du_abs_linf = 0.0

        with torch.no_grad():
            vbar = tqdm(
                enumerate(val_dl, start=1),
                total=len(val_dl),
                desc=f"val   ep{ep:03d}",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=False,
                disable=disable_bar,
            )
            for it, (x, y) in vbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    loss = loss_weighted_dnu_with_u(
                        pred, y, x,
                        eps=1e-6,
                        nb=1,
                        u_eps=float(args.u_eps),
                        s_min=float(args.s_min),
                        w_mode=str(args.w_mode),
                        w_lambda=float(args.w_lambda),
                        w_scale=float(args.w_scale),
                        w_power=float(args.w_power),
                        w_max=float(args.w_max),
                        beta_u=float(args.beta_u),
                        u_loss_kind=str(args.u_loss_kind),
                        n_floor=float(args.n_floor),
                    )

                n0 = x[:, 0:1, :]
                u0 = x[:, 1:2, :]
                T0 = x[:, 2:3, :]

                dn_p = pred[:, 0:1, :]
                dm_p = pred[:, 1:2, :]
                dT_p = pred[:, 2:3, :]

                dn_t = y[:, 0:1, :]
                dm_t = y[:, 1:2, :]
                dT_t = y[:, 2:3, :]

                n1_p = n0 + dn_p
                n1_p_safe = torch.clamp(n1_p, min=1e-12)
                u1_p = (n0 * u0 + dm_p) / n1_p_safe
                du_p = u1_p - u0

                n1_t = n0 + dn_t
                n1_t_safe = torch.clamp(n1_t, min=1e-12)
                u1_t = (n0 * u0 + dm_t) / n1_t_safe
                du_t = u1_t - u0

                denom_n = torch.abs(n0) + 1e-30
                denom_T = torch.abs(T0) + 1e-30

                val_pred_dn_over_n0_linf = max(
                    val_pred_dn_over_n0_linf,
                    float(torch.amax(torch.abs(dn_p) / denom_n).detach().cpu()),
                )
                val_pred_dT_over_T0_linf = max(
                    val_pred_dT_over_T0_linf,
                    float(torch.amax(torch.abs(dT_p) / denom_T).detach().cpu()),
                )
                val_pred_dm_abs_linf = max(
                    val_pred_dm_abs_linf,
                    float(torch.amax(torch.abs(dm_p)).detach().cpu()),
                )
                val_pred_du_abs_linf = max(
                    val_pred_du_abs_linf,
                    float(torch.amax(torch.abs(du_p)).detach().cpu()),
                )

                val_true_dn_over_n0_linf = max(
                    val_true_dn_over_n0_linf,
                    float(torch.amax(torch.abs(dn_t) / denom_n).detach().cpu()),
                )
                val_true_dT_over_T0_linf = max(
                    val_true_dT_over_T0_linf,
                    float(torch.amax(torch.abs(dT_t) / denom_T).detach().cpu()),
                )
                val_true_dm_abs_linf = max(
                    val_true_dm_abs_linf,
                    float(torch.amax(torch.abs(dm_t)).detach().cpu()),
                )
                val_true_du_abs_linf = max(
                    val_true_du_abs_linf,
                    float(torch.amax(torch.abs(du_t)).detach().cpu()),
                )

                bs = x.size(0)
                va_loss_sum += float(loss.item()) * bs
                va_n += bs

                if (it == 1) or (it == len(val_dl)):
                    vbar.set_postfix({"loss": f"{(va_loss_sum/max(va_n,1)):.3e}"})

        val_loss = va_loss_sum / max(va_n, 1)

        if scheduler is not None:
            scheduler.step(val_loss)

        # ---------------- epoch summary ----------------
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(
            f"[epoch {ep:03d}] "
            f"train={train_loss:.6e} val={val_loss:.6e} "
            f"time={train_time:.1f}s best={best_val:.6e}",
            flush=True
        )

        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_val": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "manifest": args.manifest,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")

        # append jsonl
        with log_path.open("a") as f:
            f.write(json.dumps({
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "train_time_sec": train_time,
                "lr": opt.param_groups[0]["lr"],
                "u_eps": float(args.u_eps),
                "s_min": float(args.s_min),
                "grad_clip": float(args.grad_clip),
                "val_pred_dn_over_n0_linf": float(val_pred_dn_over_n0_linf),
                "val_pred_dT_over_T0_linf": float(val_pred_dT_over_T0_linf),
                "val_pred_dm_abs_linf": float(val_pred_dm_abs_linf),
                "val_pred_du_abs_linf": float(val_pred_du_abs_linf),
                "val_true_dn_over_n0_linf": float(val_true_dn_over_n0_linf),
                "val_true_dT_over_T0_linf": float(val_true_dT_over_T0_linf),
                "val_true_dm_abs_linf": float(val_true_dm_abs_linf),
                "val_true_du_abs_linf": float(val_true_du_abs_linf),
            }) + "\n")

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
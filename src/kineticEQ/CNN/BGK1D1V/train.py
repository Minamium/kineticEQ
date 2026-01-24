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
def loss_scaled_dnu(pred: torch.Tensor,
                    y: torch.Tensor,
                    x: torch.Tensor,
                    eps: float = 1e-6,
                    u_eps: float = 5e-2,
                    s_min: float = 1e-3) -> torch.Tensor:
    """
    Scale-normalized SmoothL1 for target="dnu" outputs.

    x: (B,5,nx)  = [n0, u0, T0, logdt, logtau]
    y: (B,3,nx)  = [dn, d(nu) (or dm), dT]
    pred: same as y

    Scaling:
      dn : |n0| + eps
      dm : |n0|*(|u0| + u_eps) + eps    (prevents u0≈0 collapse)
      dT : |T0| + eps

    s_min clamps the scale to avoid exploding normalized residuals in near-vacuum/near-zero regions.
    """
    n0 = x[:, 0:1, :].abs()
    u0 = x[:, 1:2, :].abs()
    T0 = x[:, 2:3, :].abs()

    s_n = n0 + eps
    s_m = (n0 * (u0 + u_eps)) + eps
    s_T = T0 + eps

    s = torch.cat([s_n, s_m, s_T], dim=1)
    s = torch.clamp(s, min=float(s_min))

    e = F.smooth_l1_loss(pred / s, y / s, reduction="none")
    return e.mean()


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
    ap.add_argument("--s_min", type=float, default=1e-3, help="minimum scale clamp to avoid exploding normalized residuals")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="clip grad-norm (0 disables)")

    ap.add_argument("--sched_plateau", action="store_true")
    ap.add_argument("--sched_patience", type=int, default=3)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)

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
    model = MomentCNN1D(in_ch=5, hidden=512, out_ch=3, kernel=11, n_blocks=5).to(device)
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
                loss = loss_scaled_dnu(
                    pred, y, x,
                    eps=1e-6,
                    u_eps=float(args.u_eps),
                    s_min=float(args.s_min),
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
                    loss = loss_scaled_dnu(
                        pred, y, x,
                        eps=1e-6,
                        u_eps=float(args.u_eps),
                        s_min=float(args.s_min),
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
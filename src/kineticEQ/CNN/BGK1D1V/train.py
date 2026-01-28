# kineticEQ/CNN/BGK1D1V/train.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import MomentCNN1D
from .dataloader_npz import BGK1D1VNPZDeltaDataset

# --- optional warmstart debug eval (epoch-end) ---
from .eval_warmstart_debug import build_cfg, run_case_debug


# ---------------- utils ----------------
def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def make_loader(manifest: str, split: str, batch: int, workers: int, pin: bool, prefetch_factor: int):
    ds = BGK1D1VNPZDeltaDataset(
        manifest_path=manifest,
        split=split,
        mmap=True,
        cache_npz=True,
        target="dnu",
    )
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

    u1_p = (n0 * u0 + dm_p) / n1_p_safe
    u1_t = (n0 * u0 + dm_t) / n1_t_safe

    T1_p = torch.clamp(T0 + dT_p, min=float(T_floor))
    T1_t = torch.clamp(T0 + dT_t, min=float(T_floor))

    # n, T は絶対値で正規化, u は(熱速度, 絶対値)のいずれかで正規化
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


def std_w_loss_from_residuals(
    rn: torch.Tensor,
    ru: torch.Tensor,
    rT: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    kind: str = "smoothl1",
    mse_ratio: float = 0.3,
    tail_frac: float = 0.1,      # 上位tail_frac%だけMSEを当てる
    eps: float = 1e-12,
):
    """
    SmoothL1（全点） + Tail(MSE)（上位 tail_frac のみ）で平均化した損失を返す。
    rn,ru,rT は境界マスク済み（マスク部は0）を想定。
    valid_count: 1サンプルあたりの有効セル数（scalar tensor）
    """
    r = torch.cat([rn, ru, rT], dim=1)  # (B,3,nx)
    B, C, nx = r.shape

    # ---- base loss (all points) ----
    if kind == "smoothl1":
        base = F.smooth_l1_loss(r, torch.zeros_like(r), reduction="sum")
    elif kind == "mse":
        base = (r * r).sum()
    elif kind == "l1":
        base = r.abs().sum()
    else:
        raise ValueError(f"unknown kind={kind}")

    # ---- tail MSE: only top-k |r| among valid points ----
    # valid points are those not masked => r != 0 (masked region is exactly 0)
    r_flat = r.reshape(B, -1)                 # (B, 3*nx)
    abs_flat = r_flat.abs()

    valid_mask = abs_flat > 0                 # masked cells are 0
    tail_mse_sum = r_flat.new_tensor(0.0)

    # per-sample top-k (stable against different valid counts per sample)
    for b in range(B):
        vb = valid_mask[b]
        if not torch.any(vb):
            continue
        vals = abs_flat[b, vb]                # (Nv,)
        Nv = int(vals.numel())
        k = max(1, int(round(tail_frac * Nv)))
        # threshold = k-th largest
        thr = torch.topk(vals, k, largest=True, sorted=False).values.min()
        tail_sel = vb & (abs_flat[b] >= (thr - eps))
        # MSE on selected tail points
        tail_mse_sum = tail_mse_sum + (r_flat[b, tail_sel] ** 2).sum()

    e = base + float(mse_ratio) * tail_mse_sum

    # normalize by effective count (valid_count is per-sample per-channel-line count)
    denom = (valid_count * B * 3.0).clamp_min(1.0)
    return (e / denom), (base / denom), (tail_mse_sum / denom)


# ---------------- args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--save_dir", type=str, default="cnn_runs/bgk1d1v_stdW")
    ap.add_argument("--seed", type=int, default=0)

    # loss knobs
    ap.add_argument("--loss_kind", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1"])
    ap.add_argument("--loss_eps", type=float, default=1e-6)
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--n_floor", type=float, default=1e-8)
    ap.add_argument("--T_floor", type=float, default=1e-8)
    ap.add_argument("--mse_ratio", type=float, default=0.8)
    ap.add_argument("--tail_frac", type=float, default=0.05)

    # optimization knobs
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=50)

    # optional scheduler
    ap.add_argument("--sched_plateau", action="store_true")
    ap.add_argument("--sched_patience", type=int, default=3)
    ap.add_argument("--sched_factor", type=float, default=0.5)
    ap.add_argument("--sched_min_lr", type=float, default=1e-6)

    # ---- epoch-end warmstart eval (print only; uses LAST model) ----
    ap.add_argument("--warm_eval", action="store_true", help="run warmstart debug at each epoch end (print only)")
    ap.add_argument("--warm_eval_tau", type=float, default=5e-7)
    ap.add_argument("--warm_eval_dt", type=float, default=5e-5)
    ap.add_argument("--warm_eval_T_total", type=float, default=0.02)
    ap.add_argument("--warm_eval_picard_iter", type=int, default=1000)
    ap.add_argument("--warm_eval_picard_tol", type=float, default=1e-4)
    ap.add_argument("--warm_eval_abs_tol", type=float, default=1e-13)
    ap.add_argument("--warm_eval_debug_steps", type=int, default=0, help="0 disables per-step debug_log collection")
    ap.add_argument("--warm_eval_n_floor", type=float, default=1e-12)
    ap.add_argument("--warm_eval_T_floor", type=float, default=1e-12)
    return ap.parse_args()


# ---------------- main ----------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        print(f"Device_name: {torch.cuda.get_device_name(0)}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_dir / "config.json", vars(args))

    model = MomentCNN1D(in_ch=5, hidden=256, out_ch=3, kernel=11, n_blocks=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    scheduler = None
    if bool(args.sched_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(args.sched_factor),
            patience=int(args.sched_patience),
            min_lr=float(args.sched_min_lr),
        )

    train_ds, train_dl = make_loader(args.manifest, "train", args.batch, args.workers, pin=pin, prefetch_factor=args.prefetch_factor)
    val_ds,   val_dl   = make_loader(args.manifest, "val",   args.batch, args.workers, pin=pin, prefetch_factor=args.prefetch_factor)

    best_val = float("inf")
    log_path = save_dir / "log.jsonl"

    for ep in range(1, int(args.epochs) + 1):
        # ---------------- train ----------------
        model.train()
        t0 = time.time()

        tr_loss_sum = 0.0
        tr_n = 0

        # channel-wise stats on residuals
        tr_rn_abs_sum = 0.0
        tr_ru_abs_sum = 0.0
        tr_rT_abs_sum = 0.0
        tr_rn_abs_max = 0.0
        tr_ru_abs_max = 0.0
        tr_rT_abs_max = 0.0
        tr_count = 0.0  # total valid elements per channel

        pbar = tqdm(train_dl, desc=f"train ep{ep:03d}", dynamic_ncols=True, leave=False)
        for it, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)

                # residuals (for both loss + logging)
                rn, ru, rT, valid = compute_stdW_residuals(
                    pred, y, x,
                    nb=int(args.nb),
                    n_floor=float(args.n_floor),
                    T_floor=float(args.T_floor),
                    eps=float(args.loss_eps),
                )
                loss, base_loss, tail_loss = std_w_loss_from_residuals(
                    rn, ru, rT, valid,
                    kind=str(args.loss_kind),
                    mse_ratio=float(args.mse_ratio),
                    tail_frac=float(args.tail_frac)
                )

            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            tr_loss_sum += float(loss.item()) * bs
            tr_n += bs

            # accumulate stats (abs mean + abs max) on valid region
            # valid region count per sample is "valid" (scalar), per batch is valid*B
            B = rn.shape[0]
            batch_count = float(valid.item()) * float(B)
            tr_count += batch_count

            arn = rn.abs()
            aru = ru.abs()
            arT = rT.abs()
            tr_rn_abs_sum += float(arn.sum().item())
            tr_ru_abs_sum += float(aru.sum().item())
            tr_rT_abs_sum += float(arT.sum().item())
            tr_rn_abs_max = max(tr_rn_abs_max, float(arn.max().item()))
            tr_ru_abs_max = max(tr_ru_abs_max, float(aru.max().item()))
            tr_rT_abs_max = max(tr_rT_abs_max, float(arT.max().item()))

            if (it == 1) or (it % int(args.log_interval) == 0):
                lr = opt.param_groups[0]["lr"]
                # channel-wise mean abs residual so far
                rn_mae = tr_rn_abs_sum / max(tr_count, 1.0)
                ru_mae = tr_ru_abs_sum / max(tr_count, 1.0)
                rT_mae = tr_rT_abs_sum / max(tr_count, 1.0)
                pbar.set_postfix({
                    "loss": f"{(tr_loss_sum/max(tr_n,1)):.3e}",
                    "tail": f"{float(tail_loss.item()):.3e}",
                    "base": f"{float(base_loss.item()):.3e}",
                    "lr": f"{lr:.1e}",
                    "|rn|": f"{rn_mae:.2e}",
                    "|ru|": f"{ru_mae:.2e}",
                    "|rT|": f"{rT_mae:.2e}",
                    "ru_max": f"{tr_ru_abs_max:.2e}",
                })

        train_loss = tr_loss_sum / max(tr_n, 1)
        train_time = time.time() - t0
        tr_rn_mae = tr_rn_abs_sum / max(tr_count, 1.0)
        tr_ru_mae = tr_ru_abs_sum / max(tr_count, 1.0)
        tr_rT_mae = tr_rT_abs_sum / max(tr_count, 1.0)

        # ---------------- val ----------------
        model.eval()
        va_loss_sum = 0.0
        va_n = 0
        va_base_loss_sum = 0.0
        va_tail_loss_sum = 0.0

        va_rn_abs_sum = 0.0
        va_ru_abs_sum = 0.0
        va_rT_abs_sum = 0.0
        va_rn_abs_max = 0.0
        va_ru_abs_max = 0.0
        va_rT_abs_max = 0.0
        va_count = 0.0

        with torch.no_grad():
            vbar = tqdm(val_dl, desc=f"val   ep{ep:03d}", dynamic_ncols=True, leave=False)
            for x, y in vbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    rn, ru, rT, valid = compute_stdW_residuals(
                        pred, y, x,
                        nb=int(args.nb),
                        n_floor=float(args.n_floor),
                        T_floor=float(args.T_floor),
                        eps=float(args.loss_eps),
                    )
                    loss, val_base_loss, val_tail_loss = std_w_loss_from_residuals(
                        rn, ru, rT, valid,
                        kind=str(args.loss_kind),
                        mse_ratio=float(args.mse_ratio),
                        tail_frac=float(args.tail_frac)
                    )

                bs = x.size(0)
                va_loss_sum += float(loss.item()) * bs
                va_base_loss_sum += float(val_base_loss.item()) * bs
                va_tail_loss_sum += float(val_tail_loss.item()) * bs
                va_n += bs

                B = rn.shape[0]
                batch_count = float(valid.item()) * float(B)
                va_count += batch_count

                arn = rn.abs()
                aru = ru.abs()
                arT = rT.abs()
                va_rn_abs_sum += float(arn.sum().item())
                va_ru_abs_sum += float(aru.sum().item())
                va_rT_abs_sum += float(arT.sum().item())
                va_rn_abs_max = max(va_rn_abs_max, float(arn.max().item()))
                va_ru_abs_max = max(va_ru_abs_max, float(aru.max().item()))
                va_rT_abs_max = max(va_rT_abs_max, float(arT.max().item()))

        val_loss = va_loss_sum / max(va_n, 1)
        val_base_loss = va_base_loss_sum / max(va_n, 1)
        val_tail_loss = va_tail_loss_sum / max(va_n, 1)
        va_rn_mae = va_rn_abs_sum / max(va_count, 1.0)
        va_ru_mae = va_ru_abs_sum / max(va_count, 1.0)
        va_rT_mae = va_rT_abs_sum / max(va_count, 1.0)

        if scheduler is not None:
            scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(
            f"[epoch {ep:03d}] "
            f"train={train_loss:.6e}, val={val_loss:.6e}, val_base={val_base_loss:.6e}, val_tail={val_tail_loss:.6e} "
            f"time={train_time:.1f}s best={best_val:.6e} "
            f"|rn|={va_rn_mae:.2e} |ru|={va_ru_mae:.2e} |rT|={va_rT_mae:.2e} "
            f"ru_max={va_ru_abs_max:.2e}",
            flush=True,
        )

        ckpt = {
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "best_val": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_base_loss": val_base_loss,
            "val_tail_loss": val_tail_loss,
            "manifest": args.manifest,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "best.pt")

        with log_path.open("a") as f:
            f.write(json.dumps({
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_base_loss": float(val_base_loss),
                "val_tail_loss": float(val_tail_loss),
                "best_val": best_val,
                "lr": opt.param_groups[0]["lr"],
                "train_time_sec": train_time,

                # channel-wise abs residual stats (train/val)
                "train_rn_abs_mean": tr_rn_mae,
                "train_ru_abs_mean": tr_ru_mae,
                "train_rT_abs_mean": tr_rT_mae,
                "train_rn_abs_max": tr_rn_abs_max,
                "train_ru_abs_max": tr_ru_abs_max,
                "train_rT_abs_max": tr_rT_abs_max,

                "val_rn_abs_mean": va_rn_mae,
                "val_ru_abs_mean": va_ru_mae,
                "val_rT_abs_mean": va_rT_mae,
                "val_rn_abs_max": va_rn_abs_max,
                "val_ru_abs_max": va_ru_abs_max,
                "val_rT_abs_max": va_rT_abs_max,

                # config echoes
                "loss_kind": str(args.loss_kind),
                "loss_eps": float(args.loss_eps),
                "nb": int(args.nb),
                "n_floor": float(args.n_floor),
                "T_floor": float(args.T_floor),
                "grad_clip": float(args.grad_clip),
                "amp": bool(args.amp),
            }) + "\n")

        # ---------------- epoch-end warmstart debug (LATEST model; print only) ----------------
        if bool(args.warm_eval):
            try:
                # keep it deterministic & no grad
                model.eval()
                device_eval = device

                cfg = build_cfg(
                    tau=float(args.warm_eval_tau),
                    dt=float(args.warm_eval_dt),
                    T_total=float(args.warm_eval_T_total),
                    nx=int(cfg_nx := 512),   # NOTE: if you want to match training nx, set explicitly
                    nv=int(cfg_nv := 256),
                    Lx=1.0,
                    v_max=10.0,
                    picard_iter=int(args.warm_eval_picard_iter),
                    picard_tol=float(args.warm_eval_picard_tol),
                    abs_tol=float(args.warm_eval_abs_tol),
                )

                # infer n_steps from cfg
                n_steps = int(round(cfg.model_cfg.time.T_total / cfg.model_cfg.time.dt))

                for a in (1.0, 0.9, 0.8):
                    out = run_case_debug(
                        cfg=cfg,
                        model=model,                      # <-- latest weights in memory
                        n_steps=n_steps,
                        device=device_eval,
                        mix_alpha=float(a),
                        debug_steps=int(args.warm_eval_debug_steps),
                        n_floor=float(args.warm_eval_n_floor),
                        T_floor=float(args.warm_eval_T_floor),
                    )
                    base_sum = int(out["picard_iter_sum_base"])
                    warm_sum = int(out["picard_iter_sum_warm"])
                    speed = (base_sum / max(warm_sum, 1))
                    print(
                        f"[warm-eval ep{ep:03d}] "
                        f"tau={args.warm_eval_tau:.3e} alpha={a:.2f} "
                        f"picard_sum base={base_sum} warm={warm_sum} (x{speed:.2f})",
                        flush=True,
                    )
            except Exception as e:
                print(f"[warm-eval ep{ep:03d}] FAILED: {type(e).__name__}: {e}", flush=True)


    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
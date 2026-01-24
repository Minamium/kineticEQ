# kineticEQ/CNN/BGK1D1V/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import MomentCNN1D
from .dataloader_npz import BGK1D1VNPZDeltaDataset


def make_loader(manifest: str, split: str, batch: int, workers: int, pin: bool):
    ds = BGK1D1VNPZDeltaDataset(manifest_path=manifest, split=split, mmap=True, cache_npz=True)
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        drop_last=(split == "train"),
    )
    return ds, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_dir", type=str, default="cnn_runs/bgk1d1v")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = MomentCNN1D(in_ch=5, hidden=128, out_ch=3, kernel=11, n_blocks=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    train_ds, train_dl = make_loader(args.manifest, "train", args.batch, args.workers, pin=True)
    val_ds, val_dl = make_loader(args.manifest, "val", args.batch, args.workers, pin=True)

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x)
                loss = F.mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item()) * x.size(0)
            tr_n += x.size(0)

        tr_loss /= max(tr_n, 1)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred = model(x)
                    loss = F.mse_loss(pred, y)
                va_loss += float(loss.item()) * x.size(0)
                va_n += x.size(0)
        va_loss /= max(va_n, 1)

        print(f"[epoch {ep:03d}] train={tr_loss:.6e} val={va_loss:.6e}", flush=True)

        # save best
        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": best_val,
                "manifest": args.manifest,
            }
            torch.save(ckpt, save_dir / "best.pt")

    # close datasets (npz cache)
    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()

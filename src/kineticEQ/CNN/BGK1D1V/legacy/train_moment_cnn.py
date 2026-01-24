# kineticEQ/CNN/BGK1D1V/train_moment_cnn.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kineticEQ.CNN.BGK1D1V.dataset_loader_npz import BGK1D1VNPZDataset


class MomentCNN1D(nn.Module):
    """
    Simple Conv1d network:
      input:  (B, 5, nx)  [n,u,T,logdt,logtau]
      output: (B, 3, nx)  [n_next,u_next,T_next]

    Note:
    - This is a baseline. Later you can add residual blocks / normalization.
    """
    def __init__(self, ch_in: int = 5, ch_hidden: int = 64, ch_out: int = 3, kernel: int = 5) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch_in, ch_hidden, kernel_size=kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(ch_hidden, ch_hidden, kernel_size=kernel, padding=pad),
            nn.SiLU(),
            nn.Conv1d(ch_hidden, ch_out, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # baseline: MSE over all channels/space
    return F.mse_loss(pred, target)


@torch.no_grad()
def eval_loop(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        L = loss_fn(pred, y)
        total += float(L.item())
        n += 1
    return total / max(n, 1)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")

    log = {
        "train_loss": [],
        "val_loss": [],
    }

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"train ep{ep:03d}", dynamic_ncols=True)
        for step_i, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            L = loss_fn(pred, y)
            L.backward()
            opt.step()

            loss_val = float(L.item())
            running += loss_val
            n += 1

            # tqdm 表示（平均lossも表示）
            pbar.set_postfix(loss=f"{(running/max(n,1)):.3e}")


        train_loss = running / max(n, 1)
        val_loss = eval_loop(model, val_loader, device)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        print(f"[epoch {ep:03d}] train={train_loss:.6e} val={val_loss:.6e}")

        # checkpoint
        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")

    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[OK] saved logs and checkpoints under: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="mgpu_output_manifest.json")
    ap.add_argument("--out_dir", type=str, default="cnn_runs/bgk1d1v_momentcnn")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--t_stride", type=int, default=1, help="Use every k-th timestep")
    ap.add_argument("--t_max", type=int, default=None, help="Max timestep index used for sampling (<= n_steps-1)")

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--kernel", type=int, default=5)

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    # datasets
    train_ds = BGK1D1VNPZDataset(args.manifest, split="train", t_stride=args.t_stride, t_max=args.t_max)
    val_ds   = BGK1D1VNPZDataset(args.manifest, split="val",   t_stride=args.t_stride, t_max=args.t_max)

    # loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # model
    model = MomentCNN1D(ch_in=5, ch_hidden=args.hidden, ch_out=3, kernel=args.kernel).to(device)

    # run
    train_loop(
        model,
        train_loader,
        val_loader,
        device,
        lr=args.lr,
        epochs=args.epochs,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

# kineticEQ/CNN/BGK1D1V/dataloader_npz.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_manifest(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text())


def _resolve_npz_path(manifest: Dict[str, Any], rec_path: str) -> Path:
    p = Path(rec_path)
    if p.is_absolute():
        return p
    # join with data_root recorded in manifest
    root = Path(manifest.get("data_root", "."))
    return (root / p).resolve()


@dataclass
class SampleIndex:
    file_idx: int
    t: int


class BGK1D1VNPZDeltaDataset(Dataset):
    """
    1 sample = (case_id, time t) with full spatial field (nx).

    X: (5, nx)  = [n(t), u(t), T(t), log10(dt), log10(tau)]
    Y: (3, nx)  = [n(t+1)-n(t), u(t+1)-u(t), T(t+1)-T(t)]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
        cache_npz: bool = True,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")

        self.manifest_path = Path(manifest_path)
        self.manifest = _load_manifest(self.manifest_path)
        self.dtype = dtype
        self.mmap = mmap
        self.cache_npz = cache_npz

        # split case ids
        split_case_ids = set(self.manifest["splits"][split])

        # select file records
        self.records: List[Dict[str, Any]] = [
            r for r in self.manifest["files"] if int(r["case_id"]) in split_case_ids
        ]
        if len(self.records) == 0:
            raise RuntimeError(f"No records found for split={split}")

        # build global index -> (file_idx, t)
        self._offsets: List[int] = [0]
        total = 0
        for r in self.records:
            n_steps = int(r["n_steps"])
            # t = 0..n_steps-1 (needs t+1)
            total += n_steps
            self._offsets.append(total)

        self._total_len = total

        # tiny cache: last opened file
        self._cache_file_idx: int | None = None
        self._cache_npz: Any | None = None  # NpzFile

    def __len__(self) -> int:
        return self._total_len

    def _locate(self, idx: int) -> SampleIndex:
        # binary search on offsets
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._offsets[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        file_idx = lo
        t = idx - self._offsets[file_idx]
        return SampleIndex(file_idx=file_idx, t=t)

    def _open_npz(self, file_idx: int):
        if self.cache_npz and self._cache_file_idx == file_idx and self._cache_npz is not None:
            return self._cache_npz

        rec = self.records[file_idx]
        npz_path = _resolve_npz_path(self.manifest, rec["path"])

        # allow_pickle=False: safer
        z = np.load(npz_path, allow_pickle=False, mmap_mode=("r" if self.mmap else None))

        if self.cache_npz:
            # close old cache
            if self._cache_npz is not None:
                try:
                    self._cache_npz.close()
                except Exception:
                    pass
            self._cache_file_idx = file_idx
            self._cache_npz = z
        return z

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._locate(idx)
        rec = self.records[s.file_idx]
        z = self._open_npz(s.file_idx)

        # arrays: (n_steps+1, nx)
        n = z["n"]
        u = z["u"]
        T = z["T"]
        t = s.t

        # X: (5, nx)
        # W(t)
        n_t = torch.from_numpy(np.asarray(n[t], dtype=np.float32))
        u_t = torch.from_numpy(np.asarray(u[t], dtype=np.float32))
        T_t = torch.from_numpy(np.asarray(T[t], dtype=np.float32))

        # dt, tau (case constants) -> broadcast to (nx,)
        nx = int(rec["nx"])
        logdt = float(np.log10(float(rec["dt"])))
        logtau = float(np.log10(float(rec["tau_tilde"])))
        logdt_x = torch.full((nx,), logdt, dtype=torch.float32)
        logtau_x = torch.full((nx,), logtau, dtype=torch.float32)

        x = torch.stack([n_t, u_t, T_t, logdt_x, logtau_x], dim=0).to(dtype=self.dtype)

        # Y: delta (3, nx)
        n_tp1 = torch.from_numpy(np.asarray(n[t + 1], dtype=np.float32))
        u_tp1 = torch.from_numpy(np.asarray(u[t + 1], dtype=np.float32))
        T_tp1 = torch.from_numpy(np.asarray(T[t + 1], dtype=np.float32))

        y = torch.stack([n_tp1 - n_t, u_tp1 - u_t, T_tp1 - T_t], dim=0).to(dtype=self.dtype)

        return x, y

    def close(self):
        if self._cache_npz is not None:
            try:
                self._cache_npz.close()
            except Exception:
                pass
            self._cache_npz = None
            self._cache_file_idx = None
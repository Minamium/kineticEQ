# kineticEQ/CNN/BGK1D1V/dataset_loader_npz.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleIndex:
    file_idx: int   # index into manifest["files"]
    t: int          # time index (0..n_steps-1)


class BGK1D1VNPZDataset(Dataset):
    """
    Each item is a single-step mapping:
        x(t) -> y(t+1)

    x channels: [n, u, T, log10(dt), log10(tau)]
      shape: (5, nx)

    y channels: [n_next, u_next, T_next]
      shape: (3, nx)

    Notes:
    - Split is case-based via manifest["splits"].
    - Uses memory-mapped loading per file read (np.load). For performance,
      you can later add caching (e.g. LRU cache of open npz handles).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        *,
        t_stride: int = 1,
        t_max: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: Dict[str, Any] = json.loads(self.manifest_path.read_text())
        assert split in ("train", "val", "test"), f"invalid split: {split}"
        self.split = split

        self.files: List[Dict[str, Any]] = self.manifest["files"]
        self.splits: Dict[str, List[int]] = self.manifest["splits"]
        self.allowed_case_ids = set(self.splits[split])

        self.dtype = dtype
        self.device = device

        # Precompute eligible file indices (case-based)
        self.file_indices: List[int] = []
        for i, rec in enumerate(self.files):
            if int(rec["case_id"]) in self.allowed_case_ids:
                self.file_indices.append(i)

        if len(self.file_indices) == 0:
            raise RuntimeError(f"No files matched split='{split}'")

        # Build (file_idx, t) index table
        self.samples: List[SampleIndex] = []
        for fi in self.file_indices:
            n_steps = int(self.files[fi]["n_steps"])
            # we create samples for t=0..n_steps-1 so that t+1 exists
            last_t = n_steps - 1
            if t_max is not None:
                last_t = min(last_t, int(t_max))
            for t in range(0, last_t + 1, int(t_stride)):
                self.samples.append(SampleIndex(file_idx=fi, t=t))

        # Cache commonly used per-file constants (dt,tau -> log10)
        self._dt_log: Dict[int, float] = {}
        self._tau_log: Dict[int, float] = {}
        for fi in self.file_indices:
            rec = self.files[fi]
            dt = float(rec["dt"])
            tau = float(rec["tau_tilde"])
            # guard: dt, tau must be positive
            if dt <= 0 or tau <= 0:
                raise ValueError(f"non-positive dt/tau in file record: {rec['path']}")
            self._dt_log[fi] = float(np.log10(dt))
            self._tau_log[fi] = float(np.log10(tau))

        # Basic consistency checks (nx fixed)
        nx0 = int(self.files[self.file_indices[0]]["nx"])
        for fi in self.file_indices[1:10]:  # quick check subset
            if int(self.files[fi]["nx"]) != nx0:
                raise ValueError("nx is not fixed across files. Adjust dataset to handle variable nx.")
        self.nx = nx0

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npz_arrays(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.load(path, allow_pickle=False)
        return z["n"], z["u"], z["T"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        rec = self.files[s.file_idx]
        path = rec["path"]

        n, u, T = self._load_npz_arrays(path)
        t = s.t

        # (nx,) float32
        n_t = n[t].astype(np.float32, copy=False)
        u_t = u[t].astype(np.float32, copy=False)
        T_t = T[t].astype(np.float32, copy=False)

        n_tp1 = n[t + 1].astype(np.float32, copy=False)
        u_tp1 = u[t + 1].astype(np.float32, copy=False)
        T_tp1 = T[t + 1].astype(np.float32, copy=False)

        # scalar conditions, broadcast to (nx,)
        logdt = np.full((self.nx,), self._dt_log[s.file_idx], dtype=np.float32)
        logtau = np.full((self.nx,), self._tau_log[s.file_idx], dtype=np.float32)

        x = np.stack([n_t, u_t, T_t, logdt, logtau], axis=0)      # (5, nx)
        y = np.stack([n_tp1, u_tp1, T_tp1], axis=0)              # (3, nx)

        xt = torch.from_numpy(x).to(dtype=self.dtype)
        yt = torch.from_numpy(y).to(dtype=self.dtype)

        if self.device is not None:
            xt = xt.to(self.device, non_blocking=True)
            yt = yt.to(self.device, non_blocking=True)

        return xt, yt

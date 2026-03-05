from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class SampleIndex:
    case_idx: int
    t: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


class BGK1D1VShardDeltaDataset(Dataset):
    """
    Sharded PT dataset (v2): 1 sample = (case_id, t)

    X: (5, nx) = [n(t), u(t), T(t), log10(dt), log10(tau)]
    Y: (3, nx)
      - target="dw":  [Δn, Δu, ΔT]
      - target="dnu": [Δn, Δ(nu), ΔT]
    """

    def __init__(
        self,
        dataset_manifest_path: str | Path,
        split: str = "train",
        *,
        split_key: str = "split_iid",
        target: str = "dnu",
        dtype: torch.dtype = torch.float32,
        cache_shards: int = 2,
    ):
        super().__init__()

        split = str(split)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        target_norm = str(target).strip().lower()
        if target_norm not in ("dw", "dnu"):
            raise ValueError(f"target must be 'dw' or 'dnu', got {target!r}")

        self.dataset_manifest_path = Path(dataset_manifest_path).resolve()
        self.dataset_manifest = _read_json(self.dataset_manifest_path)
        self.target = target_norm
        self.dtype = dtype
        self.split_key = str(split_key)
        self.cache_shards = max(int(cache_shards), 1)

        if str(self.dataset_manifest.get("format", "")) != "kineticEQ_BGK1D1V_pt_v2":
            raise ValueError(
                "manifest format mismatch: expected 'kineticEQ_BGK1D1V_pt_v2', "
                f"got {self.dataset_manifest.get('format')!r}"
            )

        data_root = self.dataset_manifest_path.parent
        cm_rel = self.dataset_manifest.get("paths", {}).get("case_manifest", "case_manifest.jsonl")
        self.case_manifest_path = (data_root / cm_rel).resolve()

        records_all = _read_jsonl(self.case_manifest_path)
        if not records_all:
            raise RuntimeError(f"No records found in {self.case_manifest_path}")

        if self.split_key not in records_all[0]:
            raise KeyError(f"split key {self.split_key!r} not found in case manifest")

        self.records = [r for r in records_all if str(r[self.split_key]) == split]
        if not self.records:
            raise RuntimeError(f"No records for split={split}, split_key={self.split_key}")

        self._root = data_root
        self._offsets: list[int] = [0]
        total = 0
        for rec in self.records:
            n_steps = int(rec["n_steps"])
            total += max(n_steps, 0)
            self._offsets.append(total)
        self._total_len = total

        self._const_cache: dict[tuple[int, float, float], tuple[torch.Tensor, torch.Tensor]] = {}
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self._total_len

    def _locate(self, idx: int) -> SampleIndex:
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)

        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._offsets[mid + 1]:
                hi = mid
            else:
                lo = mid + 1

        case_idx = lo
        t = idx - self._offsets[case_idx]
        return SampleIndex(case_idx=case_idx, t=t)

    def _load_shard(self, shard_rel: str) -> dict[str, Any]:
        key = str(shard_rel)
        if key in self._shard_cache:
            payload = self._shard_cache.pop(key)
            self._shard_cache[key] = payload
            return payload

        shard_path = (self._root / key).resolve()
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._shard_cache[key] = payload

        while len(self._shard_cache) > self.cache_shards:
            self._shard_cache.popitem(last=False)

        return payload

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self._locate(idx)
        rec = self.records[s.case_idx]

        payload = self._load_shard(str(rec["shard_path"]))
        W = payload["W"]

        frame_start = int(rec["frame_start"])
        t = int(s.t)
        g0 = frame_start + t
        g1 = g0 + 1

        w0 = W[g0]  # (3, nx)
        w1 = W[g1]  # (3, nx)

        n_t = w0[0].to(dtype=torch.float32)
        u_t = w0[1].to(dtype=torch.float32)
        T_t = w0[2].to(dtype=torch.float32)

        n_tp1 = w1[0].to(dtype=torch.float32)
        u_tp1 = w1[1].to(dtype=torch.float32)
        T_tp1 = w1[2].to(dtype=torch.float32)

        nx = int(rec["nx"])
        logdt = float(rec["log10_dt"])
        logtau = float(rec["log10_tau"])

        ckey = (nx, logdt, logtau)
        if ckey not in self._const_cache:
            self._const_cache[ckey] = (
                torch.full((nx,), logdt, dtype=torch.float32),
                torch.full((nx,), logtau, dtype=torch.float32),
            )
        logdt_x, logtau_x = self._const_cache[ckey]

        x = torch.stack([n_t, u_t, T_t, logdt_x, logtau_x], dim=0).to(dtype=self.dtype)

        dn = n_tp1 - n_t
        dT = T_tp1 - T_t

        if self.target == "dw":
            du = u_tp1 - u_t
            y = torch.stack([dn, du, dT], dim=0).to(dtype=self.dtype)
        else:
            m_t = n_t * u_t
            m_tp1 = n_tp1 * u_tp1
            dm = m_tp1 - m_t
            y = torch.stack([dn, dm, dT], dim=0).to(dtype=self.dtype)

        return x, y

    def close(self) -> None:
        self._shard_cache.clear()
        self._const_cache.clear()

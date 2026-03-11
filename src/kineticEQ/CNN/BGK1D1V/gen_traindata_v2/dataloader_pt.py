# kineticEQ/CNN/BGK1D1V/gen_traindata_v2/dataloader_pt.py

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


@dataclass(frozen=True)
class SampleIndex:
    case_idx: int
    t: int


class BGK1D1VShardDeltaDataset(Dataset):
    def __init__(
        self,
        dataset_manifest_path: str | Path,
        split: str = "train",
        split_key: str = "split_iid",
        target: str = "dnu",
        dtype: torch.dtype = torch.float32,
        cache_shards: int = 2,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        split_key = str(split_key).strip()
        target = str(target).strip().lower()
        if target not in ("dnu", "dw"):
            raise ValueError(f"target must be 'dnu' or 'dw', got {target}")
        if int(cache_shards) < 1:
            raise ValueError("cache_shards must be >= 1")

        self.dataset_manifest_path = Path(dataset_manifest_path).resolve()
        self.dataset_root = self.dataset_manifest_path.parent
        self.dataset_manifest = _read_json(self.dataset_manifest_path)
        fmt = str(self.dataset_manifest.get("format", ""))
        if fmt != "kineticEQ_BGK1D1V_pt_v2":
            raise ValueError(f"unexpected dataset format: {fmt!r}")

        case_manifest_rel = self.dataset_manifest.get("paths", {}).get("case_manifest", "case_manifest.jsonl")
        case_manifest_path = _resolve_path(self.dataset_root, case_manifest_rel)
        records_all = _read_jsonl(case_manifest_path)

        self.records = [
            rec
            for rec in records_all
            if (not bool(rec.get("failed", False))) and str(rec.get(split_key, "")) == split
        ]
        if not self.records:
            raise RuntimeError(f"No records found for split={split!r} split_key={split_key!r}")

        self.split = split
        self.split_key = split_key
        self.target = target
        self.dtype = dtype
        self.cache_shards = int(cache_shards)
        self._const_cache: dict[tuple[int, float, float, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._shard_cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

        self._offsets: list[int] = [0]
        total = 0
        for rec in self.records:
            n_steps = int(rec["n_steps"])
            if n_steps < 1:
                raise ValueError(f"record case_id={rec.get('case_id')} has invalid n_steps={n_steps}")
            total += n_steps
            self._offsets.append(total)
        self._total_len = total

    def __len__(self) -> int:
        return self._total_len

    def _locate(self, idx: int) -> SampleIndex:
        if idx < 0 or idx >= self._total_len:
            raise IndexError(idx)
        case_idx = bisect_right(self._offsets, idx) - 1
        t = idx - self._offsets[case_idx]
        return SampleIndex(case_idx=case_idx, t=t)

    def _open_shard(self, shard_path: Path) -> dict[str, Any]:
        shard_path = shard_path.resolve()
        cached = self._shard_cache.get(shard_path)
        if cached is not None:
            self._shard_cache.move_to_end(shard_path)
            return cached

        obj = torch.load(shard_path, map_location="cpu", weights_only=False)
        fmt = str(obj.get("format", ""))
        if fmt != "kineticEQ_BGK1D1V_pt_v2":
            raise ValueError(f"unexpected shard format in {shard_path}: {fmt!r}")
        self._shard_cache[shard_path] = obj
        self._shard_cache.move_to_end(shard_path)
        while len(self._shard_cache) > self.cache_shards:
            self._shard_cache.popitem(last=False)
        return obj

    def _const_channels(self, nx: int, logdt: float, logtau: float) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(nx), float(logdt), float(logtau), self.dtype)
        cached = self._const_cache.get(key)
        if cached is not None:
            return cached
        out = (
            torch.full((nx,), float(logdt), dtype=self.dtype),
            torch.full((nx,), float(logtau), dtype=self.dtype),
        )
        self._const_cache[key] = out
        return out

    def __getitem__(self, idx: int):
        s = self._locate(idx)
        rec = self.records[s.case_idx]
        shard_path = _resolve_path(self.dataset_root, rec["shard_path"])
        shard = self._open_shard(shard_path)

        W = shard["W"]
        if not torch.is_tensor(W) or W.ndim != 3:
            raise ValueError(f"invalid W in shard {shard_path}")

        frame_start = int(rec["frame_start"])
        g0 = frame_start + int(s.t)
        g1 = g0 + 1

        n_t = W[g0, 0]
        u_t = W[g0, 1]
        T_t = W[g0, 2]
        n_tp1 = W[g1, 0]
        u_tp1 = W[g1, 1]
        T_tp1 = W[g1, 2]

        nx = int(rec["nx"])
        logdt = float(rec["log10_dt"])
        logtau = float(rec["log10_tau"])
        logdt_x, logtau_x = self._const_channels(nx=nx, logdt=logdt, logtau=logtau)

        x = torch.stack(
            [
                n_t.to(dtype=self.dtype),
                u_t.to(dtype=self.dtype),
                T_t.to(dtype=self.dtype),
                logdt_x,
                logtau_x,
            ],
            dim=0,
        )

        dn = (n_tp1 - n_t).to(dtype=self.dtype)
        dT = (T_tp1 - T_t).to(dtype=self.dtype)

        if self.target == "dw":
            du = (u_tp1 - u_t).to(dtype=self.dtype)
            y = torch.stack([dn, du, dT], dim=0)
        else:
            m_t = n_t * u_t
            m_tp1 = n_tp1 * u_tp1
            dm = (m_tp1 - m_t).to(dtype=self.dtype)
            y = torch.stack([dn, dm, dT], dim=0)

        return x, y

    def close(self) -> None:
        self._shard_cache.clear()
        self._const_cache.clear()

    def get_case_record(self, case_idx: int) -> dict[str, Any]:
        return dict(self.records[int(case_idx)])

    def get_sample_meta(self, idx: int) -> dict[str, Any]:
        s = self._locate(idx)
        rec = self.records[s.case_idx]
        return {
            "idx": int(idx),
            "case_idx": int(s.case_idx),
            "case_id": int(rec["case_id"]),
            "t": int(s.t),
            "tau_tilde": float(rec["tau_tilde"]),
            "dt": float(rec["dt"]),
            "split": str(rec.get(self.split_key, "")),
            "split_key": self.split_key,
            "target": self.target,
            "shard_path": str(rec["shard_path"]),
        }

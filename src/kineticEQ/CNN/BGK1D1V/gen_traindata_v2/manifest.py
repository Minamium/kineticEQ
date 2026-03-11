# kineticEQ/CNN/BGK1D1V/gen_traindata_v2/manifest.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


LOG_LEVEL_PRIORITY = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
}


def normalize_log_level(log_level: str) -> str:
    s = str(log_level).strip().lower()
    aliases = {
        "warn": "warning",
        "err": "error",
    }
    s = aliases.get(s, s)
    if s not in LOG_LEVEL_PRIORITY:
        raise ValueError(f"unsupported log_level: {log_level}")
    return s


def should_log(current_level: str, message_level: str) -> bool:
    return LOG_LEVEL_PRIORITY[current_level] <= LOG_LEVEL_PRIORITY[message_level]


def emit_log(log_level: str, message_level: str, message: str) -> None:
    if should_log(log_level, message_level):
        print(f"[{message_level.upper()}] {message}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning", "warn", "error", "err"])
    args = ap.parse_args()
    args.log_level = normalize_log_level(args.log_level)
    return args


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def assign_split_iid(records: list[dict[str, Any]], seed: int) -> None:
    ok = [rec for rec in records if not bool(rec.get("failed", False))]
    ok_sorted = sorted(ok, key=lambda x: int(x["case_id"]))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 2026)
    perm = torch.randperm(len(ok_sorted), generator=g).tolist()
    n = len(ok_sorted)
    n_train = int(round(0.8 * n))
    n_val = int(round(0.1 * n))
    train_cut = n_train
    val_cut = min(n, n_train + n_val)
    label_by_case: dict[int, str] = {}
    for pos, order_idx in enumerate(perm):
        case_id = int(ok_sorted[order_idx]["case_id"])
        if pos < train_cut:
            label_by_case[case_id] = "train"
        elif pos < val_cut:
            label_by_case[case_id] = "val"
        else:
            label_by_case[case_id] = "test"
    for rec in records:
        rec["split_iid"] = label_by_case.get(int(rec["case_id"]), "test")


def assign_split_ood_tau(records: list[dict[str, Any]]) -> None:
    ok = [rec for rec in records if not bool(rec.get("failed", False))]
    tau_values = sorted({float(rec["tau_tilde"]) for rec in ok})
    if len(tau_values) == 0:
        tau_to_label = {}
    elif len(tau_values) == 1:
        tau_to_label = {tau_values[0]: "train"}
    elif len(tau_values) == 2:
        tau_to_label = {tau_values[0]: "train", tau_values[1]: "test"}
    else:
        tau_to_label = {tau: "train" for tau in tau_values}
        tau_to_label[tau_values[0]] = "test"
        tau_to_label[tau_values[-1]] = "val"
    for rec in records:
        rec["split_ood_tau"] = tau_to_label.get(float(rec["tau_tilde"]), "train")


def load_generation_config(partial_dir: Path, log_level: str) -> dict[str, Any]:
    cfg_paths = sorted(partial_dir.glob("generation_config.rank*.json"))
    if not cfg_paths:
        raise FileNotFoundError(f"No generation_config.rank*.json found under {partial_dir}")

    cfgs = [read_json(p) for p in cfg_paths]
    world_sizes = {int(cfg["world_size"]) for cfg in cfgs}
    if len(world_sizes) != 1:
        raise ValueError(f"inconsistent world_size in generation configs: {sorted(world_sizes)}")
    world_size = next(iter(world_sizes))

    rank_to_cfg: dict[int, dict[str, Any]] = {}
    for cfg in cfgs:
        rank = int(cfg["rank"])
        if rank in rank_to_cfg:
            raise ValueError(f"duplicate generation config for rank={rank}")
        rank_to_cfg[rank] = cfg

    missing_ranks = [rank for rank in range(world_size) if rank not in rank_to_cfg]
    if missing_ranks:
        raise FileNotFoundError(f"missing generation config for ranks={missing_ranks}")

    canonical_cfgs = []
    for rank in range(world_size):
        cfg = dict(rank_to_cfg[rank])
        cfg.pop("rank", None)
        canonical_cfgs.append(json.dumps(cfg, sort_keys=True, ensure_ascii=False))
    if len(set(canonical_cfgs)) != 1:
        raise ValueError("generation configs are inconsistent across ranks")

    emit_log(log_level, "debug", f"loaded generation config files={len(cfg_paths)} world_size={world_size}")
    return rank_to_cfg[0]


def load_partial_records(partial_dir: Path, world_size: int, log_level: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_case_ids: set[int] = set()
    for rank in range(world_size):
        path = partial_dir / f"case_manifest.rank{rank:05d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing partial case manifest: {path}")
        rank_records = read_jsonl(path)
        emit_log(log_level, "debug", f"loaded partial case manifest rank={rank} cases={len(rank_records)}")
        for rec in rank_records:
            case_id = int(rec["case_id"])
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id detected: {case_id}")
            seen_case_ids.add(case_id)
            records.append(rec)
    records.sort(key=lambda x: int(x["case_id"]))
    return records


def validate_records(records: list[dict[str, Any]], out_dir: Path) -> None:
    shard_root = out_dir / "shards"
    if not shard_root.exists():
        raise FileNotFoundError(f"missing shard directory: {shard_root}")

    for rec in records:
        case_id = int(rec["case_id"])
        failed = bool(rec.get("failed", False))
        if failed:
            continue
        if int(rec.get("n_frames", 0)) <= 0:
            raise ValueError(f"case_id={case_id} has invalid n_frames")
        if int(rec.get("n_steps", 0)) <= 0:
            raise ValueError(f"case_id={case_id} has invalid n_steps")
        if int(rec.get("frame_start", -1)) < 0:
            raise ValueError(f"case_id={case_id} has invalid frame_start")
        shard_rel = str(rec.get("shard_path", ""))
        if not shard_rel:
            raise ValueError(f"case_id={case_id} has empty shard_path")
        shard_path = out_dir / shard_rel
        if not shard_path.exists():
            raise FileNotFoundError(f"case_id={case_id} references missing shard: {shard_path}")


def build_dataset_manifest(cfg: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [rec for rec in records if not bool(rec.get("failed", False))]
    shard_paths = sorted({str(rec["shard_path"]) for rec in ok_records if str(rec.get("shard_path", ""))})
    return {
        "format": "kineticEQ_BGK1D1V_pt_v2",
        "version": 2,
        "num_cases": int(len(ok_records)),
        "num_cases_requested": int(cfg["num_cases_requested"]),
        "num_failed_cases": int(sum(bool(rec.get("failed", False)) for rec in records)),
        "num_shards": int(len(shard_paths)),
        "cases_per_shard": int(cfg["cases_per_shard"]),
        "paths": {
            "case_manifest": "case_manifest.jsonl",
            "shard_dir": "shards",
        },
        "grid": dict(cfg["grid"]),
        "time": dict(cfg["time"]),
        "scheme": dict(cfg["scheme"]),
        "sampling": dict(cfg["sampling"]),
        "splits": {
            "iid": {"train": 0.8, "val": 0.1, "test": 0.1},
            "ood_tau_rule": "min_tau=test, max_tau=val, others=train",
        },
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    partial_dir = out_dir / "partials"
    if not partial_dir.exists():
        raise FileNotFoundError(f"missing partial directory: {partial_dir}")

    cfg = load_generation_config(partial_dir=partial_dir, log_level=args.log_level)
    world_size = int(cfg["world_size"])
    records = load_partial_records(partial_dir=partial_dir, world_size=world_size, log_level=args.log_level)
    validate_records(records=records, out_dir=out_dir)
    assign_split_iid(records, int(cfg["seed"]))
    assign_split_ood_tau(records)

    case_manifest_path = out_dir / "case_manifest.jsonl"
    dataset_manifest_path = out_dir / "dataset_manifest.json"
    write_jsonl(case_manifest_path, records)
    write_json(dataset_manifest_path, build_dataset_manifest(cfg=cfg, records=records))

    emit_log(
        args.log_level,
        "info",
        (
            f"manifest_written out_dir={str(out_dir)} num_cases={sum(not bool(rec.get('failed', False)) for rec in records)} "
            f"num_failed={sum(bool(rec.get('failed', False)) for rec in records)} num_shards={len({str(rec['shard_path']) for rec in records if str(rec.get('shard_path', ''))})}"
        ),
    )


if __name__ == "__main__":
    main()

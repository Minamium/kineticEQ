# kineticEQ/CNN/BGK1D1V/build_manifest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _read_meta(npz_path: Path) -> Dict[str, Any]:
    """
    Read meta stored as JSON bytes in the 'meta' field (np.bytes_).
    """
    with np.load(npz_path, allow_pickle=False) as z:
        meta_raw = z["meta"]

    # meta_raw is usually np.bytes_ or 0-d array; handle robustly
    if hasattr(meta_raw, "tobytes"):
        meta_str = meta_raw.tobytes().decode("utf-8")
    else:
        meta_str = str(meta_raw)

    return json.loads(meta_str)


def _infer_nsteps(npz_path: Path) -> int:
    """
    n has shape (n_steps+1, nx)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        n = z["n"]
    return int(n.shape[0] - 1)


def _collect_files(data_root: Path) -> List[Path]:
    return sorted(data_root.glob("shard_rank*/case_*.npz"))


def _split_case_ids(
    case_ids: List[int],
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, List[int]]:
    """
    Split by case_id (not by timestep), so time-series leakage is avoided.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(case_ids)), dtype=np.int64)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:n_train + n_val].tolist()
    test_ids = ids[n_train + n_val:].tolist()
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_manifest(
    data_root: Path,
    out_path: Path,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Build manifest JSON:
      - files: list of {path, case_id, dt, tau_tilde, nx, n_steps, meta}
      - splits: train/val/test lists of case_ids
      - bad_files: list of {path, error}
    strict=True:
      - if any bad file exists, raise RuntimeError (after scan) with example errors.
    strict=False:
      - keep going, and write manifest with bad_files filled.
    """
    files = _collect_files(data_root)
    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found under: {data_root}")

    records: List[Dict[str, Any]] = []
    bad_files: List[Dict[str, str]] = []
    case_ids: List[int] = []

    for p in files:
        try:
            meta = _read_meta(p)
            for k in ("case_id", "dt", "tau_tilde", "nx"):
                if k not in meta:
                    raise KeyError(f"missing meta key '{k}'")

            n_steps = _infer_nsteps(p)

            rec = {
                "path": str(p.relative_to(data_root).as_posix()),
                "case_id": int(meta["case_id"]),
                "dt": float(meta["dt"]),
                "tau_tilde": float(meta["tau_tilde"]),
                "nx": int(meta["nx"]),
                "n_steps": int(n_steps),
                "meta": meta,
            }
            records.append(rec)
            case_ids.append(int(meta["case_id"]))

        except Exception as e:
            bad_files.append({"path": str(p.relative_to(data_root).as_posix()), "error": f"{type(e).__name__}: {e}"})

    if len(records) == 0:
        raise RuntimeError(f"All files are unreadable. Example error: {bad_files[:3]}")

    if strict and len(bad_files) > 0:
        raise RuntimeError(
            f"Found {len(bad_files)} bad files under {data_root}. "
            f"Example: {bad_files[:3]}. "
            f"Use --ignore_bad to continue."
        )

    splits = _split_case_ids(case_ids, seed=seed, ratios=ratios)

    manifest = {
        "format": "kineticEQ_BGK1D1V_npz_v1",
        "data_root": str(data_root.as_posix()),
        "num_files": len(records),
        "num_bad_files": len(bad_files),
        "unique_cases": len(set(case_ids)),
        "seed": seed,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "splits": splits,
        "files": records,
        "bad_files": bad_files,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="mgpu_output",
                    help="Directory containing shard_rank*/case_*.npz")
    ap.add_argument("--out", type=str, default="mgpu_output_manifest.json",
                    help="Output manifest json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--ignore_bad", action="store_true",
                    help="If set, continue even if some npz files are unreadable.")
    args = ap.parse_args()

    ratios = (args.train, args.val, args.test)
    s = sum(ratios)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"train+val+test must be 1.0, got {s}")

    manifest = build_manifest(
        data_root=Path(args.data_root),
        out_path=Path(args.out),
        seed=args.seed,
        ratios=ratios,
        strict=(not args.ignore_bad),
    )

    print(f"[OK] wrote manifest: {args.out}")
    print(f"  files        : {manifest['num_files']}")
    print(f"  bad_files    : {manifest['num_bad_files']}")
    print(f"  unique_cases : {manifest['unique_cases']}")
    print(
        "  split sizes  : "
        f"train={len(manifest['splits']['train'])}, "
        f"val={len(manifest['splits']['val'])}, "
        f"test={len(manifest['splits']['test'])}"
    )


if __name__ == "__main__":
    main()
# kineticEQ/CNN/BGK1D1V/multi_train.py
"""Hyperparameter sweep orchestrator for train.py.

Each experiment runs as an independent subprocess on a dedicated GPU.
No changes to train.py are required.

Usage:
  python -m kineticEQ.CNN.BGK1D1V.multi_train \\
      --config config/multi_train.json \\
      --gpus 0,1,2,3,4,5,6,7 \\
      --save_root sweep_runs/exp001

  # dry-run (print commands only):
  python -m kineticEQ.CNN.BGK1D1V.multi_train --config ... --dry_run

If --config is omitted, defaults to
  kineticEQ/CNN/BGK1D1V/config/multi_train.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, IO

# ---------------------------------------------------------------------------
# train.py の store_true / nargs="+" 引数 (CLI変換で特殊扱いが必要)
# ---------------------------------------------------------------------------
_STORE_TRUE_FLAGS = frozenset({"amp", "sched_plateau", "warm_eval"})
_LIST_ARGS = frozenset({"dilation_cycle"})

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "config" / "multi_train.json"

# ---------------------------------------------------------------------------
# sweep config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def expand_grid(sweep: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of sweep axes."""
    if not sweep:
        return [{}]
    keys = sorted(sweep.keys())
    vals = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def validate_combo(params: dict[str, Any]) -> tuple[bool, str]:
    """Check parameter constraints. Returns (ok, reason)."""
    hidden = params.get("hidden")
    gn_groups = params.get("gn_groups")
    if hidden is not None and gn_groups is not None:
        if int(hidden) % int(gn_groups) != 0:
            return False, f"hidden={hidden} not divisible by gn_groups={gn_groups}"
    return True, ""


# ---------------------------------------------------------------------------
# run naming
# ---------------------------------------------------------------------------

_ABBREV = {
    "shock_ratio": "sr",
    "shock_q": "sq",
    "loss_kind": "lk",
    "loss_softmax_beta": "beta",
    "hidden": "h",
    "n_blocks": "nb",
    "kernel": "k",
    "bottleneck": "bn",
    "gn_groups": "gn",
    "dilation_cycle": "dc",
    "lr": "lr",
    "batch": "bs",
    "epochs": "ep",
    "delta_type": "dt",
    "seed": "s",
}


def make_run_name(idx: int, sweep_params: dict[str, Any]) -> str:
    """Short but informative directory name."""
    parts = [f"run_{idx:04d}"]
    for key in sorted(sweep_params.keys()):
        ab = _ABBREV.get(key, key[:4])
        val = sweep_params[key]
        if isinstance(val, float):
            parts.append(f"{ab}{val:g}")
        elif isinstance(val, list):
            parts.append(f"{ab}{'_'.join(map(str, val))}")
        else:
            parts.append(f"{ab}{val}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# CLI argument builder
# ---------------------------------------------------------------------------

def build_cli_args(
    base_args: dict[str, Any],
    sweep_params: dict[str, Any],
    save_dir: str,
) -> list[str]:
    """Merge base_args + sweep_params and convert to train.py CLI tokens."""
    merged = {**base_args, **sweep_params}
    merged["save_dir"] = save_dir

    tokens: list[str] = []
    for k, v in sorted(merged.items()):
        flag = f"--{k}"
        if k in _STORE_TRUE_FLAGS:
            if v:
                tokens.append(flag)
            # False -> omit entirely (argparse store_true default is False)
        elif k in _LIST_ARGS:
            tokens.append(flag)
            if isinstance(v, list):
                tokens.extend(str(x) for x in v)
            else:
                tokens.append(str(v))
        else:
            tokens.append(flag)
            tokens.append(str(v))
    return tokens


# ---------------------------------------------------------------------------
# GPU queue manager
# ---------------------------------------------------------------------------

class _RunningJob:
    __slots__ = ("proc", "entry", "log_fh", "gpu_id")

    def __init__(self, proc: subprocess.Popen, entry: dict, log_fh: IO, gpu_id: int):
        self.proc = proc
        self.entry = entry
        self.log_fh = log_fh
        self.gpu_id = gpu_id

    def close(self):
        try:
            self.log_fh.close()
        except Exception:
            pass


def run_sweep(
    config: dict,
    gpu_ids: list[int],
    save_root: str,
    dry_run: bool = False,
    poll_interval: float = 5.0,
):
    base_args = config.get("base_args", {})
    sweep = config.get("sweep", {})

    combos = expand_grid(sweep)

    # filter invalid
    valid_combos: list[dict[str, Any]] = []
    for i, combo in enumerate(combos):
        merged = {**base_args, **combo}
        ok, reason = validate_combo(merged)
        if ok:
            valid_combos.append(combo)
        else:
            print(f"[skip] combo {i}: {reason}")

    total = len(valid_combos)
    print(f"Total experiments: {total} (filtered from {len(combos)} grid points)")
    print(f"GPUs: {gpu_ids} (max parallel: {len(gpu_ids)})")

    save_root_path = Path(save_root)
    save_root_path.mkdir(parents=True, exist_ok=True)

    # build manifest
    manifest: list[dict[str, Any]] = []
    for idx, combo in enumerate(valid_combos):
        run_name = make_run_name(idx, combo)
        run_dir = str(save_root_path / run_name)
        manifest.append({
            "idx": idx,
            "run_name": run_name,
            "sweep_params": combo,
            "save_dir": run_dir,
        })

    manifest_path = save_root_path / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Sweep manifest: {manifest_path}")

    if dry_run:
        print("\n=== DRY RUN ===")
        for m in manifest:
            cli = build_cli_args(base_args, m["sweep_params"], m["save_dir"])
            print(f"  [{m['idx']:04d}] {sys.executable} -m kineticEQ.CNN.BGK1D1V.train "
                  + " ".join(cli))
        return

    # ---- launch loop ----
    running: dict[int, _RunningJob] = {}  # gpu_id -> job
    queue = list(manifest)
    completed: list[dict] = []
    failed: list[dict] = []
    skipped: list[dict] = []

    def _poll():
        freed: list[int] = []
        for gpu_id, job in list(running.items()):
            ret = job.proc.poll()
            if ret is not None:
                job.close()
                if ret == 0:
                    completed.append(job.entry)
                    print(f"[done] GPU {gpu_id}: {job.entry['run_name']} (rc={ret})")
                else:
                    failed.append(job.entry)
                    print(f"[FAIL] GPU {gpu_id}: {job.entry['run_name']} (rc={ret})")
                freed.append(gpu_id)
        for g in freed:
            del running[g]

    def _launch(gpu_id: int, entry: dict):
        run_dir = Path(entry["save_dir"])

        # resume: skip if last.pt already exists
        if (run_dir / "last.pt").exists():
            print(f"[skip] {entry['run_name']}: last.pt exists, resuming skipped")
            skipped.append(entry)
            return

        cli = build_cli_args(base_args, entry["sweep_params"], entry["save_dir"])
        cmd = [sys.executable, "-m", "kineticEQ.CNN.BGK1D1V.train"] + cli

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "stdout.log"
        fh = open(log_file, "w")

        print(f"[launch] GPU {gpu_id}: {entry['run_name']}")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        running[gpu_id] = _RunningJob(proc, entry, fh, gpu_id)

    t_start = time.time()

    while queue or running:
        _poll()

        # assign free GPUs
        free_gpus = [g for g in gpu_ids if g not in running]
        while free_gpus and queue:
            gpu_id = free_gpus.pop(0)
            entry = queue.pop(0)
            _launch(gpu_id, entry)

        if running:
            time.sleep(poll_interval)

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Sweep finished in {elapsed:.0f}s")
    print(f"  completed: {len(completed)}")
    print(f"  skipped (resume): {len(skipped)}")
    print(f"  failed: {len(failed)}")
    print(f"  total: {total}")

    if failed:
        print("\nFailed runs:")
        for e in failed:
            print(f"  - {e['run_name']}  (check {e['save_dir']}/stdout.log)")

    # aggregate
    aggregate_results(save_root_path, manifest)


# ---------------------------------------------------------------------------
# result aggregation
# ---------------------------------------------------------------------------

def aggregate_results(save_root: Path, manifest: list[dict] | None = None):
    """Read log.jsonl from each run and print a ranked summary."""
    if manifest is None:
        mp = save_root / "sweep_manifest.json"
        if not mp.exists():
            print(f"No sweep_manifest.json in {save_root}")
            return
        manifest = json.loads(mp.read_text())

    results: list[dict[str, Any]] = []
    for entry in manifest:
        run_dir = Path(entry["save_dir"])
        log_path = run_dir / "log.jsonl"
        if not log_path.exists():
            continue

        lines = log_path.read_text().strip().split("\n")
        if not lines:
            continue

        records = [json.loads(l) for l in lines]
        best_val = min(r.get("best_val", float("inf")) for r in records)
        last = records[-1]

        # best_speed from checkpoint (optional, torch needed)
        best_speed = _read_best_speed(run_dir)

        results.append({
            "run_name": entry["run_name"],
            "sweep_params": entry["sweep_params"],
            "best_val": best_val,
            "best_speed": best_speed,
            "final_epoch": last.get("epoch"),
            "final_lr": last.get("lr"),
            "val_rn_mae": last.get("val_rn_abs_mean"),
            "val_ru_mae": last.get("val_ru_abs_mean"),
            "val_rT_mae": last.get("val_rT_abs_mean"),
        })

    if not results:
        print("No results to aggregate.")
        return

    # rank by best_val
    results.sort(key=lambda r: r["best_val"])

    print(f"\n{'=' * 60}")
    n_show = min(10, len(results))
    print(f"Top {n_show} by best_val:")
    print(f"{'#':>3}  {'run_name':<45}  {'best_val':>12}  {'speed':>7}  sweep_params")
    print("-" * 100)
    for i, r in enumerate(results[:n_show]):
        sp = f"{r['best_speed']:.2f}" if r["best_speed"] is not None else "N/A"
        print(f"{i+1:3d}  {r['run_name']:<45}  {r['best_val']:12.6e}  {sp:>7}  {r['sweep_params']}")

    # save full summary
    summary_path = save_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFull summary: {summary_path}")


def _read_best_speed(run_dir: Path) -> float | None:
    speed_path = run_dir / "best_speed.pt"
    if not speed_path.exists():
        return None
    try:
        import torch
        ckpt = torch.load(speed_path, map_location="cpu", weights_only=False)
        return float(ckpt.get("best_speed", 0.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Hyperparameter sweep orchestrator for train.py",
    )
    ap.add_argument(
        "--config", type=str, default=None,
        help=f"Path to sweep config JSON (default: {_DEFAULT_CONFIG})",
    )
    ap.add_argument(
        "--gpus", type=str, default=None,
        help="Comma-separated GPU IDs (default: all available via torch.cuda)",
    )
    ap.add_argument(
        "--save_root", type=str, default=None,
        help="Root directory for sweep outputs (overrides config.save_root)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing",
    )
    ap.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip launching; only aggregate existing results",
    )
    ap.add_argument(
        "--poll_interval", type=float, default=5.0,
        help="Seconds between polling running processes (default: 5)",
    )
    ap.add_argument(
        "--traindata", type=str, default=None,
        help="Path to manifest.json for training data",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    print(f"Config: {config_path}")

    # --traindata: manifest.json path to inject into base_args
    if args.traindata:
        manifest_path = Path(args.traindata).resolve()
        if not manifest_path.exists():
            print(f"[error] {manifest_path} not found", file=sys.stderr)
            sys.exit(1)
        config.setdefault("base_args", {})["manifest"] = str(manifest_path)
        print(f"Training data: {manifest_path}")

    save_root = args.save_root or config.get("save_root", "sweep_runs/default")

    if args.aggregate_only:
        aggregate_results(Path(save_root))
        return

    # resolve GPU list
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        try:
            import torch
            gpu_ids = list(range(torch.cuda.device_count()))
        except Exception:
            gpu_ids = [0]

    if not gpu_ids:
        print("[error] No GPUs detected!", file=sys.stderr)
        sys.exit(1)

    run_sweep(
        config=config,
        gpu_ids=gpu_ids,
        save_root=save_root,
        dry_run=args.dry_run,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
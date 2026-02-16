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
_STORE_TRUE_FLAGS = frozenset({"amp", "sched_plateau", "warm_eval", "no_shuffle", "aa_enable"})
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
    "shock_loss_softmax_beta": "slb",
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
    "aa_enable": "aa",
}


def make_run_name(idx: int, sweep_params: dict[str, Any]) -> str:
    """Short but informative directory name."""
    parts = [f"run_{idx:04d}"]
    for key in sorted(sweep_params.keys()):
        ab = _ABBREV.get(key, key[:4])
        val = sweep_params[key]
        if isinstance(val, bool):
            parts.append(f"{ab}{int(val)}")
        elif isinstance(val, float):
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
    t_last_agg = 0.0  # last aggregate timestamp
    _AGG_INTERVAL = 60.0  # seconds between stats refresh

    while queue or running:
        _poll()

        # assign free GPUs
        free_gpus = [g for g in gpu_ids if g not in running]
        while free_gpus and queue:
            gpu_id = free_gpus.pop(0)
            entry = queue.pop(0)
            _launch(gpu_id, entry)

        # periodic stats update (files only, no console spam)
        now = time.time()
        if now - t_last_agg >= _AGG_INTERVAL:
            try:
                aggregate_results(save_root_path, manifest, quiet=True)
            except Exception:
                pass  # never crash the sweep loop for stats
            t_last_agg = now

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

def _parse_run_log(run_dir: Path) -> dict[str, Any] | None:
    """Extract comprehensive stats from a single run's log.jsonl."""
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return None
    text = log_path.read_text().strip()
    if not text:
        return None

    records = [json.loads(line) for line in text.split("\n")]
    epoch_recs = [r for r in records if not r.get("warm_eval")]
    warm_recs = [r for r in records if r.get("warm_eval")]

    if not epoch_recs:
        return None

    best_val = min(r.get("best_val", float("inf")) for r in epoch_recs)
    last_ep = epoch_recs[-1]

    # find the epoch record with best val_loss
    best_ep_rec = min(epoch_recs, key=lambda r: r.get("val_loss", float("inf")))

    total_train_sec = sum(r.get("train_time_sec", 0.0) for r in epoch_recs)

    # warm_eval stats (last available)
    picard_base = None
    picard_warm = None
    best_speed = None
    if warm_recs:
        last_warm = warm_recs[-1]
        picard_base = last_warm.get("picard_sum_base")
        picard_warm = last_warm.get("picard_sum_warm")
        best_speed = max(r.get("best_speed", 0.0) for r in warm_recs)

    # fallback: read best_speed.pt
    if best_speed is None or best_speed == 0.0:
        best_speed = _read_best_speed(run_dir)

    return {
        "best_val": best_val,
        "best_speed": best_speed,
        "final_epoch": last_ep.get("epoch"),
        "final_lr": last_ep.get("lr"),
        "total_train_sec": total_train_sec,
        "picard_base": picard_base,
        "picard_warm": picard_warm,
        "train_loss_last": last_ep.get("train_loss"),
        "val_loss_last": last_ep.get("val_loss"),
        "val_rn_mae": best_ep_rec.get("val_rn_abs_mean"),
        "val_ru_mae": best_ep_rec.get("val_ru_abs_mean"),
        "val_rT_mae": best_ep_rec.get("val_rT_abs_mean"),
        "val_rn_max": best_ep_rec.get("val_rn_abs_max"),
        "val_ru_max": best_ep_rec.get("val_ru_abs_max"),
        "val_rT_max": best_ep_rec.get("val_rT_abs_max"),
    }


def aggregate_results(save_root: Path, manifest: list[dict] | None = None, *,
                      quiet: bool = False):
    """Read log.jsonl from each run and produce ranked summary + TSV stats.

    If quiet=True, only write files (sweep_stats.tsv, sweep_summary.json)
    without printing to console.  Used for periodic mid-sweep updates.
    """
    if manifest is None:
        mp = save_root / "sweep_manifest.json"
        if not mp.exists():
            print(f"No sweep_manifest.json in {save_root}")
            return
        manifest = json.loads(mp.read_text())

    # --- collect all sweep param keys ---
    sweep_keys: list[str] = []
    if manifest:
        sweep_keys = sorted(manifest[0].get("sweep_params", {}).keys())

    results: list[dict[str, Any]] = []
    for entry in manifest:
        run_dir = Path(entry["save_dir"])
        stats = _parse_run_log(run_dir)

        sp = entry.get("sweep_params", {})
        row: dict[str, Any] = {
            "run_name": entry["run_name"],
            **{k: sp.get(k) for k in sweep_keys},
        }

        if stats is None:
            row["status"] = "no_log"
            row.update({k: None for k in [
                "best_val", "best_speed", "final_epoch", "total_train_sec",
                "picard_base", "picard_warm", "iter_ratio",
                "val_rn_mae", "val_ru_mae", "val_rT_mae", "val_rT_max",
                "train_loss_last", "val_loss_last", "final_lr",
            ]})
        else:
            pb = stats["picard_base"]
            pw = stats["picard_warm"]
            iter_ratio = (pb / max(pw, 1)) if pb is not None and pw is not None else None
            row.update({
                "status": "ok",
                "best_val": stats["best_val"],
                "best_speed": stats["best_speed"],
                "final_epoch": stats["final_epoch"],
                "total_train_sec": stats["total_train_sec"],
                "picard_base": pb,
                "picard_warm": pw,
                "iter_ratio": iter_ratio,
                "val_rn_mae": stats["val_rn_mae"],
                "val_ru_mae": stats["val_ru_mae"],
                "val_rT_mae": stats["val_rT_mae"],
                "val_rT_max": stats["val_rT_max"],
                "train_loss_last": stats["train_loss_last"],
                "val_loss_last": stats["val_loss_last"],
                "final_lr": stats["final_lr"],
            })

        results.append(row)

    if not results:
        if not quiet:
            print("No results to aggregate.")
        return

    # --- sort by best_speed desc (primary), best_val asc (secondary) ---
    def _sort_key(r):
        spd = r.get("best_speed")
        bv = r.get("best_val")
        return (-(spd if spd is not None else -1e9),
                bv if bv is not None else 1e9)
    results.sort(key=_sort_key)

    # --- write TSV ---
    tsv_cols = (
        ["rank", "run_name"] + sweep_keys +
        ["best_val", "best_speed", "iter_ratio",
         "picard_base", "picard_warm",
         "ep", "total_time_s", "train_loss", "val_loss",
         "val_rn_mae", "val_ru_mae", "val_rT_mae", "val_rT_max",
         "lr", "status"]
    )
    def _fmt(val, fmt_str=None):
        if val is None:
            return ""
        if isinstance(val, bool):
            return str(int(val))
        if fmt_str:
            return fmt_str.format(val)
        return str(val)

    tsv_path = save_root / "sweep_stats.tsv"
    with tsv_path.open("w") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for rank, r in enumerate(results, 1):
            cells = [
                str(rank),
                r["run_name"],
                *[_fmt(r.get(k)) for k in sweep_keys],
                _fmt(r.get("best_val"), "{:.6e}"),
                _fmt(r.get("best_speed"), "{:.3f}"),
                _fmt(r.get("iter_ratio"), "{:.2f}"),
                _fmt(r.get("picard_base")),
                _fmt(r.get("picard_warm")),
                _fmt(r.get("final_epoch")),
                _fmt(r.get("total_train_sec"), "{:.1f}"),
                _fmt(r.get("train_loss_last"), "{:.6e}"),
                _fmt(r.get("val_loss_last"), "{:.6e}"),
                _fmt(r.get("val_rn_mae"), "{:.4e}"),
                _fmt(r.get("val_ru_mae"), "{:.4e}"),
                _fmt(r.get("val_rT_mae"), "{:.4e}"),
                _fmt(r.get("val_rT_max"), "{:.4e}"),
                _fmt(r.get("final_lr"), "{:.2e}"),
                r.get("status", ""),
            ]
            f.write("\t".join(cells) + "\n")

    # --- save JSON ---
    summary_path = save_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if quiet:
        return

    # --- console summary ---
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_total = len(results)
    print(f"\n{'=' * 80}")
    print(f"Sweep stats: {n_ok}/{n_total} runs completed")
    print(f"{'=' * 80}")

    # top N by speed
    top_n = min(15, n_ok)
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        hdr = (f"{'#':>3}  {'AA':>2}  {'sr':>5}  {'sq':>5}  {'slb':>5}"
               f"  {'best_val':>11}  {'speed':>6}  {'iter_r':>6}"
               f"  {'ep':>3}  {'time_s':>7}  {'rT_mae':>9}  {'rT_max':>9}")
        print(f"\nTop {top_n} by best_speed:")
        print(hdr)
        print("-" * len(hdr))
        for i, r in enumerate(ok_results[:top_n]):
            aa_str = "Y" if r.get("aa_enable") else "N"
            sp_str = f"{r['best_speed']:.2f}" if r.get("best_speed") is not None else "N/A"
            ir_str = f"{r['iter_ratio']:.1f}" if r.get("iter_ratio") is not None else "N/A"
            rt_mae = f"{r['val_rT_mae']:.3e}" if r.get("val_rT_mae") is not None else "N/A"
            rt_max = f"{r['val_rT_max']:.3e}" if r.get("val_rT_max") is not None else "N/A"
            sr_v = r.get("shock_ratio", "")
            sq_v = r.get("shock_q", "")
            slb_v = r.get("shock_loss_softmax_beta", "")
            print(f"{i+1:3d}  {aa_str:>2}  {sr_v:>5}  {sq_v:>5}  {slb_v:>5}"
                  f"  {r['best_val']:11.5e}  {sp_str:>6}  {ir_str:>6}"
                  f"  {r.get('final_epoch',''):>3}  {r.get('total_train_sec',0):7.0f}"
                  f"  {rt_mae:>9}  {rt_max:>9}")

    print(f"\nFiles written:")
    print(f"  {tsv_path}")
    print(f"  {summary_path}")


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
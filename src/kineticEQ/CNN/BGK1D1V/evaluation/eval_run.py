"""CLI entry point for config-driven BGK1D1V evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .engine import EvaluationEngine
from .eval_config_builder import load_eval_spec
from .results import eval_report_to_dict
from .spec import CachePolicy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="evaluation JSON config")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cache_dir", type=str, default=None, help="override cache dir from config")
    ap.add_argument("--out", type=str, required=True, help="output JSON report path")
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    spec = load_eval_spec(args.config)
    if args.cache_dir:
        if spec.cache_policy is None:
            spec = replace(spec, cache_policy=CachePolicy(cache_dir=str(args.cache_dir)))
        else:
            spec = replace(spec, cache_policy=replace(spec.cache_policy, cache_dir=str(args.cache_dir)))

    engine = EvaluationEngine(
        device=str(args.device),
        cache_dir=(str(args.cache_dir) if args.cache_dir else None),
        verbose=(not bool(args.quiet)),
    )
    report = engine.run_spec(spec)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_report_to_dict(report), indent=2, ensure_ascii=False))
    print(f"[eval] wrote report: {out_path}", flush=True)


if __name__ == "__main__":
    main()

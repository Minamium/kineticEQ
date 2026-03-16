# kineticEQ/CNN/BGK1D1V/evaluation/cache.py
"""Run-result cache helpers for the phase-1 evaluation engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from .results import RunResult, run_result_to_dict
from .spec import EvalCase, EvalTarget


def _jsonable(x: Any) -> Any:
    """Convert nested dataclass payloads into stable JSON-serializable values."""

    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def make_cache_key(case: EvalCase, target: EvalTarget, eval_mode: str, profile: bool, device: str | None = None) -> str:
    """Build a stable cache key for one run result.

    Device is optional and omitted by default because the solver semantics are
    expected to be independent of the execution device for phase-1 evaluation.
    """

    payload = {
        'case': _jsonable(case),
        'target': _jsonable(target),
        'eval_mode': str(eval_mode),
        'profile': bool(profile),
    }
    if device is not None:
        payload['device'] = str(device)
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def get_cache_path(cache_dir: str | Path, key: str) -> Path:
    """Return the JSON cache path for a run-result key."""

    return Path(cache_dir) / f'{key}.json'


def _run_result_from_dict(obj: dict[str, Any]) -> RunResult:
    """Rehydrate a cached RunResult from JSON.

    Phase-1 keeps a single JSON schema. If results.py grows a schema version in
    later phases, this is the compatibility boundary to extend.
    """

    from .results import StepRecord, PicardRecord
    step_records = [StepRecord(**r) for r in obj.get('step_records', [])]
    picard_records_by_step_raw = obj.get('picard_records_by_step', None)
    picard_records_by_step = None
    if picard_records_by_step_raw is not None:
        picard_records_by_step = [
            [PicardRecord(**rec) for rec in recs]
            for recs in picard_records_by_step_raw
        ]
    return RunResult(
        case_name=obj['case_name'],
        target_name=obj['target_name'],
        mode=obj['mode'],
        step_records=step_records,
        picard_records_by_step=picard_records_by_step,
        final_moments=obj.get('final_moments', {}),
        meta=obj.get('meta', {}),
    )


def load_run_cache(cache_dir: str | Path, key: str) -> RunResult | None:
    """Load a cached run result if it exists."""

    path = get_cache_path(cache_dir, key)
    if not path.exists():
        return None
    return _run_result_from_dict(json.loads(path.read_text()))


def save_run_cache(cache_dir: str | Path, key: str, result: RunResult) -> None:
    """Persist one run result as JSON."""

    path = get_cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_result_to_dict(result), indent=2, ensure_ascii=False))


def load_or_run(
    *,
    cache_dir: str | Path,
    key: str,
    enabled: bool,
    runner: Callable[[], RunResult],
) -> tuple[RunResult, bool]:
    """Return a cached run result or execute and store it.

    The boolean flag indicates whether the returned result was loaded from
    cache.
    """

    if enabled:
        cached = load_run_cache(cache_dir, key)
        if cached is not None:
            return cached, True
    result = runner()
    if enabled:
        save_run_cache(cache_dir, key, result)
    return result, False

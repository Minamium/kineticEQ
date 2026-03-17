"""JSON config loader for the phase-1 evaluation engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .spec import CachePolicy, EvalCase, EvalSpec, EvalTarget, SweepSpec


def _load_json(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    obj = json.loads(cfg_path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"eval config must be a JSON object: {cfg_path}")
    return obj


def _clean_mapping(obj: dict[str, Any], *, allow: set[str], ctx: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in obj.items():
        key_s = str(key)
        if key_s.startswith("_"):
            continue
        if key_s not in allow:
            unknown.append(key_s)
            continue
        out[key_s] = value
    if unknown:
        unknown_s = ", ".join(sorted(unknown))
        raise ValueError(f"unknown keys in {ctx}: {unknown_s}")
    return out


def _parse_case(obj: dict[str, Any]) -> EvalCase:
    clean = _clean_mapping(
        obj,
        allow={"name", "tau", "dt", "T_total", "nx", "nv", "Lx", "v_max", "initial_condition", "tags"},
        ctx="case",
    )
    return EvalCase(**clean)


def _parse_target(obj: dict[str, Any], *, ctx: str) -> EvalTarget:
    clean = _clean_mapping(
        obj,
        allow={"name", "overrides", "tags"},
        ctx=ctx,
    )
    clean.setdefault("overrides", {})
    clean.setdefault("tags", [])
    return EvalTarget(**clean)


def _parse_cache_policy(obj: dict[str, Any] | None) -> CachePolicy | None:
    if obj is None:
        return None
    clean = _clean_mapping(
        obj,
        allow={"enabled", "reuse_baseline", "reuse_targets", "cache_dir"},
        ctx="cache_policy",
    )
    return CachePolicy(**clean)


def _parse_sweep(obj: dict[str, Any] | None) -> SweepSpec | None:
    if obj is None:
        return None
    clean = _clean_mapping(
        obj,
        allow={"parameters", "mode"},
        ctx="sweep",
    )
    return SweepSpec(**clean)


def load_eval_spec(path: str | Path) -> EvalSpec:
    """Load one EvalSpec from a JSON file."""

    obj = _load_json(path)
    clean = _clean_mapping(
        obj,
        allow={"name", "case", "baseline", "targets", "target", "sweep", "cache_policy", "eval_mode", "profile"},
        ctx=f"eval config {Path(path)}",
    )

    if "case" not in clean or "baseline" not in clean:
        raise ValueError("eval config requires at least 'case' and 'baseline'")

    targets_obj = clean.get("targets", None)
    target_obj = clean.get("target", None)
    if targets_obj is not None and target_obj is not None:
        raise ValueError("use either 'target' or 'targets', not both")
    if targets_obj is None:
        if target_obj is None:
            raise ValueError("eval config requires 'target' or 'targets'")
        targets_obj = [target_obj]

    if not isinstance(targets_obj, list) or not targets_obj:
        raise ValueError("'targets' must be a non-empty list")

    return EvalSpec(
        name=str(clean["name"]),
        case=_parse_case(clean["case"]),
        baseline=_parse_target(clean["baseline"], ctx="baseline"),
        targets=[_parse_target(t, ctx=f"targets[{idx}]") for idx, t in enumerate(targets_obj)],
        sweep=_parse_sweep(clean.get("sweep")),
        cache_policy=_parse_cache_policy(clean.get("cache_policy")),
        eval_mode=str(clean.get("eval_mode", "rollout_pair")),
        profile=bool(clean.get("profile", False)),
    )

"""Declarative evaluation specifications for the BGK1D1V evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """A single physical evaluation case before solver-specific overrides."""

    name: str
    tau: float
    dt: float
    T_total: float
    nx: int
    nv: int
    Lx: float = 1.0
    v_max: float = 10.0
    initial_condition: Any = field(default_factory=tuple)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalTarget:
    """A named override set applied on top of an EvalCase-derived Config."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SweepSpec:
    """Parameter sweep definition for expanding one EvalSpec into many instances."""

    parameters: dict[str, list[Any]]
    mode: str = 'grid'


@dataclass(frozen=True)
class CachePolicy:
    """Run-result cache policy for baseline and target evaluations."""

    enabled: bool = True
    reuse_baseline: bool = True
    reuse_targets: bool = False
    cache_dir: str | None = None


@dataclass(frozen=True)
class EvalSpec:
    """Top-level evaluation specification used by the engine.

    Supported eval_mode values in phase 1.1:
    - `rollout_pair`
    - `baseline_only`
    - `warm_only`

    `teacher_forcing` is planned for a later phase.
    """

    name: str
    case: EvalCase
    baseline: EvalTarget
    targets: list[EvalTarget]
    sweep: SweepSpec | None = None
    cache_policy: CachePolicy | None = None
    eval_mode: str = 'rollout_pair'
    profile: bool = False


@dataclass(frozen=True)
class EvalInstance:
    """A concrete evaluation instance after optional sweep expansion."""

    name: str
    case: EvalCase
    baseline: EvalTarget
    targets: list[EvalTarget]
    cache_policy: CachePolicy | None
    eval_mode: str
    profile: bool
    sweep_values: dict[str, Any] = field(default_factory=dict)


def clone_with_updates(spec: EvalSpec, **updates: Any) -> EvalSpec:
    """Return a shallowly updated copy of an EvalSpec."""

    return replace(spec, **updates)


def _apply_path_update(spec: EvalSpec, key: str, value: Any) -> EvalSpec:
    parts = key.split('.')
    if len(parts) < 2:
        raise ValueError(f'invalid sweep key: {key!r}')

    head = parts[0]
    tail = '.'.join(parts[1:])

    if head == 'case':
        case = replace(spec.case, **{parts[1]: value})
        return replace(spec, case=case)

    if head == 'baseline':
        ov = dict(spec.baseline.overrides)
        ov[tail] = value
        return replace(spec, baseline=replace(spec.baseline, overrides=ov))

    if head == 'target':
        if len(spec.targets) != 1:
            raise ValueError('target.* sweep is only supported when spec.targets has length 1')
        tgt = spec.targets[0]
        ov = dict(tgt.overrides)
        ov[tail] = value
        return replace(spec, targets=[replace(tgt, overrides=ov)])

    raise ValueError(f'unsupported sweep prefix: {head!r}')


def expand_sweep(spec: EvalSpec) -> list[EvalInstance]:
    """Expand an EvalSpec sweep into concrete instances.

    `target.*` sweep keys are currently supported only when `spec.targets`
    contains exactly one target.
    """

    if spec.sweep is None:
        return [
            EvalInstance(
                name=spec.name,
                case=spec.case,
                baseline=spec.baseline,
                targets=list(spec.targets),
                cache_policy=spec.cache_policy,
                eval_mode=spec.eval_mode,
                profile=spec.profile,
                sweep_values={},
            )
        ]

    if spec.sweep.mode != 'grid':
        raise ValueError(f'unsupported sweep mode: {spec.sweep.mode!r}')

    keys = list(spec.sweep.parameters.keys())
    values = [spec.sweep.parameters[k] for k in keys]
    out: list[EvalInstance] = []
    for combo in product(*values):
        cur = spec
        sweep_values = dict(zip(keys, combo))
        for k, v in sweep_values.items():
            cur = _apply_path_update(cur, k, v)
        out.append(
            EvalInstance(
                name=cur.name,
                case=cur.case,
                baseline=cur.baseline,
                targets=list(cur.targets),
                cache_policy=cur.cache_policy,
                eval_mode=cur.eval_mode,
                profile=cur.profile,
                sweep_values=sweep_values,
            )
        )
    return out

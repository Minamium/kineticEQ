"""Train-time frontend for cached warm-evaluation runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .engine import EvaluationEngine
from .spec import CachePolicy, EvalCase, EvalSpec, EvalTarget


def _default_initial_condition() -> tuple[dict[str, Any], ...]:
    """Return the fixed warm-eval shock-tube preset used during training."""

    return (
        {'x_range': (0.0, 0.5), 'n': 1.0, 'u': 0.0, 'T': 1.0},
        {'x_range': (0.5, 1.0), 'n': 0.125, 'u': 0.0, 'T': 0.8},
    )


def _aa_overrides_from_args(args: Any) -> dict[str, Any]:
    """Extract Anderson-acceleration overrides from train arguments."""

    return {
        'aa_enable': bool(args.aa_enable),
        'aa_m': int(args.aa_m),
        'aa_beta': float(args.aa_beta),
        'aa_stride': int(args.aa_stride),
        'aa_start_iter': int(args.aa_start_iter),
        'aa_reg': float(args.aa_reg),
        'aa_alpha_max': float(args.aa_alpha_max),
    }


def build_train_eval_spec_from_args(args: Any, *, cache_dir: str | None = None) -> EvalSpec:
    """Build the phase-1 train-time evaluation spec from argparse values."""

    case = EvalCase(
        name=(
            f'warm_eval_tau{float(args.warm_eval_tau):.3e}_dt{float(args.warm_eval_dt):.3e}'
            f'_nx{int(args.warm_eval_nx)}_nv{int(args.warm_eval_nv)}'
        ),
        tau=float(args.warm_eval_tau),
        dt=float(args.warm_eval_dt),
        T_total=float(args.warm_eval_T_total),
        nx=int(args.warm_eval_nx),
        nv=int(args.warm_eval_nv),
        Lx=1.0,
        v_max=10.0,
        initial_condition=_default_initial_condition(),
        tags=['train', 'warm_eval'],
    )
    common = {
        'picard_iter': int(args.warm_eval_picard_iter),
        'picard_tol': float(args.warm_eval_picard_tol),
        'abs_tol': float(args.warm_eval_abs_tol),
        'conv_type': str(args.conv_type),
        **_aa_overrides_from_args(args),
    }
    baseline = EvalTarget(name='baseline', overrides=dict(common), tags=['baseline'])
    warm = EvalTarget(name='warm', overrides=dict(common), tags=['warm'])
    cache_policy = CachePolicy(
        enabled=True,
        reuse_baseline=True,
        reuse_targets=False,
        cache_dir=cache_dir,
    )
    return EvalSpec(
        name='train_warm_eval',
        case=case,
        baseline=baseline,
        targets=[warm],
        cache_policy=cache_policy,
        eval_mode='rollout_pair',
        profile=False,
    )


class TrainEvaluator:
    """Lightweight train-time wrapper around EvaluationEngine."""

    def __init__(self, eval_spec: EvalSpec, device: str, cache_dir: str | None = None, verbose: bool = True):
        cache_dir_final = cache_dir or (eval_spec.cache_policy.cache_dir if eval_spec.cache_policy else None)
        if eval_spec.cache_policy is None:
            eval_spec = replace(eval_spec, cache_policy=CachePolicy(cache_dir=cache_dir_final))
        elif cache_dir_final is not None and eval_spec.cache_policy.cache_dir != cache_dir_final:
            eval_spec = replace(eval_spec, cache_policy=replace(eval_spec.cache_policy, cache_dir=cache_dir_final))
        self.eval_spec = eval_spec
        self.engine = EvaluationEngine(device=device, cache_dir=cache_dir_final, verbose=verbose)
        self._baseline_result = None

    def prepare_baseline(self):
        """Ensure the baseline rollout exists and is cached for later epochs."""

        self._baseline_result = self.engine.resolve_run(
            self.eval_spec.case,
            self.eval_spec.baseline,
            'baseline_only',
            self.eval_spec.profile,
            self.eval_spec.cache_policy,
        )
        return self._baseline_result

    def evaluate_checkpoint(self, ckpt_path: str | Path, extra_target_overrides: dict | None = None):
        """Evaluate one checkpoint against the cached baseline rollout."""

        if self._baseline_result is None:
            self.prepare_baseline()

        target_template = self.eval_spec.targets[0]
        overrides = dict(target_template.overrides)
        overrides['moments_cnn_modelpath'] = str(ckpt_path)
        if extra_target_overrides:
            overrides.update(extra_target_overrides)
        target = replace(target_template, overrides=overrides)
        target_result = self.engine.resolve_run(
            self.eval_spec.case,
            target,
            'warm_only',
            self.eval_spec.profile,
            self.eval_spec.cache_policy,
        )
        return self.engine.compare_runs(self._baseline_result, target_result)

    def evaluate_epoch(self, ckpt_path: str | Path, extra_target_overrides: dict | None = None) -> dict[str, Any]:
        """Return the compact per-epoch summary used by train.py logging."""

        comparison = self.evaluate_checkpoint(ckpt_path, extra_target_overrides=extra_target_overrides)
        return {
            'speedup_picard_sum': float(comparison.speedup_picard_sum),
            'picard_sum_base': int(comparison.picard_sum_base),
            'picard_sum_warm': int(comparison.picard_sum_target),
            'walltime_base_sec': float(comparison.walltime_base_sec),
            'walltime_warm_sec': float(comparison.walltime_target_sec),
            'speedup_walltime': (float(comparison.speedup_walltime) if comparison.speedup_walltime is not None else None),
        }

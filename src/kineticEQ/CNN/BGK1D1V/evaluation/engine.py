from __future__ import annotations

from dataclasses import replace

import torch

from .builders import build_cfg_for_target
from .cache import load_or_run, make_cache_key
from .results import ComparisonResult, EvalReport, RunResult, run_result_picard_sum, run_result_walltime_sum
from .runners import run_baseline_only, run_target_only
from .spec import CachePolicy, EvalCase, EvalInstance, EvalSpec, EvalTarget, expand_sweep


class EvaluationEngine:
    """Orchestrate evaluation runs and optional run-level caching."""

    def __init__(self, device: str = 'cuda', cache_dir: str | None = None, verbose: bool = True):
        self.device = str(device)
        self.cache_dir = cache_dir
        self.verbose = bool(verbose)

    def expand_spec(self, spec: EvalSpec) -> list[EvalInstance]:
        return expand_sweep(spec)

    def _resolve_cache_dir(self, cache_policy: CachePolicy | None) -> str | None:
        if cache_policy is not None and cache_policy.cache_dir:
            return str(cache_policy.cache_dir)
        if self.cache_dir:
            return str(self.cache_dir)
        return None

    def resolve_run(
        self,
        case: EvalCase,
        target: EvalTarget,
        eval_mode: str,
        profile: bool,
        cache_policy: CachePolicy | None,
    ) -> RunResult:
        cfg = build_cfg_for_target(case, target, self.device)
        n_steps = int(cfg.model_cfg.time.n_steps)
        torch_device = torch.device(self.device)
        cache_key = make_cache_key(case, target, eval_mode, profile)

        if eval_mode == 'baseline_only':
            runner = lambda: run_baseline_only(cfg, n_steps, torch_device, profile=profile, case_name=case.name, target_name=target.name)
            cache_enabled = bool(cache_policy.enabled and cache_policy.reuse_baseline) if cache_policy is not None else False
        elif eval_mode in ('warm_only', 'target_only'):
            runner = lambda: run_target_only(cfg, n_steps, torch_device, profile=profile, case_name=case.name, target_name=target.name)
            cache_enabled = bool(cache_policy.enabled and cache_policy.reuse_targets) if cache_policy is not None else False
        else:
            raise ValueError(f'unsupported resolve_run eval_mode: {eval_mode!r}')

        cache_dir = self._resolve_cache_dir(cache_policy)
        if cache_enabled and cache_dir is not None:
            result, hit = load_or_run(cache_dir=cache_dir, key=cache_key, enabled=True, runner=runner)
        else:
            result = runner()
            hit = False

        meta = dict(result.meta)
        meta['cache_hit'] = bool(hit)
        meta['cache_key'] = cache_key
        return replace(result, meta=meta)

    def compare_runs(self, base: RunResult, target: RunResult) -> ComparisonResult:
        picard_sum_base = run_result_picard_sum(base)
        picard_sum_target = run_result_picard_sum(target)
        walltime_base_sec = run_result_walltime_sum(base)
        walltime_target_sec = run_result_walltime_sum(target)
        speedup_picard_sum = float(picard_sum_base) / float(max(picard_sum_target, 1))
        speedup_walltime = None
        if walltime_target_sec > 0.0:
            speedup_walltime = float(walltime_base_sec / walltime_target_sec)
        return ComparisonResult(
            case_name=base.case_name,
            baseline_name=base.target_name,
            target_name=target.target_name,
            picard_sum_base=picard_sum_base,
            picard_sum_target=picard_sum_target,
            speedup_picard_sum=speedup_picard_sum,
            walltime_base_sec=walltime_base_sec,
            walltime_target_sec=walltime_target_sec,
            speedup_walltime=speedup_walltime,
            extra={},
        )

    def run_spec(self, spec: EvalSpec) -> EvalReport:
        run_results: list[RunResult] = []
        comparisons: list[ComparisonResult] = []

        for instance in self.expand_spec(spec):
            if instance.eval_mode == 'rollout_pair':
                baseline = self.resolve_run(instance.case, instance.baseline, 'baseline_only', instance.profile, instance.cache_policy)
                run_results.append(baseline)
                for target in instance.targets:
                    target_result = self.resolve_run(instance.case, target, 'warm_only', instance.profile, instance.cache_policy)
                    run_results.append(target_result)
                    comparisons.append(self.compare_runs(baseline, target_result))
            elif instance.eval_mode == 'baseline_only':
                baseline = self.resolve_run(instance.case, instance.baseline, 'baseline_only', instance.profile, instance.cache_policy)
                run_results.append(baseline)
            elif instance.eval_mode == 'warm_only':
                for target in instance.targets:
                    target_result = self.resolve_run(instance.case, target, 'warm_only', instance.profile, instance.cache_policy)
                    run_results.append(target_result)
            else:
                raise ValueError(f'unsupported EvalSpec.eval_mode: {instance.eval_mode!r}')

        return EvalReport(
            spec_name=spec.name,
            run_results=run_results,
            comparisons=comparisons,
            meta={'device': self.device},
        )

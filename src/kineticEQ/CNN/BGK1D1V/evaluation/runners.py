# kineticEQ/CNN/BGK1D1V/evaluation/runners.py
from __future__ import annotations

import time
from typing import Any

import torch

from kineticEQ import Engine, Config
from kineticEQ.core.schemes.BGK1D1V.bgk1d_utils.general.bgk1d_compute_moments import calculate_moments

from .results import RunResult, StepRecord


def _now_sync(device: torch.device) -> float:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _emit_progress(prefix: str, step: int, n_steps: int) -> None:
    """Print a coarse progress update for long evaluation rollouts."""

    print(f"[eval] {prefix} step {step}/{n_steps}", flush=True)


@torch.no_grad()
def _run_single(
    cfg: Config,
    n_steps: int,
    device: torch.device,
    *,
    mode: str,
    case_name: str,
    target_name: str,
    profile: bool = False,
    verbose: bool = False,
) -> RunResult:
    engine = Engine(cfg)

    step_records: list[StepRecord] = []
    if verbose:
        print(f"[eval] start mode={mode} case={case_name} target={target_name} n_steps={int(n_steps)}", flush=True)
    t0_all = _now_sync(device)
    log_stride = max(1, int(n_steps) // 10)
    for step in range(int(n_steps)):
        if verbose and (step % log_stride == 0 or step == int(n_steps) - 1):
            _emit_progress(f"mode={mode} target={target_name}", step, int(n_steps))
        t0 = _now_sync(device)
        engine.stepper(step)
        t1 = _now_sync(device)

        bench = getattr(engine.stepper, 'benchlog', None) or {}
        step_records.append(
            StepRecord(
                step=int(step),
                walltime_sec=float(t1 - t0),
                picard_iter=int(bench.get('picard_iter', -1)),
                final_std_resid=float(bench.get('std_picard_residual', float('nan'))),
                infer_time_sec=float(bench['t_infer_sec']) if 't_infer_sec' in bench else None,
                solver_time_sec=float(bench['t_picard_sec']) if 't_picard_sec' in bench else None,
            )
        )
    t1_all = _now_sync(device)

    n, u, T = calculate_moments(engine.state, engine.state.f)
    final_moments = {
        'n': n.detach().cpu().double().tolist(),
        'u': u.detach().cpu().double().tolist(),
        'T': T.detach().cpu().double().tolist(),
    }
    meta: dict[str, Any] = {
        'walltime_sec_total': float(t1_all - t0_all),
        'n_steps': int(n_steps),
        'device': str(device),
        'profile': bool(profile),
    }
    if verbose:
        print(
            f"[eval] done mode={mode} case={case_name} target={target_name} "
            f"wall={float(t1_all - t0_all):.3f}s",
            flush=True,
        )
    return RunResult(
        case_name=case_name,
        target_name=target_name,
        mode=mode,
        step_records=step_records,
        picard_records_by_step=None,
        final_moments=final_moments,
        meta=meta,
    )


def run_baseline_only(
    cfg_base: Config,
    n_steps: int,
    device: torch.device,
    profile: bool = False,
    *,
    case_name: str = 'case',
    target_name: str = 'baseline',
    verbose: bool = False,
) -> RunResult:
    return _run_single(
        cfg_base,
        n_steps,
        device,
        mode='baseline_only',
        case_name=case_name,
        target_name=target_name,
        profile=profile,
        verbose=verbose,
    )


def run_target_only(
    cfg_target: Config,
    n_steps: int,
    device: torch.device,
    profile: bool = False,
    *,
    case_name: str = 'case',
    target_name: str = 'target',
    verbose: bool = False,
) -> RunResult:
    return _run_single(
        cfg_target,
        n_steps,
        device,
        mode='warm_only',
        case_name=case_name,
        target_name=target_name,
        profile=profile,
        verbose=verbose,
    )


def run_rollout_pair(
    cfg_base: Config,
    cfg_target: Config,
    n_steps: int,
    device: torch.device,
    profile: bool = False,
    *,
    case_name: str = 'case',
    baseline_name: str = 'baseline',
    target_name: str = 'target',
    verbose: bool = False,
) -> tuple[RunResult, RunResult]:
    base = run_baseline_only(
        cfg_base,
        n_steps,
        device,
        profile=profile,
        case_name=case_name,
        target_name=baseline_name,
        verbose=verbose,
    )
    target = run_target_only(
        cfg_target,
        n_steps,
        device,
        profile=profile,
        case_name=case_name,
        target_name=target_name,
        verbose=verbose,
    )
    return base, target

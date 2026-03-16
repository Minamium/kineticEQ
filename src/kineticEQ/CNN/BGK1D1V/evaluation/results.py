# kineticEQ/CNN/BGK1D1V/evaluation/results.py
"""Result dataclasses for evaluation runs and pairwise comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepRecord:
    """Per-time-step summary emitted by a solver rollout."""

    step: int
    walltime_sec: float
    picard_iter: int
    final_std_resid: float
    infer_time_sec: float | None = None
    solver_time_sec: float | None = None


@dataclass(frozen=True)
class PicardRecord:
    """Optional per-Picard-iteration record used by later profiling phases."""

    iter: int
    residual: float
    walltime_sec: float | None = None
    aa_applied: bool | None = None


@dataclass(frozen=True)
class RunResult:
    """Result of one case-target execution.

    Phase 1 stores `picard_records_by_step` inline for JSON compatibility.
    Later phases are expected to split heavy raw traces out of the summary
    object.
    """

    case_name: str
    target_name: str
    mode: str
    step_records: list[StepRecord]
    picard_records_by_step: list[list[PicardRecord]] | None
    final_moments: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    """Derived comparison metrics between one baseline run and one target run."""

    case_name: str
    baseline_name: str
    target_name: str
    picard_sum_base: int
    picard_sum_target: int
    speedup_picard_sum: float
    walltime_base_sec: float
    walltime_target_sec: float
    speedup_walltime: float | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    """Container for all run results and derived comparisons in one spec."""

    spec_name: str
    run_results: list[RunResult]
    comparisons: list[ComparisonResult]
    meta: dict[str, Any] = field(default_factory=dict)


def run_result_picard_sum(result: RunResult) -> int:
    """Return the total Picard iteration count across all recorded steps."""

    return int(sum(max(int(r.picard_iter), 0) for r in result.step_records))


def run_result_walltime_sum(result: RunResult) -> float:
    """Return the summed step walltime across all recorded steps."""

    return float(sum(float(r.walltime_sec) for r in result.step_records))


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    return asdict(result)


def comparison_result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    return asdict(result)


def eval_report_to_dict(report: EvalReport) -> dict[str, Any]:
    return asdict(report)

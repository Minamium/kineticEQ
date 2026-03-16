# kineticEQ/CNN/BGK1D1V/evaluation/__init__.py
"""Shared evaluation core for train-time and standalone BGK1D1V runs.

The modules in `legacy/` and `eval_warmstart_debug.py` are compatibility
wrappers around the older standalone debug implementation.
"""

from .spec import EvalCase, EvalTarget, SweepSpec, CachePolicy, EvalSpec
from .results import StepRecord, PicardRecord, RunResult, ComparisonResult, EvalReport
from .builders import build_cfg_from_case, apply_target_overrides, build_cfg_for_target
from .engine import EvaluationEngine
from .train_eval import TrainEvaluator, build_train_eval_spec_from_args

__all__ = [
    'EvalCase', 'EvalTarget', 'SweepSpec', 'CachePolicy', 'EvalSpec',
    'StepRecord', 'PicardRecord', 'RunResult', 'ComparisonResult', 'EvalReport',
    'build_cfg_from_case', 'apply_target_overrides', 'build_cfg_for_target',
    'EvaluationEngine', 'TrainEvaluator', 'build_train_eval_spec_from_args',
]

# kineticEQ/CNN/BGK1D1V/evaluation/eval_warmstart_debug.py
"""Compatibility wrapper for the legacy standalone warm-eval debugger.

The main evaluation path now lives in `evaluation.engine` and
`evaluation.train_eval`. This module remains for backward compatibility until
its standalone CLI responsibilities are moved into `debug_eval.py` and
`cli_eval.py`.
"""

from .legacy.eval_warmstart_debug import *  # noqa: F401,F403

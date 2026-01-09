# kineticEQ/src/kineticEQ/api/logging_utils.py
from __future__ import annotations
import logging
from .config import Config, LogLevel

_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}

def apply_logging(config: Config) -> None:
    level = _LEVEL_MAP[config.log_level]

    pkg_logger = logging.getLogger("kineticEQ")
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False

    if not pkg_logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s : %(message)s"))
        pkg_logger.addHandler(h)

    # 既存ハンドラも含めて level を揃える（重要）
    for h in pkg_logger.handlers:
        h.setLevel(level)

# api/engine.py
from __future__ import annotations
from .result import Result
from .config import Config
import logging
from .logging_utils import apply_logging
logger = logging.getLogger(__name__)

# kineticEQの最上位wrapperクラス
class Engine:
    def __init__(self, config: Config, apply_logging_flag: bool = False):
        # configを保存
        self.config = config

        # apply_logging
        if apply_logging_flag:
            apply_logging(config)

        # info-log
        logger.info(
            "--- kineticEQ-Engine initialize complete ------\n"
            "  model    : %s\n"
            "  scheme   : %s\n"
            "  backend  : %s\n"
            "  device   : %s\n"
            "  log_level: %s",
            self.config.model_name,
            self.config.scheme_name,
            self.config.backend_name,
            self.config.device,
            self.config.log_level_name,
        )


        # debug-log
        logger.debug(f"config: {self.config.as_dict}")

    def run(self) -> Result:
        # debug-log
        logger.debug(f"run {self.config.model_name} {self.config.scheme_name}")

        return Result()

def run(config) -> Result:
    return Engine(config, apply_logging_flag=True).run()

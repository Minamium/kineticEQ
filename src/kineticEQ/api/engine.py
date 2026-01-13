# api/engine.py
from __future__ import annotations
from .result import Result
from .config import Config
import logging
from .logging_utils import apply_logging
from dataclasses import replace
import torch

# params
from kineticEQ.params.registry import default_model_cfg, expected_model_cfg_type, default_scheme_params 

# utillib
from kineticEQ.utillib.progress_bar import get_progress_bar, progress_write
from kineticEQ.utillib.pretty import format_kv_block
from kineticEQ.utillib.device_util import resolve_device

# core
from kineticEQ.core.states.registry import build_state
from kineticEQ.core.schemes.registry import build_stepper
logger = logging.getLogger(__name__)

# kineticEQの最上位wrapperクラス
class Engine:
    def __init__(self, config: Config, apply_logging_flag: bool = False):
        # configを保存
        self.config = config

        # model_cfgの設定
        if self.config.model_cfg is None:
            self.config = replace(self.config, model_cfg=default_model_cfg(self.config.model))
        else:
            t = expected_model_cfg_type(self.config.model)
            if not isinstance(self.config.model_cfg, t):
                raise TypeError(f"model_cfg type mismatch: expected {t.__name__}, got {type(self.config.model_cfg).__name__}")

        # scheme_params の自動設定
        if self.config.model_cfg.scheme_params is None:
            default_sp = default_scheme_params(self.config.model, self.config.scheme)
            if default_sp is not None:
                self.config = replace(
                    self.config,
                    model_cfg=replace(self.config.model_cfg, scheme_params=default_sp)
                )


        # apply_logging
        if apply_logging_flag:
            apply_logging(self.config)

        # info-log
        logger.info(
            "--- kineticEQ-Engine initialize complete ------\n"
            "  model    : %s\n"
            "  scheme   : %s\n"
            "  backend  : %s\n"
            "  device   : %s\n"
            "  log_level: %s\n"
            "------ Model Configuration ------\n%s",
            self.config.model_name,
            self.config.scheme_name,
            self.config.backend_name,
            self.config.device,
            self.config.log_level_name,
            format_kv_block(self.config.model_cfg),
        )

        # debug-log
        logger.debug(f"config: {self.config.as_dict}")

        # デバイスとバックエンドの例外処理
        resolve_device(self.config.device)

        # GPUのモデル名の取得と表示
        if self.config.device == "cuda":
            logger.info(f"GPU model: {torch.cuda.get_device_name(0)}")

        # steteとstepperの設定
        self.state = build_state(self.config)
        self.stepper = build_stepper(self.config, self.state)


    def run(self) -> Result:
        """
        time-evolution
        """
        # time-evolution
        with get_progress_bar(self.config.use_tqdm_bool,total=self.config.model_cfg.time.n_steps, desc="Time Evolution", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for steps in range(self.config.model_cfg.time.n_steps):
                logger.debug(f"time-evolution {self.config.model_name} {self.config.scheme_name} {self.config.backend_name} step: {steps}")

                # ============loop body===========
                # call-build_stepper
                self.stepper(steps) 

                
                # ============loop body===========
                pbar.update(1)

        return Result()

def run(config) -> Result:
    return Engine(config, apply_logging_flag=True).run()

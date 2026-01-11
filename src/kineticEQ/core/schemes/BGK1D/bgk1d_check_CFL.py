# kineticEQ/core/schemes/BGK1D/bgk1d_check_CFL.py
from kineticEQ.api.config import Config
import logging
logger = logging.getLogger(__name__)

# CFL条件チェック関数
def bgk1d_check_CFL(cfg: Config):
    if cfg.model_cfg.time.dt <= 0:
        raise ValueError("dt must be positive")

    dx = cfg.model_cfg.grid.Lx / (cfg.model_cfg.grid.nx - 1)
    cfl = cfg.model_cfg.grid.v_max * cfg.model_cfg.time.dt / dx
    cfl_limit = 0.9  # 1.0でも良いが、まずは余裕を推奨

    logger.info(f"CFL: {cfl:.3g}")

    if cfl > cfl_limit:
        raise ValueError(f"CFL violated: vmax*dt/dx = {cfl:.3g} > {cfl_limit}. "
                         f"(vmax={cfg.model_cfg.grid.v_max}, dt={cfg.model_cfg.time.dt}, dx={dx})")
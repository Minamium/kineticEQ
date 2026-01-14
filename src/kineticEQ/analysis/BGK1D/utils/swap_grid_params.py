from dataclasses import replace
from kineticEQ import Config

def with_grid(cfg: Config, *, nx: int | None = None, nv: int | None = None) -> Config:
    grid0 = cfg.model_cfg.grid
    grid1 = replace(
        grid0,
        nx=grid0.nx if nx is None else int(nx),
        nv=grid0.nv if nv is None else int(nv),
    )
    model_cfg1 = replace(cfg.model_cfg, grid=grid1)
    return replace(cfg, model_cfg=model_cfg1)

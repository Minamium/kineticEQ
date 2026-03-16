"""Config builders for phase-1 evaluation specs."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from kineticEQ import BGK1D, Config

from .spec import EvalCase, EvalTarget


def _replace_if_present(obj: Any, **updates: Any) -> Any:
    """Apply dataclass updates only for fields that exist on the object."""

    fnames = {f.name for f in fields(obj)}
    valid = {k: v for k, v in updates.items() if k in fnames}
    return replace(obj, **valid) if valid else obj


def _build_scheme_params(base_overrides: dict[str, Any] | None = None) -> Any:
    """Build implicit solver params and apply any base override fields."""

    scheme_params = BGK1D.implicit.Params()
    if not base_overrides:
        return scheme_params
    return _replace_if_present(scheme_params, **base_overrides)


def _normalize_initial_condition(initial_condition: Any) -> tuple[Any, ...]:
    """Normalize initial-condition payloads into the tuple form expected by BGK1D."""

    if initial_condition is None:
        return tuple(BGK1D.InitialCondition1D().initial_regions)
    if isinstance(initial_condition, tuple):
        return initial_condition
    if isinstance(initial_condition, list):
        return tuple(initial_condition)
    return (initial_condition,)


def build_cfg_from_case(case: EvalCase, device: str, base_overrides: dict | None = None) -> Config:
    """Build a kineticEQ Config from an EvalCase definition."""

    scheme_params = _build_scheme_params(base_overrides)
    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=int(case.nx), nv=int(case.nv), Lx=float(case.Lx), v_max=float(case.v_max)),
        time=BGK1D.TimeConfig(dt=float(case.dt), T_total=float(case.T_total)),
        params=BGK1D.BGK1D1VParams(tau_tilde=float(case.tau)),
        scheme_params=scheme_params,
        initial=BGK1D.InitialCondition1D(initial_regions=_normalize_initial_condition(case.initial_condition)),
    )
    return Config(
        model='BGK1D1V',
        scheme='implicit',
        backend='cuda_kernel',
        device=str(device),
        model_cfg=model_cfg,
        log_level='err',
        use_tqdm=False,
    )


def apply_target_overrides(cfg: Config, target: EvalTarget) -> Config:
    """Apply EvalTarget overrides onto a Config.

    Supported override key forms:
    - `scheme_params.xxx`
    - `grid.xxx`
    - `time.xxx`
    - `params.xxx`
    - `config.xxx`
    - `initial_condition`
    - `initial.initial_regions`
    - `initial_regions`

    Plain keys are still accepted when they map unambiguously onto one of the
    current dataclass objects, but prefixed forms are the preferred interface.
    """

    config = cfg
    model_cfg = config.model_cfg
    scheme_params = model_cfg.scheme_params
    grid = model_cfg.grid
    time = model_cfg.time
    params = model_cfg.params
    initial = model_cfg.initial

    for key, value in target.overrides.items():
        if '.' in key:
            prefix, subkey = key.split('.', 1)
        else:
            prefix, subkey = '', key

        if prefix == 'scheme_params' or (prefix == '' and hasattr(scheme_params, subkey)):
            scheme_params = _replace_if_present(scheme_params, **{subkey: value})
        elif prefix == 'grid' or (prefix == '' and hasattr(grid, subkey)):
            grid = _replace_if_present(grid, **{subkey: value})
        elif prefix == 'time' or (prefix == '' and hasattr(time, subkey)):
            time = _replace_if_present(time, **{subkey: value})
        elif prefix == 'params' or (prefix == '' and hasattr(params, subkey)):
            params = _replace_if_present(params, **{subkey: value})
        elif prefix == 'config' or (prefix == '' and hasattr(config, subkey)):
            config = replace(config, **{subkey: value})
        elif key in ('initial_condition', 'initial.initial_regions', 'initial_regions'):
            initial = replace(initial, initial_regions=_normalize_initial_condition(value))
        else:
            raise KeyError(
                f"unsupported target override: {key!r}. "
                "Supported prefixes: 'scheme_params.', 'grid.', 'time.', "
                "'params.', 'config.', plus initial_condition/initial_regions."
            )

    model_cfg = replace(model_cfg, scheme_params=scheme_params, grid=grid, time=time, params=params, initial=initial)
    return replace(config, model_cfg=model_cfg)


def build_cfg_for_target(case: EvalCase, target: EvalTarget, device: str) -> Config:
    """Build a Config for a concrete EvalCase/EvalTarget pair."""

    cfg = build_cfg_from_case(case, device=device)
    return apply_target_overrides(cfg, target)

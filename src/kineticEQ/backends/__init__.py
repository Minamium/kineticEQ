from importlib import import_module

try:
    _mod = import_module("kineticEQ.backends.advection1d")
    advect_upwind = _mod.sample_1d_advection_module.advec_upwind_step
except ModuleNotFoundError:
    advect_upwind = None

__all__ = ["advect_upwind"]
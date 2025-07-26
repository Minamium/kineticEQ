from importlib import import_module
try:
    advection1d = import_module("kineticEQ.backends.advection1d")
    advect_upwind = advection1d.sample_1d_advection_module.advec_upwind_step
except ModuleNotFoundError:
    advect_upwind = None
__all__ = ["advect_upwind"]
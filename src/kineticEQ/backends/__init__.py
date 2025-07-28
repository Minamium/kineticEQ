"""
High-performance Fortran backend for numerical computations
"""

try:
    from .Sample_1D_advection_by_fortran import step, is_available, get_info
except ImportError as e:
    def step(*args, **kwargs):
        raise RuntimeError("Fortran backend not available")
    def is_available():
        return False
    def get_info():
        return {"backend": "Fortran (unavailable)", "available": False}

__all__ = ["step", "is_available", "get_info"]

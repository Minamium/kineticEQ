"""
kineticEQ backends module
-------------------------
High-performance Fortran backend for numerical computations
"""

try:
    from .Sample_1D_advection_by_fortran import step, is_available, get_info
    __all__ = ["step", "is_available", "get_info"]
    
    # Check backend status
    if is_available():
        print("[kineticEQ.backends] Fortran backend is available")
    else:
        print("[kineticEQ.backends] Warning: Fortran backend is not available")
        
except ImportError as e:
    print(f"[kineticEQ.backends] Failed to import Fortran backend: {e}")
    
    # Define fallback functions
    def step(*args, **kwargs):
        raise RuntimeError("Fortran backend is not available. Check build errors.")
    
    def is_available():
        return False
    
    def get_info():
        return {"backend": "Fortran (unavailable)", "available": False, "error": str(e)}
    
    __all__ = ["step", "is_available", "get_info"]

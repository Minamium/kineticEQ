from importlib import import_module
import numpy as np

try:
    _mod = import_module("kineticEQ.backends.advection1d")
    _advec_upwind_step = _mod.sample_1d_advection_module.advec_upwind_step
    
    def advect_upwind(nx, dt, dx, u, q):
        """
        Python wrapper for Fortran advection upwind step.
        
        Parameters
        ----------
        nx : int
            Number of grid points
        dt : float
            Time step
        dx : float
            Spatial step
        u : float
            Velocity
        q : array_like
            Input array
            
        Returns
        -------
        q_new : ndarray
            Updated array after advection step
        """
        # Ensure inputs are correct types
        nx = int(nx)
        dt = np.float32(dt)
        dx = np.float32(dx)
        u = np.float32(u)
        
        # Convert input array to Fortran-compatible format
        q = np.asarray(q, dtype=np.float32, order='F')
        
        # Ensure array is contiguous and has correct size
        if q.size != nx:
            raise ValueError(f"Input array size ({q.size}) must match nx ({nx})")
        
        # Ensure q is 1D
        q = q.flatten()
        
        # Call Fortran function - F2PY handles output array automatically
        try:
            q_new = _advec_upwind_step(nx, dt, dx, u, q)
            return q_new
        except Exception as e:
            raise RuntimeError(f"Failed to call Fortran function: {e}")
    
except ModuleNotFoundError:
    advect_upwind = None

__all__ = ["advect_upwind"]
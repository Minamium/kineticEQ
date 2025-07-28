"""
Runtime Fortran backend for 1D advection
"""
import hashlib
import importlib.util
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "fortran" / "Sample_1Dadvection" / "Sample_1D_advection.f90"
BUILD_DIR = HERE.parent.parent.parent.parent / "build"
BUILD_DIR.mkdir(exist_ok=True)

def _get_flags():
    """Get compiler flags"""
    enable_openmp = os.getenv("KINEQ_ENABLE_OPENMP", "false").lower() in ("true", "1")
    if enable_openmp:
        return os.getenv("KINEQ_FFLAGS", "-O3 -fopenmp"), ["-lgomp"]
    else:
        return os.getenv("KINEQ_FFLAGS", "-O3"), []

def _build_hash():
    """Generate unique hash from source and flags"""
    flags, _ = _get_flags()
    content = SRC.read_bytes() + flags.encode()
    return hashlib.sha1(content).hexdigest()[:12]

def _build_module():
    """Build Fortran module with F2PY"""
    tag = _build_hash()
    flags, link_flags = _get_flags()
    
    # Check existing build
    so_files = list(BUILD_DIR.glob(f"advection1d_{tag}*.so"))
    if so_files and not os.getenv("KINEQ_FORCE_REBUILD"):
        return so_files[0]
    
    print(f"[kineticEQ] Building Fortran backend...")
    
    # Build in temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_src = tmpdir / "Sample_1D_advection.f90"
        shutil.copy2(SRC, tmp_src)
        
        # F2PY command
        cmd = [
            "python", "-m", "numpy.f2py", "-c", str(tmp_src),
            "-m", "advection1d", "--f90exec=gfortran",
            f"--f90flags={flags}"
        ]
        if link_flags:
            cmd.extend(link_flags)
        
        result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"F2PY build failed: {result.stderr}")
        
        # Copy built module
        built_files = list(tmpdir.glob("advection1d*.so"))
        if not built_files:
            raise RuntimeError("No .so file generated")
        
        target = BUILD_DIR / f"advection1d_{tag}.so"
        shutil.copy2(built_files[0], target)
        print(f"[kineticEQ] Build completed: {target.name}")
        return target

def _load_module():
    """Load the built Fortran module"""
    so_file = _build_module()
    spec = importlib.util.spec_from_file_location("advection1d", so_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Initialize module
try:
    _module = _load_module()
    _func = None
    
    # Find the correct function
    for attr_name in dir(_module):
        attr = getattr(_module, attr_name)
        if hasattr(attr, 'advec_upwind'):
            _func = attr.advec_upwind
            break
    
    if _func is None:
        raise AttributeError("Function advec_upwind not found")
        
    print("[kineticEQ] Fortran backend loaded successfully")
    
except Exception as e:
    print(f"[kineticEQ] Failed to initialize Fortran backend: {e}")
    _module = None
    _func = None

# Public API
def step(q: np.ndarray, dt: float, dx: float, u: float, nt: int = 1) -> np.ndarray:
    """1D upwind advection for nt steps
    
    Args:
        q: Initial concentration [nx]
        dt: Time step
        dx: Spatial step
        u: Advection velocity
        nt: Number of time steps
        
    Returns:
        Final concentration [nx]
    """
    if _func is None:
        raise RuntimeError("Fortran backend not available")
    
    q = np.asarray(q, dtype=np.float64)
    return _func(nt, dt, dx, u, q)

def is_available() -> bool:
    """Check if Fortran backend is available"""
    return _func is not None

def get_info() -> dict:
    """Get backend information"""
    flags, _ = _get_flags()
    openmp_enabled = "openmp" in flags.lower()
    
    return {
        "backend": f"Fortran (F2PY{' + OpenMP' if openmp_enabled else ''})",
        "available": is_available(),
        "source_file": str(SRC),
        "build_dir": str(BUILD_DIR),
        "compiler": "gfortran",
        "flags": flags,
        "openmp_enabled": openmp_enabled
    }

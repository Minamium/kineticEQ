"""
Runtime-built Fortran backend (OpenMP)
-------------------------------------
初回 import 時に f2py でビルドし、build/ に .so をキャッシュ。
"""
from __future__ import annotations
import hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC  = HERE / "fortran" / "Sample_1D_advection.f90"
BUILD_DIR = HERE.parent / ".." / ".." / ".." / ".." / "build"  # kineticEQ/build
BUILD_DIR = BUILD_DIR.resolve()
BUILD_DIR.mkdir(exist_ok=True)

# — ハッシュ（ソース + フラグ）で一意に
def _tag() -> str:
    txt = SRC.read_bytes()
    flags = os.environ.get("KINEQ_FFLAGS", "-O3 -fopenmp").encode()
    return hashlib.sha1(txt + flags).hexdigest()[:12]

def _build() -> Path:
    so_name = f"advection1d_{_tag()}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    so_path = BUILD_DIR / so_name
    if so_path.exists() and not os.getenv("KINEQ_FORCE_REBUILD"):
        return so_path

    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", str(SRC),
        "-m", "advection1d",
        f"--f90exec={os.getenv('FC', 'gfortran')}",
        f"--f90flags={os.getenv('KINEQ_FFLAGS', '-O3 -fopenmp')}",
        "-lgomp",
        f"--build-dir={BUILD_DIR}",
        "-o", str(so_path)
    ]
    print(f"[kineticEQ] building Fortran backend → {so_path.name}")
    subprocess.check_call(cmd)
    return so_path

# — 動的 import
_so = _build()
_spec = importlib.util.spec_from_file_location("advection1d", _so)
_mod  = importlib.util.module_from_spec(_spec)          # type: ignore
_spec.loader.exec_module(_mod)                         # type: ignore

# — 公開 API
def step(q: np.ndarray, dt: float, dx: float, u: float, nt: int = 1) -> np.ndarray:
    """一次風上差分を nt ステップ進める（周期境界）"""
    q = np.ascontiguousarray(q, dtype=np.float64)
    _mod.advection1d_mod.advec_upwind(nt, q.size, dt, dx, u, q)  # in-place
    return q

# cuda_kernel/compile.py
from __future__ import annotations
import torch, os
import sysconfig
import functools
from pathlib import Path
from torch.utils.cpp_extension import load

_KERNEL_DIR = Path(__file__).parent
_BUILD_DIR = _KERNEL_DIR / "build"

def _setup_cuda_arch():
    """GPUアーキテクチャを取得, 設定"""
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}+PTX")

def _common_flags():
    return {
        "extra_cflags": ["-O3"],
        "extra_cuda_cflags": ["-O3"],
        "extra_include_paths": [sysconfig.get_paths()["include"]],
        "build_directory": str(_BUILD_DIR),
        "verbose": True,
    }

@functools.lru_cache(maxsize=1)
def load_explicit_fused():
    _setup_cuda_arch()
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="explicit_fused",
        sources=[
            str(_KERNEL_DIR / "explicit_fused" / "explicit_binding.cpp"),
            str(_KERNEL_DIR / "explicit_fused" / "explicit_kernel.cu"),
        ],
        **_common_flags(),
    )

@functools.lru_cache(maxsize=1)
def load_implicit_fused():
    _setup_cuda_arch()
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="implicit_fused",
        sources=[
            str(_KERNEL_DIR / "implicit_fused" / "implicit_binding.cpp"),
            str(_KERNEL_DIR / "implicit_fused" / "implicit_kernels.cu"),
        ],
        extra_ldflags=["-lcusparse"],
        **_common_flags(),
    )

@functools.lru_cache(maxsize=1)
def load_gtsv():
    _setup_cuda_arch()
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="gtsv_batch",
        sources=[
            str(_KERNEL_DIR / "gtsv" / "gtsv_binding.cpp"),
            str(_KERNEL_DIR / "gtsv" / "gtsv_batch.cu"),
        ],
        extra_ldflags=["-lcusparse"],
        **_common_flags(),
    )

@functools.lru_cache(maxsize=1)
def load_lo_blocktridiag():
    _setup_cuda_arch()
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="lo_blocktridiag",
        sources=[
            str(_KERNEL_DIR / "lo_blocktridiag" / "block_tridiag_binding.cpp"),
            str(_KERNEL_DIR / "lo_blocktridiag" / "block_tridiag_kernel.cu"),
        ],
        **_common_flags(),
    )
from __future__ import annotations

import functools
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load

_KERNEL_DIR = Path(__file__).parent
_BUILD_DIR = _KERNEL_DIR / "build"


def _common_flags() -> dict[str, object]:
    return {
        "extra_cflags": ["-O3", "-std=c++17"],
        "extra_include_paths": [sysconfig.get_paths()["include"]],
        "build_directory": str(_BUILD_DIR),
        "verbose": False,
    }


@functools.lru_cache(maxsize=1)
def load_implicit_fused_cpu():
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="implicit_fused_cpu",
        sources=[
            str(_KERNEL_DIR / "BGK1D1V" / "implicit_fused_cpu" / "implicit_binding.cpp"),
        ],
        **_common_flags(),
    )


@functools.lru_cache(maxsize=1)
def load_gtsv_cpu():
    _BUILD_DIR.mkdir(exist_ok=True)
    return load(
        name="gtsv_cpu",
        sources=[
            str(_KERNEL_DIR / "BGK1D1V" / "gtsv_cpu" / "gtsv_binding.cpp"),
        ],
        **_common_flags(),
    )

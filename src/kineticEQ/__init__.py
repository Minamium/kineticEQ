# src/kineticEQ/__init__.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = version("kineticEQ")  

# APIディレクトリの公開
from .api import Config, Engine, run, Result
from .params import BGK1D1V_params as BGK1D
from .params import BGK2D2V_params as BGK2D2V

__all__ = ["__version__", "Config", "Engine", "run", "Result", "BGK1D", "BGK2D2V"]
# src/kineticEQ/__init__.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = version("kineticEQ")  

# APIディレクトリの公開
from .api import Config, Engine, run, Result

__all__ = ["__version__", "Config", "Engine", "run", "Result"]
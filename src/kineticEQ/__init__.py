# src/kineticEQ/__init__.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = version("kineticEQ")  

# レガシーの公開
from ._legacy.BGK1Dsim import BGK1D
from ._legacy.BGK1DPlotUtil import BGK1DPlotMixin

# APIディレクトリの公開
from .api import Config, Engine, run, Result
from kineticEQ.params import BGK1D
from kineticEQ.params import BGK2D2V

__all__ = ["__version__", "Config", "Engine", "run", "Result", "BGK1D", "BGK2D2V"]
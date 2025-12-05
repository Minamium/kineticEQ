##############################
# Local端末で実行する可視化関数群 #
##############################

import torch
import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Any, Union

# BGK1Dbaseクラスを継承するためのimport
from .BGK1Dsim import BGK1D
from .BGK2D2V_core import BGK2D2V

# 可視化関数群
class BGK2D2VPlotMixin:
    """可視化, 解析用の関数群"""
    #状態可視化メソッド
    def plot_state(self):
        """状態を可視化するメソッド"""
        pass
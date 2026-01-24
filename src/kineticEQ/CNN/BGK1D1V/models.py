# kineticEQ/CNN/BGK1D1V/models.py
from __future__ import annotations

import torch
import torch.nn as nn

# MomentCNN1D
class MomentCNN1D(nn.Module):
    """
    MomentCNN1D:
      入力:  (B, 5, nx)  = (n, u, T, log10dt, log10tau)
      出力:  (B, 3, nx)  = (Δn, Δu, ΔT)

      CNN構造:
        基本入力はモーメントベクトルWと物理空間分割数nxによる(W, nx), その他dt, tau_tildeをログスケールで入力
        1次元畳み込みでカーネルサイズは基本を7として調整, paddingは"same padding"相当
        隠れ層は~~層で各層のニューロン数は~~, 活性化関数は~~を基本として調整, 

        ResNet構造については~~

    学習方針:
      入力データはk時点でのモーメント値の絶対値, 教師データはPicard収束後のk+1時点のモーメント値との更新量とする(Δ学習).

    """
    def __init__(self, in_ch=5, hidden=128, out_ch=3, kernel=11, n_blocks=4):
        super().__init__()
        pad = kernel // 2

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel, padding=pad),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(*[
            ResBlock1D(hidden, kernel=kernel) for _ in range(n_blocks)
        ])

        self.head = nn.Conv1d(hidden, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        y = self.head(x)
        return y

# ResNet構造
class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad)

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        return x + h  # hidden 空間で残差結合
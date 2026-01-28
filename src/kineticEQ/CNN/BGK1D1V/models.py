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
    def __init__(self, in_ch=5, hidden=128, out_ch=3, kernel=5, n_blocks=2, gn_groups=32):
        super().__init__()
        assert hidden % gn_groups == 0, "hidden must be divisible by gn_groups"
        self.kernel = kernel

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel, padding=kernel//2),
            nn.GroupNorm(gn_groups, hidden),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            ResBlock1D(hidden, kernel=1, gn_groups=gn_groups, res_scale=0.1)
            for _ in range(n_blocks)
        ])

        self.head = nn.Conv1d(hidden, out_ch, kernel_size=1)

        # optional: small init for head to start near-zero updates
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel, gn_groups=32, res_scale=0.1):
        super().__init__()
        pad = kernel // 2
        self.res_scale = float(res_scale)

        self.gn1  = nn.GroupNorm(gn_groups, ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad)

        self.gn2  = nn.GroupNorm(gn_groups, ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad)

        # optional: stabilize residual branch init
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(self.act1(self.gn1(x)))
        h = self.conv2(self.act2(self.gn2(h)))
        return x + self.res_scale * h
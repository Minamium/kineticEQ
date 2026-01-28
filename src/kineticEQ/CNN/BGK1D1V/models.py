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
    def __init__(self, in_ch=5, hidden=128, out_ch=3, kernel=5, n_blocks=2, gn_groups=32,
                 bottleneck=0.5, dilation_cycle=(1, 2), use_gate_head=True):
        super().__init__()
        # GroupNorm groups auto-fix (hidden < 32 対策)
        gn_groups = int(min(gn_groups, hidden))
        while hidden % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel, padding=(kernel//2)),
            nn.GroupNorm(gn_groups, hidden),
            nn.SiLU(),
        )

        blocks = []
        for i in range(int(n_blocks)):
            dil = int(dilation_cycle[i % len(dilation_cycle)])
            blocks.append(LiteResBlock1D(hidden, kernel=kernel, dilation=dil,
                                         gn_groups=gn_groups, res_scale=0.1, bottleneck=bottleneck))
        self.blocks = nn.Sequential(*blocks)

        # ---- head ----
        # base head: always there
        self.head_base = nn.Conv1d(hidden, out_ch, kernel_size=1)
        nn.init.zeros_(self.head_base.weight)
        if self.head_base.bias is not None:
            nn.init.zeros_(self.head_base.bias)

        # optional: gate + tail head (外れ値専用の補正)
        self.use_gate_head = bool(use_gate_head)
        if self.use_gate_head:
            self.gate = nn.Sequential(
                nn.Conv1d(hidden, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.head_tail = nn.Conv1d(hidden, out_ch, kernel_size=1)
            # tail側は最初は0に近く、学習で必要な時だけ効くように
            nn.init.zeros_(self.head_tail.weight)
            if self.head_tail.bias is not None:
                nn.init.zeros_(self.head_tail.bias)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        y = self.head_base(h)
        if self.use_gate_head:
            g = self.gate(h)          # (B,1,nx) : 外れ値っぽい場所だけ 1 に寄る
            y = y + g * self.head_tail(h)
        return y


class LiteResBlock1D(nn.Module):
    """
    ch -> mid (1x1) -> depthwise(k, dilation) -> ch (1x1)
    """
    def __init__(self, ch, kernel=5, dilation=1, gn_groups=32, res_scale=0.1, bottleneck=0.5):
        super().__init__()
        self.res_scale = float(res_scale)

        mid = int(max(8, int(round(ch * float(bottleneck)))))
        # mid も GroupNorm の割り切りを安全化
        gn_mid = int(min(gn_groups, mid))
        while mid % gn_mid != 0 and gn_mid > 1:
            gn_mid -= 1

        pad = (kernel // 2) * dilation

        self.pre = nn.Sequential(
            nn.GroupNorm(gn_groups, ch),
            nn.SiLU(),
            nn.Conv1d(ch, mid, kernel_size=1),
        )
        self.dw = nn.Sequential(
            nn.GroupNorm(gn_mid, mid),
            nn.SiLU(),
            nn.Conv1d(mid, mid, kernel_size=kernel, padding=pad, dilation=dilation, groups=mid),  # depthwise
        )
        self.post = nn.Sequential(
            nn.GroupNorm(gn_mid, mid),
            nn.SiLU(),
            nn.Conv1d(mid, ch, kernel_size=1),
        )

        # 安定化：残差枝の最後は小さく開始
        nn.init.zeros_(self.post[-1].weight)
        if self.post[-1].bias is not None:
            nn.init.zeros_(self.post[-1].bias)

    def forward(self, x):
        h = self.pre(x)
        h = self.dw(h)
        h = self.post(h)
        return x + self.res_scale * h
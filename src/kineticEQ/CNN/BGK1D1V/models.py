# kineticEQ/CNN/BGK1D1V/models.py
from __future__ import annotations

import torch
import torch.nn as nn


class MomentCNN1D(nn.Module):
    """
    MomentCNN1D:
      入力:  (B, 5, nx)  = (n, u, T, log10dt, log10tau)
      出力:  (B, 3, nx)  = (Δn, (Δnu or Δu), ΔT)  引数で選択可能

    設計意図:
      - stem + 軽量ResBlock（bottleneck + depthwise conv + dilation）で推論を軽く
      - base head は全域の滑らかな更新を担当
      - tail head は外れ値（大残差）だけを補正する
      - gate は「普段閉じる」初期化にして、tail が最初から全点へ混入しないようにする
    """
    def __init__(
        self,
        in_ch: int = 5,
        hidden: int = 128,
        out_ch: int = 3,
        kernel: int = 5,
        n_blocks: int = 2,
        gn_groups: int = 32,
        bottleneck: float = 0.5,
        dilation_cycle: tuple[int, ...] = (1, 2),
        use_gate_head: bool = True,
        gate_bias_init: float = -3.0,   # <- (sigmoid(-3)≈0.047)
        gate_scale: float = 1.0,        # <- tail寄与の全体係数（必要なら0.1とか）
        gate_per_channel: bool = False, # <- Trueで(B,3,nx)ゲート（より強いが少し重い）
    ):
        super().__init__()

        hidden = int(hidden)
        n_blocks = int(n_blocks)
        out_ch = int(out_ch)

        # ---- GroupNorm groups auto-fix ----
        gn_groups = int(min(int(gn_groups), hidden))
        while gn_groups > 1 and (hidden % gn_groups) != 0:
            gn_groups -= 1

        self.use_gate_head = bool(use_gate_head)
        self.gate_scale = float(gate_scale)
        self.gate_per_channel = bool(gate_per_channel)

        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel, padding=(kernel // 2)),
            nn.GroupNorm(gn_groups, hidden),
            nn.SiLU(),
        )

        # ---- blocks ----
        blocks = []
        dcyc = tuple(int(d) for d in dilation_cycle) if len(dilation_cycle) > 0 else (1,)
        for i in range(n_blocks):
            dil = int(dcyc[i % len(dcyc)])
            blocks.append(
                LiteResBlock1D(
                    ch=hidden,
                    kernel=kernel,
                    dilation=dil,
                    gn_groups=gn_groups,
                    res_scale=0.1,
                    bottleneck=bottleneck,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # ---- base head ----
        self.head_base = nn.Conv1d(hidden, out_ch, kernel_size=1)
        nn.init.zeros_(self.head_base.weight)
        if self.head_base.bias is not None:
            nn.init.zeros_(self.head_base.bias)

        # ---- optional: gate + tail head ----
        if self.use_gate_head:
            gate_out = out_ch if self.gate_per_channel else 1
            self.gate = nn.Sequential(
                nn.Conv1d(hidden, gate_out, kernel_size=1),
                nn.Sigmoid(),
            )
            self.head_tail = nn.Conv1d(hidden, out_ch, kernel_size=1)

            # tail head: start near zero
            nn.init.zeros_(self.head_tail.weight)
            if self.head_tail.bias is not None:
                nn.init.zeros_(self.head_tail.bias)

            # IMPORTANT: gate should start "closed"
            gate_conv = self.gate[0]
            nn.init.zeros_(gate_conv.weight)
            if gate_conv.bias is not None:
                nn.init.constant_(gate_conv.bias, float(gate_bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)

        y = self.head_base(h)
        if self.use_gate_head:
            g = self.gate(h)  # (B,1,nx) or (B,3,nx)
            # gate_per_channel=False のときは broadcast で (B,3,nx) に拡張される
            y = y + (self.gate_scale * g) * self.head_tail(h)
        return y


class LiteResBlock1D(nn.Module):
    """
    ch -> mid (1x1) -> depthwise(k, dilation) -> ch (1x1)
    軽量・高速化狙いの ResBlock
    """
    def __init__(
        self,
        ch: int,
        kernel: int = 5,
        dilation: int = 1,
        gn_groups: int = 32,
        res_scale: float = 0.1,
        bottleneck: float = 0.5,
    ):
        super().__init__()
        self.res_scale = float(res_scale)

        ch = int(ch)
        mid = int(max(8, int(round(ch * float(bottleneck)))))

        # GroupNorm for mid
        gn_mid = int(min(int(gn_groups), mid))
        while gn_mid > 1 and (mid % gn_mid) != 0:
            gn_mid -= 1

        pad = (kernel // 2) * int(dilation)

        self.pre = nn.Sequential(
            nn.GroupNorm(int(gn_groups), ch),
            nn.SiLU(),
            nn.Conv1d(ch, mid, kernel_size=1),
        )
        self.dw = nn.Sequential(
            nn.GroupNorm(gn_mid, mid),
            nn.SiLU(),
            nn.Conv1d(
                mid,
                mid,
                kernel_size=kernel,
                padding=pad,
                dilation=int(dilation),
                groups=mid,  # depthwise
            ),
        )
        self.post = nn.Sequential(
            nn.GroupNorm(gn_mid, mid),
            nn.SiLU(),
            nn.Conv1d(mid, ch, kernel_size=1),
        )

        # stabilize: last conv in residual branch starts at ~0
        nn.init.zeros_(self.post[-1].weight)
        if self.post[-1].bias is not None:
            nn.init.zeros_(self.post[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.dw(h)
        h = self.post(h)
        return x + self.res_scale * h
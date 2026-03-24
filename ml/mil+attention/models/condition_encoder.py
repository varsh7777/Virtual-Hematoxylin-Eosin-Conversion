# models/condition_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionEncoder(nn.Module):
    """
    Encodes the conditioning raw image into multi-scale features.
    Input:  [B,3,H,W]
    Output:
      f1: [B, base, H, W]
      f2: [B, 2*base, H/2, W/2]
      f3: [B, 4*base, H/4, W/4]
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.block1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.block2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.block3 = ConvBlock(base_ch * 4, base_ch * 4)

    def forward(self, x: torch.Tensor):
        f1 = self.block1(x)
        f2 = self.block2(self.down1(f1))
        f3 = self.block3(self.down2(f2))
        return f1, f2, f3
# models/unet_diffusion.py
from __future__ import annotations

import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding, TimeMLP
from .condition_encoder import ConditionEncoder


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU(inplace=True)

        self.time_proj = nn.Linear(t_dim, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, out_ch, t_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x


class UNetDiffusion(nn.Module):
    """
    Predicts bridge noise (or another chosen target) given:
      - x_t (noisy intermediate state)
      - cond (raw input image)
      - timestep t
    """
    def __init__(
        self,
        in_ch: int = 3,
        cond_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 32,
        time_dim: int = 128,
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim, time_dim)

        self.cond_encoder = ConditionEncoder(in_ch=cond_ch, base_ch=base_ch)

        self.in_proj = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 8, time_dim)
        self.mid2 = ResBlock(base_ch * 8, base_ch * 8, time_dim)

        self.up2 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, time_dim)
        self.up1 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, time_dim)

        self.out_block = ResBlock(base_ch * 2, base_ch, time_dim)
        self.out_proj = nn.Conv2d(base_ch, out_ch, 1)

        # project condition features to matching channels
        self.cond1_proj = nn.Conv2d(base_ch, base_ch, 1)
        self.cond2_proj = nn.Conv2d(base_ch * 2, base_ch * 2, 1)
        self.cond3_proj = nn.Conv2d(base_ch * 4, base_ch * 4, 1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))

        c1, c2, c3 = self.cond_encoder(cond)

        x = self.in_proj(x_t)
        x = x + self.cond1_proj(c1)

        x, s1 = self.down1(x, t_emb)
        x = x + self.cond2_proj(c2)

        x, s2 = self.down2(x, t_emb)
        x = x + self.cond3_proj(c3)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)

        x = self.out_block(x, t_emb)
        x = self.out_proj(x)
        return x
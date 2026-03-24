# models/time_embedding.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding.
    Input:  t [B] integer or float
    Output: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class TimeMLP(nn.Module):
    """
    Turns sinusoidal embedding into a richer learned embedding.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)
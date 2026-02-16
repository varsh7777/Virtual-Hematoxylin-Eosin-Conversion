# mil
# Wraps:
#     - patch encoder
#     - attention module
#     - pooling to get bag embedding z
#     - (optional) bag classifier head (if you have bag labels like tumor/subtype)

import torch
import torch.nn as nn

from .encoder import SmallEncoder
from .attention import PlainAttention, GatedAttention


class MILAttention(nn.Module):
    """
    MIL attention model producing attention weights + bag embedding.
    Input is a bag of patches:
      x: [B, N, 3, H, W]
    Output:
      attn: [B, N]
      bag_emb: [B, D]
    """
    def __init__(self, embed_dim: int = 256, attn_type: str = "gated"):
        super().__init__()
        self.encoder = SmallEncoder(embed_dim=embed_dim)
        attn_type = attn_type.lower()
        if attn_type == "plain":
            self.attn = PlainAttention(embed_dim=embed_dim)
        elif attn_type == "gated":
            self.attn = GatedAttention(embed_dim=embed_dim)
        else:
            raise ValueError("attn_type must be 'plain' or 'gated'")

    def forward(self, x: torch.Tensor):
        B, N, C, H, W = x.shape
        x_flat = x.view(B * N, C, H, W)
        h = self.encoder(x_flat).view(B, N, -1)   # [B,N,D]
        attn = self.attn(h)                       # [B,N]
        bag_emb = (h * attn.unsqueeze(-1)).sum(dim=1)  # [B,D]
        return {"attn": attn, "bag_emb": bag_emb}
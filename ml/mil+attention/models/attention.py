# attention
# attention pooling:
#      - standard attention MIL
#      - gated attention MIL (usually better)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainAttention(nn.Module):
    """
    Ilse et al attention:
      a_i = softmax(w^T tanh(V h_i))
    """
    def __init__(self, embed_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, N, D]
        A = self.w(torch.tanh(self.V(H)))  # [B,N,1]
        A = A.squeeze(-1)                  # [B,N]
        return F.softmax(A, dim=1)


class GatedAttention(nn.Module):
    """
    Gated attention:
      a_i ∝ w^T (tanh(V h_i) ⊙ sigmoid(U h_i))
    """
    def __init__(self, embed_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(embed_dim, attn_dim)
        self.U = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, N, D]
        v = torch.tanh(self.V(H))
        u = torch.sigmoid(self.U(H))
        A = self.w(v * u).squeeze(-1)      # [B,N]
        return F.softmax(A, dim=1)
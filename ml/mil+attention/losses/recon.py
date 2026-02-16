# recon
# pixel losses: L1/L2, Charbonnier, SSIM loss

import torch
import torch.nn as nn


class PatchL1Loss(nn.Module):
    """
    pred, target: [B, N, 3, H, W]
    returns:
      patch_loss_map: [B, N]
      mean_loss: scalar
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l = self.l1(pred, target)         # [B,N,3,H,W]
        l = l.mean(dim=(2, 3, 4))         # [B,N]
        return l, l.mean()

# losses/diffusion.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeDiffusionLoss(nn.Module):
    """
    Combines:
      - epsilon prediction loss
      - optional image reconstruction loss on predicted y
    """
    def __init__(
        self,
        lambda_eps: float = 1.0,
        lambda_x0: float = 0.0,
        use_l1_for_x0: bool = True,
    ):
        super().__init__()
        self.lambda_eps = float(lambda_eps)
        self.lambda_x0 = float(lambda_x0)
        self.use_l1_for_x0 = bool(use_l1_for_x0)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        y_pred: torch.Tensor | None = None,
        y_true: torch.Tensor | None = None,
    ):
        loss_eps = F.mse_loss(eps_pred, eps_true)

        if self.lambda_x0 > 0.0:
            if y_pred is None or y_true is None:
                raise ValueError("y_pred and y_true required when lambda_x0 > 0")
            if self.use_l1_for_x0:
                loss_x0 = F.l1_loss(y_pred, y_true)
            else:
                loss_x0 = F.mse_loss(y_pred, y_true)
        else:
            loss_x0 = eps_pred.new_tensor(0.0)

        total = self.lambda_eps * loss_eps + self.lambda_x0 * loss_x0

        return {
            "loss": total,
            "loss_eps": loss_eps,
            "loss_x0": loss_x0,
        }
# gan 
# adversarial losses if you want realistic H&E appearance

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    Standard GAN loss wrapper.
    gan_mode:
      - "lsgan": least squares GAN
      - "bce": BCEWithLogits
    """
    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        gan_mode = gan_mode.lower()
        if gan_mode not in ("lsgan", "bce"):
            raise ValueError("gan_mode must be 'lsgan' or 'bce'")
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.crit = nn.MSELoss()
        else:
            self.crit = nn.BCEWithLogitsLoss()

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return torch.ones_like(pred) if is_real else torch.zeros_like(pred)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return self.crit(pred, self._target(pred, is_real))

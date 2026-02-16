import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for [B,3,H,W] -> [B,1,h',w'] logits
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c, 2*c, 4, stride=2, padding=1),
            nn.InstanceNorm2d(2*c),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2*c, 4*c, 4, stride=2, padding=1),
            nn.InstanceNorm2d(4*c),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4*c, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
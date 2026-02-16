# encoder
# CNN that maps a patch → embedding: hi = Encoder(xi)
# (ResNet, ConvNeXt, lightweight UNet encoder, etc.)

import torch
import torch.nn as nn


class SmallEncoder(nn.Module):
    """
    Patch encoder: [B*N,3,H,W] -> [B*N, D]
    Lightweight CNN with global average pooling.
    """
    def __init__(self, embed_dim: int = 256, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c, 2*c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(2*c, 4*c, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(4*c, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)
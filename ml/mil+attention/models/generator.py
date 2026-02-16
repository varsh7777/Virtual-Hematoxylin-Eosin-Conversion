# generator
# a generator G(x) that outputs virtual H&E patch (UNet / ResNet generator)

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    """
    Simple UNet-like generator.
    Input:  [B,3,H,W] float in [0,1]
    Output: [B,3,H,W] float in [0,1]
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.down1 = _ConvBlock(3, c)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = _ConvBlock(c, 2 * c)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = _ConvBlock(2 * c, 4 * c)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = _ConvBlock(4 * c, 8 * c)

        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = _ConvBlock(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = _ConvBlock(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = _ConvBlock(2 * c, c)

        self.out = nn.Sequential(
            nn.Conv2d(c, 3, 1),
            nn.Sigmoid(),  # keep [0,1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.mid(self.pool3(d3))

        u3 = self.up3(m)
        x3 = torch.cat([u3, d3], dim=1)
        x3 = self.dec3(x3)

        u2 = self.up2(x3)
        x2 = torch.cat([u2, d2], dim=1)
        x2 = self.dec2(x2)

        u1 = self.up1(x2)
        x1 = torch.cat([u1, d1], dim=1)
        x1 = self.dec1(x1)

        return self.out(x1)
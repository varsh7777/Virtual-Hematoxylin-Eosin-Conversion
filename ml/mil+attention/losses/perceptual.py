# losses/perceptual.py
# VGG perceptual loss for image-to-image translation.
# Assumes inputs are RGB float tensors in [0,1], shape [B,3,H,W].

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 feature maps.

    Args:
        layer_ids: indices of feature blocks to use from torchvision VGG16 features.
                   Common useful defaults:
                     (3, 8, 15, 22)  -> relu1_2, relu2_2, relu3_3, relu4_3-ish boundaries
        layer_weights: weights for each chosen layer
        resize: if not None, bilinearly resize inputs to this square size before VGG
                (useful if training patch sizes vary; set None to keep original size)
        use_l1: if True, L1 feature loss; else L2/MSE
    """
    def __init__(
        self,
        layer_ids: Sequence[int] = (3, 8, 15, 22),
        layer_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        resize: int | None = None,
        use_l1: bool = True,
    ):
        super().__init__()

        if len(layer_ids) != len(layer_weights):
            raise ValueError("layer_ids and layer_weights must have same length")

        self.layer_ids = tuple(int(x) for x in layer_ids)
        self.layer_weights = tuple(float(x) for x in layer_weights)
        self.resize = resize
        self.use_l1 = use_l1

        # Newer torchvision prefers weights=...
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = models.vgg16(pretrained=True).features

        self.vgg = vgg.eval()

        # Freeze VGG weights
        for p in self.vgg.parameters():
            p.requires_grad = False

        # Register ImageNet normalization stats as buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [0,1]
        return (x - self.mean) / self.std

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize is None:
            return x
        return F.interpolate(
            x,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B,3,H,W] in [0,1]
        returns scalar perceptual loss
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must match shape, got {pred.shape} vs {target.shape}")

        pred = self._maybe_resize(pred)
        target = self._maybe_resize(target)

        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = pred.new_tensor(0.0)

        x = pred
        y = target

        max_layer = max(self.layer_ids)
        chosen = set(self.layer_ids)

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)

            if i in chosen:
                w = self.layer_weights[self.layer_ids.index(i)]
                if self.use_l1:
                    loss = loss + w * F.l1_loss(x, y)
                else:
                    loss = loss + w * F.mse_loss(x, y)

            if i >= max_layer:
                break

        return loss
# ml_infer.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# ---------------------------------------------------------------------
# Make sure we can import your training package at:
#   HPL/ml/mil+attention/models
# even when pipeline.py is executed from HPL root.
# ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MIL_ROOT = os.path.join(_THIS_DIR, "ml", "mil+attention")  # contains models/, utils/, etc.

if _MIL_ROOT not in sys.path:
    sys.path.insert(0, _MIL_ROOT)


@dataclass
class MLParams:
    checkpoint: str
    device: str = "cuda"        # "cuda" or "cpu"
    tile_size: int = 512
    overlap: int = 32
    use_amp: bool = True
    normalize: str = "0_1"      # "none" | "0_1" | "imagenet"
    base_channels: int = 32     # must match training config


def _normalize_rgb(rgb_f32_0_1: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("none", "0_1"):
        return rgb_f32_0_1
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (rgb_f32_0_1 - mean) / std
    raise ValueError(f"Unknown normalize mode: {mode}")


def _denormalize_rgb(rgb_f32: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("none", "0_1"):
        return rgb_f32
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return rgb_f32 * std + mean
    raise ValueError(f"Unknown normalize mode: {mode}")


def _tile_coords(H: int, W: int, tile: int, overlap: int):
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    return ys, xs


def _make_weight_mask(tile: int) -> np.ndarray:
    # Smooth cosine blending mask [tile,tile,1]
    yy = np.linspace(0, 1, tile, dtype=np.float32)
    xx = np.linspace(0, 1, tile, dtype=np.float32)
    wy = 0.5 - 0.5 * np.cos(np.clip(yy, 0, 1) * np.pi)
    wx = 0.5 - 0.5 * np.cos(np.clip(xx, 0, 1) * np.pi)
    w = (wy[:, None] * wx[None, :]).astype(np.float32)
    return w[..., None]


def _load_model_from_checkpoint(checkpoint_path: str, device: "torch.device", base_channels: int) -> "nn.Module":
    """
    Expects checkpoints saved by your train.py like:
      torch.save({"G": G.state_dict(), "MIL": MIL.state_dict()}, "best.pt")

    Loads G only for inference.
    """
    from models import UNetGenerator  # imported from ml/mil+attention/models

    model = UNetGenerator(base_channels=base_channels).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "G" in ckpt:
        state_g = ckpt["G"]
    else:
        # fallback: checkpoint itself is a state_dict
        state_g = ckpt

    model.load_state_dict(state_g, strict=True)
    return model


@torch.no_grad()
def apply_bgr_ml(bgr_u8: np.ndarray, params: MLParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch to use --method ml")

    if bgr_u8 is None:
        raise ValueError("Input image is None")

    if bgr_u8.dtype != np.uint8:
        bgr_u8 = np.clip(bgr_u8, 0, 255).astype(np.uint8)

    if bgr_u8.ndim == 2:
        bgr_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_GRAY2BGR)
    elif bgr_u8.ndim == 3 and bgr_u8.shape[2] == 4:
        bgr_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGRA2BGR)

    device = torch.device(params.device if (params.device == "cpu" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(params.use_amp and device.type == "cuda")

    model = _load_model_from_checkpoint(params.checkpoint, device, base_channels=int(params.base_channels))

    H, W = bgr_u8.shape[:2]
    tile = int(params.tile_size)
    overlap = int(params.overlap)

    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_n = _normalize_rgb(rgb, params.normalize)

    ys, xs = _tile_coords(H, W, tile, overlap)
    wmask = _make_weight_mask(tile)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W, 1), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            patch = rgb_n[y0:y0 + tile, x0:x0 + tile, :]  # [tile,tile,3]
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = model(t)
            else:
                y = model(t)

            # assume model outputs in [0,1]. If your generator uses tanh, uncomment:
            # y = (y + 1.0) / 2.0

            y = y[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # [tile,tile,3]
            y = _denormalize_rgb(y, params.normalize)
            y = np.clip(y, 0.0, 1.0)

            out_acc[y0:y0 + tile, x0:x0 + tile, :] += y * wmask
            w_acc[y0:y0 + tile, x0:x0 + tile, :] += wmask

    out = out_acc / np.maximum(w_acc, 1e-8)
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

    meta = {
        "method": "ml",
        "checkpoint": params.checkpoint,
        "device": str(device),
        "tile_size": tile,
        "overlap": overlap,
        "normalize": params.normalize,
        "use_amp": use_amp,
        "base_channels": int(params.base_channels),
        "shape_in": (int(H), int(W)),
    }
    return out_bgr, meta
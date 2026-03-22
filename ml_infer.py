# ml_infer.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

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
_MIL_ROOT = os.path.join(_THIS_DIR, "ml", "mil+attention")
if _MIL_ROOT not in sys.path:
    sys.path.insert(0, _MIL_ROOT)


@dataclass
class MLParams:
    checkpoint: str
    device: str = "cuda"
    tile_size: int = 512
    overlap: int = 128
    use_amp: bool = True
    normalize: str = "0_1"           # "none" | "0_1" | "imagenet"
    base_channels: int = 32          # must match training config
    output_activation: str = "sigmoid"  # "sigmoid", "none_0_1", or "tanh"


# Simple in-process model cache so repeated calls don't reload weights
_MODEL_CACHE: Dict[Tuple[str, str, int], "nn.Module"] = {}


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


def _postprocess_model_output(y: np.ndarray, output_activation: str) -> np.ndarray:
    """
    Convert model output to [0,1] RGB float.
    """
    output_activation = output_activation.lower()
    if output_activation in ("sigmoid", "none_0_1"):
        return np.clip(y, 0.0, 1.0)
    if output_activation == "tanh":
        return np.clip((y + 1.0) / 2.0, 0.0, 1.0)
    raise ValueError(f"Unknown output_activation: {output_activation}")


def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: "torch.device",
    base_channels: int,
) -> "nn.Module":
    """
    Expects checkpoints saved by train.py like:
      torch.save({"G": G.state_dict(), "MIL": MIL.state_dict()}, "best.pt")

    Also supports:
      - {"model": state_dict}
      - raw state_dict
    """
    cache_key = (os.path.abspath(checkpoint_path), str(device), int(base_channels))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from models import UNetGenerator  # imported from ml/mil+attention/models

    model = UNetGenerator(base_channels=base_channels).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "G" in ckpt:
            state_g = ckpt["G"]
        elif "model" in ckpt:
            state_g = ckpt["model"]
        else:
            state_g = ckpt
    else:
        state_g = ckpt

    model.load_state_dict(state_g, strict=True)
    _MODEL_CACHE[cache_key] = model
    return model


def _start_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


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

    device = torch.device(
        params.device if (params.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    use_amp = bool(params.use_amp and device.type == "cuda")

    model = _load_model_from_checkpoint(
        params.checkpoint,
        device,
        base_channels=int(params.base_channels),
    )

    H, W = bgr_u8.shape[:2]
    tile = int(params.tile_size)
    overlap = int(params.overlap)

    if tile <= 0:
        raise ValueError(f"tile_size must be positive, got {tile}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap * 2 >= tile:
        raise ValueError(f"overlap must be less than half the tile size; got tile={tile}, overlap={overlap}")

    # Center-crop inference:
    # run model on full tile, trust only interior region except at image borders
    stride = tile - 2 * overlap

    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_n = _normalize_rgb(rgb, params.normalize)

    ys = _start_positions(H, tile, stride)
    xs = _start_positions(W, tile, stride)

    out = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            patch = rgb_n[y0:y0 + tile, x0:x0 + tile, :]  # [ph,pw,3]

            ph, pw = patch.shape[:2]
            if ph != tile or pw != tile:
                pad = np.zeros((tile, tile, 3), dtype=np.float32)
                pad[:ph, :pw, :] = patch
                patch = pad

            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = model(t)
            else:
                y = model(t)

            y = y[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            y = _postprocess_model_output(y, params.output_activation)
            y = _denormalize_rgb(y, params.normalize)
            y = np.clip(y, 0.0, 1.0)

            # Keep center crop except on slide borders
            top_crop = 0 if y0 == 0 else overlap
            left_crop = 0 if x0 == 0 else overlap
            bottom_crop = 0 if y0 + tile >= H else overlap
            right_crop = 0 if x0 + tile >= W else overlap

            src_y0 = top_crop
            src_y1 = tile - bottom_crop
            src_x0 = left_crop
            src_x1 = tile - right_crop

            dst_y0 = y0 + top_crop
            dst_y1 = min(y0 + src_y1, H)
            dst_x0 = x0 + left_crop
            dst_x1 = min(x0 + src_x1, W)

            h_keep = dst_y1 - dst_y0
            w_keep = dst_x1 - dst_x0
            if h_keep <= 0 or w_keep <= 0:
                continue

            src = y[src_y0:src_y0 + h_keep, src_x0:src_x0 + w_keep, :]

            out[dst_y0:dst_y1, dst_x0:dst_x1, :] += src
            count[dst_y0:dst_y1, dst_x0:dst_x1, :] += 1.0

    out = out / np.maximum(count, 1e-8)
    out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

    meta = {
        "method": "ml",
        "checkpoint": params.checkpoint,
        "device": str(device),
        "tile_size": tile,
        "overlap": overlap,
        "stride": stride,
        "normalize": params.normalize,
        "output_activation": params.output_activation,
        "use_amp": use_amp,
        "base_channels": int(params.base_channels),
        "shape_in": (int(H), int(W)),
        "num_tiles_y": len(ys),
        "num_tiles_x": len(xs),
        "num_tiles_total": len(ys) * len(xs),
    }
    return out_bgr, meta
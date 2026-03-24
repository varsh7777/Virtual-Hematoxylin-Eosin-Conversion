# bb_infer.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MIL_ROOT = os.path.join(_THIS_DIR, "ml", "mil+attention")
if _MIL_ROOT not in sys.path:
    sys.path.insert(0, _MIL_ROOT)

from models.unet_diffusion import UNetDiffusion
from models.scheduler import BrownianBridgeScheduler
from models.bb_diffusion import BrownianBridgeDiffusion


@dataclass
class BBParams:
    checkpoint: str
    device: str = "cuda"
    tile_size: int = 512
    overlap: int = 128
    base_channels: int = 32
    time_dim: int = 128
    num_steps: int = 50
    sigma_min: float = 1e-4
    sigma_max: float = 0.05
    eta: float = 0.0


def _readable_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _bgr_to_rgb01(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _rgb01_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _start_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.no_grad()
def apply_bgr_bb(bgr_u8: np.ndarray, params: BBParams) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    bgr_u8 = _readable_bgr(bgr_u8)

    device = torch.device(
        params.device if (params.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    ckpt = torch.load(params.checkpoint, map_location=device)

    denoiser = UNetDiffusion(
        in_ch=3,
        cond_ch=3,
        out_ch=3,
        base_ch=int(params.base_channels),
        time_dim=int(params.time_dim),
    ).to(device)

    scheduler = BrownianBridgeScheduler(
        num_steps=int(params.num_steps),
        sigma_min=float(params.sigma_min),
        sigma_max=float(params.sigma_max),
        device=str(device),
    ).to(device)

    model = BrownianBridgeDiffusion(denoiser=denoiser, scheduler=scheduler).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    H, W = bgr_u8.shape[:2]
    tile = int(params.tile_size)
    overlap = int(params.overlap)

    if overlap * 2 >= tile:
        raise ValueError(f"overlap must be less than half tile size; got tile={tile}, overlap={overlap}")

    stride = tile - 2 * overlap
    ys = _start_positions(H, tile, stride)
    xs = _start_positions(W, tile, stride)

    rgb = _bgr_to_rgb01(bgr_u8)

    out = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            patch = rgb[y0:y0 + tile, x0:x0 + tile, :]
            ph, pw = patch.shape[:2]

            if ph != tile or pw != tile:
                pad = np.zeros((tile, tile, 3), dtype=np.float32)
                pad[:ph, :pw, :] = patch
                patch = pad

            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            yhat = model.sample(t, num_steps=int(params.num_steps), eta=float(params.eta))[0]
            yhat = yhat.permute(1, 2, 0).cpu().numpy()

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

            src = yhat[src_y0:src_y0 + h_keep, src_x0:src_x0 + w_keep, :]
            out[dst_y0:dst_y1, dst_x0:dst_x1, :] += src
            count[dst_y0:dst_y1, dst_x0:dst_x1, :] += 1.0

    out = out / np.maximum(count, 1e-8)
    out_bgr = _rgb01_to_bgr_u8(out)

    meta = {
        "method": "bb",
        "checkpoint": params.checkpoint,
        "device": str(device),
        "tile_size": tile,
        "overlap": overlap,
        "num_steps": int(params.num_steps),
        "shape_in": (int(H), int(W)),
        "num_tiles_total": len(ys) * len(xs),
    }
    return out_bgr, meta
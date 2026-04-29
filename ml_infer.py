# ml_infer.py  (WSI-optimised)

from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


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
    normalize: str = "0_1"            # "none" | "0_1" | "imagenet"
    base_channels: int = 32           # must match training config
    output_activation: str = "sigmoid"  # "sigmoid" | "none_0_1" | "tanh"

    batch_size: int = 16
    tissue_min_frac: float = 0.05
    tissue_white_thresh: int = 230


_MODEL_CACHE: Dict[Tuple[str, str, int], "nn.Module"] = {}

def _normalize_rgb(rgb_f32_0_1: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("none", "0_1"):
        return rgb_f32_0_1
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (rgb_f32_0_1 - mean) / std
    raise ValueError(f"Unknown normalize mode: {mode}")


def _denormalize_rgb(rgb_f32: np.ndarray, mode: str) -> np.ndarray:
    if mode in ("none", "0_1"):
        return rgb_f32
    if mode == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return rgb_f32 * std + mean
    raise ValueError(f"Unknown normalize mode: {mode}")


def _postprocess_model_output(y: np.ndarray, output_activation: str) -> np.ndarray:
    """Convert model output tensor (numpy) to a [0, 1] float RGB array."""
    act = output_activation.lower()
    if act in ("sigmoid", "none_0_1"):
        return np.clip(y, 0.0, 1.0)
    if act == "tanh":
        return np.clip((y + 1.0) / 2.0, 0.0, 1.0)
    raise ValueError(f"Unknown output_activation: {output_activation}")


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _is_small_unet(state_dict: dict) -> bool:
    """
    Returns True if the state_dict looks like it came from SmallUNet
    (train_paired_wsi.py) rather than UNetGenerator (MIL training).

    SmallUNet keys start with "inc.", "bot.", "down1.conv.", "up4.", "outc."
    UNetGenerator keys start with "down1.net.", "mid.", "dec3.", "out."
    """
    keys = set(state_dict.keys())
    # SmallUNet-specific keys
    if any(k.startswith("inc.") for k in keys):
        return True
    if any(k.startswith("bot.") for k in keys):
        return True
    if any(k.startswith("outc.") for k in keys):
        return True
    return False


def _build_small_unet(base_channels: int, device: "torch.device") -> "nn.Module":
    """Instantiate SmallUNet from train_paired_wsi without importing the whole
    training module — we duplicate the tiny architecture here so ml_infer has
    no hard dependency on train_paired_wsi at import time."""
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class Down(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.conv = DoubleConv(c_in, c_out)
        def forward(self, x): return self.conv(self.pool(x))

    class Up(nn.Module):
        def __init__(self, c_in, c_skip, c_out):
            super().__init__()
            self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((c_in // 2) + c_skip, c_out)
        def forward(self, x, skip):
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class SmallUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=3, base=32):
            super().__init__()
            self.inc   = DoubleConv(in_channels, base)
            self.down1 = Down(base,     base * 2)
            self.down2 = Down(base * 2, base * 4)
            self.down3 = Down(base * 4, base * 8)
            self.bot   = DoubleConv(base * 8, base * 16)
            self.up1   = Up(base * 16, base * 8, base * 8)
            self.up2   = Up(base * 8,  base * 4, base * 4)
            self.up3   = Up(base * 4,  base * 2, base * 2)
            self.up4   = Up(base * 2,  base,     base)
            self.outc  = nn.Conv2d(base, out_channels, kernel_size=1)
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            xb = self.bot(x4)
            x  = self.up1(xb, x4)
            x  = self.up2(x,  x3)
            x  = self.up3(x,  x2)
            x  = self.up4(x,  x1)
            return torch.sigmoid(self.outc(x))

    return SmallUNet(base=base_channels).to(device).eval()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: "torch.device",
    base_channels: int,
) -> "nn.Module":
    """
    Loads (and caches) a model from a checkpoint.

    Auto-detects architecture from state_dict keys:
      - SmallUNet  (saved by train_paired_wsi.py)  keys include "inc.", "bot.", "outc."
      - UNetGenerator (saved by MIL train.py)       keys include "down1.net.", "mid.", "dec3."

    Supported checkpoint formats
    ----------------------------
    - {"G": state_dict, ...}   (saved by train.py with MIL)
    - {"model": state_dict}    (saved by train_paired_wsi.py)
    - raw state_dict
    """
    cache_key = (os.path.abspath(checkpoint_path), str(device), int(base_channels))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "G" in ckpt:
            state_dict = ckpt["G"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if _is_small_unet(state_dict):
        print("  [ml_infer] detected SmallUNet checkpoint (train_paired_wsi)")
        model = _build_small_unet(base_channels, device)
    else:
        print("  [ml_infer] detected UNetGenerator checkpoint (MIL)")
        from models import UNetGenerator  # noqa: PLC0415
        model = UNetGenerator(base_channels=base_channels).to(device).eval()

    model.load_state_dict(state_dict, strict=True)

    # --- torch.compile (PyTorch >= 2.0) -----------------------------------
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  [ml_infer] torch.compile enabled (reduce-overhead)")
        except Exception as exc:
            print(f"  [ml_infer] torch.compile skipped: {exc}")

    _MODEL_CACHE[cache_key] = model
    return model


def _start_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def _is_tissue(patch_bgr_u8: np.ndarray, white_thresh: int, min_frac: float) -> bool:
    if min_frac <= 0.0:
        return True
    gray = cv2.cvtColor(patch_bgr_u8, cv2.COLOR_BGR2GRAY)
    tissue_px = int(np.count_nonzero(gray < white_thresh))
    return tissue_px / gray.size >= min_frac


def _tile_producer(
    positions: List[Tuple[int, int]],
    rgb_n: np.ndarray,
    bgr_u8: np.ndarray,
    tile: int,
    batch_size: int,
    white_thresh: int,
    min_tissue_frac: float,
    out_q: "queue.Queue[Optional[Tuple[List, np.ndarray, List[bool]]]]",
) -> None:
    for batch_start in range(0, len(positions), batch_size):
        batch_pos = positions[batch_start : batch_start + batch_size]
        patches: List[np.ndarray] = []
        flags: List[bool] = []

        for y0, x0 in batch_pos:
            patch = rgb_n[y0 : y0 + tile, x0 : x0 + tile, :]
            ph, pw = patch.shape[:2]
            needs_pad = ph != tile or pw != tile

            if needs_pad:
                pad = np.zeros((tile, tile, 3), dtype=np.float32)
                pad[:ph, :pw] = patch
                patch = pad

            bgr_patch = bgr_u8[y0 : y0 + tile, x0 : x0 + tile]
            if needs_pad:
                bgr_pad = np.full((tile, tile, 3), 255, dtype=np.uint8)
                bgr_pad[:ph, :pw] = bgr_patch
                bgr_patch = bgr_pad

            flags.append(_is_tissue(bgr_patch, white_thresh, min_tissue_frac))
            patches.append(patch)

        stacked = np.stack(patches, axis=0)
        out_q.put((batch_pos, stacked, flags))

    out_q.put(None)


def _accumulate(
    out: np.ndarray,
    count: np.ndarray,
    y: np.ndarray,
    y0: int,
    x0: int,
    tile: int,
    overlap: int,
    H: int,
    W: int,
) -> None:
    top_crop    = 0 if y0 == 0           else overlap
    left_crop   = 0 if x0 == 0           else overlap
    bottom_crop = 0 if y0 + tile >= H    else overlap
    right_crop  = 0 if x0 + tile >= W    else overlap

    src_y0 = top_crop;      src_y1 = tile - bottom_crop
    src_x0 = left_crop;     src_x1 = tile - right_crop

    dst_y0 = y0 + top_crop; dst_y1 = min(y0 + src_y1, H)
    dst_x0 = x0 + left_crop; dst_x1 = min(x0 + src_x1, W)

    h_keep = dst_y1 - dst_y0
    w_keep = dst_x1 - dst_x0
    if h_keep <= 0 or w_keep <= 0:
        return

    out  [dst_y0:dst_y1, dst_x0:dst_x1, :] += y [src_y0:src_y0+h_keep, src_x0:src_x0+w_keep, :]
    count[dst_y0:dst_y1, dst_x0:dst_x1, :]  += 1.0


@torch.no_grad()
def apply_bgr_ml(
    bgr_u8: np.ndarray,
    params: MLParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.  Install torch to use --method ml.")
    if bgr_u8 is None:
        raise ValueError("Input image is None.")

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

    H, W   = bgr_u8.shape[:2]
    tile   = int(params.tile_size)
    overlap = int(params.overlap)
    batch_size = max(1, int(params.batch_size))

    if tile <= 0:
        raise ValueError(f"tile_size must be positive, got {tile}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap * 2 >= tile:
        raise ValueError(
            f"overlap must be less than half the tile size; "
            f"got tile={tile}, overlap={overlap}"
        )

    stride = tile - 2 * overlap
    ys_pos = _start_positions(H, tile, stride)
    xs_pos = _start_positions(W, tile, stride)
    all_positions = [(y0, x0) for y0 in ys_pos for x0 in xs_pos]
    total_tiles   = len(all_positions)

    rgb   = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_n = _normalize_rgb(rgb, params.normalize)

    out   = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    prefetch_q: "queue.Queue" = queue.Queue(maxsize=4)
    producer_thread = threading.Thread(
        target=_tile_producer,
        args=(
            all_positions,
            rgb_n,
            bgr_u8,
            tile,
            batch_size,
            params.tissue_white_thresh,
            params.tissue_min_frac,
            prefetch_q,
        ),
        daemon=True,
    )
    producer_thread.start()

    tiles_done     = 0
    tiles_skipped  = 0

    while True:
        item = prefetch_q.get()
        if item is None:
            break

        batch_pos, stacked_patches, tissue_flags = item

        tissue_idx = [i for i, f in enumerate(tissue_flags) if f]
        bg_idx     = [i for i, f in enumerate(tissue_flags) if not f]

        for i in bg_idx:
            y0, x0 = batch_pos[i]
            y = np.clip(_denormalize_rgb(stacked_patches[i], params.normalize), 0.0, 1.0)
            _accumulate(out, count, y, y0, x0, tile, overlap, H, W)
            tiles_skipped += 1

        if tissue_idx:
            tissue_patches = stacked_patches[tissue_idx]
            t = (
                torch.from_numpy(tissue_patches)
                .permute(0, 3, 1, 2)
                .pin_memory()
                .to(device=device, dtype=torch.float32, non_blocking=True)
            )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y_batch = model(t)
            else:
                y_batch = model(t)

            y_np = y_batch.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(tissue_idx):
                y0, x0 = batch_pos[global_i]
                y = _postprocess_model_output(y_np[local_i], params.output_activation)
                y = _denormalize_rgb(y, params.normalize)
                y = np.clip(y, 0.0, 1.0)
                _accumulate(out, count, y, y0, x0, tile, overlap, H, W)

        tiles_done += len(batch_pos)
        print(
            f"  [ml_infer] tiles {tiles_done}/{total_tiles}"
            f"  (skipped background: {tiles_skipped})",
            end="\r",
        )

    producer_thread.join()
    print()

    out     = out / np.maximum(count, 1e-8)
    out_u8  = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)

    meta: Dict[str, Any] = {
        "method":           "ml",
        "checkpoint":       params.checkpoint,
        "device":           str(device),
        "tile_size":        tile,
        "overlap":          overlap,
        "stride":           stride,
        "batch_size":       batch_size,
        "normalize":        params.normalize,
        "output_activation": params.output_activation,
        "use_amp":          use_amp,
        "base_channels":    int(params.base_channels),
        "shape_in":         (int(H), int(W)),
        "num_tiles_y":      len(ys_pos),
        "num_tiles_x":      len(xs_pos),
        "num_tiles_total":  total_tiles,
        "tiles_skipped_bg": tiles_skipped,
        "tissue_min_frac":  params.tissue_min_frac,
        "tissue_white_thresh": params.tissue_white_thresh,
    }
    return out_bgr, meta
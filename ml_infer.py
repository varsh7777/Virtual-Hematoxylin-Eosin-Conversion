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
# Model loading
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: "torch.device",
    base_channels: int,
) -> "nn.Module":
    """
    Loads (and caches) a UNetGenerator from a checkpoint.

    Supported checkpoint formats
    ----------------------------
    - {"G": state_dict, ...}   (saved by train.py with MIL)
    - {"model": state_dict}
    - raw state_dict
    """
    cache_key = (os.path.abspath(checkpoint_path), str(device), int(base_channels))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from models import UNetGenerator  # noqa: PLC0415  (local import intentional)

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

    # --- torch.compile (PyTorch >= 2.0) -----------------------------------
    # `reduce-overhead` fuses small elementwise ops and removes repeated Python
    # dispatch cost — especially useful when running thousands of same-size
    # tiles.  Falls back gracefully on older PyTorch.
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
    """
    Returns True when enough of the patch looks like tissue rather than glass /
    background.  Uses a fast single-channel brightness check — same heuristic
    used by StainNet/ParamNet to skip blank tiles.
    """
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
    """
    Background thread: slices patches out of the (already normalised) numpy
    array, checks tissue content, and pushes batches into *out_q*.

    Each item pushed is  (batch_positions, stacked_patches_f32, is_tissue_flags).
    A None sentinel signals end-of-data to the consumer.
    """
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

            # Tissue check is done on uint8 BGR (faster, avoids float conversion)
            bgr_patch = bgr_u8[y0 : y0 + tile, x0 : x0 + tile]
            if needs_pad:
                bgr_pad = np.full((tile, tile, 3), 255, dtype=np.uint8)
                bgr_pad[:ph, :pw] = bgr_patch
                bgr_patch = bgr_pad

            flags.append(_is_tissue(bgr_patch, white_thresh, min_tissue_frac))
            patches.append(patch)

        stacked = np.stack(patches, axis=0)  # [B, H, W, 3]
        out_q.put((batch_pos, stacked, flags))

    out_q.put(None)  # sentinel


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
    """
    Run the stain-normalisation model over *bgr_u8* using tiled inference.

    Returns
    -------
    out_bgr : np.ndarray  uint8 BGR, same spatial size as input
    meta    : dict        provenance / diagnostic info
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed.  Install torch to use --method ml.")
    if bgr_u8 is None:
        raise ValueError("Input image is None.")

    # --- Normalise input dtype / channels ----------------------------------
    if bgr_u8.dtype != np.uint8:
        bgr_u8 = np.clip(bgr_u8, 0, 255).astype(np.uint8)
    if bgr_u8.ndim == 2:
        bgr_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_GRAY2BGR)
    elif bgr_u8.ndim == 3 and bgr_u8.shape[2] == 4:
        bgr_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGRA2BGR)

    # --- Device / AMP setup ------------------------------------------------
    device = torch.device(
        params.device if (params.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    use_amp = bool(params.use_amp and device.type == "cuda")

    # --- Load (and optionally compile) model --------------------------------
    model = _load_model_from_checkpoint(
        params.checkpoint,
        device,
        base_channels=int(params.base_channels),
    )

    # --- Tile geometry ------------------------------------------------------
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

    # --- Pre-compute normalised float image (read-only in producer thread) --
    rgb   = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_n = _normalize_rgb(rgb, params.normalize)

    # --- Output canvas -------------------------------------------------------
    out   = np.zeros((H, W, 3), dtype=np.float32)
    count = np.zeros((H, W, 1), dtype=np.float32)

    # --- Prefetch queue (maxsize=4 keeps ~4 batches in RAM ahead of GPU) ----
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

    # --- Consumer: GPU inference loop ---------------------------------------
    tiles_done     = 0
    tiles_skipped  = 0

    while True:
        item = prefetch_q.get()
        if item is None:
            break  # sentinel — producer finished

        batch_pos, stacked_patches, tissue_flags = item
        # stacked_patches: [B, tile, tile, 3]  float32

        # Separate tissue tiles from background tiles
        tissue_idx = [i for i, f in enumerate(tissue_flags) if f]
        bg_idx     = [i for i, f in enumerate(tissue_flags) if not f]

        # Background tiles: copy input patch directly (no model call)
        for i in bg_idx:
            y0, x0 = batch_pos[i]
            # Use the normalised float patch as-is (it's already "correct" —
            # blank glass regions don't need stain normalisation)
            y = np.clip(_denormalize_rgb(stacked_patches[i], params.normalize), 0.0, 1.0)
            _accumulate(out, count, y, y0, x0, tile, overlap, H, W)
            tiles_skipped += 1

        if tissue_idx:
            # Stack only the tissue subset → [T, 3, tile, tile]
            tissue_patches = stacked_patches[tissue_idx]  # [T, tile, tile, 3]
            t = (
                torch.from_numpy(tissue_patches)
                .permute(0, 3, 1, 2)   # [T, 3, tile, tile]
                .pin_memory()          # faster CPU→GPU DMA
                .to(device=device, dtype=torch.float32, non_blocking=True)
            )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y_batch = model(t)   # [T, 3, tile, tile]
            else:
                y_batch = model(t)

            # Move results back to CPU as a single transfer
            y_np = y_batch.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
            # y_np: [T, tile, tile, 3]

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
    print()  # newline after \r progress

    # --- Finalise -----------------------------------------------------------
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
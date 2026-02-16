# infer.py
# runs inference on a new slide/image: extracts patches, runs generator, stitches output, saves virtual H&E

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
import torch

from models import UNetGenerator
from utils import load_checkpoint
from utils.tiling import tile_coords


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    return img


def _bgr_to_rgb01(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _rgb01_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    u8 = (rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def run_infer(image_path: str, ckpt_path: str, out_path: str, tile: int = 512, overlap: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, map_location=device)

    G = UNetGenerator()
    G.load_state_dict(ckpt["G"], strict=False)
    G.to(device).eval()

    bgr = _read_bgr(image_path)
    H, W = bgr.shape[:2]
    rgb = _bgr_to_rgb01(bgr)

    ys, xs = tile_coords(H, W, tile, overlap)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W, 1), dtype=np.float32)

    # blending mask
    yy = np.linspace(0, 1, tile, dtype=np.float32)
    xx = np.linspace(0, 1, tile, dtype=np.float32)
    wy = 0.5 - 0.5 * np.cos(yy * np.pi)
    wx = 0.5 - 0.5 * np.cos(xx * np.pi)
    wmask = (wy[:, None] * wx[None, :]).astype(np.float32)[..., None]

    for y0 in ys:
        for x0 in xs:
            patch = rgb[y0:y0+tile, x0:x0+tile, :]
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            y = G(t)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            out_acc[y0:y0+tile, x0:x0+tile, :] += y * wmask
            w_acc[y0:y0+tile, x0:x0+tile, :] += wmask

    out = out_acc / np.maximum(w_acc, 1e-8)
    out_bgr = _rgb01_to_bgr_u8(out)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, out_bgr)
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()
    run_infer(args.image, args.ckpt, args.out, tile=args.tile, overlap=args.overlap)


if __name__ == "__main__":
    main()

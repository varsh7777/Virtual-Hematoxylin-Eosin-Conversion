# infer_bb.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml
import torch

from models.unet_diffusion import UNetDiffusion
from models.scheduler import BrownianBridgeScheduler
from models.bb_diffusion import BrownianBridgeDiffusion


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


def _start_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


@torch.no_grad()
def run_infer(
    image_path: str,
    ckpt_path: str,
    out_path: str,
    base_channels: int = 32,
    time_dim: int = 128,
    num_steps: int = 50,
    sigma_min: float = 1e-4,
    sigma_max: float = 0.05,
    tile: int = 512,
    overlap: int = 128,
    eta: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    denoiser = UNetDiffusion(
        in_ch=3,
        cond_ch=3,
        out_ch=3,
        base_ch=base_channels,
        time_dim=time_dim,
    ).to(device)

    scheduler = BrownianBridgeScheduler(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=str(device),
    ).to(device)

    model = BrownianBridgeDiffusion(denoiser=denoiser, scheduler=scheduler).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    bgr = _read_bgr(image_path)
    H, W = bgr.shape[:2]
    rgb = _bgr_to_rgb01(bgr)

    if overlap * 2 >= tile:
        raise ValueError("overlap must be less than half of tile")

    stride = tile - 2 * overlap
    ys = _start_positions(H, tile, stride)
    xs = _start_positions(W, tile, stride)

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
            yhat = model.sample(t, num_steps=num_steps, eta=eta)[0].permute(1, 2, 0).cpu().numpy()

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

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, out_bgr)
    print("saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--time-dim", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--sigma-min", type=float, default=1e-4)
    ap.add_argument("--sigma-max", type=float, default=0.05)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--eta", type=float, default=0.0)
    args = ap.parse_args()

    run_infer(
        image_path=args.image,
        ckpt_path=args.ckpt,
        out_path=args.out,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        tile=args.tile,
        overlap=args.overlap,
        eta=args.eta,
    )


if __name__ == "__main__":
    main()
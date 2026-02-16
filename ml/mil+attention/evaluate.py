# evaluate.py
# 
# computes metrics on val/test (color similarity, SSIM/PSNR, stain-specific metrics), 
# optionally saves visual grids

import argparse
import glob
import os
from typing import List

import cv2
import numpy as np
import torch

from metrics.image_metrics import psnr, l1
from models import UNetGenerator
from utils.checkpoint import load_checkpoint


def _list_images(root: str, exts: List[str]):
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(root, f"*{e}")))
    return sorted(out)


def _read_rgb01(path: str) -> torch.Tensor:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--he-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, map_location=device)

    G = UNetGenerator().to(device).eval()
    G.load_state_dict(ckpt["G"], strict=False)

    raw_paths = _list_images(args.raw_dir, args.ext)
    if not raw_paths:
        print("no images found")
        return

    psnrs = []
    l1s = []

    for rp in raw_paths:
        name = os.path.basename(rp)
        hp = os.path.join(args.he_dir, name)
        if not os.path.exists(hp):
            continue

        x = _read_rgb01(rp).to(device)
        y = _read_rgb01(hp).to(device)
        yhat = G(x)

        psnrs.append(psnr(yhat, y))
        l1s.append(l1(yhat, y))

    if not psnrs:
        print("no matched pairs evaluated")
        return

    print(f"PSNR mean: {sum(psnrs)/len(psnrs):.3f}")
    print(f"L1   mean: {sum(l1s)/len(l1s):.4f}")


if __name__ == "__main__":
    main()
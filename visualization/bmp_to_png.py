import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow very large images


def read_bmp_safe(path):
    """
    Try OpenCV first (fast).
    Fall back to Pillow for huge BMPs.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.astype(np.uint8)

    # Pillow fallback
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            rgb = np.array(im, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
    except Exception as e:
        raise RuntimeError(f"Failed to read BMP: {path}\n{e}")


def convert_folder(in_dir, out_dir, recursive=False):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    if recursive:
        bmp_files = list(in_dir.rglob("*.bmp"))
    else:
        bmp_files = list(in_dir.glob("*.bmp"))

    print(f"Found {len(bmp_files)} BMP files")

    for i, bmp in enumerate(bmp_files, 1):
        rel = bmp.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix(".png")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(bmp_files)}] Converting {bmp}")

        try:
            img = read_bmp_safe(bmp)
            ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
        except Exception as e:
            print(f"FAILED: {bmp}")
            print(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder with BMP files")
    ap.add_argument("--output", required=True, help="Output folder for PNG files")
    ap.add_argument("--recursive", action="store_true", help="Search subfolders")
    args = ap.parse_args()

    convert_folder(args.input, args.output, args.recursive)


if __name__ == "__main__":
    main()
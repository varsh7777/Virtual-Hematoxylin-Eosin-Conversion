# pipeline.py (Version G - with safe BMP read + pink bias params)
# Python 3.8/3.9 compatible

import argparse
from pathlib import Path
from typing import List, Union, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import stitch
from hsv import hsb_adjust

# NEW: ML method
from ml_infer import MLParams, apply_bgr_ml

# need to do: create + save color histograms?
# or maybe add a param to optionally save histograms

JPEG_QUALITY = 98


def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def list_subfolders(root: Union[str, Path]) -> List[Path]:
    root_p = Path(root)
    return sorted([p for p in root_p.iterdir() if p.is_dir()])


def list_bmps(root: Union[str, Path]) -> List[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    return sorted([p for p in root_p.iterdir() if p.is_file() and p.suffix.lower() == ".bmp"])


def _ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape for save: {img.shape}")

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)

    return img


def save_bmp(path: Union[str, Path], bgr: np.ndarray) -> None:
    bgr = _ensure_bgr_u8(bgr)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write BMP: {path}")


def _try_save_jpeg_opencv(path: Union[str, Path], bgr: np.ndarray, quality: int) -> bool:
    bgr = _ensure_bgr_u8(bgr)
    return cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def _try_save_jpeg_pillow(path: Union[str, Path], bgr: np.ndarray, quality: int) -> bool:
    try:
        bgr = _ensure_bgr_u8(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.save(str(path), format="JPEG", quality=int(quality), subsampling=0, optimize=True)
        return True
    except Exception:
        return False


def save_png(path: Union[str, Path], bgr: np.ndarray) -> None:
    bgr = _ensure_bgr_u8(bgr)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {path}")


def save_final_image_with_fallback(
    final_jpg_path: Union[str, Path],
    bgr: np.ndarray,
    quality: int,
) -> Tuple[Path, str]:
    bgr = _ensure_bgr_u8(bgr)
    h, w = bgr.shape[:2]

    if w > 65000 or h > 65000:
        png_path = Path(final_jpg_path).with_suffix(".png")
        save_png(png_path, bgr)
        return png_path, "png (auto, JPEG dimension limit)"

    if _try_save_jpeg_opencv(final_jpg_path, bgr, quality):
        return Path(final_jpg_path), "jpg (opencv)"

    if _try_save_jpeg_pillow(final_jpg_path, bgr, quality):
        return Path(final_jpg_path), "jpg (pillow)"

    png_path = Path(final_jpg_path).with_suffix(".png")
    save_png(png_path, bgr)
    return png_path, "png (fallback, JPEG write failed)"


def _read_bmp(path: Union[str, Path]) -> np.ndarray:
    """
    OpenCV BMP decoder fails for very large BMPs (~>=1GB decoded).
    This function tries OpenCV first, then falls back to Pillow.
    """
    path = Path(path)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return _ensure_bgr_u8(img)

    # Pillow fallback
    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            rgb = np.array(im, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return _ensure_bgr_u8(bgr)
    except Exception as e:
        raise RuntimeError(f"Failed to read BMP with OpenCV and Pillow: {path}\n{e}")


def run_pipeline(
    raw_root: Optional[str],
    stitched_out: str,
    final_out: str,
    num_rows: int = 0,
    overwrite: bool = False,
    skip_stitch: bool = False,
    stitched_root: Optional[str] = None,
    # NEW:
    method: str = "hsv",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 32,
    normalize: str = "0_1",
) -> None:
    ensure_dir(stitched_out)
    ensure_dir(final_out)

    # Tuned params (pink-leaning, less purple)
    hsb_params = {
        "chunk_width": 1024,
        "do_hue": True,
        "do_sat": True,
        "do_bri": True,

        # keep saturation under control
        "sat_floor_tissue": 20,
        "sat_push_nuclei": 0.05,
        "sat_push_cyto": 0.22,

        # brighten nuclei
        "v_gamma_nuclei": 0.30,
        "v_gamma_cyto": 0.82,
        "v_floor_nuclei": 97,

        # conservative hue pulls + NEW pink bias (negative values -> more pink)
        "hue_pull_nuclei": 0.03,
        "hue_pull_cyto": 0.05,
        "hue_bias_nuclei": -10,
        "hue_bias_cyto": -14,

        # background pink stabilization
        "white_hue_center": 140,
        "white_hue_strength": 0.22,
        "white_s_hi": 55,
        "white_v_lo": 150,
    }

    # Helper to apply chosen method
    def _apply_method(bgr: np.ndarray, meta_in: dict):
        if method == "hsv":
            bgr_adj, meta = hsb_adjust.adjust_bgr_image_hsb(
                bgr,
                params=hsb_params,
                metadata=meta_in,
            )
            return _ensure_bgr_u8(bgr_adj), meta

        if method == "ml":
            if not checkpoint:
                raise ValueError("--checkpoint is required when --method ml")

            ml_params = MLParams(
                checkpoint=checkpoint,
                device=device,
                tile_size=tile_size,
                overlap=overlap,
                use_amp=True,
                normalize=normalize,
            )
            bgr_adj, meta = apply_bgr_ml(bgr, ml_params)
            return _ensure_bgr_u8(bgr_adj), meta

        raise ValueError(f"Unknown method: {method}")

    if skip_stitch:
        if not stitched_root:
            raise ValueError("--skip-stitch requires --stitched-root")

        bmp_paths = list_bmps(stitched_root)
        if not bmp_paths:
            print(f"No .bmp files found under: {stitched_root}")
            return

        print(f"Found {len(bmp_paths)} stitched BMPs under: {stitched_root}")

        for idx, bmp_path in enumerate(bmp_paths, 1):
            name = bmp_path.stem
            stitched_bmp_path = Path(stitched_out) / f"{name}.bmp"
            final_jpg_path = Path(final_out) / f"{name}.jpg"

            if not overwrite:
                if final_jpg_path.exists() or final_jpg_path.with_suffix(".png").exists():
                    print(f"[{idx}/{len(bmp_paths)}] SKIP (exists): {final_jpg_path} or .png")
                    continue

            print(f"\n[{idx}/{len(bmp_paths)}] Processing stitched BMP: {bmp_path.name}")

            bgr = _read_bmp(bmp_path)

            # NOTE: For huge stitched BMPs, rewriting a copy can be slow / fail.
            save_bmp(stitched_bmp_path, bgr)
            print(f"  stitched (input) -> {stitched_bmp_path}  shape={bgr.shape} dtype={bgr.dtype}")

            bgr_adj, meta = _apply_method(bgr, {"source_bmp": str(bmp_path), "method": method})

            if method == "hsv":
                print(f"  adjust counts: {meta.get('counts')}")
                print(f"  version: {meta.get('version')}")
            else:
                print(f"  ml meta: {meta}")

            written_path, written_fmt = save_final_image_with_fallback(
                final_jpg_path, bgr_adj, quality=JPEG_QUALITY
            )
            print(f"  saved -> {written_path}  ({written_fmt}, quality={JPEG_QUALITY})")

        return

    # ---- Original mode: stitch from raw folders of tiles ----
    if raw_root is None:
        raise ValueError("raw_root is required unless --skip-stitch is used")

    folders = list_subfolders(raw_root)
    if not folders:
        print(f"No subfolders found under: {raw_root}")
        return

    print(f"Found {len(folders)} raw folders under: {raw_root}")

    for idx, folder in enumerate(folders, 1):
        name = folder.name
        stitched_bmp_path = Path(stitched_out) / f"{name}.bmp"
        final_jpg_path = Path(final_out) / f"{name}.jpg"

        if not overwrite:
            if final_jpg_path.exists() or final_jpg_path.with_suffix(".png").exists():
                print(f"[{idx}/{len(folders)}] SKIP (exists): {final_jpg_path} or .png")
                continue

        print(f"\n[{idx}/{len(folders)}] Processing: {name}")

        bgr = stitch.stitch_folder_to_image(str(folder), num_rows=num_rows)
        bgr = _ensure_bgr_u8(bgr)

        save_bmp(stitched_bmp_path, bgr)
        print(f"  stitched -> {stitched_bmp_path}  shape={bgr.shape} dtype={bgr.dtype}")

        bgr_adj, meta = _apply_method(bgr, {"source_folder": name, "method": method})

        if method == "hsv":
            print(f"  adjust counts: {meta.get('counts')}")
            print(f"  version: {meta.get('version')}")
        else:
            print(f"  ml meta: {meta}")

        written_path, written_fmt = save_final_image_with_fallback(
            final_jpg_path, bgr_adj, quality=JPEG_QUALITY
        )
        print(f"  saved -> {written_path}  ({written_fmt}, quality={JPEG_QUALITY})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stitching + HSB adjustment pipeline (supports stitched BMP mode)")

    ap.add_argument("--raw-root", help="Folder containing raw tile subfolders (stitch mode)")
    ap.add_argument("--stitched-out", default="stitched_bmps", help="Where to write stitched BMPs (and/or copies)")
    ap.add_argument("--final-out", default="final_outputs", help="Where to write final JPG/PNG")
    ap.add_argument("--num-rows", type=int, default=0, help="Optional limit on rows (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    ap.add_argument("--skip-stitch", action="store_true", help="Skip stitching; process already-stitched BMPs")
    ap.add_argument("--stitched-root", help="Folder containing stitched .bmp files (required with --skip-stitch)")

    # NEW: method switch
    ap.add_argument("--method", choices=["hsv", "ml"], default="hsv", help="Colorization method")

    # NEW: ML params (used if --method ml)
    ap.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint for ML method (.pt)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--tile-size", type=int, default=512, help="ML inference tile size")
    ap.add_argument("--overlap", type=int, default=32, help="ML inference overlap")
    ap.add_argument("--normalize", choices=["none", "0_1", "imagenet"], default="0_1", help="ML normalization mode")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        raw_root=args.raw_root,
        stitched_out=args.stitched_out,
        final_out=args.final_out,
        num_rows=args.num_rows,
        overwrite=args.overwrite,
        skip_stitch=args.skip_stitch,
        stitched_root=args.stitched_root,
        method=args.method,
        checkpoint=args.checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()

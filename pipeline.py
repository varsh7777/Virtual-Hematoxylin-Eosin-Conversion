# pipeline.py (Version G - ML compatible; safe BMP read; uses your DEFAULTS + brightness lift)
# Python 3.8/3.9 compatible

import argparse
from pathlib import Path
from typing import List, Union, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import stitch
from hsv import hsb_adjust

from ml_infer import MLParams, apply_bgr_ml

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


def list_svs(root: Union[str, Path]) -> List[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    return sorted([p for p in root_p.rglob("*.svs") if p.is_file()])


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
    """
    Always save as PNG (lossless).
    No JPEG.
    No lossy compression.
    Full resolution preserved.
    """

    bgr = _ensure_bgr_u8(bgr)

    final_png_path = Path(final_jpg_path).with_suffix(".png")

    ok = cv2.imwrite(
        str(final_png_path),
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],  # 0 = no compression (lossless)
    )

    if not ok:
        raise RuntimeError(f"Failed to write PNG: {final_png_path}")

    return final_png_path, "png (lossless, no compression)"


def _read_bmp(path: Union[str, Path]) -> np.ndarray:
    """
    OpenCV BMP decoder can fail for very large BMPs (~>=1GB decoded).
    Try OpenCV first, then fall back to Pillow.
    """
    path = Path(path)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return _ensure_bgr_u8(img)

    try:
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            rgb = np.array(im, dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return _ensure_bgr_u8(bgr)
    except Exception as e:
        raise RuntimeError(f"Failed to read BMP with OpenCV and Pillow: {path}\n{e}")


def _read_svs(path: Union[str, Path], level: int = 0) -> np.ndarray:
    from openslide import OpenSlide
    path = Path(path)
    slide = OpenSlide(str(path))
    w, h = slide.level_dimensions[level]
    rgba = np.array(slide.read_region((0, 0), level, (w, h)), dtype=np.uint8)  # RGBA
    rgb = rgba[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _ensure_bgr_u8(bgr)


def run_pipeline(
    raw_root: Optional[str],
    stitched_out: str,
    final_out: str,
    num_rows: int = 0,
    overwrite: bool = False,
    skip_stitch: bool = False,
    stitched_root: Optional[str] = None,
    method: str = "hsv",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 32,
    normalize: str = "0_1",
    svs_root: Optional[str] = None,
    svs_level: int = 0,
) -> None:
    ensure_dir(stitched_out)
    ensure_dir(final_out)

    # Your exact defaults + NEW brightness lift
    hsb_params = {
        "chunk_width": 1024,
        "do_hue": True,
        "do_sat": True,
        "do_bri": True,

        "white_s_hi": 55,
        "white_v_lo": 150,
        "min_sat_for_hue": 15,
        "nuclei_dark_v_thresh": 115,

        "gamma_lift": 0.75,
        "gamma_compress": 2.2,

        "sat_floor_tissue": 0,
        "sat_push_nuclei": 0.05,
        "sat_push_cyto": 0.12,

        "v_gamma_nuclei": 0.55,
        "v_gamma_cyto": 0.92,
        "v_floor_nuclei": 95,
        "v_floor_cyto": 0,

        "hue_pull_nuclei": 0.03,
        "hue_pull_cyto": 0.05,

        "hue_bias_nuclei": -10,
        "hue_bias_cyto": -14,

        "white_hue_center": 140,
        "white_hue_strength": 0.22,

        # NEW: brighten everything a bit
        "v_offset_all": 45,  # try 10..30
    }

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

    # ------------------------------------------------------------------ #
    #  SVS mode                                                            #
    # ------------------------------------------------------------------ #
    if svs_root:
        svs_paths = list_svs(svs_root)
        if not svs_paths:
            print(f"No .svs files found under: {svs_root}")
            return

        print(f"Found {len(svs_paths)} SVS files under: {svs_root}")

        for idx, svs_path in enumerate(svs_paths, 1):
            name = svs_path.stem
            final_jpg_path = Path(final_out) / f"{name}.jpg"

            if not overwrite:
                if final_jpg_path.exists() or final_jpg_path.with_suffix(".png").exists():
                    print(f"[{idx}/{len(svs_paths)}] SKIP (exists): {name}")
                    continue

            print(f"\n[{idx}/{len(svs_paths)}] Processing SVS: {svs_path.name}")

            bgr = _read_svs(svs_path, level=svs_level)
            print(f"  read SVS -> shape={bgr.shape} dtype={bgr.dtype} level={svs_level}")

            bgr_adj, meta = _apply_method(bgr, {"source_svs": str(svs_path), "method": method})

            if method == "hsv":
                print(f"  adjust counts: {meta.get('counts')}")
                print(f"  version: {meta.get('version')}")
            else:
                print(f"  ml meta: {meta}")

            written_path, written_fmt = save_final_image_with_fallback(
                final_jpg_path, bgr_adj, quality=JPEG_QUALITY
            )
            print(f"  saved -> {written_path}  ({written_fmt})")

        return

    # ------------------------------------------------------------------ #
    #  Skip-stitch mode (already-stitched BMPs)                           #
    # ------------------------------------------------------------------ #
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

            try:
                save_bmp(stitched_bmp_path, bgr)
                print(f"  stitched (input) -> {stitched_bmp_path}  shape={bgr.shape} dtype={bgr.dtype}")
            except Exception as e:
                print(f"  [WARN] Could not write stitched BMP (skipping): {stitched_bmp_path}")
                print(f"         Reason: {e}")
                print(f"  stitched -> (not saved)  shape={bgr.shape} dtype={bgr.dtype}")

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

    # ------------------------------------------------------------------ #
    #  Normal stitch mode (raw tile subfolders)                           #
    # ------------------------------------------------------------------ #
    if raw_root is None:
        raise ValueError("raw_root is required unless --skip-stitch or --svs-root is used")

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

        try:
            save_bmp(stitched_bmp_path, bgr)
            print(f"  stitched -> {stitched_bmp_path}  shape={bgr.shape} dtype={bgr.dtype}")
        except Exception as e:
            print(f"  [WARN] Could not write stitched BMP (skipping): {stitched_bmp_path}")
            print(f"         Reason: {e}")
            print(f"  stitched -> (not saved)  shape={bgr.shape} dtype={bgr.dtype}")

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
    ap = argparse.ArgumentParser(description="Stitching + HSB adjustment pipeline (supports stitched BMP mode and SVS mode)")

    ap.add_argument("--raw-root", help="Folder containing raw tile subfolders (stitch mode)")
    ap.add_argument("--stitched-out", default="stitched_bmps", help="Where to write stitched BMPs (and/or copies)")
    ap.add_argument("--final-out", default="final_outputs", help="Where to write final JPG/PNG")
    ap.add_argument("--num-rows", type=int, default=0, help="Optional limit on rows (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    ap.add_argument("--skip-stitch", action="store_true", help="Skip stitching; process already-stitched BMPs")
    ap.add_argument("--stitched-root", help="Folder containing stitched .bmp files (required with --skip-stitch)")

    ap.add_argument("--svs-root", help="Folder containing .svs files to process directly (no stitching needed)")
    ap.add_argument("--svs-level", type=int, default=0, help="SVS pyramid level to read (0 = full resolution)")

    ap.add_argument("--method", choices=["hsv", "ml"], default="hsv", help="Colorization method")

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
        svs_root=args.svs_root,
        svs_level=args.svs_level,
    )


if __name__ == "__main__":
    main()
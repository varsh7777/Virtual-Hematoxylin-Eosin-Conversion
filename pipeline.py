# pipeline.py (Version H+++ - current pipeline + paired TIFF training mode + TIFF inference mode)
# Python 3.8/3.9 compatible

import argparse
from pathlib import Path
from typing import List, Union, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import warnings

import stitch
from hsv import hsb_adjust
from ml_infer import MLParams, apply_bgr_ml
from bb_infer import BBParams, apply_bgr_bb

from train_paired_wsi import train_paired_wsi

JPEG_QUALITY = 98
SVS_MAX_DIM = 30000


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


def list_tiffs(root: Union[str, Path]) -> List[Path]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    tiffs = []
    for suffix in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tiffs.extend(root_p.rglob(suffix))
    return sorted(set(tiffs))


def _ensure_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape: {img.shape}")

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


def save_final_image_with_fallback(
    final_jpg_path: Union[str, Path],
    bgr: np.ndarray,
    quality: int,
    save_png: bool = True,
) -> Tuple[Path, str]:
    bgr = _ensure_bgr_u8(bgr)

    if save_png:
        final_png_path = Path(final_jpg_path).with_suffix(".png")
        ok = cv2.imwrite(str(final_png_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f"Failed to write PNG: {final_png_path}")
        return final_png_path, "png (lossless, compression=1)"

    final_jpg_path = Path(final_jpg_path).with_suffix(".jpg")
    if _try_save_jpeg_opencv(final_jpg_path, bgr, quality):
        return final_jpg_path, f"jpeg (opencv, quality={quality})"

    print("  [WARN] OpenCV JPEG write failed, trying Pillow...")
    if _try_save_jpeg_pillow(final_jpg_path, bgr, quality):
        return final_jpg_path, f"jpeg (pillow, quality={quality})"

    print("  [WARN] Pillow JPEG write failed, falling back to PNG (light compression)...")
    final_png_path = Path(final_jpg_path).with_suffix(".png")
    ok = cv2.imwrite(str(final_png_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {final_png_path}")
    return final_png_path, "png (fallback, compression=1)"


def _read_bmp(path: Union[str, Path]) -> np.ndarray:
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
        raise RuntimeError(f"Failed to read BMP: {path}\n{e}")


def _read_tif(path: Union[str, Path]) -> np.ndarray:
    """
    Read huge TIF/TIFF as BGR uint8.

    1) Try tifffile (best for large TIFFs).
    2) Fall back to Pillow, disabling MAX_IMAGE_PIXELS limit for trusted local files.
    """
    path = Path(path)

    try:
        import tifffile  # type: ignore
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            rgb = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[2] >= 3:
                rgb = arr[:, :, :3]
            else:
                raise ValueError(f"Unexpected tif array shape: {arr.shape}")
        else:
            raise ValueError(f"Unexpected tif array ndim: {arr.ndim} shape={arr.shape}")

        if rgb.dtype != np.uint8:
            if rgb.dtype == np.uint16:
                rgb = (rgb / 257.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return _ensure_bgr_u8(bgr)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [WARN] tifffile failed on {path.name}, falling back to Pillow. Reason: {e}")

    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(str(path)) as im:
                im = im.convert("RGB")
                rgb = np.array(im, dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return _ensure_bgr_u8(bgr)
    except Exception as e:
        raise RuntimeError(f"Failed to read TIF: {path}\n{e}")


def _sample_paired_patches(
    raw_bgr: np.ndarray,
    tgt_bgr: np.ndarray,
    patch: int,
    count: int,
    seed: int = 1337,
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    if raw_bgr.shape[:2] != tgt_bgr.shape[:2]:
        raise ValueError(f"raw/target size mismatch: raw={raw_bgr.shape[:2]} tgt={tgt_bgr.shape[:2]}")

    H, W = raw_bgr.shape[:2]
    if H < patch or W < patch:
        raise ValueError(f"Image smaller than patch: img=({H},{W}) patch={patch}")

    rng = np.random.RandomState(seed)
    out = []
    for _ in range(count):
        y0 = int(rng.randint(0, H - patch + 1))
        x0 = int(rng.randint(0, W - patch + 1))
        rp = raw_bgr[y0:y0 + patch, x0:x0 + patch].copy()
        tp = tgt_bgr[y0:y0 + patch, x0:x0 + patch].copy()
        out.append((rp, tp, (x0, y0)))
    return out


def _read_svs(path: Union[str, Path], level: int = 0) -> np.ndarray:
    from openslide import OpenSlide

    path = Path(path)
    slide = OpenSlide(str(path))

    num_levels = slide.level_count
    print(f"  SVS levels: {num_levels}  dimensions per level: {slide.level_dimensions}")

    chosen_level = min(level, num_levels - 1)
    w, h = slide.level_dimensions[chosen_level]
    while max(w, h) > SVS_MAX_DIM and chosen_level < num_levels - 1:
        chosen_level += 1
        w, h = slide.level_dimensions[chosen_level]
        print(f"  Image too large; stepping up to level {chosen_level} ({w}x{h})")

    if chosen_level != level:
        print(f"  [INFO] Requested level {level} -> using level {chosen_level} ({w}x{h})")
    else:
        print(f"  Reading level {chosen_level}: {w}x{h} px")

    TILE = 4096
    bgr_out = np.empty((h, w, 3), dtype=np.uint8)
    scale = slide.level_downsamples[chosen_level]

    total_tiles = ((h + TILE - 1) // TILE) * ((w + TILE - 1) // TILE)
    tile_idx = 0

    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            tw = min(TILE, w - x)
            th = min(TILE, h - y)

            x0 = int(x * scale)
            y0 = int(y * scale)

            region = slide.read_region((x0, y0), chosen_level, (tw, th))
            rgba = np.array(region, dtype=np.uint8)
            bgr_out[y:y + th, x:x + tw] = rgba[:, :, 2::-1]

            tile_idx += 1
            if tile_idx % 50 == 0 or tile_idx == total_tiles:
                print(f"  ... tile {tile_idx}/{total_tiles}", end="\r")

    print()
    slide.close()
    return _ensure_bgr_u8(bgr_out)


def run_pipeline(
    raw_root: Optional[str],
    stitched_out: str,
    final_out: str,
    num_rows: int = 0,
    overwrite: bool = False,
    save_png: bool = True,
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
    tiff_root: Optional[str] = None,
    export_train_pairs: bool = False,
    targets_root: Optional[str] = None,
    train_raw_out: str = "train_pairs/raw",
    train_he_out: str = "train_pairs/he",
    patches_per_slide: int = 200,
    patch_size: int = 256,
    patch_seed: int = 1337,
    task: str = "process",
    pairs: Optional[List[List[str]]] = None,
    epochs: int = 20,
    tiles_per_epoch: int = 1000,
    val_tiles_per_slide: int = 128,
    batch_size: int = 8,
    lr: float = 1e-4,
    num_workers: int = 0,
    resume_checkpoint: Optional[str] = None,
    stride: Optional[int] = None,
    perceptual_weight: float = 0.0,
) -> None:
    ensure_dir(stitched_out)
    ensure_dir(final_out)

    if task == "train_paired_wsi":
        if not pairs:
            raise ValueError("At least one --pair RAW_TIF TARGET_TIF must be provided for training.")
        train_paired_wsi(
            pairs=[(p[0], p[1]) for p in pairs],
            out_dir=final_out,
            epochs=epochs,
            tiles_per_epoch=tiles_per_epoch,
            val_tiles_per_slide=val_tiles_per_slide,
            batch_size=batch_size,
            tile_size=tile_size,
            lr=lr,
            num_workers=num_workers,
            resume_checkpoint=resume_checkpoint,
            stride=stride,
            perceptual_weight=perceptual_weight,
        )
        return

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

        "v_offset_all": 45,
    }

    def _apply_method(bgr: np.ndarray, meta_in: dict):
        if method == "hsv":
            bgr_adj, meta = hsb_adjust.adjust_bgr_image_hsb(bgr, params=hsb_params, metadata=meta_in)
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
                base_channels=32,
                output_activation="sigmoid"
            )
            bgr_adj, meta = apply_bgr_ml(bgr, ml_params)
            return _ensure_bgr_u8(bgr_adj), meta

        if method == "bb":
            if not checkpoint:
                raise ValueError("--checkpoint is required when --method bb")

            bb_params = BBParams(
                checkpoint=checkpoint,
                device=device,
                tile_size=tile_size,
                overlap=overlap,
                base_channels=32,
                time_dim=128,
                num_steps=20,
                sigma_min=1e-4,
                sigma_max=0.05,
                eta=0.0,
            )
            bgr_adj, meta = apply_bgr_bb(bgr, bb_params)
            return _ensure_bgr_u8(bgr_adj), meta

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------ #
    #  TIFF inference mode                                                 #
    # ------------------------------------------------------------------ #
    if tiff_root:
        tiff_paths = list_tiffs(tiff_root)
        if not tiff_paths:
            print(f"No .tif/.tiff files found under: {tiff_root}")
            return

        print(f"Found {len(tiff_paths)} TIFF files under: {tiff_root}")
        failed = []

        for idx, tiff_path in enumerate(tiff_paths, 1):
            name = tiff_path.stem
            final_jpg_path = Path(final_out) / f"{name}.jpg"

            if not overwrite:
                if final_jpg_path.exists() or final_jpg_path.with_suffix(".png").exists():
                    print(f"[{idx}/{len(tiff_paths)}] SKIP (exists): {name}")
                    continue

            print(f"\n[{idx}/{len(tiff_paths)}] Processing TIFF: {tiff_path.name}")

            try:
                bgr = _read_tif(tiff_path)
                print(f"  read TIFF -> shape={bgr.shape} dtype={bgr.dtype}")
            except Exception as e:
                print(f"  [ERROR] Could not read TIFF, skipping: {tiff_path.name}")
                print(f"          Reason: {e}")
                failed.append((tiff_path.name, f"read error: {e}"))
                continue

            try:
                bgr_adj, meta = _apply_method(bgr, {"source_tiff": str(tiff_path), "method": method})
            except Exception as e:
                print(f"  [ERROR] Adjustment failed, skipping: {tiff_path.name}")
                print(f"          Reason: {e}")
                failed.append((tiff_path.name, f"adjustment error: {e}"))
                continue

            try:
                written_path, written_fmt = save_final_image_with_fallback(final_jpg_path, bgr_adj, quality=JPEG_QUALITY, save_png=save_png)
                print(f"  saved -> {written_path}  ({written_fmt})")
            except Exception as e:
                print(f"  [ERROR] Could not save output, skipping: {tiff_path.name}")
                print(f"          Reason: {e}")
                failed.append((tiff_path.name, f"save error: {e}"))
                continue

        if failed:
            print(f"\n{'=' * 60}")
            print(f"FAILED FILES ({len(failed)}):")
            for fname, reason in failed:
                print(f"  {fname}")
                print(f"    {reason}")
            print(f"{'=' * 60}")
        else:
            print(f"\nAll {len(tiff_paths)} files processed successfully.")
        return

    if export_train_pairs:
        if raw_root is None:
            raise ValueError("--export-train-pairs requires --raw-root (folders of tiles)")
        if not targets_root:
            raise ValueError("--export-train-pairs requires --targets-root")

        ensure_dir(train_raw_out)
        ensure_dir(train_he_out)

        folders = list_subfolders(raw_root)
        if not folders:
            print(f"No subfolders found under: {raw_root}")
            return

        print(f"Exporting training pairs from {len(folders)} raw folders...")
        failed = []

        for idx, folder in enumerate(folders, 1):
            name = folder.name

            tif_path = Path(targets_root) / f"{name}.tif"
            if not tif_path.exists():
                tif_path = Path(targets_root) / f"{name}.tiff"

            if not tif_path.exists():
                print(f"[{idx}/{len(folders)}] [WARN] No target tif for {name}; skipping.")
                failed.append((name, "missing target tif"))
                continue

            print(f"\n[{idx}/{len(folders)}] Exporting pairs for: {name}")

            try:
                raw_bgr = stitch.stitch_folder_to_image(str(folder), num_rows=num_rows)
                raw_bgr = _ensure_bgr_u8(raw_bgr)
                print(f"  stitched raw -> shape={raw_bgr.shape} dtype={raw_bgr.dtype}")
            except Exception as e:
                print(f"  [ERROR] Stitch failed for {name}: {e}")
                failed.append((name, f"stitch failed: {e}"))
                continue

            try:
                tgt_bgr = _read_tif(tif_path)
                tgt_bgr = _ensure_bgr_u8(tgt_bgr)
                print(f"  read target -> shape={tgt_bgr.shape} dtype={tgt_bgr.dtype}")
            except Exception as e:
                print(f"  [ERROR] Read target TIF failed for {name}: {e}")
                failed.append((name, f"target read failed: {e}"))
                continue

            Hr, Wr = raw_bgr.shape[:2]
            Ht, Wt = tgt_bgr.shape[:2]

            H = min(Hr, Ht)
            W = min(Wr, Wt)

            if H < patch_size or W < patch_size:
                msg = (
                    f"overlap too small for patch_size={patch_size}: "
                    f"overlap=({H},{W}) raw=({Hr},{Wr}) tgt=({Ht},{Wt})"
                )
                print(f"  [ERROR] {msg}")
                failed.append((name, msg))
                continue

            if (Hr, Wr) != (Ht, Wt):
                print(f"  [WARN] size mismatch raw=({Hr},{Wr}) tgt=({Ht},{Wt}) -> using overlap crop=({H},{W})")

            raw_use = raw_bgr[:H, :W, :]
            tgt_use = tgt_bgr[:H, :W, :]

            try:
                pairs_out = _sample_paired_patches(
                    raw_bgr=raw_use,
                    tgt_bgr=tgt_use,
                    patch=patch_size,
                    count=patches_per_slide,
                    seed=patch_seed,
                )
            except Exception as e:
                print(f"  [ERROR] Patch sampling failed for {name}: {e}")
                failed.append((name, f"patch sampling failed: {e}"))
                continue

            for j, (rp, tp, (x0, y0)) in enumerate(pairs_out):
                stem = f"{name}__{j:06d}_x{x0:08d}_y{y0:08d}.png"
                raw_out = Path(train_raw_out) / stem
                he_out = Path(train_he_out) / stem
                cv2.imwrite(str(raw_out), rp)
                cv2.imwrite(str(he_out), tp)

            print(f"  wrote {len(pairs_out)} patch pairs")

        if failed:
            print("\nFAILED:")
            for n, r in failed:
                print(f"  {n}: {r}")
        else:
            print("\nAll slides exported successfully.")
        return

    if svs_root:
        svs_paths = list_svs(svs_root)
        if not svs_paths:
            print(f"No .svs files found under: {svs_root}")
            return

        print(f"Found {len(svs_paths)} SVS files under: {svs_root}")
        failed = []

        for idx, svs_path in enumerate(svs_paths, 1):
            name = svs_path.stem
            final_jpg_path = Path(final_out) / f"{name}.jpg"

            if not overwrite:
                if final_jpg_path.exists() or final_jpg_path.with_suffix(".png").exists():
                    print(f"[{idx}/{len(svs_paths)}] SKIP (exists): {name}")
                    continue

            print(f"\n[{idx}/{len(svs_paths)}] Processing SVS: {svs_path.name}")

            try:
                bgr = _read_svs(svs_path, level=svs_level)
                print(f"  read SVS -> shape={bgr.shape} dtype={bgr.dtype}")
            except Exception as e:
                print(f"  [ERROR] Could not read SVS, skipping: {svs_path.name}")
                print(f"          Reason: {e}")
                failed.append((svs_path.name, f"read error: {e}"))
                continue

            try:
                bgr_adj, meta = _apply_method(bgr, {"source_svs": str(svs_path), "method": method})
            except Exception as e:
                print(f"  [ERROR] Adjustment failed, skipping: {svs_path.name}")
                print(f"          Reason: {e}")
                failed.append((svs_path.name, f"adjustment error: {e}"))
                continue

            try:
                written_path, written_fmt = save_final_image_with_fallback(final_jpg_path, bgr_adj, quality=JPEG_QUALITY, save_png=save_png)
                print(f"  saved -> {written_path}  ({written_fmt})")
            except Exception as e:
                print(f"  [ERROR] Could not save output, skipping: {svs_path.name}")
                print(f"          Reason: {e}")
                failed.append((svs_path.name, f"save error: {e}"))
                continue

        if failed:
            print(f"\n{'=' * 60}")
            print(f"FAILED FILES ({len(failed)}):")
            for fname, reason in failed:
                print(f"  {fname}")
                print(f"    {reason}")
            print(f"{'=' * 60}")
        else:
            print(f"\nAll {len(svs_paths)} files processed successfully.")
        return

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
                print(f"  [WARN] Could not write stitched BMP: {e}")

            bgr_adj, meta = _apply_method(bgr, {"source_bmp": str(bmp_path), "method": method})
            written_path, written_fmt = save_final_image_with_fallback(final_jpg_path, bgr_adj, quality=JPEG_QUALITY, save_png=save_png)
            print(f"  saved -> {written_path}  ({written_fmt})")

        return

    if raw_root is None:
        raise ValueError(
            "raw_root is required unless --skip-stitch or --svs-root or --tiff-root or "
            "--export-train-pairs or --task train_paired_wsi is used"
        )

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
            print(f"  [WARN] Could not write stitched BMP: {e}")

        bgr_adj, meta = _apply_method(bgr, {"source_folder": name, "method": method})
        written_path, written_fmt = save_final_image_with_fallback(final_jpg_path, bgr_adj, quality=JPEG_QUALITY, save_png=save_png)
        print(f"  saved -> {written_path}  ({written_fmt})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stitching + adjustment pipeline (supports stitched BMP, SVS, TIFF, raw tile modes, exporting train pairs, and paired TIFF training)"
    )

    ap.add_argument("--task", choices=["process", "train_paired_wsi"], default="process")

    ap.add_argument("--raw-root", help="Folder containing raw tile subfolders (stitch mode)")
    ap.add_argument("--stitched-out", default="stitched_bmps", help="Where to write stitched BMPs (and/or copies)")
    ap.add_argument("--final-out", default="final_outputs", help="Where to write final JPG/PNG")
    ap.add_argument("--num-rows", type=int, default=0, help="Optional limit on rows (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--save-jpeg", action="store_true", help="Save outputs as JPEG instead of the default lossless PNG")

    ap.add_argument("--skip-stitch", action="store_true", help="Skip stitching; process already-stitched BMPs")
    ap.add_argument("--stitched-root", help="Folder containing stitched .bmp files (required with --skip-stitch)")

    ap.add_argument("--svs-root", help="Folder containing .svs files to process directly (no stitching needed)")
    ap.add_argument("--svs-level", type=int, default=0, help="SVS pyramid level (0=full res). Auto-steps up if exceeds SVS_MAX_DIM.")

    ap.add_argument("--tiff-root", help="Folder containing .tif/.tiff files to process directly (no stitching needed)")

    ap.add_argument("--method", choices=["hsv", "ml", "bb"], default="hsv", help="Colorization method")
    ap.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint for ML method (.pt)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--tile-size", type=int, default=512, help="ML inference tile size, and training tile size")
    ap.add_argument("--overlap", type=int, default=32, help="ML inference overlap")
    ap.add_argument("--normalize", choices=["none", "0_1", "imagenet"], default="0_1", help="ML normalization mode")

    ap.add_argument("--export-train-pairs", action="store_true", help="Export paired patches from raw tile folders + target TIFFs, then exit.")
    ap.add_argument("--targets-root", default=None, help="Folder containing target .tif/.tiff files matching raw folder names.")
    ap.add_argument("--train-raw-out", default="train_pairs/raw", help="Output folder for exported raw patches.")
    ap.add_argument("--train-he-out", default="train_pairs/he", help="Output folder for exported target patches.")
    ap.add_argument("--patches-per-slide", type=int, default=200, help="How many patch pairs to export per slide.")
    ap.add_argument("--patch-size", type=int, default=256, help="Patch size for exported training patches.")
    ap.add_argument("--patch-seed", type=int, default=1337, help="Seed for patch sampling reproducibility.")

    ap.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("RAW_TIF", "TARGET_TIF"),
        help="Paired raw/target TIFF paths. Repeat for multiple slide pairs."
    )
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs for --task train_paired_wsi")
    ap.add_argument("--tiles-per-epoch", type=int, default=1000, help="Number of training tiles consumed per epoch")
    ap.add_argument("--val-tiles-per-slide", type=int, default=128, help="Validation tiles reserved per slide")
    ap.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Training learning rate")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for training")
    ap.add_argument("--resume-checkpoint", default=None, help="Resume paired TIFF training from checkpoint")
    ap.add_argument("--stride", type=int, default=None, help="Stride for tile manifest generation; defaults to tile-size")
    ap.add_argument("--perceptual-weight", type=float, default=0.0, help="Weight for perceptual (VGG) loss during training. 0 = disabled. Recommended: 0.1")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        raw_root=args.raw_root,
        stitched_out=args.stitched_out,
        final_out=args.final_out,
        num_rows=args.num_rows,
        overwrite=args.overwrite,
        save_png=not args.save_jpeg,
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
        tiff_root=args.tiff_root,
        export_train_pairs=args.export_train_pairs,
        targets_root=args.targets_root,
        train_raw_out=args.train_raw_out,
        train_he_out=args.train_he_out,
        patches_per_slide=args.patches_per_slide,
        patch_size=args.patch_size,
        patch_seed=args.patch_seed,
        task=args.task,
        pairs=args.pair,
        epochs=args.epochs,
        tiles_per_epoch=args.tiles_per_epoch,
        val_tiles_per_slide=args.val_tiles_per_slide,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint,
        stride=args.stride,
        perceptual_weight=args.perceptual_weight,
    )


if __name__ == "__main__":
    main()

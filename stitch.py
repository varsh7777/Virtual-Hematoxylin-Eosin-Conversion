# stitch.py  (WSI-optimised)
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import cv2
import numpy as np

def find_overlap_start(img_a, img_b, direction="horizontal"):
    """
    Find stitch overlap between two images.

    direction = "horizontal"  → left + right images
    direction = "vertical"    → top + bottom images
    """
    if direction not in ("horizontal", "vertical"):
        raise ValueError('direction must be "horizontal" or "vertical"')

    # ----- vertical stitching (top_img, bottom_img) -----
    if direction == "vertical":
        height, width = max(img_a.shape[:2], img_b.shape[:2])

        haystack_offset_y = height // 8
        needle_height     = height // 32

        haystack = img_a[height - haystack_offset_y:, :]
        needle   = img_b[:needle_height, width // 4 : width - width // 4]

        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        max_loc = (max_loc[0] - 200, max_loc[1] + (height - haystack_offset_y))
        return max_loc[1]

    # ----- horizontal stitching (left_img, right_img) -----
    else:
        if img_a.shape != img_b.shape:
            raise ValueError("Horizontal mode requires same image shape.")

        height, width = img_a.shape[:2]

        haystack_offset_x = width // 8
        needle_width      = width // 32

        haystack = img_a[:, width - haystack_offset_x:]
        needle   = img_b[200 : height - 200, :needle_width]

        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        max_loc = (max_loc[0] + (width - haystack_offset_x), max_loc[1] - 200)
        return max_loc[0]


def find_overlaps(images: List[np.ndarray], direction: str = "horizontal") -> List[int]:
    """
    Compute overlap start positions for every adjacent pair in *images*.

    Each pair is processed in a separate thread so that all matchTemplate
    calls execute concurrently.  Results are assembled in order before the
    final image's full dimension is appended (same contract as the original).
    """
    n = len(images)
    if n == 1:
        dim = images[0].shape[1] if direction == "horizontal" else images[0].shape[0]
        return [dim]

    # Submit all pair-wise comparisons in parallel
    overlap_starts = [None] * (n - 1)
    with ThreadPoolExecutor(max_workers=min(n - 1, os.cpu_count() or 4)) as ex:
        future_to_idx = {
            ex.submit(find_overlap_start, images[i], images[i + 1], direction): i
            for i in range(n - 1)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            overlap_starts[idx] = future.result()

    # Append the last image's full width/height (unchanged semantics)
    if direction == "horizontal":
        overlap_starts.append(images[-1].shape[1])
    else:
        overlap_starts.append(images[-1].shape[0])

    print("Overlap starts:", overlap_starts)
    return overlap_starts


def stitch_images(images, overlap_starts, direction="horizontal", blend_size=100):
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    is_color = len(images[0].shape) == 3
    channels = images[0].shape[2] if is_color else 1

    if direction == "horizontal":
        total_width  = max(overlap_starts[0] + sum(overlap_starts[i] for i in range(1, len(overlap_starts))),
                           sum(overlap_starts))
        total_height = max(img.shape[0] for img in images)
        shape        = (total_height, total_width, channels) if is_color else (total_height, total_width)
    else:
        total_height = max(overlap_starts[0] + sum(overlap_starts[i] for i in range(1, len(overlap_starts))),
                           sum(overlap_starts))
        total_width  = max(img.shape[1] for img in images)
        shape        = (total_height, total_width, channels) if is_color else (total_height, total_width)

    result      = np.zeros(shape, dtype=images[0].dtype)
    current_pos = 0

    for i, start in enumerate(overlap_starts):
        img  = images[i]
        h, w = img.shape[:2]

        if direction == "horizontal":
            if i > 0 and blend_size > 0:
                blend_start = current_pos
                blend_end   = min(current_pos + blend_size, result.shape[1], current_pos + w)
                alpha       = (np.linspace(0, 1, blend_end - blend_start)[None, :, None]
                               if is_color else np.linspace(0, 1, blend_end - blend_start)[None, :])

                result[:h, blend_start:blend_end] = (
                    (1 - alpha) * result[:h, blend_start:blend_end]
                    + alpha     * img[:, :blend_end - blend_start]
                ).astype(img.dtype)

                result[:h, blend_end : current_pos + w] = img[:, blend_end - blend_start:]
            else:
                result[:h, current_pos : current_pos + w] = img
        else:
            if i > 0 and blend_size > 0:
                blend_start = current_pos
                blend_end   = min(current_pos + blend_size, result.shape[0], current_pos + h)
                alpha       = (np.linspace(0, 1, blend_end - blend_start)[:, None, None]
                               if is_color else np.linspace(0, 1, blend_end - blend_start)[:, None])

                result[blend_start:blend_end, :w] = (
                    (1 - alpha) * result[blend_start:blend_end, :w]
                    + alpha     * img[:blend_end - blend_start, :]
                ).astype(img.dtype)

                result[blend_end : current_pos + h, :w] = img[blend_end - blend_start:, :]
            else:
                result[current_pos : current_pos + h, :w] = img

        current_pos += start

    return result



def image_reader(image_dir, num_rows=0, base_dir="images"):
    """
    Read tile images from base_dir/image_dir.
    Returns a 2-D list: rows of images, grouped by the Y index in the filename.
    Filenames expected like: "<whatever> Y00000 X00000.bmp"
    """
    image_rows  = []
    reading_row = 0
    current_row = []

    folder_path = os.path.join(base_dir, image_dir)
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".bmp"):
            y_val = int(fname.split()[1][1:])   # token like "Y00000"
            if y_val != reading_row:
                reading_row = y_val
                image_rows.append(current_row)
                current_row = []
                if num_rows > 0 and reading_row >= num_rows:
                    break

            print("Reading image:", fname)
            img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_COLOR)
            current_row.append(img)

    if current_row:
        image_rows.append(current_row)

    return image_rows


def stitch_image(path, save_path=None):
    images = image_reader(path)
    rows   = []

    for row in images:
        overlap_starts = find_overlaps(row, direction="horizontal")
        rows.append(stitch_images(row, overlap_starts, direction="horizontal"))

    final_overlap_starts = find_overlaps(rows, direction="vertical")
    final_image          = stitch_images(rows, final_overlap_starts, direction="vertical")

    if save_path:
        cv2.imwrite(save_path, final_image)

    return final_image


def stitch_folder_to_image(image_dir: str, num_rows: int = 0) -> np.ndarray:
    """
    Stitch a folder of tiles into a single BGR image (returned in memory).

    image_dir : path to the folder containing tile BMPs
    num_rows  : optional row limit (0 = all)
    """
    images = image_reader(image_dir, num_rows=num_rows)

    if not images or not images[0]:
        raise ValueError(f"No images found in folder: {image_dir}")

    rows = []
    for row in images:
        overlap_starts = find_overlaps(row, direction="horizontal")
        rows.append(stitch_images(row, overlap_starts, direction="horizontal"))

    final_overlap_starts = find_overlaps(rows, direction="vertical")
    return stitch_images(rows, final_overlap_starts, direction="vertical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder",      help="Folder name containing tile BMPs (under --base_dir)")
    ap.add_argument("--base_dir",  default="images", help="Root containing folders of tiles")
    ap.add_argument("--out",       default="stitch.bmp", help="Output stitched BMP")
    ap.add_argument("--show",      action="store_true",  help="Show stitched image window")
    args = ap.parse_args()

    final = stitch_image(args.folder, save_path=args.out)

    if args.show:
        cv2.imshow("Final Image", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
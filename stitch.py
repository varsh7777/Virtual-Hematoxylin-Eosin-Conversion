import cv2
import numpy as np
import os
import argparse

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
        needle_height = height // 32

        haystack = img_a[height - haystack_offset_y:, :]
        needle = img_b[:needle_height, width//4 : width - width//4]

        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Adjust
        max_loc = (max_loc[0] - 200, max_loc[1] + (height - haystack_offset_y))
        return max_loc[1]

    # ----- horizontal stitching (left_img, right_img) -----
    else:
        if img_a.shape != img_b.shape:
            raise ValueError("Horizontal mode requires same image shape.")

        height, width = img_a.shape[:2]

        haystack_offset_x = width // 8
        needle_width = width // 32

        haystack = img_a[:, width - haystack_offset_x:]
        needle = img_b[200:height-200, :needle_width]

        res = cv2.matchTemplate(haystack, needle, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Adjust
        max_loc = (max_loc[0] + (width - haystack_offset_x), max_loc[1] - 200)
        return max_loc[0]


def find_overlaps(images, direction="horizontal"):
    overlap_starts = []
    for i in range(len(images) - 1):
        if direction == "horizontal":
            overlap_starts.append(find_overlap_start(images[i], images[i+1], direction="horizontal"))
        else:
            overlap_starts.append(find_overlap_start(images[i], images[i+1], direction="vertical"))

    # and the last image is used whole
    if direction == "horizontal":
        overlap_starts.append(images[-1].shape[1])
    else:
        overlap_starts.append(images[-1].shape[0])

    print("Overlap starts:", overlap_starts)
    return overlap_starts


def stitch_images(images, overlap_starts, direction="horizontal", blend_size=100):
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    # check for gray or color images
    is_color = len(images[0].shape) == 3
    channels = images[0].shape[2] if is_color else 1

    # get size of final image
    if direction == "horizontal":
        total_width = overlap_starts[0] + sum(overlap_starts[i] for i in range(1, len(overlap_starts)))
        total_width = max(total_width, sum(overlap_starts))  # just in case
        total_height = max(img.shape[0] for img in images)
        shape = (total_height, total_width, channels) if is_color else (total_height, total_width)
    else:
        total_height = overlap_starts[0] + sum(overlap_starts[i] for i in range(1, len(overlap_starts)))
        total_height = max(total_height, sum(overlap_starts))
        total_width = max(img.shape[1] for img in images)
        shape = (total_height, total_width, channels) if is_color else (total_height, total_width)

    # create blank canvas
    result = np.zeros(shape, dtype=images[0].dtype)

    current_pos = 0
    for i, start in enumerate(overlap_starts):
        img = images[i]
        h, w = img.shape[:2]

        if direction == "horizontal":
            # blending for horizontal stitching
            if i > 0 and blend_size > 0:
                blend_start = current_pos
                blend_end = min(current_pos + blend_size, result.shape[1], current_pos + w)
                alpha = (np.linspace(0, 1, blend_end - blend_start)[None, :, None]
                         if is_color else np.linspace(0, 1, blend_end - blend_start)[None, :])

                result[:h, blend_start:blend_end] = (
                    (1 - alpha) * result[:h, blend_start:blend_end] +
                    alpha * img[:, :blend_end - blend_start]
                ).astype(img.dtype)

                result[:h, blend_end:current_pos + w] = img[:, blend_end - blend_start:]
            else:
                result[:h, current_pos:current_pos + w] = img
        else:
            # blending for vertical stitching
            if i > 0 and blend_size > 0:
                blend_start = current_pos
                blend_end = min(current_pos + blend_size, result.shape[0], current_pos + h)
                alpha = (np.linspace(0, 1, blend_end - blend_start)[:, None, None]
                         if is_color else np.linspace(0, 1, blend_end - blend_start)[:, None])

                result[blend_start:blend_end, :w] = (
                    (1 - alpha) * result[blend_start:blend_end, :w] +
                    alpha * img[:blend_end - blend_start, :]
                ).astype(img.dtype)

                result[blend_end:current_pos + h, :w] = img[blend_end - blend_start:, :]
            else:
                result[current_pos:current_pos + h, :w] = img

        current_pos += start

    return result


def image_reader(image_dir, num_rows=0, base_dir="images"):
    """
    Read tile images from base_dir/image_dir.
    Returns 2D list: rows of images, based on Y index in filename.
    Filenames expected like: "<whatever> Y00000 X00000.bmp"
    """
    image_rows = []
    reading_row = 0
    current_row = []

    folder_path = os.path.join(base_dir, image_dir)
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".bmp"):
            y_val = int(fname.split()[1][1:])  # expects token like "Y00000"
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

"""
def stitch_image(path, out_path=None, base_dir="images", show=False):
    # 
    # Stitches tiles under base_dir/path into one image.
    # - If out_path is provided: writes stitched BMP.
    # - Returns the final stitched image (BGR numpy array).
    # 
    images = image_reader(path, base_dir=base_dir)
    rows = []
    for row in images:
        overlap_starts = find_overlaps(row, direction="horizontal")
        row_img = stitch_images(row, overlap_starts, direction="horizontal")
        rows.append(row_img)

    final_overlap_starts = find_overlaps(rows, direction="vertical")
    final_image = stitch_images(rows, final_overlap_starts, direction="vertical")

    if out_path is not None:
        cv2.imwrite(out_path, final_image)

    if show:
        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_image
"""

def stitch_image(path, save_path=None):
    images = image_reader(path)
    rows = []

    for row in images:
        overlap_starts = find_overlaps(row, direction="horizontal")
        rows.append(stitch_images(row, overlap_starts, direction="horizontal"))

    final_overlap_starts = find_overlaps(rows, direction="vertical")
    final_image = stitch_images(rows, final_overlap_starts, direction="vertical")

    if save_path:
        cv2.imwrite(save_path, final_image)

    return final_image

def stitch_folder_to_image(image_dir, num_rows=0):
    """
    Stitch a folder of tiles into a single BGR image (returned in memory).

    image_dir: path to folder containing tile BMPs
    num_rows: optional limit on rows (0 = all)
    """
    images = image_reader(image_dir, num_rows=num_rows)

    if not images or not images[0]:
        raise ValueError(f"No images found in folder: {image_dir}")

    rows = []
    for row in images:
        overlap_starts = find_overlaps(row, direction="horizontal")
        stitched_row = stitch_images(row, overlap_starts, direction="horizontal")
        rows.append(stitched_row)

    final_overlap_starts = find_overlaps(rows, direction="vertical")
    final_image = stitch_images(rows, final_overlap_starts, direction="vertical")

    return final_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder name containing tile bmps (under --base_dir)")
    ap.add_argument("--base_dir", default="images", help="Root containing folders of tiles")
    ap.add_argument("--out", default="stitch.bmp", help="Output stitched bmp")
    ap.add_argument("--show", action="store_true", help="Show stitched image window")
    args = ap.parse_args()

    stitch_image(args.folder, out_path=args.out, base_dir=args.base_dir, show=args.show)


if __name__ == "__main__":
    main()

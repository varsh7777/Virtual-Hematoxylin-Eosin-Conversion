from typing import Union
from pathlib import Path
import cv2
import numpy as np


def save_image_bgr(path: Union[str, Path], bgr_u8: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
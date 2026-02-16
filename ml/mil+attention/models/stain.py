# stain
# losses that encourage realistic H&E stain distribution
# (e.g., in stain space / histogram losses / differentiable color transform)

"""
Optional helpers for stain-ish transforms.
Not required for training/inference in this baseline,
but left here for future stain-space losses.
"""

import numpy as np


def rgb_to_od(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rgb = np.clip(rgb, eps, 1.0)
    return -np.log(rgb)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    return np.exp(-od)
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpatialFreqParams:
    # Breakpoints in normalized radius r in [0,1]
    # 0 = DC/lowest freq, 1 = Nyquist/highest freq
    # Example curve shaped like your arrows: down / up / up / down
    r_points: List[float] = None
    gains: List[float] = None

    # Smoothness of the mask edge blending (bigger = smoother)
    smooth_sigma: float = 3.0

    # Apply on brightness channel (HSB B == HSV V)
    apply_to: str = "brightness"  # keep for clarity

    def __post_init__(self):
        if self.r_points is None:
            # You should tune these to match your chart “zones”
            self.r_points = [0.0, 0.15, 0.35, 0.65, 0.85, 1.0]
        if self.gains is None:
            # down, up, up, down pattern (example)
            self.gains = [1.0, 0.85, 1.15, 1.10, 0.80, 0.75]


def _radial_grid(h: int, w: int) -> np.ndarray:
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    r /= (r.max() + 1e-9)  # normalize 0..1
    return r.astype(np.float32)


def _build_gain_mask(r: np.ndarray, r_points: List[float], gains: List[float], smooth_sigma: float) -> np.ndarray:
    # Piecewise-linear interpolation of gain vs normalized radius
    r_points = np.array(r_points, dtype=np.float32)
    gains = np.array(gains, dtype=np.float32)

    gain = np.interp(r, r_points, gains).astype(np.float32)

    # Smooth the mask so it doesn't create ringing
    if smooth_sigma and smooth_sigma > 0:
        gain = cv2.GaussianBlur(gain, ksize=(0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)

    return gain


def apply_spatial_frequency_shaping(gray_u8: np.ndarray, params: SpatialFreqParams) -> np.ndarray:
    """
    Apply frequency-domain gain shaping to a single-channel uint8 image (brightness).
    Returns uint8 brightness channel.
    """
    if gray_u8.ndim != 2:
        raise ValueError("apply_spatial_frequency_shaping expects a single-channel image")

    h, w = gray_u8.shape
    img_f = gray_u8.astype(np.float32)

    # FFT
    F = np.fft.fft2(img_f)
    F_shift = np.fft.fftshift(F)

    # Build radial gain mask
    r = _radial_grid(h, w)
    gain = _build_gain_mask(r, params.r_points, params.gains, params.smooth_sigma)

    # Apply mask
    F_filt = F_shift * gain

    # Inverse FFT
    out = np.fft.ifft2(np.fft.ifftshift(F_filt)).real

    # Normalize back to uint8 range safely
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def apply_to_bgr_brightness(bgr: np.ndarray, params: SpatialFreqParams) -> np.ndarray:
    """
    Convert BGR -> HSV, modify V (brightness), convert back.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v2 = apply_spatial_frequency_shaping(v, params)

    hsv2 = cv2.merge([h, s, v2])
    out_bgr = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return out_bgr

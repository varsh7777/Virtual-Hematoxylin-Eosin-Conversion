# hsb_adjust.py (SV first; hue clamp/pull/bias last; + optional global brightness lift)
#
# Behavior:
#  1) Classify pixels into white/background vs tissue.
#  2) Within tissue: nuclei vs cytoplasm using hue-arc where reliable; else dark->nuclei fallback.
#  3) Saturation & Brightness adjusted FIRST (soft clamp into POST ranges + optional pushes/gamma/floor).
#  4) Optional global brightness lift (v_offset_all) AFTER per-class V shaping.
#  5) Hue adjusted LAST (clamp to arc -> pull -> bias -> clamp), wrap-aware.
#  6) Background hue can be gently pulled toward a light-pink center.

from typing import Dict, Optional, Tuple
import numpy as np
import cv2

# Targets/goals for POST-processing ranges (0..255 hue space; OpenCV hue converted internally)
TARGETS = {
    "cytoplasm": {
        "h_lower": 210, "h_upper": 6,     # wrap-around
        "s_lower": 13,  "s_upper": 120,
        "b_lower": 83,  "b_upper": 166,
    },
    "nuclei": {
        "h_lower": 172, "h_upper": 255,   # no wrap
        "s_lower": 13,  "s_upper": 120,
        "b_lower": 38,  "b_upper": 160,
    },
    "white": {
        "h_lower": 90,  "h_upper": 152,
        "s_lower": 0,   "s_upper": 44,
        "b_lower": 172, "b_upper": 207,
    },
}

# --- REQUIRED: your exact constants (unchanged) ---
DEFAULTS = {
    # background classification (looser so background doesn't get treated as tissue)
    "white_s_hi": 55,
    "white_v_lo": 150,

    # hue reliability threshold (hue unstable at very low saturation)
    "min_sat_for_hue": 15,

    # if hue unreliable in tissue, darker -> nuclei
    "nuclei_dark_v_thresh": 115,

    # soft clamp strengths (only affects out-of-range)
    "gamma_lift": 0.75,
    "gamma_compress": 2.2,

    # saturation controls (keep small to avoid oversaturation)
    "sat_floor_tissue": 0,
    "sat_push_nuclei": 0.05,
    "sat_push_cyto": 0.12,

    # brightness shaping (fix dark nuclei)
    "v_gamma_nuclei": 0.55,     # <1 brightens midtones
    "v_gamma_cyto": 0.92,
    "v_floor_nuclei": 95,       # set 0 to disable
    "v_floor_cyto": 0,

    # hue pull toward class arc centers (keep small)
    "hue_pull_nuclei": 0.03,
    "hue_pull_cyto": 0.05,

    # hue bias shifts (0..255 hue space)
    # negative -> more pink (lower hue); Positive -> more purple (higher hue)
    # start around -8..-18; avoid below -25 (can drift toward salmon/orange)
    "hue_bias_nuclei": -10,
    "hue_bias_cyto": -14,

    # background hue: gently pull toward light pink to prevent salmon
    "white_hue_center": 140,     # tweak 135..155
    "white_hue_strength": 0.22,  # 0 disables

    # NEW: optional global brightness lift (+5..+30 recommended)
    "v_offset_all": 0,
}


# ---------- Hue conversions (OpenCV hue is 0..179; we work in 0..255) ----------

def _h179_to_h255(h179_u8: np.ndarray) -> np.ndarray:
    h = h179_u8.astype(np.uint16)
    return ((h * 255) + 89) // 179


def _h255_to_h179(h255: np.ndarray) -> np.ndarray:
    h = h255.astype(np.int32)
    h179 = (h * 179 + 127) // 255
    return np.clip(h179, 0, 179).astype(np.uint8)


def _hue_in_arc(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    h = h.astype(np.uint16)
    lo = int(lo) & 0xFF
    hi = int(hi) & 0xFF
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def _white_mask(s_u8: np.ndarray, v_u8: np.ndarray, s_hi: int, v_lo: int) -> np.ndarray:
    return (s_u8 <= int(s_hi)) & (v_u8 >= int(v_lo))


# ---------- Saturation / brightness shaping ----------

def _soft_clamp_u8(x_u8: np.ndarray, lo: int, hi: int, gamma_lift: float, gamma_compress: float) -> np.ndarray:
    """
    Soft clamp into [lo..hi] where:
      - values already inside remain unchanged
      - values below lo are lifted toward lo with a power curve
      - values above hi are compressed toward hi with a power curve
    """
    x = x_u8.astype(np.float32)
    lo_f = float(lo)
    hi_f = float(hi)
    y = x.copy()

    m_lo = x < lo_f
    if np.any(m_lo):
        r = np.clip(x[m_lo] / max(lo_f, 1.0), 0.0, 1.0)
        y_lift = lo_f * (r ** float(gamma_lift))
        y[m_lo] = np.maximum(y_lift, lo_f)

    m_hi = x > hi_f
    if np.any(m_hi):
        denom = max(255.0 - hi_f, 1.0)
        r = np.clip((x[m_hi] - hi_f) / denom, 0.0, 1.0)
        y_comp = hi_f + denom * (r ** float(gamma_compress))
        y[m_hi] = np.minimum(y_comp, hi_f)

    y = np.clip(y, lo_f, hi_f)
    return np.rint(y).astype(np.uint8)


def _gamma_in_range_u8(x_u8: np.ndarray, lo: int, hi: int, gamma: float) -> np.ndarray:
    """
    Apply gamma only within [lo..hi] (keeps endpoints fixed).
    gamma < 1 brightens midtones; gamma > 1 darkens midtones.
    """
    if gamma is None or abs(float(gamma) - 1.0) < 1e-6:
        return x_u8
    x = x_u8.astype(np.float32)
    lo_f = float(lo)
    hi_f = float(hi)
    denom = max(hi_f - lo_f, 1.0)
    r = np.clip((x - lo_f) / denom, 0.0, 1.0)
    y = lo_f + (r ** float(gamma)) * denom
    return np.clip(np.rint(y), lo_f, hi_f).astype(np.uint8)


def _push_toward_upper_u8(x_u8: np.ndarray, hi: int, strength: float) -> np.ndarray:
    """
    x <- x + k*(hi-x). Small k (0.05..0.15) nudges up without blowing out.
    """
    k = float(strength)
    if k <= 0.0:
        return x_u8
    x = x_u8.astype(np.float32)
    y = x + k * (float(hi) - x)
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def _add_v_offset_u8(x_u8: np.ndarray, offset: int) -> np.ndarray:
    off = int(offset)
    if off == 0:
        return x_u8
    x = x_u8.astype(np.int16) + off
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------- Hue clamp/pull/bias (wrap-aware) ----------

_HUE_CLAMP_LUT_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _nearest_boundary_hue_scalar(x: int, lo: int, hi: int) -> int:
    x = int(x) & 0xFF
    lo = int(lo) & 0xFF
    hi = int(hi) & 0xFF

    if lo <= hi:
        if lo <= x <= hi:
            return x
        return lo if x < lo else hi

    # wrap arc
    if x >= lo or x <= hi:
        return x

    dist_to_hi = x - hi
    dist_to_lo = lo - x
    return hi if dist_to_hi <= dist_to_lo else lo


def _get_hue_clamp_lut(lo: int, hi: int) -> np.ndarray:
    key = (int(lo) & 0xFF, int(hi) & 0xFF)
    lut = _HUE_CLAMP_LUT_CACHE.get(key)
    if lut is not None:
        return lut
    arr = np.empty((256,), dtype=np.uint8)
    for x in range(256):
        arr[x] = _nearest_boundary_hue_scalar(x, lo, hi)
    _HUE_CLAMP_LUT_CACHE[key] = arr
    return arr


def _arc_center(lo: int, hi: int) -> int:
    lo = int(lo) & 0xFF
    hi = int(hi) & 0xFF
    if lo <= hi:
        return int((lo + hi) // 2)
    arc_len = (256 - lo) + (hi + 1)
    mid = arc_len // 2
    c = lo + mid
    if c > 255:
        c -= 256
    return int(c)


def _pull_hue_toward_center(h_u8: np.ndarray, lo: int, hi: int, strength: float) -> np.ndarray:
    k = float(strength)
    if k <= 0.0:
        return h_u8
    center = _arc_center(lo, hi)
    h = h_u8.astype(np.int16)
    delta = ((center - h + 128) % 256) - 128
    h2 = (h + np.rint(k * delta).astype(np.int16)) % 256
    h2 = h2.astype(np.uint8)
    lut = _get_hue_clamp_lut(lo, hi)
    return lut[h2]


def _apply_hue_bias(h_u8: np.ndarray, bias: int) -> np.ndarray:
    b = int(bias)
    if b == 0:
        return h_u8
    h = h_u8.astype(np.int16)
    h2 = (h + b) % 256
    return h2.astype(np.uint8)


def _pull_hue_toward_value(h_u8: np.ndarray, target: int, strength: float) -> np.ndarray:
    k = float(strength)
    if k <= 0.0:
        return h_u8
    target = int(target) & 0xFF
    h = h_u8.astype(np.int16)
    delta = ((target - h + 128) % 256) - 128
    h2 = (h + np.rint(k * delta).astype(np.int16)) % 256
    return h2.astype(np.uint8)


# ---------- Main strip adjust ----------

def _adjust_strip_hsb(
    bgr_strip: np.ndarray,
    do_hue: bool,
    do_sat: bool,
    do_bri: bool,
    cfg: Dict,
) -> Tuple[np.ndarray, Dict[str, int]]:
    hsv = cv2.cvtColor(bgr_strip, cv2.COLOR_BGR2HSV)
    h179 = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h255 = _h179_to_h255(h179).astype(np.uint8)

    # --- classification (BEFORE modifications) ---
    m_white = _white_mask(s, v, s_hi=cfg["white_s_hi"], v_lo=cfg["white_v_lo"])
    m_tissue = ~m_white

    # Hue-based tissue class when hue is reliable
    m_hue_ok = (s >= int(cfg["min_sat_for_hue"])) & m_tissue

    m_nuclei_hue = _hue_in_arc(h255, TARGETS["nuclei"]["h_lower"], TARGETS["nuclei"]["h_upper"]) & m_hue_ok
    m_cyto_hue = _hue_in_arc(h255, TARGETS["cytoplasm"]["h_lower"], TARGETS["cytoplasm"]["h_upper"]) & m_hue_ok

    # Uncertain tissue gets dark->nuclei fallback
    m_uncertain = m_tissue & (~(m_nuclei_hue | m_cyto_hue))
    m_dark = m_uncertain & (v < int(cfg["nuclei_dark_v_thresh"]))

    m_nuclei = m_nuclei_hue | m_dark
    m_cyto = m_tissue & (~m_nuclei)

    # --- PASS 1: SATURATION & BRIGHTNESS FIRST ---
    if do_sat:
        # background saturation clamp
        if np.any(m_white):
            t = TARGETS["white"]
            s[m_white] = _soft_clamp_u8(s[m_white], t["s_lower"], t["s_upper"], cfg["gamma_lift"], cfg["gamma_compress"])

        # tissue saturation clamp
        if np.any(m_nuclei):
            t = TARGETS["nuclei"]
            s[m_nuclei] = _soft_clamp_u8(s[m_nuclei], t["s_lower"], t["s_upper"], cfg["gamma_lift"], cfg["gamma_compress"])
        if np.any(m_cyto):
            t = TARGETS["cytoplasm"]
            s[m_cyto] = _soft_clamp_u8(s[m_cyto], t["s_lower"], t["s_upper"], cfg["gamma_lift"], cfg["gamma_compress"])

        # optional floor
        sat_floor = int(cfg.get("sat_floor_tissue", 0))
        if sat_floor > 0 and np.any(m_tissue):
            s[m_tissue] = np.maximum(s[m_tissue], sat_floor).astype(np.uint8)

        # gentle pushes toward upper (small)
        if np.any(m_nuclei):
            t = TARGETS["nuclei"]
            s[m_nuclei] = _push_toward_upper_u8(s[m_nuclei], t["s_upper"], float(cfg.get("sat_push_nuclei", 0.0)))
            s[m_nuclei] = np.clip(s[m_nuclei], t["s_lower"], t["s_upper"]).astype(np.uint8)
        if np.any(m_cyto):
            t = TARGETS["cytoplasm"]
            s[m_cyto] = _push_toward_upper_u8(s[m_cyto], t["s_upper"], float(cfg.get("sat_push_cyto", 0.0)))
            s[m_cyto] = np.clip(s[m_cyto], t["s_lower"], t["s_upper"]).astype(np.uint8)

    if do_bri:
        # background brightness clamp
        if np.any(m_white):
            t = TARGETS["white"]
            v[m_white] = _soft_clamp_u8(v[m_white], t["b_lower"], t["b_upper"], cfg["gamma_lift"], cfg["gamma_compress"])

        # tissue brightness clamp + gammas/floors
        if np.any(m_nuclei):
            t = TARGETS["nuclei"]
            v[m_nuclei] = _soft_clamp_u8(v[m_nuclei], t["b_lower"], t["b_upper"], cfg["gamma_lift"], cfg["gamma_compress"])
            v[m_nuclei] = _gamma_in_range_u8(v[m_nuclei], t["b_lower"], t["b_upper"], float(cfg.get("v_gamma_nuclei", 1.0)))

            v_floor = int(cfg.get("v_floor_nuclei", 0))
            if v_floor > 0:
                v_floor = max(t["b_lower"], min(t["b_upper"], v_floor))
                v[m_nuclei] = np.maximum(v[m_nuclei], v_floor).astype(np.uint8)

        if np.any(m_cyto):
            t = TARGETS["cytoplasm"]
            v[m_cyto] = _soft_clamp_u8(v[m_cyto], t["b_lower"], t["b_upper"], cfg["gamma_lift"], cfg["gamma_compress"])
            v[m_cyto] = _gamma_in_range_u8(v[m_cyto], t["b_lower"], t["b_upper"], float(cfg.get("v_gamma_cyto", 1.0)))

            v_floor = int(cfg.get("v_floor_cyto", 0))
            if v_floor > 0:
                v_floor = max(t["b_lower"], min(t["b_upper"], v_floor))
                v[m_cyto] = np.maximum(v[m_cyto], v_floor).astype(np.uint8)

        # NEW: global brightness lift
        v_offset = int(cfg.get("v_offset_all", 0))
        if v_offset != 0:
            v[:] = _add_v_offset_u8(v, v_offset)

    # --- PASS 2: HUE LAST (clamp -> pull -> bias -> clamp) ---
    if do_hue:
        # background: pull toward light pink to avoid salmon/orange
        if np.any(m_white):
            k = float(cfg.get("white_hue_strength", 0.0))
            if k > 0.0:
                h255[m_white] = _pull_hue_toward_value(
                    h255[m_white],
                    target=int(cfg.get("white_hue_center", 140)),
                    strength=k,
                )

        # nuclei hue corrections
        if np.any(m_nuclei):
            t = TARGETS["nuclei"]
            lut = _get_hue_clamp_lut(t["h_lower"], t["h_upper"])
            h255[m_nuclei] = lut[h255[m_nuclei]]
            h255[m_nuclei] = _pull_hue_toward_center(
                h255[m_nuclei], t["h_lower"], t["h_upper"],
                strength=float(cfg.get("hue_pull_nuclei", 0.0))
            )
            h255[m_nuclei] = _apply_hue_bias(h255[m_nuclei], int(cfg.get("hue_bias_nuclei", 0)))
            h255[m_nuclei] = lut[h255[m_nuclei]]

        # cytoplasm hue corrections
        if np.any(m_cyto):
            t = TARGETS["cytoplasm"]
            lut = _get_hue_clamp_lut(t["h_lower"], t["h_upper"])
            h255[m_cyto] = lut[h255[m_cyto]]
            h255[m_cyto] = _pull_hue_toward_center(
                h255[m_cyto], t["h_lower"], t["h_upper"],
                strength=float(cfg.get("hue_pull_cyto", 0.0))
            )
            h255[m_cyto] = _apply_hue_bias(h255[m_cyto], int(cfg.get("hue_bias_cyto", 0)))
            h255[m_cyto] = lut[h255[m_cyto]]

    # write back / convert
    hsv[:, :, 0] = _h255_to_h179(h255)
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    out_strip = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    counts = {
        "white": int(m_white.sum()),
        "nuclei": int(m_nuclei.sum()),
        "cytoplasm": int(m_cyto.sum()),
        "unclassified": 0,
    }
    return out_strip, counts


# ---------- Public API ----------

def adjust_bgr_image_hsb(
    bgr: np.ndarray,
    params: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    if metadata is None:
        metadata = {}

    do_hue = True
    do_sat = True
    do_bri = True
    chunk_width = 1024
    cfg = dict(DEFAULTS)

    if params:
        do_hue = bool(params.get("do_hue", do_hue))
        do_sat = bool(params.get("do_sat", do_sat))
        do_bri = bool(params.get("do_bri", do_bri))
        chunk_width = int(params.get("chunk_width", chunk_width))

        # Allow overriding any DEFAULTS key via params
        for k in list(cfg.keys()):
            if k in params:
                cfg[k] = params[k]

    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR uint8 image, got shape={None if bgr is None else bgr.shape}")
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    _, w = bgr.shape[:2]
    total_counts = {"white": 0, "nuclei": 0, "cytoplasm": 0, "unclassified": 0}

    for x0 in range(0, w, chunk_width):
        x1 = min(w, x0 + chunk_width)
        strip = bgr[:, x0:x1, :]
        out_strip, counts = _adjust_strip_hsb(strip, do_hue, do_sat, do_bri, cfg)
        bgr[:, x0:x1, :] = out_strip
        for k in total_counts:
            total_counts[k] += counts.get(k, 0)

    meta_out = dict(metadata)
    meta_out.update({
        "chunk_width": chunk_width,
        "counts": total_counts,
        "targets": TARGETS,
        "do_hue": do_hue,
        "do_sat": do_sat,
        "do_bri": do_bri,
        "config": cfg,
        "version": "SV first; hue clamp/pull/bias last; + v_offset_all global lift",
    })
    return bgr, meta_out
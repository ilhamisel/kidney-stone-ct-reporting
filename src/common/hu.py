"""HU <-> 16-bit PNG encoding and windowing.

Archive contract:  png_uint16 = HU + HU_OFFSET

HU_OFFSET is 4096 rather than the conventional 1024 because the scanner writes
HU values down to -3024 in the out-of-field padding (measured by 00_env_check);
an offset of 1024 would overflow uint16. The encoding is lossless and exactly
invertible.
"""
from __future__ import annotations

import numpy as np

HU_OFFSET = 4096
HU_MIN = -HU_OFFSET
HU_MAX = 65535 - HU_OFFSET

# 8-bit derivative windows, as (window level, window width)
WINDOWS: dict[str, tuple[int, int]] = {
    "soft": (40, 400),     # -160..240   parenchyma / collecting system
    "stone": (400, 1500),  # -350..1150  PRIMARY: stone density stays on a monotone ramp
    "wide": (500, 3000),   # -1000..2000 global context including air and bone
}
MW3_ORDER = ("soft", "stone", "wide")


def hu_to_png16(hu: np.ndarray) -> np.ndarray:
    if hu.min() < HU_MIN or hu.max() > HU_MAX:
        raise ValueError(
            f"HU range not encodable: [{hu.min()}, {hu.max()}], limit [{HU_MIN}, {HU_MAX}]")
    return (hu.astype(np.int32) + HU_OFFSET).astype(np.uint16)


def png16_to_hu(png: np.ndarray) -> np.ndarray:
    return png.astype(np.int32) - HU_OFFSET


def window8(hu: np.ndarray, wl: int, ww: int) -> np.ndarray:
    """Linear windowing. No gamma, no resampling, no padding."""
    lo = wl - ww / 2.0
    v = np.clip((hu.astype(np.float32) - lo) / float(ww), 0.0, 1.0)
    return np.round(v * 255.0).astype(np.uint8)


def multiwindow3(hu: np.ndarray) -> np.ndarray:
    """R=soft, G=stone, B=wide  -> (H, W, 3) uint8."""
    return np.stack([window8(hu, *WINDOWS[n]) for n in MW3_ORDER], axis=-1)

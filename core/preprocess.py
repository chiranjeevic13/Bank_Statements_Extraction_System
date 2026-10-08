"""
preprocess.py — scan clean-up applied BEFORE OCR.

  * despeckle  : removes salt-and-pepper noise that scanners/photocopies add
  * deskew     : straightens a slightly tilted page so table rows are horizontal

Row grouping in parser.py relies on every word of a table row sharing the same
y-position.  A tilt of only ~0.5° moves the right-hand columns several points
away from the left-hand ones and rows start to merge / split.
"""
from __future__ import annotations
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_SKEW_DEG = 5.0      # ignore anything larger (probably not a plain tilt)
MIN_SKEW_DEG = 0.10     # below this, rotating only adds blur


def despeckle(gray: np.ndarray) -> np.ndarray:
    """Median filter: kills isolated dots, keeps text strokes."""
    return cv2.medianBlur(gray, 3)


def _row_sharpness(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST)
    prof = rot.sum(axis=1).astype(np.float64)
    return float(np.sum(np.diff(prof) ** 2))   # sharp row edges => text lines level


def estimate_skew(gray: np.ndarray) -> float:
    """Projection-profile skew estimate in degrees (coarse -> fine search)."""
    scale = 1000.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    small = cv2.medianBlur(small, 3)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # drop remaining specks so they cannot vote
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if binary.sum() == 0:
        return 0.0

    coarse = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 0.01, 0.25)
    best = max(coarse, key=lambda a: _row_sharpness(binary, a))
    fine = np.arange(best - 0.25, best + 0.2501, 0.05)
    best = max(fine, key=lambda a: _row_sharpness(binary, a))
    base = _row_sharpness(binary, 0.0)
    gain = _row_sharpness(binary, best) / base if base else 1.0
    return float(best) if gain > 1.02 else 0.0   # require a real improvement


def deskew(gray: np.ndarray) -> np.ndarray:
    angle = estimate_skew(gray)
    if abs(angle) < MIN_SKEW_DEG or abs(angle) > MAX_SKEW_DEG:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    log.info("Deskew: rotated page by %.2f°", angle)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def invert_dark_bands(gray: np.ndarray) -> np.ndarray:
    """
    Table headers are often white text on a dark bar, which OCR engines read poorly.
    Find wide, short, dark bands and flip them to black-on-white.
    """
    h, w = gray.shape
    mask = (gray < 110).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 7)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = gray.copy()
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 0.35 * w or not (0.008 * h < bh < 0.08 * h):
            continue
        roi = gray[y:y + bh, x:x + bw]
        if roi.mean() < 120:                      # genuinely dark bar, not a bold paragraph
            flipped = 255 - roi
            # crisp black-on-white text instead of grey-on-grey
            _, flipped = cv2.threshold(flipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out[y:y + bh, x:x + bw] = flipped
            log.info("Inverted dark band at y=%d (h=%d)", y, bh)
    return out


def prepare_scan(gray: np.ndarray) -> np.ndarray:
    """Grayscale page image -> despeckled, straightened, header-bar-corrected image."""
    return invert_dark_bands(deskew(despeckle(gray)))

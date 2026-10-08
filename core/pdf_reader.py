"""
PDF → structured Page objects (text-based or OCR)

Handles:
  • Text-based PDFs  (PyMuPDF word extraction)
  • Image/scanned PDFs (adaptive-threshold → Tesseract OCR)
  • Password-protected PDFs
  • Blank / corrupted pages (graceful skip)
"""

from __future__ import annotations
import logging
import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
from dataclasses import dataclass
import cv2
import fitz           # PyMuPDF
import numpy as np

log = logging.getLogger(__name__)


#  OCR Engines Initialization 
try:
    from paddleocr import PaddleOCR
    # Initialize globally to avoid reloading the model for every page
    OCR_ENGINE = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_det_limit_type="max",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        cpu_threads=4,
    )
    log.info("PaddleOCR loaded successfully.")
except Exception as exc:
    log.warning("PaddleOCR unavailable (%s). Falling back to Tesseract.", exc)

import pytesseract
_TESS_PATH = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(_TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH


MIN_WORDS_FOR_TEXT = 20

#  Data classes 

@dataclass
class Word:
    text:   str
    x0:     float
    x1:     float
    top:    float
    bottom: float

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class Page:
    number: int
    kind:   str
    words:  list[Word]


class PasswordRequired(Exception):
    pass


#  Public API 

def read_pdf(data: bytes, password: str | None = None, dpi: int = 300) -> list[Page]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    if doc.needs_pass:
        if not doc.authenticate(password or ""):
            raise PasswordRequired("Encrypted PDF — password missing or incorrect.")

    pages: list[Page] = []
    for i, fitz_page in enumerate(doc):
        page_num = i + 1
        try:
            words = _extract_text_words(fitz_page)
            if len(words) >= MIN_WORDS_FOR_TEXT:
                pages.append(Page(page_num, "text", words))
                log.debug("Page %d → text (%d words)", page_num, len(words))
            else:
                ocr_words = _ocr_page(fitz_page, dpi)
                if ocr_words:
                    pages.append(Page(page_num, "ocr", ocr_words))
                    log.debug("Page %d → OCR (%d words)", page_num, len(ocr_words))
                else:
                    pages.append(Page(page_num, "blank", []))
                    log.warning("Page %d → blank / unreadable", page_num)
        except Exception as exc:
            log.warning("Page %d: error during extraction — %s", page_num, exc)
            pages.append(Page(page_num, "blank", []))

    doc.close()
    return pages


#  Internal helpers 

def _extract_text_words(page: fitz.Page) -> list[Word]:
    raw = page.get_text("words")
    return [Word(w[4], w[0], w[2], w[1], w[3]) for w in raw if w[4].strip()]


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(img, 3)
    binary  = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    kernel = np.ones((1, 1), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def _ocr_page(page: fitz.Page, dpi: int) -> list[Word]:
    """Render PDF page to image and route through OCR with hard fallbacks."""
    # RESOURCE LIMIT: Cap DPI at 300 to prevent Out-Of-Memory (OOM) crashes
    global OCR_ENGINE
    safe_dpi = min(dpi, 300)
    scale  = safe_dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix    = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
    img    = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    
    words: list[Word] = []

    if OCR_ENGINE is not None:
        try:
            paddle_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            words = _paddle_ocr(paddle_img, scale)
        except Exception as e:
            log.warning("PaddleOCR inference failed: %s. Disabling it for this run.", e)
            OCR_ENGINE = None

    if not words:
        try:
            img_prep = _preprocess_image(img)
            words = _tesseract_ocr(img_prep, scale)
        except Exception as e:
            log.error("Tesseract fallback failed: %s", e)

    return words


def _paddle_ocr(img: np.ndarray, scale: float) -> list[Word]:
    """Convert PaddleOCR 3.x result objects into geometry-aware words."""
    results = OCR_ENGINE.predict(img, return_word_box=True)
    if not results:
        return []

    result = results[0]
    if not hasattr(result, "get"):
        raise TypeError("Unexpected PaddleOCR result; expected a mapping-like result object.")

    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    word_lines = result.get("text_word", [])
    word_regions = result.get("text_word_boxes", [])
    out: list[Word] = []

    for line_index, text in enumerate(texts):
        if not str(text).strip():
            continue

        if line_index < len(scores) and float(scores[line_index]) < 0.60:
            continue

        line_words = word_lines[line_index] if line_index < len(word_lines) else []
        line_regions = word_regions[line_index] if line_index < len(word_regions) else []
        if len(line_words) == len(line_regions) and line_words:
            for token, region in zip(line_words, line_regions):
                bounds = _box_bounds(region)
                if bounds is not None and str(token).strip():
                    x0, top, x1, bottom = bounds
                    out.append(Word(str(token).strip(), x0 / scale, x1 / scale, top / scale, bottom / scale))
            continue

        # PaddleOCR can omit word boxes for unsupported text or recognition modes.
        line_boxes = result.get("rec_boxes", [])
        if line_index >= len(line_boxes):
            line_boxes = result.get("rec_polys", result.get("dt_polys", []))
        if line_index >= len(line_boxes):
            continue

        bounds = _box_bounds(line_boxes[line_index])
        tokens = str(text).split()
        if bounds is None or not tokens:
            continue
        x0, top, x1, bottom = bounds
        token_width = (x1 - x0) / len(tokens)
        for token_index, token in enumerate(tokens):
            token_x0 = x0 + token_index * token_width
            out.append(Word(token, token_x0 / scale, (token_x0 + token_width) / scale, top / scale, bottom / scale))

    return out


def _box_bounds(box) -> tuple[float, float, float, float] | None:
    """Return (x0, top, x1, bottom) from a Paddle box or polygon."""
    points = np.asarray(box, dtype=float)
    if points.size == 4:
        x0, top, x1, bottom = points.reshape(-1).tolist()
    elif points.ndim == 2 and points.shape[1] == 2 and points.shape[0] >= 2:
        x0 = float(points[:, 0].min())
        x1 = float(points[:, 0].max())
        top = float(points[:, 1].min())
        bottom = float(points[:, 1].max())
    else:
        return None
    return float(x0), float(top), float(x1), float(bottom)


def _tesseract_ocr(img: np.ndarray, scale: float) -> list[Word]:
    """Adapter for Tesseract. Provides native word-level bounding boxes."""
    try:
        d = pytesseract.image_to_data(
            img, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        log.error("Tesseract binary not found.")
        return []

    out = []
    for text, x, y, w, h, conf in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
        if text.strip() and float(conf) > 25.0:
            out.append(Word(
                text.strip(),
                x / scale,
                (x + w) / scale,
                y / scale,
                (y + h) / scale,
            ))
    return out


#  Line grouping utilities 

def group_lines(words: list[Word]) -> tuple[list[tuple[float, list[Word]]], float]:
    if not words:
        return [], 10.0

    heights = [w.bottom - w.top for w in words]
    med_h   = float(np.median(heights))
    tol     = max(2.0, 0.6 * med_h) 

    buckets: list[dict] = []
    for w in sorted(words, key=lambda w: w.cy):
        if buckets and abs(w.cy - buckets[-1]["cy"]) <= tol:
            bucket = buckets[-1]
            bucket["words"].append(w)
            bucket["cy"] = float(np.mean([x.cy for x in bucket["words"]]))
        else:
            buckets.append({"cy": w.cy, "words": [w]})

    lines = [(b["cy"], sorted(b["words"], key=lambda w: w.x0)) for b in buckets]
    return lines, med_h


def page_text(page: Page) -> str:
    lines, _ = group_lines(page.words)
    return "\n".join(" ".join(w.text for w in ws) for _, ws in lines)
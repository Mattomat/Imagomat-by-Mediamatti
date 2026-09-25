"""Rückennummern und Trikotnummern erkennen (OCR).

Backends: Apple Vision (macOS, über ocrmac, schnell und gut), EasyOCR (plattformübergreifend).
Ohne Backend liefert die Erkennung eine leere Liste.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import load_settings
from .models import has_module

log = logging.getLogger(__name__)
_NUM = re.compile(r"^\d{1,2}$")


@dataclass
class NumberHit:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]   # normiert x0 y0 x1 y1


class _VisionOCR:
    def __call__(self, img: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        from ocrmac import ocrmac
        from PIL import Image

        res = ocrmac.OCR(Image.fromarray(img), recognition_level="accurate",
                         language_preference=["en-US"]).recognize()
        out = []
        for text, conf, (x, y, w, h) in res:
            # Vision: Ursprung unten links
            out.append((text, float(conf), (x, 1 - y - h, x + w, 1 - y)))
        return out


class _EasyOCR:
    def __init__(self):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def __call__(self, img: np.ndarray):
        h, w = img.shape[:2]
        out = []
        for box, text, conf in self.reader.readtext(img, allowlist="0123456789"):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append((text, float(conf), (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)))
        return out


@lru_cache(maxsize=1)
def _backend():
    pref = load_settings().ocr_backend
    if pref == "none":
        return None
    if pref in ("auto", "vision") and sys.platform == "darwin" and has_module("ocrmac"):
        return _VisionOCR()
    if pref in ("auto", "easyocr") and has_module("easyocr"):
        try:
            return _EasyOCR()
        except Exception as e:  # noqa: BLE001
            log.warning("EasyOCR nicht verfügbar: %s", e)
    return None


def available() -> bool:
    return _backend() is not None


def find_numbers(img: np.ndarray, min_height: float = 0.012, min_conf: float = 0.45) -> list[NumberHit]:
    ocr = _backend()
    if ocr is None:
        return []
    hits = []
    try:
        results = ocr(img)
    except Exception as e:  # noqa: BLE001 - OCR darf die Analyse nie abbrechen
        log.warning("OCR fehlgeschlagen: %s", e)
        return []
    for text, conf, box in results:
        t = text.strip().replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        if not _NUM.match(t) or conf < min_conf or (box[3] - box[1]) < min_height:
            continue
        hits.append(NumberHit(t.lstrip("0") or "0", conf, box))
    return hits


def assign_to_bodies(hits: list[NumberHit], body_boxes: list[tuple[float, ...]]) -> dict[int, NumberHit]:
    """Ordnet Nummern der Körperbox zu, in der sie liegen (Index der Box -> Nummer)."""
    out: dict[int, NumberHit] = {}
    for h in sorted(hits, key=lambda x: -x.confidence):
        cx, cy = (h.bbox[0] + h.bbox[2]) / 2, (h.bbox[1] + h.bbox[3]) / 2
        for i, (x0, y0, x1, y1) in enumerate(body_boxes):
            if i not in out and x0 <= cx <= x1 and y0 <= cy <= y1:
                out[i] = h
                break
    return out

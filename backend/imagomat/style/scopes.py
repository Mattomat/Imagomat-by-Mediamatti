"""Weiss- und Schwarzpunkt wie am Waveform (Lumetri-Scopes) setzen.

Ziel: oben und unten schlägt jeweils ein kleiner Teil des Bildes leicht an (fast reines Weiss bzw.
Schwarz), der Rest bleibt dazwischen. Gemessen wird an einer kleinen Vorschau-Entwicklung mit den
vorhergesagten Einstellungen (inkl. Masken); dann werden nur Weiss (Whites2012) und Schwarz (Blacks2012)
per Bisektion nachgeführt. Die Messung nutzt die linearisierte Kamera-Vorschau und funktioniert so für
jede Kamera; sie ist eine Näherung an Lightroom.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..analysis import PREVIEW_TO_RAW_EV
from ..lightroom.params import to_number
from ..io.color import XYZ_TO_SRGB
from ..render.pipeline import render

SIDE = 192
HI_Q, HI_TARGET = 0.997, 0.975      # 0.3 % der Pixel ab ~97.5 % Helligkeit: "schlägt leicht an"
LO_Q, LO_TARGET = 0.003, 0.035      # 0.3 % der Pixel unter ~3.5 % (Schwarz satt, aber nicht zugelaufen)
LO_ONLY_ABOVE = 0.07                # Schwarz nur nachführen, wenn das Bild unten wirklich flau ist
MAX_DEEPEN = 25.0
WHITES_RANGE = (-40.0, 70.0)
BLACKS_RANGE = (-60.0, 25.0)
MAX_CHANGE = 50.0                   # nie mehr als so viel von deinem (gelernten) Wert weg


def preview_linear(path: str, side: int = SIDE) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return (lin * 2.0 ** -PREVIEW_TO_RAW_EV).astype(np.float32)   # auf RAW-Skala wie die Analyse


def _luma(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _measure(lin: np.ndarray, crs: dict[str, Any], orientation: int, seg: dict[str, np.ndarray]) -> tuple[float, float]:
    # Vorschau ist schon weissabgeglichen und in Anzeige-Orientierung: WB "wie aufgenommen", sRGB-Matrix
    c = {**crs, "WhiteBalance": "As Shot"}
    y = _luma(render(lin, XYZ_TO_SRGB, np.ones(3), c, orientation, seg, None))
    return float(np.quantile(y, HI_Q)), float(np.quantile(y, LO_Q))


def _bisect(fn, lo: float, hi: float, target: float, increasing: bool = True, steps: int = 5) -> float:
    for _ in range(steps):
        mid = (lo + hi) / 2
        v = fn(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_scopes(crs: dict[str, Any], lin: np.ndarray, orientation: int = 1,
               subject: np.ndarray | None = None, amount: float = 1.0, hi_target: float = HI_TARGET,
               lo_target: float = LO_TARGET, free_blacks: bool = False) -> list[str]:
    """Setzt Whites2012/Blacks2012 in ``crs`` so, dass oben und unten leicht angeschlagen wird.
    ``free_blacks``: Schwarz auch anheben (wenn ein Referenz-Look weichere Tiefen hat)."""
    if amount <= 0:
        return []
    seg = {"subject": subject} if subject is not None else {}
    w0 = float(to_number(crs.get("Whites2012")) or 0.0)
    b0 = float(to_number(crs.get("Blacks2012")) or 0.0)

    def hi_at(w: float) -> float:
        return _measure(lin, {**crs, "Whites2012": w}, orientation, seg)[0]

    w = _bisect(hi_at, max(WHITES_RANGE[0], w0 - MAX_CHANGE), min(WHITES_RANGE[1], w0 + MAX_CHANGE),
                hi_target, increasing=True)
    w = w0 + (w - w0) * min(amount, 1.0)
    crs["Whites2012"] = int(round(w))

    def lo_at(b: float) -> float:
        return _measure(lin, {**crs, "Blacks2012": b}, orientation, seg)[1]

    # Schwarz nur vertiefen, nie anheben (Nachthimmel bleibt schwarz), und nur bei flauen Bildern, sanft
    if free_blacks:
        b = _bisect(lo_at, max(BLACKS_RANGE[0], b0 - MAX_DEEPEN), min(BLACKS_RANGE[1], b0 + MAX_DEEPEN),
                    lo_target, increasing=True)
    elif lo_at(b0) > LO_ONLY_ABOVE:
        b = _bisect(lo_at, max(BLACKS_RANGE[0], b0 - MAX_DEEPEN), b0, lo_target, increasing=True)
    else:
        b = b0
    b = b0 + (b - b0) * min(amount, 1.0)
    crs["Blacks2012"] = int(round(b))
    notes = []
    if abs(w - w0) >= 2:
        notes.append(f"Weiss {w0:+.0f}→{w:+.0f}")
    if abs(b - b0) >= 2:
        notes.append(f"Schwarz {b0:+.0f}→{b:+.0f}")
    return notes


def scopes(img_u8: np.ndarray) -> dict[str, float]:
    """Kennzahlen wie am Waveform: Anteil angeschlagener Pixel oben/unten (für Tests und Anzeige)."""
    y = _luma(img_u8)
    return {"white_clip": float((y >= 0.985).mean()), "black_clip": float((y <= 0.015).mean()),
            "p_hi": float(np.quantile(y, HI_Q)), "p_lo": float(np.quantile(y, LO_Q))}

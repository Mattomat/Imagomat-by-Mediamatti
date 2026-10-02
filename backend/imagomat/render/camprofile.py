"""Profil "Kamera (wie Original)": bei Reglern auf 0 sieht die RAW-Entwicklung aus wie das Kamera-JPEG.

Zwei Teile, pro Bild aus RAW und eingebettetem Kamera-JPEG bestimmt:
- Kurven pro Kanal (Tonwerte, Farbe der Kamera),
- eine grobe Karte (Helligkeit/Farbe pro Bildbereich) für das, was die Kamera lokal macht (z. B. Sony DRO).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import cache_dir
from .pipeline import camera_curve

RATIO_SIDE = 72
VERSION = 2


def profile_path(key: str) -> Path:
    d = cache_dir() / "camprof"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}_v{VERSION}.npz"


def compute(raw_disp: np.ndarray, cam: np.ndarray) -> tuple[list[list[float]], np.ndarray]:
    """raw_disp: neutrale RAW-Entwicklung (Adobe-Basis), cam: Kamera-JPEG gleicher Grösse, beide 0..1."""
    cc = camera_curve(raw_disp, cam)
    raw_cc = np.stack([np.interp(raw_disp[..., c], np.linspace(0, 1, len(cc[c])), cc[c]) for c in range(3)], -1)
    sig = max(2.0, max(raw_disp.shape[:2]) / 80)
    ratio = (cv2.GaussianBlur(cam.astype(np.float32), (0, 0), sig) + 0.02) / \
        (cv2.GaussianBlur(raw_cc.astype(np.float32), (0, 0), sig) + 0.02)
    h, w = ratio.shape[:2]
    s = RATIO_SIDE / max(h, w)
    ratio = cv2.resize(np.clip(ratio, 0.6, 1.6), (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
    return cc, ratio.astype(np.float32)


def save(key: str, cc: list[list[float]], ratio: np.ndarray) -> None:
    np.savez(profile_path(key), cc=np.asarray(cc, np.float32), ratio=ratio.astype(np.float32))


def load(key: str) -> tuple[list[list[float]], np.ndarray] | None:
    p = profile_path(key)
    if not p.exists():
        return None
    z = np.load(p)
    return z["cc"].tolist(), z["ratio"]

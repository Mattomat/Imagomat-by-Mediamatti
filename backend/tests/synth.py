"""Synthetische Test-Shoots als Bayer-DNG (echte RAW-Pipeline über LibRaw)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from imagomat.io.dng import mosaic_rggb, write_dng


def scene(seed: int, w: int = 900, h: int = 600) -> np.ndarray:
    """Lineares RGB 0..1: Rasen, Himmel, Streifen/Text als Details, ein "Spieler"."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    img[: h // 3] = [0.25, 0.35, 0.6]                                  # Himmel
    img[h // 3:] = [0.08, 0.22, 0.06]                                  # Rasen
    grass = rng.normal(0, 0.02, (h - h // 3, w, 1)).astype(np.float32)
    img[h // 3:] += grass
    for i in range(0, w, 60):                                          # Linien, Details
        cv2.line(img, (i, h // 3), (i + 30, h), (0.3, 0.3, 0.3), 2)
    x = 200 + (seed % 5) * 60
    cv2.rectangle(img, (x, 180), (x + 110, 520), (0.5, 0.05, 0.05), -1)  # Trikot
    cv2.circle(img, (x + 55, 140), 42, (0.45, 0.3, 0.22), -1)          # Kopf
    txt = np.zeros((h, w), np.uint8)
    cv2.putText(txt, str(7 + seed % 3), (x + 20, 330), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 6)
    img[txt > 0] = 0.9
    return np.clip(img, 0, 1)


def write_shoot(folder: Path, n: int = 12, start: dt.datetime | None = None) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    start = start or dt.datetime(2026, 5, 1, 19, 30, 0)
    paths = []
    wb = np.array([0.5, 1.0, 0.65])
    for i in range(n):
        series = i // 4                                                # 3 Serien à 4 Bilder
        img = scene(series)
        kind = i % 4
        if kind == 1:
            img = cv2.GaussianBlur(img, (0, 0), 3.0)                   # unscharf
        elif kind == 2:
            k = np.zeros((1, 25), np.float32)
            k[0, :] = 1 / 25
            img = cv2.filter2D(img, -1, k)                             # Bewegungsunschärfe
        elif kind == 3 and series == 2:
            img = np.clip(img * 6, 0, 1)                               # überbelichtet
        cam = img * wb
        rng = np.random.default_rng(i)
        raw = cam * 50000 + 800 + rng.normal(0, 250, cam.shape)
        raw = np.clip(raw, 0, 65535).astype(np.uint16)
        t = start + dt.timedelta(seconds=series * 30 + kind * 0.2)
        preview = (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        p = folder / f"DSC{i:05d}.dng"
        write_dng(p, mosaic_rggb(raw), cfa=True, black=800, white=60000, as_shot_neutral=tuple(wb),
                  iso=6400, exposure_time=1 / 1000, fnumber=2.8, focal_length=200, capture_time=t,
                  preview=cv2.resize(preview, (450, 300)))
        paths.append(p)
    return paths

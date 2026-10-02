"""Klassische Bildqualitäts-Metriken (ohne KI-Modelle, schnell auf der CPU).

Alle Funktionen erwarten RGB uint8 (Vorschau) bzw. float-Graustufen.
"""

from __future__ import annotations

import cv2
import numpy as np


def gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32) / (255.0 if g.dtype == np.uint8 else 1.0)


def sharpness(g: np.ndarray, reference_side: int = 1024) -> float:
    """Normierte Schärfe: Varianz des Laplace-Operators nach Skalierung auf eine Referenzgrösse.

    Kleine Ausschnitte (Gesichter) werden hochskaliert, damit Werte vergleichbar bleiben.
    """
    if g.size == 0:
        return 0.0
    h, w = g.shape[:2]
    s = reference_side / max(h, w)
    if abs(s - 1) > 0.05 and max(h, w) > 16:
        g = cv2.resize(g, (max(8, int(w * s)), max(8, int(h * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    # Kontrastnormierung: Bilder mit wenig Kontrast sind nicht automatisch unscharf
    contrast = float(np.std(g)) + 1e-3
    return float(lap.var() / contrast)


def edge_sharpness(g: np.ndarray) -> float:
    """Schärfe der stärksten Kanten (robuster bei viel Unschärfe im Hintergrund, z. B. Sport-Tele)."""
    if g.size == 0:
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    top = np.percentile(mag, 99.5)
    return float(top / (np.std(g) + 1e-3))


def motion_blur(g: np.ndarray) -> dict[str, float]:
    """Bewegungsunschärfe über die Anisotropie des Strukturtensors auf Kantenpixeln.

    Rückgabe: anisotropy (0 = isotrop, 1 = alle Kanten in einer Richtung verschmiert) und die
    Verschmierungsrichtung in Grad.
    """
    if g.size == 0:
        return {"anisotropy": 0.0, "angle": 0.0}
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag > np.percentile(mag, 90)
    if m.sum() < 50:
        return {"anisotropy": 0.0, "angle": 0.0}
    jxx, jyy, jxy = float((gx[m] ** 2).mean()), float((gy[m] ** 2).mean()), float((gx[m] * gy[m]).mean())
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    anis = (l1 - l2) / (l1 + l2 + 1e-9)
    # Die Unschärfe verläuft senkrecht zur dominanten Gradientenrichtung
    angle = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy)) + 90.0
    return {"anisotropy": float(anis), "angle": float(angle % 180)}


def exposure_stats(img: np.ndarray) -> dict[str, float]:
    """Belichtungsstatistik auf der (gamma-kodierten) Vorschau."""
    g = gray(img)
    hist = np.histogram(g, bins=64, range=(0, 1))[0].astype(np.float64)
    hist /= max(hist.sum(), 1)
    p = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    clip_hi = float((g > 0.98).mean())
    clip_lo = float((g < 0.02).mean())
    # Score: Median nahe 0.45, wenig Clipping
    score = 1.0 - min(1.0, abs(p[3] - 0.45) * 1.6) - min(0.5, clip_hi * 5) - min(0.4, clip_lo * 2)
    return {
        "p01": float(p[0]), "p05": float(p[1]), "p25": float(p[2]), "median": float(p[3]), "p75": float(p[4]),
        "p95": float(p[5]), "p99": float(p[6]), "mean": float(g.mean()), "clip_hi": clip_hi, "clip_lo": clip_lo,
        "exposure_score": float(max(0.0, score)), "hist": hist.tolist(),
    }


def linear_stats(lin: np.ndarray, wb: np.ndarray | None = None) -> dict[str, float]:
    """Statistik auf linearen RAW-Daten (echtes Sensor-Clipping, Szenenhelligkeit)."""
    x = lin if wb is None else lin * wb[None, None, :3]
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    y = np.clip(y, 1e-6, None)
    logy = np.log2(y)
    p = np.percentile(logy, [1, 5, 25, 50, 75, 95, 99])
    raw_clip = float((lin.max(axis=-1) > 0.985).mean())
    hist = np.histogram(logy, bins=32, range=(-14, 0))[0].astype(np.float64)
    hist /= max(hist.sum(), 1)
    return {
        "lin_log_p01": float(p[0]), "lin_log_p05": float(p[1]), "lin_log_p25": float(p[2]),
        "lin_log_median": float(p[3]), "lin_log_p75": float(p[4]), "lin_log_p95": float(p[5]),
        "lin_log_p99": float(p[6]), "lin_log_mean": float(logy.mean()), "raw_clip": raw_clip,
        "lin_hist": hist.tolist(),
        "lin_rgb_mean": [float(v) for v in x.reshape(-1, 3).mean(axis=0)],
    }


def color_stats(img: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    return {
        "sat_mean": float(sat.mean()), "sat_p95": float(np.percentile(sat, 95)),
        "lab_a": float(lab[..., 1].mean() - 128), "lab_b": float(lab[..., 2].mean() - 128),
        # Anteil stark gesättigter, heller Pixel: typisch für LED-Bühnenlicht
        "neon_fraction": float(((sat > 0.75) & (hsv[..., 2] > 150)).mean()),
    }


def crop_region(img: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.0) -> np.ndarray:
    """Ausschnitt mit normierter Box (x0, y0, x1, y1)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0.0, x0 - pad * bw), min(1.0, x1 + pad * bw)
    y0, y1 = max(0.0, y0 - pad * bh), min(1.0, y1 + pad * bh)
    return img[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]

"""Weissabgleich auf neutrales Weiss (Trikots, Linien, Hosen) statt Grauwelt.

Unter Flutlicht ist die Kamera-Automatik oft grün- oder violettstichig, und eine Grauwelt-Korrektur kippt
Rasenbilder ins Violette (der viele grüne Rasen zieht den Mittelwert). Gemessen wird deshalb nur an hellen,
fast farblosen, nicht ausgefressenen Flächen in der Kamera-Vorschau (die schon mit dem Kamera-Weissabgleich
entwickelt ist). Ergebnis ist eine Verschiebung in Mired/Tint gegenüber der Kamera-Einstellung.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io.color import XYZ_TO_SRGB, mired, multipliers_to_temp_tint

MIN_SHARE = 0.001          # mind. 0.1 % der Pixel müssen "weiss" sein
MAX_SAT = 0.30             # (max-min)/max in der Kamera-Vorschau
MAX_MIRED, MAX_TINT = 90.0, 45.0


def white_gains(img: np.ndarray, subject: np.ndarray | None = None, rounds: int = 1) -> np.ndarray | None:
    """Kanal-Faktoren (G = 1), die helles Weiss an den Spielern neutral machen. Ohne Personen-Maske: None
    (am ganzen Bild gemessen verwechselt es blassen Rasen und Lichthöfe mit Weiss)."""
    if subject is None or float((subject > 0.5).mean()) < 0.01:
        return None
    x = img.astype(np.float32) / 255.0
    lin0 = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    total = np.ones(3)
    for _ in range(rounds):
        cur = np.clip(lin0 * total[None, None, :].astype(np.float32), 0, 1)
        enc = np.where(cur <= 0.0031308, 12.92 * cur, 1.055 * cur ** (1 / 2.4) - 0.055)
        g = _gains_once((enc * 255 + 0.5).astype(np.uint8), subject)
        if g is None:
            return None if np.allclose(total, 1) else total
        total = total * g
        if np.abs(g - 1).max() < 0.01:
            break
    return total


def _gains_once(img: np.ndarray, subject: np.ndarray | None = None) -> np.ndarray | None:
    """img: Kamera-Vorschau RGB uint8 (Anzeige). -> Kanal-Faktoren (G = 1), die helles Weiss neutral machen.
    ``subject``: Personen-Maske (0..1); wenn vorhanden, zählt nur Weiss an den Spielern (Trikots, Hosen,
    Stutzen) – Rasen unter Flutlicht, Lichthöfe und Werbebanden stören dann nicht."""
    x = img.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    s = 512 / max(h, w)
    if s < 1:
        x = cv2.resize(x, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    mx, mn = x.max(-1), x.min(-1)
    y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    clipped = (mx >= 0.96).astype(np.uint8)
    # Lichthöfe um Flutlichter/LED-Banden ausschliessen (dort ist die Farbe die der Lampe, nicht Weiss)
    k = max(3, int(max(x.shape[:2]) * 0.02)) | 1
    near_light = cv2.dilate(clipped, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    usable = ~near_light
    if usable.sum() < 100:
        return None
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    pale = usable & (sat < MAX_SAT) & (x.mean(-1) > 0.35)
    if subject is not None:
        sm = cv2.resize(subject.astype(np.float32), (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        if (pale & sm).sum() >= max(30, MIN_SHARE * sm.size * 0.5):
            pale = pale & sm
    if pale.sum() < 30:
        return None
    # Weisse Trikots/Hosen sind die hellsten nicht ausgefressenen, fast farblosen Flächen
    cand = pale & (y >= np.quantile(y[pale], 0.7))
    if cand.sum() < 20:
        return None
    m = np.median(lin[cand], axis=0)
    if m.min() <= 1e-4:
        return None
    return (m[1] / m).astype(np.float64)


def white_patch_shift(img: np.ndarray, subject: np.ndarray | None = None,
                      strength: float = 0.9) -> tuple[float, float] | None:
    """-> (Mired-Verschiebung, Tint-Verschiebung) gegenüber dem Kamera-Weissabgleich, oder None."""
    g = white_gains(img, subject)
    if g is None:
        return None
    try:
        t0, tint0 = multipliers_to_temp_tint(XYZ_TO_SRGB, np.ones(3))
        t1, tint1 = multipliers_to_temp_tint(XYZ_TO_SRGB, g)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, AssertionError):
        return None
    dm = float(np.clip((mired(t1) - mired(t0)) * strength, -MAX_MIRED, MAX_MIRED))
    dt = float(np.clip((tint1 - tint0) * strength, -MAX_TINT, MAX_TINT))
    return dm, dt

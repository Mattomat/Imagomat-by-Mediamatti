"""Jedes Bild einzeln beurteilen (für die mitgelieferten Presets).

Die Preset-Werte sind nur der Ausgangspunkt. Danach wird das Bild klein entwickelt (aus der Kamera-Vorschau,
mit allen Masken) und angeschaut, wie es aussieht:

- Helligkeit: das hellere Viertel des Bildes (75 %-Quantil, L*) auf die Helligkeit deiner Bilder bringen
  (Nacht: an deinen Lightroom-Bearbeitungen gemessen, L ≈ 35, Streuung ±4; Tag heller). Das ist robuster als
  Spieler oder Rasen allein: ein weisses oder dunkles Trikot, viel oder wenig Himmel verschieben es kaum.
- Lichter: weisse Trikots dürfen nicht ausfressen (oberste 3 % der Spieler unter L 93), sonst Lichter runter.
- Tiefen: zu dunkle Spieler (Gegenlicht, dunkle Ecke) bekommen mehr Tiefen.

Dreimal messen und nachregeln; Weiss/Schwarz setzt danach ``fit_white_black`` wie am Waveform.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..io.color import XYZ_TO_SRGB
from ..lightroom.params import to_number
from ..render.pipeline import render

log = logging.getLogger(__name__)

ROUNDS = 3
L75_NIGHT, L75_DAY = 35.0, 58.0     # 75 %-Quantil L* (Nacht: an deinen 4 Lightroom-Bearbeitungen geeicht)
SUBJ_HI_MAX = 93.0                  # oberste 3 % der Spieler: Weiss mit Zeichnung
SUBJ_DARK = 18.0                    # Spieler-Median darunter: Tiefen anheben
MAX_EXPOSURE_CHANGE = 1.0           # gegenüber der Schätzung aus den RAW-Daten


def _ev(target_l: float, now_l: float) -> float:
    def y(L: float) -> float:
        L = max(L, 1.0)
        return ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
    return float(np.log2(y(target_l) / y(now_l)))


def _num(crs: dict[str, Any], k: str) -> float:
    return float(to_number(crs.get(k)) or 0.0)


def judge(crs: dict[str, Any], lin: np.ndarray, analysis: dict[str, Any], orientation: int = 1,
          subject: np.ndarray | None = None, target_l75: float | None = None) -> list[str]:
    """Passt Belichtung, Lichter und Tiefen an das tatsächliche Aussehen dieses Bildes an."""
    from .look import _WB, measure
    from .presets import night_weight

    seg = {"subject": subject} if subject is not None else {}
    wb = _WB(crs, analysis.get("as_shot_temp"), analysis.get("as_shot_tint"))
    nw = night_weight(analysis)
    target = target_l75 if target_l75 is not None else (1 - nw) * L75_DAY + nw * L75_NIGHT
    e0, h0, s0 = _num(crs, "Exposure2012"), _num(crs, "Highlights2012"), _num(crs, "Shadows2012")
    for rnd in range(ROUNDS):
        c = {**crs, **wb.preview_keys(), "HasCrop": "False", "CropAngle": 0}
        img = render(lin, XYZ_TO_SRGB, np.ones(3), c, orientation, seg, None)
        m = measure(img, subject)
        lab_l = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[..., 0]
        damp = 0.85 if rnd < ROUNDS - 1 else 0.6
        d = float(np.clip(_ev(target, float(np.quantile(lab_l, 0.75))) * damp, -1.0, 1.0))
        crs["Exposure2012"] = round(float(np.clip(_num(crs, "Exposure2012") + d, e0 - MAX_EXPOSURE_CHANGE,
                                                  e0 + MAX_EXPOSURE_CHANGE)), 2)
        if subject is not None and "subj_L50" in m:
            sm = cv2.resize(subject.astype(np.float32), (img.shape[1], img.shape[0])) > 0.5
            hi = float(np.quantile(lab_l[sm], 0.97))
            if hi > SUBJ_HI_MAX:
                crs["Highlights2012"] = int(round(max(-100.0, _num(crs, "Highlights2012") -
                                                      min(20.0, (hi - SUBJ_HI_MAX) * 3.0))))
            if m["subj_L50"] < SUBJ_DARK:
                crs["Shadows2012"] = int(round(min(70.0, _num(crs, "Shadows2012") + 8.0)))
    notes = []
    de = _num(crs, "Exposure2012") - e0
    if abs(de) >= 0.05:
        notes.append(f"beurteilt: Belichtung {de:+.2f}")
    if _num(crs, "Highlights2012") < h0 - 0.5:
        notes.append(f"Lichter {_num(crs, 'Highlights2012') - h0:+.0f} (Trikots nicht ausgefressen)")
    if _num(crs, "Shadows2012") > s0 + 0.5:
        notes.append(f"Tiefen +{_num(crs, 'Shadows2012') - s0:.0f} (Spieler zu dunkel)")
    return notes

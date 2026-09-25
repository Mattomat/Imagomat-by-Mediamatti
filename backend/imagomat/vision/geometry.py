"""Begradigen und Zuschneiden.

- ``estimate_tilt``: Neigung aus Liniensegmenten (bevorzugt Senkrechte wie Torpfosten,
  Masten, Lautsprechertürme; Waagrechte nur, wenn sie lang sind).
- ``plan_crop``: Zuschnitt im gedrehten Bild (Seitenverhältnis, Motivposition, Zoom).
- ``to_lightroom_crop``: Umrechnung in crs:CropLeft/Top/Right/Bottom/Angle. Lightroom
  speichert die zwei Eckpunkte des gedrehten Rechtecks normiert im *Sensor*-Koordinatensystem
  (vor der EXIF-Drehung); vgl. darktable src/develop/lightroom.c.

Vorzeichen von CropAngle: positiv = Bild im Uhrzeigersinn drehen (wie der Lightroom-
Geraderichten-Regler). Muss im Roundtrip-Test bestätigt werden (docs/lightroom-roundtrip.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CROP_ANGLE_SIGN = 1.0


@dataclass
class Tilt:
    angle: float          # Grad, im Uhrzeigersinn drehen, um das Bild zu begradigen
    confidence: float     # 0..1
    source: str           # vertical | horizontal | none


def _segments(gray: np.ndarray) -> np.ndarray:
    g = gray if gray.dtype == np.uint8 else np.clip(gray * 255, 0, 255).astype(np.uint8)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(g)[0]
    except (cv2.error, AttributeError):
        lines = None
    if lines is None:
        edges = cv2.Canny(g, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 60, minLineLength=g.shape[1] // 12, maxLineGap=6)
    return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4).astype(np.float32)


def estimate_tilt(img: np.ndarray, max_deg: float = 8.0) -> Tilt:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    s = 1024 / max(h, w)
    if s < 1:
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = g.shape[:2]
    seg = _segments(g)
    if len(seg) == 0:
        return Tilt(0.0, 0.0, "none")
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))            # y nach unten
    ang = (ang + 180) % 180                         # 0..180
    diag = math.hypot(w, h)

    def robust(dev: np.ndarray, wts: np.ndarray) -> tuple[float, float]:
        order = np.argsort(dev)
        cw = np.cumsum(wts[order])
        med = float(dev[order][np.searchsorted(cw, cw[-1] / 2)])
        support = float(wts[np.abs(dev - med) < 0.75].sum() / max(wts.sum(), 1e-6))
        return med, support

    # Senkrechte: Abweichung von 90°. Eine Linie mit Winkel > 90 kippt oben nach links
    vmask = (np.abs(ang - 90) < max_deg + 2) & (length > 0.04 * diag)
    hmask = ((ang < max_deg + 2) | (ang > 180 - max_deg - 2)) & (length > 0.12 * diag)
    if vmask.sum() >= 3:
        dev = ang[vmask] - 90.0
        med, support = robust(dev, length[vmask])
        total = float(length[vmask].sum() / diag)
        conf = support * min(1.0, total / 1.5)
        # Inhalt um a gegen den Uhrzeigersinn verdreht -> Senkrechte hat Winkel 90 - a
        # -> Korrektur: um a = -dev im Uhrzeigersinn drehen (gilt analog für Waagrechte).
        return Tilt(float(np.clip(-med, -max_deg, max_deg)), conf, "vertical")
    if hmask.sum() >= 1:
        dev = np.where(ang[hmask] > 90, ang[hmask] - 180, ang[hmask])
        med, support = robust(dev, length[hmask])
        total = float(length[hmask].sum() / diag)
        conf = support * min(1.0, total / 1.0) * 0.8
        return Tilt(float(np.clip(-med, -max_deg, max_deg)), conf, "horizontal")
    return Tilt(0.0, 0.0, "none")


# ---------------------------------------------------------------------------
# Zuschnitt
# ---------------------------------------------------------------------------

@dataclass
class CropPlan:
    cx: float             # Mittelpunkt im gedrehten Bild, Pixel
    cy: float
    width: float          # Pixel
    height: float
    angle: float          # Grad im Uhrzeigersinn
    image_w: int
    image_h: int

    @property
    def area_fraction(self) -> float:
        return (self.width * self.height) / max(self.image_w * self.image_h, 1)


def _rot(x: float, y: float, deg: float) -> tuple[float, float]:
    """Drehung im Bildkoordinatensystem (y nach unten), positiv = sichtbar im Uhrzeigersinn."""
    a = math.radians(deg)
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def _corners(plan: CropPlan) -> list[tuple[float, float]]:
    """Ecken des Zuschnitts in Originalkoordinaten (Anzeigeorientierung, Pixel)."""
    W, H = plan.image_w, plan.image_h
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        qx, qy = plan.cx + sx * plan.width / 2 - W / 2, plan.cy + sy * plan.height / 2 - H / 2
        px, py = _rot(qx, qy, -plan.angle)       # zurück ins ungedrehte Bild
        out.append((px + W / 2, py + H / 2))
    return out


def _fits(plan: CropPlan) -> bool:
    return all(-0.5 <= x <= plan.image_w + 0.5 and -0.5 <= y <= plan.image_h + 0.5 for x, y in _corners(plan))


def max_crop(W: int, H: int, angle: float, aspect: float, cx: float | None = None,
             cy: float | None = None) -> CropPlan:
    """Grösster Zuschnitt mit Seitenverhältnis ``aspect`` (Breite/Höhe) um (cx, cy)."""
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    lo, hi = 0.0, float(max(W, H)) * 2
    for _ in range(40):
        mid = (lo + hi) / 2
        p = CropPlan(cx, cy, mid, mid / aspect, angle, W, H)
        if _fits(p):
            lo = mid
        else:
            hi = mid
    return CropPlan(cx, cy, lo, lo / aspect, angle, W, H)


def plan_crop(W: int, H: int, angle: float = 0.0, aspect: float | None = None, zoom: float = 1.0,
              subject: tuple[float, float, float, float] | None = None,
              face: tuple[float, float] | None = None) -> CropPlan:
    """Zuschnitt planen.

    zoom: Anteil der maximal möglichen Breite (1.0 = kein zusätzliches Einengen).
    subject: normierte Motivbox; muss vollständig im Zuschnitt liegen.
    face: normierter Gesichtsmittelpunkt, der auf das obere Drittel gesetzt wird.
    """
    aspect = aspect or (W / H)
    full = max_crop(W, H, angle, aspect)
    width = max(full.width * min(max(zoom, 0.2), 1.0), 16)
    if subject is not None:
        sw = (subject[2] - subject[0]) * W * 1.08
        sh = (subject[3] - subject[1]) * H * 1.08
        width = max(width, sw, sh * aspect)
    width = min(width, full.width)
    height = width / aspect
    # Zielmittelpunkt
    cx, cy = W / 2, H / 2
    if subject is not None:
        cx = (subject[0] + subject[2]) / 2 * W
        cy = (subject[1] + subject[3]) / 2 * H
    if face is not None:
        cy = face[1] * H + height / 6          # Gesicht auf dem oberen Drittel
        cx = face[0] * W if subject is None else cx
    # In den gültigen Bereich schieben, sonst verkleinern
    for _ in range(60):
        p = CropPlan(cx, cy, width, height, angle, W, H)
        if _fits(p):
            return p
        # Richtung Bildmitte ziehen
        cx += (W / 2 - cx) * 0.15
        cy += (H / 2 - cy) * 0.15
        if abs(cx - W / 2) < 1 and abs(cy - H / 2) < 1:
            width *= 0.97
            height = width / aspect
    return max_crop(W, H, angle, aspect)


def display_to_sensor(x: float, y: float, orientation: int) -> tuple[float, float]:
    """Normierte Anzeige- -> Sensorkoordinaten (Umkehrung der EXIF-Orientierung)."""
    if orientation == 3:
        return 1 - x, 1 - y
    if orientation == 6:
        return y, 1 - x
    if orientation == 8:
        return 1 - y, x
    if orientation == 2:
        return 1 - x, y
    if orientation == 4:
        return x, 1 - y
    if orientation == 5:
        return y, x
    if orientation == 7:
        return 1 - y, 1 - x
    return x, y


def to_lightroom_crop(plan: CropPlan, orientation: int = 1) -> dict[str, object]:
    """CropPlan -> crs-Felder. Liefert HasCrop=False, wenn nichts zu tun ist."""
    full = plan.area_fraction > 0.999 and abs(plan.angle) < 0.01
    if full:
        return {"HasCrop": False, "CropAngle": 0.0}
    pts = [(x / plan.image_w, y / plan.image_h) for x, y in _corners(plan)]
    pts = [display_to_sensor(x, y, orientation) for x, y in pts]
    # Diagonal gegenüberliegende Ecken: die "oben links" und "unten rechts" liegenden im Sensorraum
    a = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    b = (a + 2) % 4
    angle = plan.angle * CROP_ANGLE_SIGN
    if orientation in (2, 4, 5, 7):
        angle = -angle
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    return {
        "HasCrop": True,
        "CropLeft": clamp(pts[a][0]), "CropTop": clamp(pts[a][1]),
        "CropRight": clamp(pts[b][0]), "CropBottom": clamp(pts[b][1]),
        "CropAngle": round(angle, 4), "CropConstrainToWarp": 0,
    }


def from_lightroom_crop(crs: dict[str, object], orientation: int = 1) -> dict[str, float] | None:
    """Liest einen Lightroom-Zuschnitt als Merkmale (Fläche, Seitenverhältnis, Winkel, Lage)."""
    from ..lightroom.params import to_number

    if str(crs.get("HasCrop", "False")).lower() != "true":
        return {"crop_area": 1.0, "crop_angle": float(to_number(crs.get("CropAngle")) or 0.0),
                "crop_cx": 0.5, "crop_cy": 0.5, "crop_aspect_rel": 1.0}
    l, t = to_number(crs.get("CropLeft")) or 0.0, to_number(crs.get("CropTop")) or 0.0
    r, b = to_number(crs.get("CropRight")) or 1.0, to_number(crs.get("CropBottom")) or 1.0
    ang = to_number(crs.get("CropAngle")) or 0.0
    w, h = abs(r - l), abs(b - t)
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return {"crop_area": float(w * h), "crop_angle": float(ang), "crop_cx": float((l + r) / 2),
            "crop_cy": float((t + b) / 2), "crop_aspect_rel": float(w / h) if h else 1.0}

"""Lokale Anpassungen: Lightroom-Masken erzeugen und aus Beispielen lernen.

Maskenarten, die wir schreiben:
- KI-Masken (Motiv, Himmel, Personen, Hintergrund = Motiv invertiert) als *Deklaration*.
  Lightroom berechnet die Maske selbst ("KI-Einstellungen aktualisieren").
  Die exakten Felder stammen aus dem gelernten Dialekt (eigene XMPs), sonst Fallback.
- Radialfilter und Verläufe: rein parametrisch, exakt.
- Pinsel (Mask/Paint): Fallback für eigene Segmentierungen, Maske -> Tupfer ("Dabs").

Aus Trainingsdaten entstehen *Masken-Vorlagen*: gleiche Maskenkombination (Signatur) +
typische Korrekturwerte. Das Stilmodell sagt pro Bild voraus, welche Vorlagen passen und mit
welchen Werten.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..lightroom.develop import new_sync_id, refresh_sync_ids, strip_digests
from ..lightroom.dialect import Dialect
from ..lightroom.params import LOCAL_PARAMS, to_number
from ..vision.geometry import display_to_sensor, sensor_to_display

# Annahme (zu verifizieren): Pinselradius relativ zur langen Bildseite
PAINT_RADIUS_BASIS = "long_side"


def _local_values(local_ui: dict[str, float]) -> dict[str, float]:
    out = {}
    for k, v in local_ui.items():
        key = k if k.startswith("Local") else f"Local{k}"
        scale = LOCAL_PARAMS.get(key, 100.0)
        out[key] = float(np.clip(v / scale, -1.0, 1.0))
    return out


def correction(name: str, local_ui: dict[str, float], components: list[dict[str, Any]],
               amount: float = 1.0) -> dict[str, Any]:
    c: dict[str, Any] = {"What": "Correction", "CorrectionAmount": float(amount), "CorrectionActive": True,
                         "CorrectionName": name, "CorrectionSyncID": new_sync_id()}
    c.update(_local_values(local_ui))
    c["CorrectionMasks"] = components
    return c


def _component_base(name: str, inverted: bool = False, blend: int = 0) -> dict[str, Any]:
    return {"MaskActive": True, "MaskName": name, "MaskBlendMode": blend, "MaskInverted": inverted,
            "MaskSyncID": new_sync_id()}


def ai_component(kind: str, dialect: Dialect, name: str | None = None, inverted: bool = False) -> dict[str, Any]:
    if kind == "background":
        kind, inverted = "subject", not inverted
    tmpl = dialect.ai_masks.get(kind) or dialect.ai_masks["subject"]
    comp = {k: v for k, v in strip_digests(copy.deepcopy(tmpl)).items()}
    comp.update(_component_base(name or kind, inverted))
    return comp


def radial_component(box: tuple[float, float, float, float], orientation: int = 1, feather: float = 60,
                     inverted: bool = False, name: str = "Radial") -> dict[str, Any]:
    """box: normierte Anzeige-Koordinaten (x0, y0, x1, y1)."""
    x0, y0 = display_to_sensor(box[0], box[1], orientation)
    x1, y1 = display_to_sensor(box[2], box[3], orientation)
    comp = {"What": "Mask/CircularGradient", "MaskValue": 1.0, "Top": min(y0, y1), "Left": min(x0, x1),
            "Bottom": max(y0, y1), "Right": max(x0, x1), "Angle": 0.0, "Midpoint": 50.0, "Roundness": 0.0,
            "Feather": float(feather), "Flipped": False, "Version": 2}
    comp.update(_component_base(name, inverted))
    return comp


def gradient_component(zero: tuple[float, float], full: tuple[float, float], orientation: int = 1,
                       name: str = "Verlauf") -> dict[str, Any]:
    """Linearer Verlauf: bei ``zero`` 0 % Wirkung, bei ``full`` 100 % (Anzeige-Koordinaten)."""
    zx, zy = display_to_sensor(*zero, orientation)
    fx, fy = display_to_sensor(*full, orientation)
    comp = {"What": "Mask/Gradient", "MaskValue": 1.0, "ZeroX": zx, "ZeroY": zy, "FullX": fx, "FullY": fy}
    comp.update(_component_base(name))
    return comp


def mask_to_dabs(mask: np.ndarray, max_dabs: int = 1200, coverage: float = 0.93
                 ) -> list[tuple[float, float, float]]:
    """Binärmaske -> Kreise (x, y, r) in normierten Anzeige-Koordinaten (r relativ zur langen Seite)."""
    m = (mask > 0.5).astype(np.uint8)
    h, w = m.shape
    total = int(m.sum())
    if total < 20:
        return []
    remaining = m.copy()
    covered = np.zeros_like(m)
    dabs = []
    long_side = max(h, w)
    while len(dabs) < max_dabs and covered.sum() < coverage * total:
        dt = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        _, maxv, _, (x, y) = cv2.minMaxLoc(dt)
        if maxv < 1.0:
            break
        r = max(1.5, maxv * 1.05)
        cv2.circle(remaining, (x, y), int(round(r * 0.8)), 0, -1)
        cv2.circle(covered, (x, y), int(round(r)), 1, -1)
        covered &= m
        dabs.append((x / w, y / h, r / long_side))
    return dabs


def paint_components(mask: np.ndarray, orientation: int = 1, name: str = "Pinsel",
                     max_dabs: int = 1200) -> list[dict[str, Any]]:
    """Pinsel-Komponenten (eine pro Radiusklasse), da Lightroom pro Strich nur einen Radius kennt."""
    dabs = mask_to_dabs(mask, max_dabs)
    if not dabs:
        return []
    radii = np.array([d[2] for d in dabs])
    edges = np.quantile(radii, [0.0, 0.5, 0.8, 1.0])
    comps = []
    for i in range(3):
        sel = [d for d in dabs if edges[i] <= d[2] <= edges[i + 1] + 1e-9 and (i == 0 or d[2] > edges[i])]
        if not sel:
            continue
        pts = []
        for x, y, _ in sel:
            sx, sy = display_to_sensor(x, y, orientation)
            pts.append(f"d {sx:.6f} {sy:.6f}")
        comp = {"What": "Mask/Paint", "MaskValue": 1.0, "Radius": float(np.median([d[2] for d in sel])),
                "Flow": 1.0, "CenterWeight": 0.0, "Dabs": pts}
        comp.update(_component_base(f"{name} {i + 1}", blend=0))
        comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
# Vorlagen aus Trainingsdaten
# ---------------------------------------------------------------------------

def component_kind(m: dict[str, Any]) -> str:
    what = str(m.get("What", ""))
    inv = "-inv" if str(m.get("MaskInverted", "false")).lower() == "true" else ""
    if what == "Mask/Image":
        sub = m.get("MaskSubType", "")
        cat = m.get("MaskSubCategoryID", "")
        return f"image:{sub}:{cat}{inv}"
    if what == "Mask/CircularGradient":
        return f"radial{inv}"
    if what == "Mask/Gradient":
        return "gradient"
    if what == "Mask/Paint":
        return "brush"
    if what == "Mask/RangeMask":
        rm = m.get("CorrectionRangeMask") or {}
        return f"range:{rm.get('Type', '')}"
    return what or "unknown"


def signature(corr: dict[str, Any]) -> str:
    comps = corr.get("CorrectionMasks") or []
    kinds = sorted({component_kind(m) for m in comps if isinstance(m, dict)} - {"brush"})
    if not kinds and any(isinstance(m, dict) and m.get("What") == "Mask/Paint" for m in comps):
        kinds = ["brush"]
    return "+".join(kinds) or "leer"


def local_values(corr: dict[str, Any]) -> dict[str, float]:
    return {k: float(to_number(v) or 0.0) for k, v in corr.items() if k.startswith("Local") and to_number(v) is not None}


@dataclass
class MaskTemplate:
    sig: str
    name: str
    count: int
    example: dict[str, Any]
    params: list[str]
    medians: dict[str, float]
    radial_rel: list[list[float]] = field(default_factory=list)   # [dx, dy, sw, sh] relativ zum Motiv
    gradient_geo: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MaskTemplate":
        return cls(**d)


def _subject_box(a: dict[str, Any]) -> tuple[float, float, float, float]:
    b = a.get("subject_bbox") or [0.3, 0.2, 0.7, 0.9]
    return tuple(float(x) for x in b)  # type: ignore[return-value]


def learn_templates(samples: list[tuple[dict[str, Any], dict[str, Any]]], min_frac: float = 0.03,
                    min_count: int = 3) -> list[MaskTemplate]:
    """samples: (crs, analysis) je Trainingsbild."""
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for crs, a in samples:
        for corr in crs.get("MaskGroupBasedCorrections", []) or []:
            if isinstance(corr, dict) and str(corr.get("CorrectionActive", "true")).lower() != "false":
                groups.setdefault(signature(corr), []).append((corr, a))
    n = max(len(samples), 1)
    out = []
    for sig, items in groups.items():
        if len(items) < min_count or len(items) / n < min_frac or sig == "leer":
            continue
        params = sorted({k for corr, _ in items for k in local_values(corr)})
        med = {p: float(np.median([local_values(c).get(p, 0.0) for c, _ in items])) for p in params}
        # Repräsentatives Beispiel: das mit den mediannächsten Werten
        ex = min(items, key=lambda it: sum(abs(local_values(it[0]).get(p, 0) - med[p]) for p in params))[0]
        names = Counter(str(c.get("CorrectionName") or sig) for c, _ in items)
        t = MaskTemplate(sig, names.most_common(1)[0][0], len(items), strip_digests(copy.deepcopy(ex)), params, med)
        for corr, a in items:
            sb = _subject_box(a)
            sw, sh = max(sb[2] - sb[0], 1e-3), max(sb[3] - sb[1], 1e-3)
            for m in corr.get("CorrectionMasks", []) or []:
                if not isinstance(m, dict):
                    continue
                if m.get("What") == "Mask/CircularGradient":
                    o = int(a.get("orientation") or 1)
                    l, tp = to_number(m.get("Left")) or 0, to_number(m.get("Top")) or 0
                    r, b = to_number(m.get("Right")) or 1, to_number(m.get("Bottom")) or 1
                    (dl, dt), (dr, db) = sensor_to_display(l, tp, o), sensor_to_display(r, b, o)
                    l, r, tp, b = min(dl, dr), max(dl, dr), min(dt, db), max(dt, db)
                    cx, cy = (l + r) / 2, (tp + b) / 2
                    t.radial_rel.append([(cx - (sb[0] + sb[2]) / 2) / sw, (cy - (sb[1] + sb[3]) / 2) / sh,
                                         (r - l) / sw, (b - tp) / sh])
                elif m.get("What") == "Mask/Gradient":
                    t.gradient_geo.append([to_number(m.get(k)) or 0.0 for k in ("ZeroX", "ZeroY", "FullX", "FullY")])
        out.append(t)
    return sorted(out, key=lambda t: -t.count)


def instantiate(t: MaskTemplate, values: dict[str, float], analysis: dict[str, Any], orientation: int,
                dialect: Dialect, subject_mask: np.ndarray | None = None) -> dict[str, Any] | None:
    """Vorlage auf ein neues Bild übertragen (Werte vorhergesagt, Geometrie angepasst)."""
    corr = refresh_sync_ids(strip_digests(copy.deepcopy(t.example)))
    for k in list(corr):
        if k.startswith("Local"):
            corr.pop(k)
    for k, v in values.items():
        corr[k] = float(np.clip(v, -1, 1))
    comps = []
    sb = _subject_box(analysis)
    for m in corr.get("CorrectionMasks", []) or []:
        if not isinstance(m, dict):
            continue
        what = m.get("What")
        if what == "Mask/CircularGradient" and t.radial_rel:
            dx, dy, rw, rh = np.median(np.array(t.radial_rel), axis=0)
            sw, sh = sb[2] - sb[0], sb[3] - sb[1]
            cx, cy = (sb[0] + sb[2]) / 2 + dx * sw, (sb[1] + sb[3]) / 2 + dy * sh
            box = (cx - rw * sw / 2, cy - rh * sh / 2, cx + rw * sw / 2, cy + rh * sh / 2)
            new = radial_component(box, orientation, float(to_number(m.get("Feather")) or 60),
                                   str(m.get("MaskInverted", "false")).lower() == "true", str(m.get("MaskName", "Radial")))
            comps.append(new)
        elif what == "Mask/Paint":
            if subject_mask is not None and not any(c.get("What") == "Mask/Paint" for c in comps):
                comps += paint_components(subject_mask, orientation, "Motiv (Pinsel)")
        else:
            comps.append(m)
    if not comps:
        return None
    corr["CorrectionMasks"] = comps
    return corr

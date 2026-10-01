"""Look aus deinen fertigen Bildern: messen statt raten.

Aus einem Ordner mit fertig bearbeiteten Bildern (JPEG-Export aus Lightroom) wird eine *Look-Signatur*
gemessen: wie hell Personen, oberer Bildteil (Himmel/Dach), Hintergrund (Zuschauer) und der untere Rand sind,
wie neutral helles Weiss ist, wie satt die einzelnen Farben sind und wo Weiss und Schwarz am Waveform
anschlagen. Beim Entwickeln wird jedes Bild mit einer kleinen Vorschau-Entwicklung genau auf diese Werte
gebracht (Belichtung, Weissabgleich, Masken für oben/Hintergrund/unten, HSL-Sättigung, Kontrast, Weiss und
Schwarz). Die Werte landen als normale Lightroom-Einstellungen im XMP.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import data_dir
from ..io.color import XYZ_TO_SRGB, mired, multipliers_to_temp_tint
from ..lightroom.params import to_number
from ..render.pipeline import HUE_CENTERS, render

log = logging.getLogger(__name__)

PREFIX = "look:"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
TOP, BOTTOM = 0.30, 0.85            # oberer Bildteil / unterer Rand (Anteil der Bildhöhe)
MIN_REGION = 0.02
ROUNDS = 4
LOOK_NAMES = {"top": "Look: oben", "bg": "Look: Hintergrund"}


def looks_dir() -> Path:
    p = data_dir() / "looks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _file(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in " -_äöüÄÖÜ").strip() or "Look"
    return looks_dir() / f"{safe}.json"


def list_looks() -> list[dict[str, Any]]:
    out = []
    for p in sorted(looks_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text())
            out.append({"name": d["name"], "n": d.get("n"), "base": d.get("base")})
        except (OSError, ValueError, KeyError):
            continue
    return out


def load_look(name: str) -> dict[str, Any] | None:
    p = _file(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def delete_look(name: str) -> bool:
    p = _file(name)
    if p.exists():
        p.unlink()
        return True
    return False


def look_version(name: str) -> float:
    p = _file(name)
    return p.stat().st_mtime if p.exists() else 0.0


# ---------------------------------------------------------------------------
# Messen
# ---------------------------------------------------------------------------

def _regions(shape: tuple[int, int], subject: np.ndarray | None) -> dict[str, np.ndarray]:
    h, w = shape
    yy = (np.arange(h, dtype=np.float32) / max(h - 1, 1))[:, None].repeat(w, 1)
    subj = np.zeros((h, w), bool) if subject is None else \
        cv2.resize(subject.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    top = (yy < TOP) & ~subj
    bottom = (yy >= BOTTOM) & ~subj
    bg = ~subj & ~top & ~bottom
    return {"subj": subj, "top": top, "bg": bg, "bottom": bottom}


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def measure(img: np.ndarray, subject: np.ndarray | None = None) -> dict[str, float]:
    """Look-Kennzahlen eines fertigen Bildes (RGB uint8 oder float 0..1)."""
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else np.clip(img.astype(np.float32), 0, 1)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(A, B)
    out: dict[str, float] = {}
    n = L.size
    for name, m in _regions(L.shape, subject).items():
        if m.sum() < MIN_REGION * n:
            continue
        v = L[m]
        out[f"{name}_L10"], out[f"{name}_L50"], out[f"{name}_L90"] = (float(q) for q in np.quantile(v, [.1, .5, .9]))
        if name in ("top", "bg"):
            out[f"{name}_a"], out[f"{name}_b"] = float(np.median(A[m])), float(np.median(B[m]))
    y = _luma(x)
    out["y_hi"], out["y_lo"] = float(np.quantile(y, 0.997)), float(np.quantile(y, 0.003))
    # Weiss: helle, fast farblose Flächen (Trikots, Linien, Flutlicht-Reflexe)
    neutral = (L > 60) & (C < 14)
    if neutral.sum() >= 0.002 * n:
        out["n_a"], out["n_b"] = float(np.median(A[neutral])), float(np.median(B[neutral]))
    # Sättigung je Farbton (HSV wie die HSL-Regler)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (S > 0.12) & (V > 0.12)
    for hue, center in HUE_CENTERS.items():
        d = np.abs(((H - center + 180) % 360) - 180)
        m = colored & (d < 22)
        if m.sum() >= 0.01 * n:
            out[f"sat_{hue}"] = float(np.mean(S[m]))
    return out


def _segment_subject(img: np.ndarray) -> np.ndarray | None:
    try:
        from ..vision.segmentation import segment

        return segment(img).subject
    except Exception as e:  # noqa: BLE001 - dann ohne Personen-Bereich
        log.debug("Segmentierung fehlgeschlagen: %s", e)
        return None


def learn_look(name: str, folder: Path, base: str = "fb_signature", progress=None) -> dict[str, Any]:
    """Look-Signatur aus einem Ordner mit fertigen Bildern messen und speichern."""
    files = [p for p in sorted(Path(folder).expanduser().rglob("*"))
             if p.suffix.lower() in IMAGE_EXT and p.is_file() and not p.name.startswith("._")]
    if not files:
        raise ValueError(f"Im Ordner „{Path(folder).name}“ sind keine fertigen Bilder (JPEG/PNG/TIFF).")
    rows: list[dict[str, float]] = []
    for i, p in enumerate(files[:200]):
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = 768 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rows.append(measure(img, _segment_subject(img)))
        if progress:
            progress(i + 1, len(files))
    if not rows:
        raise ValueError("Keines der Bilder konnte gelesen werden.")
    keys = sorted({k for r in rows for k in r})
    stats = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys
             if sum(k in r for r in rows) >= max(1, len(rows) // 4)}
    look = {"name": name, "n": len(rows), "base": base, "stats": stats, "folder": str(folder),
            "created": time.time()}
    _file(name).write_text(json.dumps(look, indent=1, ensure_ascii=False))
    return look


# ---------------------------------------------------------------------------
# Anpassen
# ---------------------------------------------------------------------------

def _Y(L: float) -> float:
    return ((L + 16) / 116) ** 3


def _ev(target_L: float, now_L: float) -> float:
    return math.log2(max(_Y(target_L), 1e-4) / max(_Y(now_L), 1e-4))


def _local_ev(corr: dict[str, Any]) -> float:
    return float(to_number(corr.get("LocalExposure2012")) or 0.0) * 4.0


def _set_local_ev(corr: dict[str, Any], ev: float) -> None:
    corr["LocalExposure2012"] = float(np.clip(ev, -4, 4) / 4.0)


class _WB:
    """Weissabgleich als Verschiebung (Mired/Tint) gegenüber der Kamera-Einstellung."""

    def __init__(self, crs: dict[str, Any], as_temp: float | None, as_tint: float | None):
        self.as_temp, self.as_tint = as_temp, as_tint or 0.0
        t, tint = to_number(crs.get("Temperature")), to_number(crs.get("Tint"))
        custom = str(crs.get("WhiteBalance", "As Shot")) != "As Shot" and t
        self.dm0 = (mired(float(t)) - mired(as_temp)) if (custom and as_temp) else 0.0
        self.dt0 = (float(tint or 0) - self.as_tint) if custom else 0.0
        self.dm = self.dt = 0.0
        # Messung an der (bereits weissabgeglichenen) Kamera-Vorschau: Bezug ist deren neutrale Temperatur
        self.t0, self.tint0 = multipliers_to_temp_tint(XYZ_TO_SRGB, np.ones(3))

    def preview_keys(self) -> dict[str, Any]:
        dm, dt = self.dm0 + self.dm, self.dt0 + self.dt
        if abs(dm) < 0.5 and abs(dt) < 0.5:
            return {"WhiteBalance": "As Shot"}
        return {"WhiteBalance": "Custom", "Temperature": 1e6 / (mired(self.t0) + dm), "Tint": self.tint0 + dt}

    def final_keys(self) -> dict[str, Any]:
        if not self.as_temp or (abs(self.dm) < 0.5 and abs(self.dt) < 0.5):
            return {}
        dm, dt = self.dm0 + self.dm, self.dt0 + self.dt
        return {"WhiteBalance": "Custom", "Temperature": int(round(1e6 / (mired(self.as_temp) + dm) / 50) * 50),
                "Tint": int(round(self.as_tint + dt))}


def _corr(crs: dict[str, Any], name: str) -> dict[str, Any] | None:
    for c in crs.get("MaskGroupBasedCorrections") or []:
        if c.get("CorrectionName") == name:
            return c
    return None


def _ensure_corrections(crs: dict[str, Any], orientation: int, dialect: Any) -> None:
    from . import masks as mk

    corrs = list(crs.get("MaskGroupBasedCorrections") or [])
    if _corr(crs, LOOK_NAMES["top"]) is None:
        corrs.append(mk.correction(LOOK_NAMES["top"], {},
                                   [mk.gradient_component((0.5, TOP + 0.12), (0.5, 0.0), orientation,
                                                          LOOK_NAMES["top"])]))
    if dialect is not None and _corr(crs, LOOK_NAMES["bg"]) is None:
        corrs.append(mk.correction(LOOK_NAMES["bg"], {}, [mk.ai_component("background", dialect, LOOK_NAMES["bg"])]))
    crs["MaskGroupBasedCorrections"] = corrs


def _bottom_corr(crs: dict[str, Any], orientation: int) -> dict[str, Any] | None:
    from .develop import _is_bottom_gradient

    for c in crs.get("MaskGroupBasedCorrections") or []:
        if _is_bottom_gradient(c, orientation):
            return c
    return None


def fit_look(crs: dict[str, Any], lin: np.ndarray, look: dict[str, Any], orientation: int = 1,
             subject: np.ndarray | None = None, as_temp: float | None = None, as_tint: float | None = None,
             dialect: Any = None) -> list[str]:
    """Passt ``crs`` so an, dass die Vorschau-Entwicklung die Kennzahlen des Looks trifft."""
    ref = look.get("stats") or {}
    if not ref:
        return []
    _ensure_corrections(crs, orientation, dialect if subject is not None else None)
    seg = {"subject": subject} if subject is not None else {}
    wb = _WB(crs, as_temp, as_tint)
    e0 = float(to_number(crs.get("Exposure2012")) or 0.0)
    c0 = float(to_number(crs.get("Contrast2012")) or 0.0)
    top, bg = _corr(crs, LOOK_NAMES["top"]), _corr(crs, LOOK_NAMES["bg"])
    wb_orig = {k: crs[k] for k in ("WhiteBalance", "Temperature", "Tint") if k in crs}
    bottom = _bottom_corr(crs, orientation)

    def shot() -> dict[str, float]:
        c = {**crs, **wb.preview_keys(), "HasCrop": "False", "CropAngle": 0}
        return measure(render(lin, XYZ_TO_SRGB, np.ones(3), c, orientation, seg, None), subject)

    for rnd in range(ROUNDS):
        m = shot()
        damp = 0.85 if rnd < ROUNDS - 1 else 0.6
        # 1) Belichtung: Personen (sonst Hintergrund) auf die Helligkeit deiner Bilder
        key = "subj_L50" if "subj_L50" in m and "subj_L50" in ref else "bg_L50"
        if key in m and key in ref:
            dev = float(np.clip(_ev(ref[key], m[key]) * damp, -1.2, 1.2))
            crs["Exposure2012"] = round(float(np.clip(float(to_number(crs.get("Exposure2012")) or 0) + dev,
                                                      e0 - 2.5, e0 + 2.5)), 2)
        # 2) Bereiche relativ dazu: oben, Hintergrund, unten
        for corr, k in ((top, "top_L50"), (bg, "bg_L50"), (bottom, "bottom_L50")):
            if corr is not None and k in m and k in ref:
                d = float(np.clip(_ev(ref[k], m[k]) * damp, -1.0, 1.0))
                lo, hi = (-4.0, 1.0) if corr is bottom else (-2.5, 1.5)
                _set_local_ev(corr, float(np.clip(_local_ev(corr) + d, lo, hi)))
        # 3) Kontrast in den Personen (Spreizung hell/dunkel)
        if subject is not None and all(k in m and k in ref for k in ("subj_L10", "subj_L90")):
            spread_ref, spread = ref["subj_L90"] - ref["subj_L10"], m["subj_L90"] - m["subj_L10"]
            dc = float(np.clip((spread_ref - spread) * 1.2 * damp, -12, 12))
            crs["Contrast2012"] = int(round(np.clip(float(to_number(crs.get("Contrast2012")) or 0) + dc,
                                                    c0 - 30, min(c0 + 45, 80))))
        # 4) Weissabgleich: Weiss so neutral (bzw. so getönt) wie in deinen Bildern
        if as_temp and "n_a" in m and "n_b" in ref and "n_b" in m:
            wb.dm = float(np.clip(wb.dm + (m["n_b"] - ref["n_b"]) * 1.6 * damp, -45, 45))
            wb.dt = float(np.clip(wb.dt - (m["n_a"] - ref["n_a"]) * 1.2 * damp, -30, 30))
        # 5) Farbe des oberen Bildteils (z. B. Nachthimmel tiefblau statt violett/grau)
        if top is not None and "top_b" in m and "top_b" in ref:
            t = float(to_number(top.get("LocalTemperature")) or 0.0)
            top["LocalTemperature"] = float(np.clip(t + (ref["top_b"] - m["top_b"]) * 0.025 * damp, -0.6, 0.6))
            ti = float(to_number(top.get("LocalTint")) or 0.0)
            top["LocalTint"] = float(np.clip(ti + (ref.get("top_a", 0) - m.get("top_a", 0)) * 0.02 * damp,
                                             -0.5, 0.5))
        # 6) Sättigung je Farbe
        for hue in HUE_CENTERS:
            k = f"sat_{hue}"
            if k in m and k in ref and m[k] > 0.08 and ref[k] > 0.15:     # nur echte Farbflächen
                cur = float(to_number(crs.get(f"SaturationAdjustment{hue}")) or 0.0)
                d = float(np.clip((ref[k] / m[k] - 1) * 100 * 0.7 * damp, -15, 15))
                crs[f"SaturationAdjustment{hue}"] = int(round(np.clip(cur + d, -45, 50)))
    crs.update(wb.preview_keys())
    # 7) Weiss und Schwarz wie am Waveform deiner Bilder
    from .scopes import fit_scopes

    hi, lo = ref.get("y_hi"), ref.get("y_lo")
    notes = fit_scopes(crs, lin, orientation, subject, 1.0, hi_target=hi, lo_target=lo, free_blacks=True) \
        if hi is not None and lo is not None else []
    for k in ("WhiteBalance", "Temperature", "Tint"):
        crs.pop(k, None)
    crs.update(wb.final_keys() or wb_orig)
    if wb.final_keys():
        notes.append(f"Weissabgleich {wb.dm:+.0f} Mired / Tint {wb.dt:+.0f}")
    for corr, label in ((top, "oben"), (bg, "Hintergrund"), (bottom, "unten")):
        if corr is not None and abs(_local_ev(corr)) >= 0.1:
            notes.append(f"Look {label} {_local_ev(corr):+.1f} EV")
    # Leere Look-Masken nicht ins XMP schreiben
    crs["MaskGroupBasedCorrections"] = [
        c for c in crs.get("MaskGroupBasedCorrections") or []
        if not (str(c.get("CorrectionName", "")).startswith("Look:") and abs(_local_ev(c)) < 0.05
                and abs(float(to_number(c.get("LocalTemperature")) or 0)) < 0.02)]
    if not crs["MaskGroupBasedCorrections"]:
        crs.pop("MaskGroupBasedCorrections")
    return notes

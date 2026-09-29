"""Eigene Render-Pipeline: Annäherung an den Lightroom-Look für Vorschau und Galerie-JPEGs.

WICHTIG: Das ist eine Annäherung. Adobes Prozessversion, Profile (DCP/Looks) und die lokalen
Tonwert-Algorithmen sind nicht öffentlich. Farben und Kontraste weichen sichtbar ab; die UI
kennzeichnet die Vorschau entsprechend. Massgeblich ist immer Lightroom.

Reihenfolge: Weissabgleich -> Kamera->sRGB linear -> Belichtung -> lokale Tonwerte
(Lichter/Tiefen) -> Weiss/Schwarz -> Basiskurve + Kontrast -> Präsenz (Klarheit, Textur,
Dunst) -> Farbe (Dynamik, Sättigung, HSL, Color Grading) -> Gradationskurve -> lokale Masken
-> Vignette -> Drehen/Zuschneiden -> sRGB.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..io.color import XYZ_TO_SRGB, wb_multipliers
from ..lightroom.params import LOCAL_PARAMS, parse_curve, to_number
from ..vision.geometry import sensor_to_display

HUE_CENTERS = {"Red": 0, "Orange": 30, "Yellow": 60, "Green": 120, "Aqua": 180, "Blue": 240, "Purple": 275,
               "Magenta": 315}
BASE_GAIN = 2.0 ** 0.5   # grobe Entsprechung von BaselineExposure (Mittelgrau im RAW ~ 0.12)


def _n(crs: dict[str, Any], k: str, default: float = 0.0) -> float:
    v = to_number(crs.get(k))
    return default if v is None else float(v)


def cam_to_srgb(xyz_to_cam: np.ndarray) -> np.ndarray:
    srgb_to_cam = xyz_to_cam @ np.linalg.inv(XYZ_TO_SRGB)
    srgb_to_cam = srgb_to_cam / srgb_to_cam.sum(axis=1, keepdims=True)
    return np.linalg.inv(srgb_to_cam)


def _lum(x: np.ndarray) -> np.ndarray:
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), max(sigma, 0.1))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def base_curve(x: np.ndarray, contrast: float) -> np.ndarray:
    """Filmische Basiskurve (linear -> Anzeige 0..1) mit Kontrastregler."""
    k = 1.0 + contrast / 100.0 * 0.6
    white = 4.0
    y = x * (1 + x / (white * white)) / (1 + x) * 1.18   # erweiterter Reinhard, Mittelgrau bleibt ~0.18
    g = _srgb_encode(np.clip(y, 0, 1))
    # S-Kurve um Mittelgrau
    mid = 0.46
    return np.clip(mid + (g - mid) * k - (k - 1) * 0.35 * (g - mid) ** 3 / 0.25, 0, 1)


def _tone_local(lin: np.ndarray, crs: dict[str, Any], scale: float) -> np.ndarray:
    hl, sh = _n(crs, "Highlights2012") / 100, _n(crs, "Shadows2012") / 100
    L = np.maximum(_lum(lin), 1e-6)
    logL = np.log2(L)
    base = _blur(logL.astype(np.float32), 12 * scale)
    # Lichter: hohe Basis komprimieren; Tiefen: niedrige Basis anheben (in EV)
    w_hi = _smoothstep(-2.8, 0.5, base)
    w_lo = 1 - _smoothstep(-7.5, -3.0, base)
    delta = hl * 1.4 * w_hi + sh * 1.6 * w_lo
    return lin * np.power(2.0, delta)[..., None]


def _white_black(disp: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    """Weiss/Schwarz verschieben die Endpunkte (Anzeige-Raum): Weiss hebt/senkt vor allem das obere
    Drittel bis zum Anschlag, Schwarz das untere. Näherung an Lightroom (PV2012)."""
    wh, bl = _n(crs, "Whites2012") / 100, _n(crs, "Blacks2012") / 100
    if not wh and not bl:
        return disp
    Y = np.clip(_lum(disp), 1e-4, 1.0)
    Y2 = Y * (1 + (0.45 if wh > 0 else 0.3) * wh * Y ** 1.5)
    Y2 = Y2 + (0.25 if bl < 0 else 0.15) * bl * (1 - np.clip(Y2, 0, 1)) ** 3
    return disp * (np.maximum(Y2, 0) / Y)[..., None]


def _presence(disp: np.ndarray, crs: dict[str, Any], scale: float) -> np.ndarray:
    clar, tex, dehaze = _n(crs, "Clarity2012") / 100, _n(crs, "Texture") / 100, _n(crs, "Dehaze") / 100
    Y = _lum(disp).astype(np.float32)
    out = disp
    if abs(clar) > 1e-3:
        detail = Y - _blur(Y, 25 * scale)
        mid = 1 - np.abs(Y - 0.5) * 2
        out = out + (clar * 0.9 * detail * np.clip(mid, 0.2, 1))[..., None]
    if abs(tex) > 1e-3:
        detail = Y - _blur(Y, 2.5 * scale)
        out = out + (tex * 0.8 * detail)[..., None]
    if abs(dehaze) > 1e-3:
        dark = _blur(np.min(out, axis=-1).astype(np.float32), 30 * scale)
        out = (out - dehaze * 0.6 * dark[..., None]) / (1 - dehaze * 0.6 * np.clip(dark, 0, 0.9)[..., None])
        out = out + (out - _lum(out)[..., None]) * dehaze * 0.3
    return np.clip(out, 0, 1)


def _hsl_and_color(disp: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(disp.astype(np.float32), cv2.COLOR_RGB2HSV)   # H 0..360, S/V 0..1
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vib, sat = _n(crs, "Vibrance") / 100, _n(crs, "Saturation") / 100
    S = S * (1 + sat) * (1 + vib * (1 - S) * 1.2)
    dh = np.zeros_like(H)
    ds = np.ones_like(S)
    dv = np.ones_like(V)
    for name, center in HUE_CENTERS.items():
        d = np.abs(((H - center + 180) % 360) - 180)
        w = np.clip(1 - d / 40.0, 0, 1) ** 1.5 * np.clip(S * 3, 0, 1)
        dh += w * _n(crs, f"HueAdjustment{name}") * 0.3
        ds *= 1 + w * _n(crs, f"SaturationAdjustment{name}") / 100
        dv *= 1 + w * _n(crs, f"LuminanceAdjustment{name}") / 100 * 0.5
    hsv = np.stack([(H + dh) % 360, np.clip(S * ds, 0, 1), np.clip(V * dv, 0, 1)], -1)
    out = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    # Color Grading
    Y = _lum(out)
    bal = _n(crs, "SplitToningBalance") / 100
    blend = _n(crs, "ColorGradeBlending", 50) / 100
    w_sh = 1 - _smoothstep(0.1, 0.5 + bal * 0.3 + blend * 0.2, Y)
    w_hi = _smoothstep(0.5 + bal * 0.3 - blend * 0.2, 0.9, Y)
    w_mid = np.clip(1 - w_sh - w_hi, 0, 1)
    for hue_k, sat_k, lum_k, w in (("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum", w_sh),
                                   ("ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum", w_mid),
                                   ("SplitToningHighlightHue", "SplitToningHighlightSaturation",
                                    "ColorGradeHighlightLum", w_hi),
                                   ("ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum", 1.0)):
        s = _n(crs, sat_k) / 100
        lum = _n(crs, lum_k) / 100
        wv = np.asarray(w)[..., None] if np.ndim(w) else w
        if s > 0:
            tint = cv2.cvtColor(np.array([[[_n(crs, hue_k), 1.0, 1.0]]], np.float32), cv2.COLOR_HSV2RGB)[0, 0]
            out = out + wv * s * 0.35 * (tint - tint.mean())
        if lum:
            out = out * (1 + wv * lum * 0.4)
    return np.clip(out, 0, 1)


def _apply_curve(disp: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    out = disp
    pts = parse_curve(crs.get("ToneCurvePV2012"))
    if len(pts) >= 2:
        xs, ys = zip(*sorted(pts))
        lut = np.interp(np.arange(256), xs, ys) / 255.0
        out = lut[np.clip(out * 255, 0, 255).astype(np.uint8)]
    # parametrische Kurve
    p = [_n(crs, k) / 100 for k in ("ParametricShadows", "ParametricDarks", "ParametricLights",
                                   "ParametricHighlights")]
    if any(abs(v) > 1e-3 for v in p):
        x = out
        bumps = [(0.125, 0.12), (0.375, 0.14), (0.625, 0.14), (0.875, 0.12)]
        d = sum(v * 0.12 * np.exp(-((x - c) ** 2) / (2 * w * w)) for v, (c, w) in zip(p, bumps))
        out = np.clip(x + d, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Masken
# ---------------------------------------------------------------------------

def _component_mask(m: dict[str, Any], shape: tuple[int, int], orientation: int,
                    seg: dict[str, np.ndarray]) -> np.ndarray:
    h, w = shape
    what = m.get("What")
    mask = np.zeros((h, w), np.float32)
    if what == "Mask/Image":
        name = str(m.get("MaskName", "")).lower()
        key = "sky" if ("himmel" in name or "sky" in name) else "subject"
        src = seg.get(key)
        if src is not None:
            mask = cv2.resize(src.astype(np.float32), (w, h))
    elif what == "Mask/CircularGradient":
        (l, t), (r, b) = (sensor_to_display(_n(m, "Left"), _n(m, "Top"), orientation),
                          sensor_to_display(_n(m, "Right", 1), _n(m, "Bottom", 1), orientation))
        cx, cy = (l + r) / 2 * w, (t + b) / 2 * h
        rx, ry = max(abs(r - l) / 2 * w, 1), max(abs(b - t) / 2 * h, 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        feather = max(_n(m, "Feather", 50) / 100, 0.02)
        mask = 1 - _smoothstep(1 - feather, 1.0, d)
    elif what == "Mask/Gradient":
        zx, zy = sensor_to_display(_n(m, "ZeroX"), _n(m, "ZeroY"), orientation)
        fx, fy = sensor_to_display(_n(m, "FullX"), _n(m, "FullY"), orientation)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        vx, vy = (fx - zx) * w, (fy - zy) * h
        ln2 = max(vx * vx + vy * vy, 1e-6)
        t = ((xx - zx * w) * vx + (yy - zy * h) * vy) / ln2
        mask = np.clip(t, 0, 1).astype(np.float32)
    elif what == "Mask/Paint":
        r = max(int(_n(m, "Radius", 0.02) * max(h, w)), 1)
        for dab in m.get("Dabs", []) or []:
            parts = str(dab).split()
            if len(parts) >= 3:
                x, y = sensor_to_display(float(parts[1]), float(parts[2]), orientation)
                cv2.circle(mask, (int(x * w), int(y * h)), r, 1.0, -1)
        mask = _blur(mask, r * 0.3)
    if str(m.get("MaskInverted", "false")).lower() == "true":
        mask = 1 - mask
    return np.clip(mask, 0, 1)


def correction_mask(corr: dict[str, Any], shape: tuple[int, int], orientation: int,
                    seg: dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for m in corr.get("CorrectionMasks", []) or []:
        if not isinstance(m, dict) or str(m.get("MaskActive", "true")).lower() == "false":
            continue
        cm = _component_mask(m, shape, orientation, seg)
        mode = int(_n(m, "MaskBlendMode", 0))
        if total is None:
            total = cm
        elif mode == 1:          # subtrahieren
            total = total * (1 - cm)
        elif mode == 2:          # schneiden
            total = total * cm
        else:                    # addieren
            total = np.maximum(total, cm)
    return np.zeros(shape, np.float32) if total is None else total * _n(corr, "CorrectionAmount", 1.0)


def _apply_local(lin: np.ndarray, corrections: list[dict[str, Any]], orientation: int,
                 seg: dict[str, np.ndarray], scale: float) -> np.ndarray:
    for corr in corrections:
        if not isinstance(corr, dict) or str(corr.get("CorrectionActive", "true")).lower() == "false":
            continue
        m = correction_mask(corr, lin.shape[:2], orientation, seg)[..., None]
        if m.max() <= 0:
            continue
        ev = _n(corr, "LocalExposure2012") * LOCAL_PARAMS["LocalExposure2012"]
        lin = lin * np.power(2.0, ev * m)
        temp, tint = _n(corr, "LocalTemperature"), _n(corr, "LocalTint")
        if temp or tint:
            gain = np.array([1 + 0.25 * temp, 1 - 0.2 * tint, 1 - 0.25 * temp], np.float32)
            lin = lin * (1 + (gain - 1) * m)
        sat = _n(corr, "LocalSaturation")
        if sat:
            L = _lum(lin)[..., None]
            lin = L + (lin - L) * (1 + sat * m)
        con = _n(corr, "LocalContrast2012") + 0.5 * _n(corr, "LocalClarity2012")
        if con:
            logL = np.log2(np.maximum(_lum(lin), 1e-6)).astype(np.float32)
            base = _blur(logL, 20 * scale)
            lin = lin * np.power(2.0, (logL - base) * con * 0.6 * m[..., 0])[..., None]
        hl, sh = _n(corr, "LocalHighlights2012"), _n(corr, "LocalShadows2012")
        if hl or sh:
            logL = np.log2(np.maximum(_lum(lin), 1e-6))
            d = hl * 1.2 * _smoothstep(-2.8, 0.5, logL) + sh * 1.4 * (1 - _smoothstep(-7.5, -3.0, logL))
            lin = lin * np.power(2.0, d * m[..., 0])[..., None]
        soft = -min(0.0, _n(corr, "LocalSharpness")) + 0.5 * -min(0.0, _n(corr, "LocalTexture"))
        if soft:
            # negative Schärfe/Struktur: weichzeichnen (Vorschau-Näherung an Lightroom)
            lin = lin + (_blur(lin, 3.0 * scale) - lin) * np.clip(soft, 0, 1) * m
    return lin


# ---------------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------------

def crop_rect(crs: dict[str, Any], orientation: int, W: int, H: int) -> tuple[float, tuple[float, ...]] | None:
    """-> (Winkel, (x0, y0, x1, y1) im gedrehten Bild in Pixel) oder None."""
    angle = _n(crs, "CropAngle")
    if orientation in (2, 4, 5, 7):
        angle = -angle
    if str(crs.get("HasCrop", "False")).lower() != "true":
        return (angle, (0, 0, W, H)) if abs(angle) > 1e-3 else None
    a = sensor_to_display(_n(crs, "CropLeft"), _n(crs, "CropTop"), orientation)
    b = sensor_to_display(_n(crs, "CropRight", 1), _n(crs, "CropBottom", 1), orientation)
    th = math.radians(angle)
    pts = []
    for x, y in (a, b):
        px, py = x * W - W / 2, y * H - H / 2
        pts.append((px * math.cos(th) - py * math.sin(th) + W / 2, px * math.sin(th) + py * math.cos(th) + H / 2))
    x0, x1 = sorted([pts[0][0], pts[1][0]])
    y0, y1 = sorted([pts[0][1], pts[1][1]])
    return angle, (x0, y0, x1, y1)


def _geometry(img: np.ndarray, crs: dict[str, Any], orientation: int) -> np.ndarray:
    H, W = img.shape[:2]
    cr = crop_rect(crs, orientation, W, H)
    if cr is None:
        return img
    angle, (x0, y0, x1, y1) = cr
    if abs(angle) > 1e-3:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle, 1.0)   # OpenCV: positiv = gegen Uhrzeigersinn
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img
    return img[y0:y1, x0:x1]


def _vignette(disp: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    amt = _n(crs, "PostCropVignetteAmount") / 100
    if abs(amt) < 1e-3:
        return disp
    h, w = disp.shape[:2]
    mid = _n(crs, "PostCropVignetteMidpoint", 50) / 100
    feather = max(_n(crs, "PostCropVignetteFeather", 50) / 100, 0.05)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    wgt = _smoothstep(mid * 0.9, mid * 0.9 + feather, d)[..., None]
    return np.clip(disp * (1 + amt * 0.8 * wgt), 0, 1)


# ---------------------------------------------------------------------------

def render(lin_cam: np.ndarray, xyz_to_cam: np.ndarray, camera_wb: np.ndarray, crs: dict[str, Any],
           orientation: int = 1, seg: dict[str, np.ndarray] | None = None, max_side: int | None = 1600
           ) -> np.ndarray:
    """lin_cam: lineares Kamera-RGB (Anzeigeorientierung, ohne WB). Rückgabe: sRGB uint8."""
    seg = seg or {}
    img = lin_cam.astype(np.float32)
    if max_side:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    scale = max(img.shape[:2]) / 1600
    if str(crs.get("WhiteBalance", "As Shot")) == "As Shot" or to_number(crs.get("Temperature")) is None:
        mult = np.asarray(camera_wb[:3], dtype=np.float32)
    else:
        mult = wb_multipliers(xyz_to_cam, _n(crs, "Temperature", 5500), _n(crs, "Tint")).astype(np.float32)
    mult = mult / mult[1]
    img = img * mult[None, None, :]
    img = np.clip(img @ cam_to_srgb(xyz_to_cam).T.astype(np.float32), 0, None)
    img = img * (BASE_GAIN * 2.0 ** _n(crs, "Exposure2012"))
    img = _tone_local(img, crs, scale)
    img = _apply_local(img, crs.get("MaskGroupBasedCorrections") or [], orientation, seg, scale)
    disp = base_curve(img, _n(crs, "Contrast2012")).astype(np.float32)
    disp = _white_black(disp, crs)
    disp = _presence(disp, crs, scale)
    disp = _hsl_and_color(disp, crs)
    disp = _apply_curve(disp, crs)
    disp = _vignette(disp, crs)
    disp = _geometry(disp, crs, orientation)
    return (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)

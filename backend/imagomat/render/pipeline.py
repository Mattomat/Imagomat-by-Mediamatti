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
from ..vision.geometry import CROP_ANGLE_SIGN, sensor_to_display

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


AS_SHOT_TEMP, AS_SHOT_TINT = "_AsShotTemp", "_AsShotTint"     # nur intern beim Rendern, nie im XMP


def _wb_mult(xyz_to_cam: np.ndarray, camera_wb: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    """Weissabgleich-Faktoren. Eigene Temperatur/Tint werden RELATIV zur Kamera-Einstellung umgesetzt: so führt
    eine ungenaue Umrechnung Temperatur -> Kamerafaktoren nicht zu einem Farbstich."""
    base = np.asarray(camera_wb[:3], dtype=np.float32)
    base = base / max(float(base[1]), 1e-6)
    if str(crs.get("WhiteBalance", "As Shot")) == "As Shot" or to_number(crs.get("Temperature")) is None:
        return base
    try:
        from ..io.color import multipliers_to_temp_tint

        if to_number(crs.get(AS_SHOT_TEMP)):
            # Daten mit schon angewendetem Kamera-Weissabgleich (Apple RAW-Engine, DNG Converter, Vorschau):
            # Ausgangspunkt ist der Weissabgleich der Aufnahme, nicht D65
            t0, tint0 = _n(crs, AS_SHOT_TEMP), _n(crs, AS_SHOT_TINT)
        else:
            t0, tint0 = multipliers_to_temp_tint(xyz_to_cam, base)
        rel = wb_multipliers(xyz_to_cam, _n(crs, "Temperature", 5500), _n(crs, "Tint")) / \
            wb_multipliers(xyz_to_cam, t0, tint0)
        m = (base * rel).astype(np.float32)
        return m / max(float(m[1]), 1e-6)
    except Exception:  # noqa: BLE001
        return base


def _black_floor(lin: np.ndarray) -> np.ndarray:
    """Rausch-Sockel abziehen. Der RAW-Leser schneidet negative Rauschwerte am Schwarzpunkt ab; in dunklen
    Flächen (Nachthimmel) bleibt dadurch pro Kanal ein positiver Sockel, den Weissabgleich und Farbmatrix zu
    einem kräftigen Farbschleier (meist blau) verstärken. Lightroom rechnet ohne dieses Abschneiden.
    Geschätzt wird der Sockel aus den dunkelsten Flächen eines geglätteten Bildes (Rauschen gemittelt)."""
    small = cv2.blur(lin.astype(np.float32), (9, 9))
    flat = small.reshape(-1, lin.shape[-1])
    step = max(1, flat.shape[0] // 200_000)
    floor = np.clip(np.percentile(flat[::step], 0.5, axis=0), 0, 0.01).astype(np.float32)
    return lin - floor                       # bewusst ohne Abschneiden: Rauschen bleibt mittelwertfrei


def _calm_shadows(lin: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Farbrauschen entfernen, solange die Werte noch nicht abgeschnitten sind: Farbanteil glätten und in tiefen
    Schatten ganz entsättigen (dort ist Farbe fast nur Rauschen; Lightroom hält Schwarz neutral)."""
    L = _lum(lin)[..., None]
    chroma = lin - L
    sigma = max(1.0, 2.0 * scale)
    chroma = np.stack([cv2.GaussianBlur(chroma[..., c], (0, 0), sigma) for c in range(chroma.shape[-1])], -1)
    Ls = cv2.GaussianBlur(L[..., 0], (0, 0), sigma)[..., None]
    w = _smoothstep(0.006, 0.05, Ls)
    return L + chroma * w


def _chroma_nr(disp: np.ndarray, crs: dict[str, Any], scale: float) -> np.ndarray:
    """Farbrauschen entfernen (Lightroom: Farbe-Rauschreduzierung, Standard 25)."""
    amt = to_number(crs.get("ColorNoiseReduction"))
    amt = 25.0 if amt is None else float(amt)
    if amt <= 0:
        return disp
    ycc = cv2.cvtColor(np.clip(disp, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    sigma = max(0.6, (0.6 + amt / 25.0) * scale * 1.6)
    for c in (1, 2):
        ycc[..., c] = cv2.GaussianBlur(ycc[..., c], (0, 0), sigma)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def denoise_amount(crs: dict[str, Any]) -> float:
    """Entrausch-Stärke (0..100) aus den Lightroom-Feldern (Schlüssel je nach Version verschieden)."""
    for k, v in crs.items():
        if "Denoise" in k and "Amount" in k:
            x = to_number(v)
            if x is not None:
                return float(x)
    return 0.0


def _denoise(lin: np.ndarray, crs: dict[str, Any]) -> np.ndarray:
    """Vorschau wie nach "KI-Einstellungen aktualisieren": Rauschen entsprechend der Denoise-Stärke entfernen
    (klassisches Verfahren, schnell genug für die Vorschau)."""
    amt = denoise_amount(crs)
    if amt <= 0 or min(lin.shape[:2]) < 16:
        return lin
    from ..denoise.local import classical

    k = max(float(np.percentile(lin, 99.9)), 1e-3)
    g = np.power(np.clip(lin / k, 0, 1), 1 / 2.2).astype(np.float32)
    # Stärke nach gemessenem Rauschen (robust: Median des Hochpasses) und gewünschter Denoise-Stärke
    y = g.mean(axis=2)
    hp = y - cv2.blur(y, (3, 3))
    sigma = float(np.median(np.abs(hp))) / 0.6745 * 1.5
    out = classical(g, max(0.004 + 0.022 * min(amt, 100) / 100, sigma * (0.3 + 0.4 * min(amt, 100) / 100)))
    return (np.power(np.clip(out, 0, 1), 2.2) * k).astype(np.float32)


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
        wh, bl = _n(corr, "LocalWhites2012"), _n(corr, "LocalBlacks2012")
        if wh or bl:
            # Weiss: hellste Töne (z. B. Trikots in der Spieler-Maske), Schwarz: tiefste Töne
            logL = np.log2(np.maximum(_lum(lin), 1e-6))
            d = wh * 0.9 * _smoothstep(-3.0, 0.0, logL) + bl * 1.0 * (1 - _smoothstep(-8.5, -4.5, logL))
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
    angle = _n(crs, "CropAngle") * CROP_ANGLE_SIGN     # Lightroom-Wert -> intern (positiv = im Uhrzeigersinn)
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
    mult = _wb_mult(xyz_to_cam, camera_wb, crs)
    clipped = _smoothstep(0.90, 0.99, img.max(axis=-1))[..., None]   # Sensor gesättigt (vor dem Weissabgleich)
    # Ohne Zwischen-Abschneiden bis das Farbrauschen weg ist (sonst entsteht ein Farbsockel in dunklen Flächen)
    img = _black_floor(img)
    img = img * mult[None, None, :]
    # Ausgefressene Lichter (Flutlicht, LED-Tafel) neutral weiss statt magenta: ist ein Kanal gesättigt, sind
    # die anderen nach dem Weissabgleich zu stark -> auf den hellsten gemeinsamen Wert ziehen
    if clipped.max() > 0:
        neutral = np.repeat(np.minimum(img, 1.0).max(axis=-1, keepdims=True), 3, axis=-1)
        img = img * (1 - clipped) + neutral * clipped
    img = img @ cam_to_srgb(xyz_to_cam).T.astype(np.float32)
    img = img * (BASE_GAIN * 2.0 ** _n(crs, "Exposure2012"))
    img = _calm_shadows(img, scale)
    img = np.clip(img, 0, None)
    img = _denoise(img, crs)
    img = _tone_local(img, crs, scale)
    img = _apply_local(img, crs.get("MaskGroupBasedCorrections") or [], orientation, seg, scale)
    disp = base_curve(img, _n(crs, "Contrast2012")).astype(np.float32)
    disp = _white_black(disp, crs)
    disp = _presence(disp, crs, scale)
    disp = _chroma_nr(disp, crs, scale)
    disp = _hsl_and_color(disp, crs)
    disp = _apply_curve(disp, crs)
    disp = _vignette(disp, crs)
    disp = _geometry(disp, crs, orientation)
    return (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Saubere Vorschau: Details aus dem Kamera-JPEG, Licht und Farbe aus der RAW-Entwicklung
# ---------------------------------------------------------------------------

RENDER_VERSION = "h3"          # ändern, wenn die Vorschau anders aussieht: alte Zwischenspeicher verfallen
HYBRID_LO_SIDE = 560          # Auflösung der RAW-Entwicklung für Licht/Farbe (rauscht dort kaum)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(x, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Kantenerhaltendes Glätten (He et al.): ``src`` folgt den Kanten von ``guide`` (keine Lichthöfe).
    Mehrkanalig: Kanal c von ``src`` folgt Kanal c von ``guide`` (trennt z. B. Rot von Grün gleicher Helligkeit)."""
    out = []
    for c in range(src.shape[-1]):
        g = guide[..., c] if guide.ndim == 3 else guide
        p = src[..., c]
        mi, mp = _box(g, r), _box(p, r)
        a = (_box(g * p, r) - mi * mp) / (_box(g * g, r) - mi * mi + eps)
        b = mp - a * mi
        out.append(_box(a, r) * g + _box(b, r))
    return np.stack(out, -1)


def _soften_local(img: np.ndarray, corrections: list[dict[str, Any]], orientation: int,
                  seg: dict[str, np.ndarray], scale: float) -> np.ndarray:
    """Negative Schärfe/Struktur/Klarheit in Masken (z. B. Verlauf unten) als echte Unschärfe."""
    for corr in corrections:
        if not isinstance(corr, dict) or str(corr.get("CorrectionActive", "true")).lower() == "false":
            continue
        soft = (-min(0.0, _n(corr, "LocalSharpness")) + 0.5 * -min(0.0, _n(corr, "LocalTexture"))
                + 0.3 * -min(0.0, _n(corr, "LocalClarity2012")))
        if soft <= 0:
            continue
        m = correction_mask(corr, img.shape[:2], orientation, seg)[..., None]
        if m.max() <= 0:
            continue
        img = img + (_blur(img, 3.5 * scale) - img) * np.clip(soft * 1.2, 0, 1) * m
    return img


def _match_curves(src: np.ndarray, dst: np.ndarray, bins: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    """Monotone Kurve pro Kanal (log-Werte), die ``src`` (Kamera-JPEG) auf ``dst`` (RAW-Entwicklung) abbildet."""
    out = []
    for c in range(src.shape[-1]):
        x, y = src[..., c].ravel(), dst[..., c].ravel()
        qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(qs) < 3:
            out.append((np.array([x.min(), x.max() + 1e-3]), np.array([0.0, 0.0])))
            continue
        idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        xs, ys = [], []
        for i in range(len(qs) - 1):
            sel = idx == i
            if sel.sum() >= 8:
                xs.append(float(np.median(x[sel])))
                ys.append(float(np.median(y[sel] - x[sel])))       # Versatz statt Wert: glatter
        if len(xs) < 2:
            out.append((np.array([x.min(), x.max() + 1e-3]), np.full(2, float(np.median(y - x)))))
            continue
        xs_a, ys_a = np.array(xs), np.array(ys)
        # monoton halten (Ausgabe x + Versatz darf nicht fallen), leicht glätten
        ys_a = np.convolve(np.pad(ys_a, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        v = xs_a + ys_a
        # Steigung begrenzen: wo das JPEG die Tiefen abgeschnitten hat, würde eine steile Kurve nur dessen
        # Rauschen und Blockartefakte verstärken (Rest gleicht das grossflächige Verhältnis aus)
        dx = np.maximum(np.diff(xs_a), 1e-6)
        dv = np.clip(np.diff(v), 0.3 * dx, 1.6 * dx)
        mid = len(v) // 2
        v2 = np.empty_like(v)
        v2[mid] = v[mid]
        v2[mid + 1:] = v[mid] + np.cumsum(dv[mid:])
        v2[:mid] = v[mid] - np.cumsum(dv[:mid][::-1])[::-1]
        out.append((xs_a, v2 - xs_a))
    return out


def _apply_curves(x: np.ndarray, curves: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.stack([x[..., c] + np.interp(x[..., c], xs, off).astype(np.float32)
                     for c, (xs, off) in enumerate(curves)], -1).astype(np.float32)


def render_hybrid(lin_cam: np.ndarray, xyz_to_cam: np.ndarray, camera_wb: np.ndarray, crs: dict[str, Any],
                  detail: np.ndarray | None, orientation: int = 1, seg: dict[str, np.ndarray] | None = None,
                  max_side: int | None = 1600) -> np.ndarray:
    """Vorschau ohne RAW-Rauschen: Die Bildstruktur stammt aus der (in der Kamera entrauschten und geschärften)
    eingebetteten JPEG-Vorschau ``detail`` (RGB uint8, Anzeigeorientierung). Helligkeit, Weissabgleich, Farben,
    Masken und Verläufe stammen aus der RAW-Entwicklung, übertragen als kantenerhaltendes, grossflächiges
    Verhältnis. Ohne passende Vorschau: normale RAW-Entwicklung."""
    seg = seg or {}
    if detail is None or detail.ndim != 3:
        return render(lin_cam, xyz_to_cam, camera_wb, crs, orientation, seg, max_side)
    h, w = lin_cam.shape[:2]
    dh, dw = detail.shape[:2]
    if abs(math.log((dw / dh) / (w / h))) > 0.03:        # anderes Seitenverhältnis: nicht dasselbe Bild
        return render(lin_cam, xyz_to_cam, camera_wb, crs, orientation, seg, max_side)
    flat = {**crs, "HasCrop": "False", "CropAngle": 0}
    side = min(max_side or max(dh, dw), max(dh, dw))
    s = side / max(dh, dw)
    det = detail if s >= 1 else cv2.resize(detail, (int(dw * s), int(dh * s)), interpolation=cv2.INTER_AREA)
    det = det.astype(np.float32) / 255.0
    lo = render(lin_cam, xyz_to_cam, camera_wb, flat, orientation, seg, min(HYBRID_LO_SIDE, side))
    lo = _srgb_decode(lo.astype(np.float32) / 255.0)
    det_lin = _srgb_decode(det)
    lo_det = cv2.resize(det_lin, (lo.shape[1], lo.shape[0]), interpolation=cv2.INTER_AREA)
    eps = 0.004
    # Kamera-JPEG hat eigene Tonkurve/Farbe (Sony: steiler, DRO, satter). Erst punktweise pro Kanal auf die
    # RAW-Entwicklung abbilden, sonst hängt das Verhältnis von der Helligkeit ab und erzeugt Lichthöfe/Flecken.
    curves = _match_curves(np.log(lo_det + eps), np.log(lo + eps))
    det_log = _apply_curves(np.log(det_lin + eps), curves)
    lo_log = _apply_curves(np.log(lo_det + eps), curves)
    det_lin = np.exp(det_log) - eps
    logr = (np.log(lo + eps) - lo_log).astype(np.float32)
    logr = np.clip(cv2.GaussianBlur(logr, (0, 0), 0.7), -3.5, 3.5)
    H, W = det.shape[:2]
    logr = cv2.resize(logr, (W, H), interpolation=cv2.INTER_LINEAR)
    guide = det.astype(np.float32)
    r = max(4, int(round(max(H, W) / 150)))
    logr = guided_filter(guide, logr, r, 4e-4)
    out = (det_lin + eps) * np.exp(logr) - eps      # aufgehellte Tiefen kommen auch aus reinem Schwarz zurück
    scale = max(H, W) / 1600
    out = _soften_local(out, crs.get("MaskGroupBasedCorrections") or [], orientation, seg, scale)
    disp = _srgb_encode(np.clip(out, 0, 1))
    disp = _geometry(disp, crs, orientation)
    return (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)

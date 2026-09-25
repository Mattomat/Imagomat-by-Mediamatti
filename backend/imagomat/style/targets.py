"""Lernziele: Lightroom-Einstellungen <-> Zahlenvektor.

Besonderheiten:
- Weissabgleich relativ zum As-Shot-Wert (Mired- und Tint-Differenz).
- Farbtöne (Color Grading) als Vektor (Sättigung * cos/sin), damit 359° und 1° nah beieinander liegen.
- Punktkurven werden an festen Stellen abgetastet (Differenz zur Diagonalen).
- Denoise-Stärke über den gelernten Dialekt-Schlüssel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..io.color import from_mired, mired
from ..lightroom.params import PARAMS, format_curve, parse_curve, to_number

CURVE_X = [32, 64, 96, 128, 160, 192, 224]

DIRECT = [
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
    "Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation",
    "ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights",
    "ColorGradeShadowLum", "ColorGradeMidtoneLum", "ColorGradeHighlightLum", "ColorGradeGlobalLum",
    "ColorGradeBlending", "SplitToningBalance",
    "Sharpness", "SharpenRadius", "SharpenDetail", "SharpenEdgeMasking", "LuminanceSmoothing",
    "ColorNoiseReduction", "PostCropVignetteAmount", "PostCropVignetteMidpoint", "PostCropVignetteFeather",
    "GrainAmount", "ShadowTint", "RedHue", "RedSaturation", "GreenHue", "GreenSaturation", "BlueHue",
    "BlueSaturation",
] + [f"{kind}Adjustment{c}" for kind in ("Hue", "Saturation", "Luminance")
     for c in ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")]

WHEELS = [("SplitToningShadowHue", "SplitToningShadowSaturation", "grade_shadow"),
          ("ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "grade_mid"),
          ("SplitToningHighlightHue", "SplitToningHighlightSaturation", "grade_high"),
          ("ColorGradeGlobalHue", "ColorGradeGlobalSat", "grade_global")]

TARGETS: list[str] = (
    ["wb_dmired", "wb_dtint", "wb_custom"] + DIRECT
    + [f"{p}_{ax}" for _, _, p in WHEELS for ax in ("a", "b")]
    + [f"curve_{x}" for x in CURVE_X]
    + ["denoise", "crop_area", "crop_angle", "has_crop"]
)


def _num(crs: dict[str, Any], k: str, default: float | None = None) -> float:
    v = to_number(crs.get(k))
    if v is None:
        return PARAMS[k].default if k in PARAMS else (default if default is not None else 0.0)
    return v


def curve_samples(points: list[tuple[float, float]]) -> list[float]:
    if len(points) < 2:
        return [0.0] * len(CURVE_X)
    xs, ys = zip(*sorted(points))
    return [float(np.interp(x, xs, ys) - x) for x in CURVE_X]


def encode(crs: dict[str, Any], as_shot_temp: float | None, as_shot_tint: float | None,
           denoise_key: str | None = None) -> dict[str, float]:
    """crs-Dict -> Zielwerte (fehlende Werte = Lightroom-Standard)."""
    t: dict[str, float] = {}
    wbmode = str(crs.get("WhiteBalance", "As Shot"))
    temp, tint = to_number(crs.get("Temperature")), to_number(crs.get("Tint"))
    if wbmode != "As Shot" and temp and as_shot_temp:
        t["wb_dmired"] = mired(temp) - mired(as_shot_temp)
        t["wb_dtint"] = (tint or 0.0) - (as_shot_tint or 0.0)
        t["wb_custom"] = 1.0
    else:
        t["wb_dmired"], t["wb_dtint"], t["wb_custom"] = 0.0, 0.0, 0.0
    for k in DIRECT:
        t[k] = _num(crs, k)
    for hue_k, sat_k, name in WHEELS:
        h, s = _num(crs, hue_k), _num(crs, sat_k)
        t[f"{name}_a"] = s * math.cos(math.radians(h))
        t[f"{name}_b"] = s * math.sin(math.radians(h))
    for x, v in zip(CURVE_X, curve_samples(parse_curve(crs.get("ToneCurvePV2012")))):
        t[f"curve_{x}"] = v
    t["denoise"] = 0.0
    if denoise_key and to_number(crs.get(denoise_key)) is not None:
        t["denoise"] = float(to_number(crs.get(denoise_key)) or 0.0)
    from ..vision.geometry import from_lightroom_crop

    crop = from_lightroom_crop(crs) or {}
    t["crop_area"] = crop.get("crop_area", 1.0)
    t["crop_angle"] = crop.get("crop_angle", 0.0)
    t["has_crop"] = float(str(crs.get("HasCrop", "False")).lower() == "true")
    return t


def decode(t: dict[str, float], as_shot_temp: float | None, as_shot_tint: float | None) -> dict[str, Any]:
    """Zielwerte -> crs-Dict (ohne Zuschnitt/Denoise/Masken, die separat entstehen)."""
    crs: dict[str, Any] = {}
    if t.get("wb_custom", 0) >= 0.5 and as_shot_temp:
        crs["WhiteBalance"] = "Custom"
        crs["Temperature"] = PARAMS["Temperature"].clamp(from_mired(mired(as_shot_temp) + t.get("wb_dmired", 0)))
        crs["Tint"] = PARAMS["Tint"].clamp((as_shot_tint or 0) + t.get("wb_dtint", 0))
    else:
        crs["WhiteBalance"] = "As Shot"
    for k in DIRECT:
        if k in t:
            crs[k] = PARAMS[k].clamp(t[k]) if k in PARAMS else t[k]
    for hue_k, sat_k, name in WHEELS:
        a, b = t.get(f"{name}_a", 0.0), t.get(f"{name}_b", 0.0)
        sat = math.hypot(a, b)
        if sat >= 0.5:
            crs[hue_k] = round(math.degrees(math.atan2(b, a)) % 360) % 360
            crs[sat_k] = min(100, round(sat))
        else:
            crs[hue_k], crs[sat_k] = 0, 0
    samples = [t.get(f"curve_{x}", 0.0) for x in CURVE_X]
    if any(abs(v) >= 1.0 for v in samples):
        pts = [(0.0, max(0.0, samples[0] * 0.25))] + [(x, min(255, max(0, x + v))) for x, v in zip(CURVE_X, samples)]
        pts.append((255.0, 255.0))
        # monoton halten
        ys = np.maximum.accumulate([p[1] for p in pts])
        crs["ToneCurvePV2012"] = format_curve([(x, y) for (x, _), y in zip(pts, ys)])
        crs["ToneCurveName2012"] = "Custom"
    else:
        crs["ToneCurvePV2012"] = format_curve([(0, 0), (255, 255)])
        crs["ToneCurveName2012"] = "Linear"
    return crs

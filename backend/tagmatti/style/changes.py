"""Bearbeitung in Worten: welche Regler und Masken ein Bild konkret verändern (für die Anzeige in der App)."""

from __future__ import annotations

from typing import Any

from ..lightroom.params import LOCAL_PARAMS, to_number

GLOBAL = [
    ("Exposure2012", "Belichtung", "{:+.2f}"), ("Contrast2012", "Kontrast", "{:+.0f}"),
    ("Highlights2012", "Lichter", "{:+.0f}"), ("Shadows2012", "Tiefen", "{:+.0f}"),
    ("Whites2012", "Weiss", "{:+.0f}"), ("Blacks2012", "Schwarz", "{:+.0f}"),
    ("Texture", "Struktur", "{:+.0f}"), ("Clarity2012", "Klarheit", "{:+.0f}"), ("Dehaze", "Dunst", "{:+.0f}"),
    ("Vibrance", "Dynamik", "{:+.0f}"), ("Saturation", "Sättigung", "{:+.0f}"),
    ("PostCropVignetteAmount", "Vignette", "{:+.0f}"),
]
HSL_COLORS = {"Red": "Rot", "Orange": "Orange", "Yellow": "Gelb", "Green": "Grün", "Aqua": "Aqua", "Blue": "Blau",
              "Purple": "Lila", "Magenta": "Magenta"}
LOCAL = [("LocalExposure2012", "Belichtung", 4.0, "{:+.1f} EV"), ("LocalContrast2012", "Kontrast", 100, "{:+.0f}"),
         ("LocalHighlights2012", "Lichter", 100, "{:+.0f}"), ("LocalShadows2012", "Tiefen", 100, "{:+.0f}"),
         ("LocalWhites2012", "Weiss", 100, "{:+.0f}"), ("LocalBlacks2012", "Schwarz", 100, "{:+.0f}"),
         ("LocalClarity2012", "Klarheit", 100, "{:+.0f}"), ("LocalTexture", "Struktur", 100, "{:+.0f}"),
         ("LocalSharpness", "Schärfe", 100, "{:+.0f}"), ("LocalSaturation", "Sättigung", 100, "{:+.0f}"),
         ("LocalTemperature", "Temperatur", 100, "{:+.0f}"), ("LocalDehaze", "Dunst", 100, "{:+.0f}")]


def _num(d: dict[str, Any], k: str) -> float:
    return float(to_number(d.get(k)) or 0.0)


def describe(crs: dict[str, Any], as_shot_temp: float | None = None,
             as_shot_tint: float | None = None) -> dict[str, Any]:
    sliders: list[dict[str, Any]] = []
    wb = str(crs.get("WhiteBalance", "As Shot"))
    if wb != "As Shot" and to_number(crs.get("Temperature")):
        t, tint = _num(crs, "Temperature"), _num(crs, "Tint")
        txt = f"{t:.0f} K / Tint {tint:+.0f}"
        if as_shot_temp:
            txt += f" (Kamera {float(as_shot_temp):.0f} K / {float(as_shot_tint or 0):+.0f})"
        sliders.append({"label": "Weissabgleich", "value": txt})
    else:
        sliders.append({"label": "Weissabgleich", "value": "wie aufgenommen"})
    for k, label, fmt in GLOBAL:
        v = _num(crs, k)
        if abs(v) >= (0.01 if k == "Exposure2012" else 1):
            sliders.append({"label": label, "value": fmt.format(v), "key": k, "num": v})
    hsl = []
    for c, name in HSL_COLORS.items():
        parts = []
        for pre, short in (("HueAdjustment", "Farbton"), ("SaturationAdjustment", "Sättigung"),
                           ("LuminanceAdjustment", "Luminanz")):
            v = _num(crs, f"{pre}{c}")
            if abs(v) >= 1:
                parts.append(f"{short} {v:+.0f}")
        if parts:
            hsl.append(f"{name}: " + ", ".join(parts))
    if hsl:
        sliders.append({"label": "HSL", "value": " · ".join(hsl)})
    if str(crs.get("HasCrop", "False")).lower() == "true" or abs(_num(crs, "CropAngle")) > 0.05:
        a = _num(crs, "CropAngle")
        sliders.append({"label": "Zuschnitt", "value": f"gedreht {a:+.1f}°" if abs(a) > 0.05 else "zugeschnitten"})
    masks = []
    for corr in crs.get("MaskGroupBasedCorrections") or []:
        if not isinstance(corr, dict):
            continue
        eff = []
        for k, label, scale, fmt in LOCAL:
            v = _num(corr, k) * (scale if k == "LocalExposure2012" else LOCAL_PARAMS.get(k, 100.0))
            if abs(v) >= (0.05 if k == "LocalExposure2012" else 1):
                eff.append(f"{label} {fmt.format(v)}")
        if eff:
            masks.append({"name": str(corr.get("CorrectionName") or "Maske"), "effects": eff})
    return {"sliders": sliders, "masks": masks}

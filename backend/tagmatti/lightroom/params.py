"""Spezifikation der Lightroom-Entwicklungsparameter (Prozessversion 2012 und neuer).

Die Namen entsprechen den XMP-Namen im Namespace ``crs`` bzw. den Schlüsseln der
Lua-Tabelle im Lightroom-Katalog. Die Formatierung (Vorzeichen, Dezimalstellen) bildet
nach, was Lightroom selbst schreibt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    decimals: int = 0
    signed: bool = True
    group: str = "basic"

    def clamp(self, v: float) -> float:
        return float(min(self.hi, max(self.lo, v)))

    def fmt(self, v: float) -> str:
        v = self.clamp(v)
        if self.decimals == 0:
            iv = int(round(v))
            return f"{iv:+d}" if self.signed and iv != 0 else str(iv)
        s = f"{v:.{self.decimals}f}"
        if self.signed and v > 0:
            s = "+" + s
        return s


def _p(name: str, lo: float, hi: float, default: float = 0, decimals: int = 0, signed: bool = True,
       group: str = "basic") -> Param:
    return Param(name, lo, hi, default, decimals, signed, group)


HSL_COLORS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

_PARAMS: list[Param] = [
    # Weissabgleich
    _p("Temperature", 2000, 50000, 5500, signed=False, group="wb"),
    _p("Tint", -150, 150, 0, group="wb"),
    # Grundeinstellungen
    _p("Exposure2012", -5, 5, 0, decimals=2),
    _p("Contrast2012", -100, 100),
    _p("Highlights2012", -100, 100),
    _p("Shadows2012", -100, 100),
    _p("Whites2012", -100, 100),
    _p("Blacks2012", -100, 100),
    _p("Texture", -100, 100, group="presence"),
    _p("Clarity2012", -100, 100, group="presence"),
    _p("Dehaze", -100, 100, group="presence"),
    _p("Vibrance", -100, 100, group="presence"),
    _p("Saturation", -100, 100, group="presence"),
    # Parametrische Kurve
    _p("ParametricShadows", -100, 100, group="curve"),
    _p("ParametricDarks", -100, 100, group="curve"),
    _p("ParametricLights", -100, 100, group="curve"),
    _p("ParametricHighlights", -100, 100, group="curve"),
    _p("ParametricShadowSplit", 10, 70, 25, signed=False, group="curve"),
    _p("ParametricMidtoneSplit", 20, 80, 50, signed=False, group="curve"),
    _p("ParametricHighlightSplit", 30, 90, 75, signed=False, group="curve"),
    # Color Grading (alte SplitToning-Namen werden von Lightroom weiter genutzt)
    _p("SplitToningShadowHue", 0, 359, 0, signed=False, group="grading"),
    _p("SplitToningShadowSaturation", 0, 100, 0, signed=False, group="grading"),
    _p("SplitToningHighlightHue", 0, 359, 0, signed=False, group="grading"),
    _p("SplitToningHighlightSaturation", 0, 100, 0, signed=False, group="grading"),
    _p("SplitToningBalance", -100, 100, group="grading"),
    _p("ColorGradeMidtoneHue", 0, 359, 0, signed=False, group="grading"),
    _p("ColorGradeMidtoneSat", 0, 100, 0, signed=False, group="grading"),
    _p("ColorGradeShadowLum", -100, 100, group="grading"),
    _p("ColorGradeMidtoneLum", -100, 100, group="grading"),
    _p("ColorGradeHighlightLum", -100, 100, group="grading"),
    _p("ColorGradeBlending", 0, 100, 50, signed=False, group="grading"),
    _p("ColorGradeGlobalHue", 0, 359, 0, signed=False, group="grading"),
    _p("ColorGradeGlobalSat", 0, 100, 0, signed=False, group="grading"),
    _p("ColorGradeGlobalLum", -100, 100, group="grading"),
    # Details
    _p("Sharpness", 0, 150, 40, signed=False, group="detail"),
    _p("SharpenRadius", 0.5, 3.0, 1.0, decimals=1, group="detail"),
    _p("SharpenDetail", 0, 100, 25, signed=False, group="detail"),
    _p("SharpenEdgeMasking", 0, 100, 0, signed=False, group="detail"),
    _p("LuminanceSmoothing", 0, 100, 0, signed=False, group="detail"),
    _p("ColorNoiseReduction", 0, 100, 25, signed=False, group="detail"),
    # Effekte
    _p("PostCropVignetteAmount", -100, 100, group="effects"),
    _p("PostCropVignetteMidpoint", 0, 100, 50, signed=False, group="effects"),
    _p("PostCropVignetteFeather", 0, 100, 50, signed=False, group="effects"),
    _p("PostCropVignetteRoundness", -100, 100, group="effects"),
    _p("PostCropVignetteHighlightContrast", 0, 100, 0, signed=False, group="effects"),
    _p("GrainAmount", 0, 100, 0, signed=False, group="effects"),
    # Kalibrierung
    _p("ShadowTint", -100, 100, group="calibration"),
    _p("RedHue", -100, 100, group="calibration"),
    _p("RedSaturation", -100, 100, group="calibration"),
    _p("GreenHue", -100, 100, group="calibration"),
    _p("GreenSaturation", -100, 100, group="calibration"),
    _p("BlueHue", -100, 100, group="calibration"),
    _p("BlueSaturation", -100, 100, group="calibration"),
]
for _c in HSL_COLORS:
    _PARAMS += [
        _p(f"HueAdjustment{_c}", -100, 100, group="hsl"),
        _p(f"SaturationAdjustment{_c}", -100, 100, group="hsl"),
        _p(f"LuminanceAdjustment{_c}", -100, 100, group="hsl"),
    ]

PARAMS: dict[str, Param] = {p.name: p for p in _PARAMS}

CROP_KEYS = ["CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle"]
TONE_CURVE_KEYS = ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]

# Lokale Korrekturen: Lightroom speichert sie normiert auf [-1, 1].
# Faktor = Wert in UI-Einheiten pro XMP-Einheit.
LOCAL_PARAMS: dict[str, float] = {
    "LocalExposure2012": 4.0,
    "LocalContrast2012": 100.0,
    "LocalHighlights2012": 100.0,
    "LocalShadows2012": 100.0,
    "LocalWhites2012": 100.0,
    "LocalBlacks2012": 100.0,
    "LocalClarity2012": 100.0,
    "LocalTexture": 100.0,
    "LocalDehaze": 100.0,
    "LocalSaturation": 100.0,
    "LocalTemperature": 100.0,
    "LocalTint": 100.0,
    "LocalSharpness": 100.0,
    "LocalLuminanceNoise": 100.0,
    "LocalMoire": 100.0,
    "LocalDefringe": 100.0,
    "LocalHue": 100.0,
    "LocalToningHue": 1.0,
    "LocalToningSaturation": 100.0,
}

# Diese Schlüssel sagen Lightroom, dass Einstellungen vorhanden sind.
BASE_FLAGS = {
    "HasSettings": "True",
    "AlreadyApplied": "False",
}


def fmt_value(name: str, v: Any) -> str:
    """Formatiert einen crs-Wert so, wie Lightroom ihn schreibt."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if name in PARAMS and isinstance(v, (int, float)):
        return PARAMS[name].fmt(float(v))
    if name in ("CropTop", "CropLeft", "CropBottom", "CropRight"):
        return f"{min(1.0, max(0.0, float(v))):.6f}"
    if name == "CropAngle":
        return f"{float(v):.6f}"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def fmt_local(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def to_number(v: Any) -> float | None:
    """Liest einen crs-Wert als Zahl (\"+0.35\" -> 0.35, \"True\" -> 1)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return 1.0 if s.lower() == "true" else 0.0
    try:
        return float(s)
    except ValueError:
        return None


def parse_curve(v: Any) -> list[tuple[float, float]]:
    """Tonkurve aus XMP (["0, 0", "255, 255"]) oder Lua (flache Zahlenliste)."""
    if not v:
        return []
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
        return [(float(v[i]), float(v[i + 1])) for i in range(0, len(v) - 1, 2)]
    pts = []
    for item in v:
        parts = str(item).replace(";", ",").split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def format_curve(points: list[tuple[float, float]]) -> list[str]:
    return [f"{int(round(x))}, {int(round(y))}" for x, y in points]

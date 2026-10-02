"""Eigener Edit in der App: Regler und Masken wie im Entwickeln-Modul von Lightroom.

Die Oberfläche arbeitet mit einem einfachen *Editor-Modell* in Anzeige-Koordinaten (0..1, so wie das Bild auf
dem Bildschirm steht). Hier wird es in Lightroom-Einstellungen (crs, Sensor-Koordinaten) umgerechnet und
zurück. Alles, was der Editor nicht kennt (Profil, Tonkurve, Zuschnitt, Denoise, unbekannte Masken), bleibt
unverändert erhalten.
"""

from __future__ import annotations

import copy
from typing import Any

from ..lightroom.dialect import Dialect
from ..lightroom.params import LOCAL_PARAMS, format_curve, parse_curve, to_number
from ..vision.geometry import display_to_sensor, sensor_to_display
from . import masks as mk

HSL_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
GLOBAL_KEYS = ["Temperature", "Tint", "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
               "Whites2012", "Blacks2012", "Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation",
               "Sharpness", "LuminanceSmoothing", "ColorNoiseReduction", "PostCropVignetteAmount"] + \
    [f"{p}{c}" for p in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment") for c in HSL_COLORS] + \
    ["SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum", "ColorGradeMidtoneHue",
     "ColorGradeMidtoneSat", "ColorGradeMidtoneLum", "SplitToningHighlightHue", "SplitToningHighlightSaturation",
     "ColorGradeHighlightLum", "ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum",
     "ColorGradeBlending", "SplitToningBalance", "ParametricShadows", "ParametricDarks", "ParametricLights",
     "ParametricHighlights", "ParametricShadowSplit", "ParametricMidtoneSplit", "ParametricHighlightSplit",
     "PostCropVignetteMidpoint", "PostCropVignetteFeather", "SharpenRadius", "SharpenDetail",
     "LuminanceNoiseReductionDetail", "ColorNoiseReductionDetail"]
CURVE_KEYS = {"main": "ToneCurvePV2012", "red": "ToneCurvePV2012Red", "green": "ToneCurvePV2012Green",
              "blue": "ToneCurvePV2012Blue"}
CROP_KEYS = ("HasCrop", "CropLeft", "CropTop", "CropRight", "CropBottom", "CropAngle")
LOCAL_KEYS = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
              "Texture", "Clarity2012", "Dehaze", "Saturation", "Temperature", "Tint", "Sharpness"]
MASK_KINDS = ("subject", "background", "sky", "person", "gradient", "radial", "brush")


def _num(v: Any, default: float = 0.0) -> float:
    n = to_number(v)
    return float(default if n is None else n)


def _local_ui(corr: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k in LOCAL_KEYS:
        key = f"Local{k}"
        if key in corr:
            out[k] = round(_num(corr[key]) * LOCAL_PARAMS.get(key, 100.0), 2 if k == "Exposure2012" else 0)
    return out


def _kind(comp: dict[str, Any]) -> str | None:
    what = comp.get("What")
    inv = str(comp.get("MaskInverted", "false")).lower() == "true"
    if what == "Mask/Image":
        sub = str(comp.get("MaskSubType", ""))
        if sub == "1":
            return "background" if inv else "subject"
        return {"2": "sky", "3": "person"}.get(sub)
    if what == "Mask/Gradient":
        return "gradient"
    if what == "Mask/CircularGradient":
        return "radial"
    return None


def crs_to_model(crs: dict[str, Any], orientation: int) -> dict[str, Any]:
    g: dict[str, Any] = {}
    for k in GLOBAL_KEYS:
        if k in crs and to_number(crs.get(k)) is not None:
            g[k] = _num(crs[k])
    model: dict[str, Any] = {"global": g, "wb_custom": str(crs.get("WhiteBalance", "As Shot")) != "As Shot",
                             "masks": [], "curves": {}, "crop": {}}
    for name, key in CURVE_KEYS.items():
        pts = parse_curve(crs.get(key))
        if len(pts) >= 2:
            model["curves"][name] = [[float(x), float(y)] for x, y in sorted(pts)]
    has = str(crs.get("HasCrop", "False")).lower() == "true"
    model["crop"] = {"HasCrop": has, "CropAngle": _num(crs.get("CropAngle")),
                     **{k: _num(crs.get(k), 1.0 if k in ("CropRight", "CropBottom") else 0.0)
                        for k in ("CropLeft", "CropTop", "CropRight", "CropBottom")}}
    for corr in crs.get("MaskGroupBasedCorrections") or []:
        if not isinstance(corr, dict):
            continue
        comps = [c for c in corr.get("CorrectionMasks") or [] if isinstance(c, dict)]
        kinds = [_kind(c) for c in comps]
        entry: dict[str, Any] = {"name": str(corr.get("CorrectionName") or "Maske"), "local": _local_ui(corr),
                                 "amount": _num(corr.get("CorrectionAmount"), 1.0)}
        paint = [c for c in comps if c.get("What") == "Mask/Paint"]
        if comps and len(paint) == len(comps) and all(to_number(c.get("MaskValue")) in (None, 1.0) for c in comps):
            entry["kind"] = "brush"
            dabs = []
            for c in paint:
                r = _num(c.get("Radius"), 0.02)
                for d in c.get("Dabs") or []:
                    parts = str(d).split()
                    if len(parts) >= 3:
                        x, y = sensor_to_display(float(parts[1]), float(parts[2]), orientation)
                        dabs.append([round(x, 5), round(y, 5), round(r, 5)])
            entry["dabs"] = dabs
            entry["feather"] = round((1 - _num(paint[0].get("CenterWeight"), 0.0)) * 100)
        elif len(comps) == 1 and kinds[0] is not None:
            c = comps[0]
            entry["kind"] = kinds[0]
            if kinds[0] == "gradient":
                entry["zero"] = list(sensor_to_display(_num(c.get("ZeroX")), _num(c.get("ZeroY")), orientation))
                entry["full"] = list(sensor_to_display(_num(c.get("FullX")), _num(c.get("FullY")), orientation))
            elif kinds[0] == "radial":
                a = sensor_to_display(_num(c.get("Left")), _num(c.get("Top")), orientation)
                b = sensor_to_display(_num(c.get("Right"), 1), _num(c.get("Bottom"), 1), orientation)
                entry["box"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
                entry["feather"] = _num(c.get("Feather"), 60)
                entry["invert"] = str(c.get("MaskInverted", "false")).lower() == "true"
        else:
            entry["kind"] = "other"               # z. B. Pinsel oder kombinierte Masken: unverändert behalten
            entry["raw"] = copy.deepcopy(corr)
        model["masks"].append(entry)
    return model


def model_to_crs(model: dict[str, Any], base: dict[str, Any], orientation: int,
                 dialect: Dialect | None = None) -> dict[str, Any]:
    """Editor-Modell -> crs. ``base``: bisherige Einstellungen (unbekannte Schlüssel bleiben erhalten)."""
    dialect = dialect or Dialect.load()
    crs = {k: copy.deepcopy(v) for k, v in base.items() if k != "MaskGroupBasedCorrections"}
    for k, v in (model.get("global") or {}).items():
        if k not in GLOBAL_KEYS or v is None:
            continue
        crs[k] = round(float(v), 2) if k == "Exposure2012" else int(round(float(v)))
    if model.get("wb_custom") and "Temperature" in (model.get("global") or {}):
        crs["WhiteBalance"] = "Custom"
    elif not model.get("wb_custom"):
        crs["WhiteBalance"] = "As Shot"
        crs.pop("Temperature", None)
        crs.pop("Tint", None)
    curves = model.get("curves")
    if isinstance(curves, dict):
        for name, key in CURVE_KEYS.items():
            pts = curves.get(name)
            if pts and len(pts) >= 2 and any(abs(float(x) - float(y)) > 0.5 for x, y in pts):
                crs[key] = format_curve(sorted((float(x), float(y)) for x, y in pts))
            else:
                crs[key] = format_curve([(0, 0), (255, 255)])
        crs["ToneCurveName2012"] = "Custom"
    crop = model.get("crop")
    if isinstance(crop, dict) and "HasCrop" in crop:
        if crop.get("HasCrop"):
            crs["HasCrop"] = True
            for k in ("CropLeft", "CropTop", "CropRight", "CropBottom"):
                crs[k] = round(min(1.0, max(0.0, float(crop.get(k, 0.0)))), 6)
            crs["CropAngle"] = round(float(crop.get("CropAngle") or 0.0), 4)
        else:
            crs["HasCrop"] = False
            crs["CropAngle"] = round(float(crop.get("CropAngle") or 0.0), 4)
            for k in ("CropLeft", "CropTop", "CropRight", "CropBottom"):
                crs.pop(k, None)
    corrections = []
    for m in model.get("masks") or []:
        kind = m.get("kind")
        local = {k: float(v) for k, v in (m.get("local") or {}).items() if k in LOCAL_KEYS and v}
        name = str(m.get("name") or "Maske")
        if kind == "other" and isinstance(m.get("raw"), dict):
            corr = copy.deepcopy(m["raw"])
            for k in [k for k in corr if k.startswith("Local")]:
                if k[5:] in LOCAL_KEYS:
                    corr.pop(k)
            for k, v in local.items():
                key = f"Local{k}"
                corr[key] = max(-1.0, min(1.0, v / LOCAL_PARAMS.get(key, 100.0)))
            corr["CorrectionName"], corr["CorrectionAmount"] = name, float(m.get("amount", 1.0))
            corrections.append(corr)
            continue
        if kind in ("subject", "background", "sky", "person"):
            comp = mk.ai_component(kind, dialect, name)
        elif kind == "gradient":
            zero, full = m.get("zero") or [0.5, 0.6], m.get("full") or [0.5, 1.0]
            comp = mk.gradient_component((float(zero[0]), float(zero[1])), (float(full[0]), float(full[1])),
                                         orientation, name)
        elif kind == "brush":
            comps = _brush_components(m.get("dabs") or [], orientation, float(m.get("feather", 50)), name)
            if comps:
                corrections.append(mk.correction(name, local, comps, float(m.get("amount", 1.0))))
            continue
        elif kind == "radial":
            box = m.get("box") or [0.3, 0.2, 0.7, 0.9]
            comp = mk.radial_component(tuple(float(x) for x in box), orientation, float(m.get("feather", 60)),
                                       bool(m.get("invert")), name)
        else:
            continue
        corrections.append(mk.correction(name, local, [comp], float(m.get("amount", 1.0))))
    if corrections:
        crs["MaskGroupBasedCorrections"] = corrections
    return crs


def _brush_components(dabs: list[Any], orientation: int, feather: float, name: str) -> list[dict[str, Any]]:
    """Pinsel-Tupfer (Anzeige x, y, Radius relativ zur langen Seite) -> Lightroom Mask/Paint (ein Radius pro
    Komponente, wie Lightroom einen Strich speichert)."""
    groups: dict[float, list[str]] = {}
    for d in dabs:
        try:
            x, y, r = float(d[0]), float(d[1]), float(d[2])
        except (TypeError, ValueError, IndexError):
            continue
        sx, sy = display_to_sensor(min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), orientation)
        groups.setdefault(round(r, 4), []).append(f"d {sx:.6f} {sy:.6f}")
    comps = []
    for i, (r, pts) in enumerate(sorted(groups.items())):
        comp = {"What": "Mask/Paint", "MaskValue": 1.0, "Radius": r, "Flow": 1.0,
                "CenterWeight": round(max(0.0, min(1.0, 1 - feather / 100)), 3), "Dabs": pts}
        comp.update(mk._component_base(f"{name} {i + 1}"))
        comps.append(comp)
    return comps


def template_from_edit(name: str, crs: dict[str, Any], analysis: dict[str, Any], exif: dict[str, Any],
                       orientation: int) -> dict[str, Any]:
    """Eigener Edit eines Bildes als Vorlage: gilt 1:1 für die anderen Bilder, mit Bezug genau dieses Bild
    (Belichtung/Weissabgleich werden nur so weit verschoben, wie sich die anderen Bilder davon unterscheiden)."""
    import json

    from .changes import describe
    from .template import _clean, _denoised, _file, scene_level

    tcrs = _clean(crs)
    ref = {"iso": exif.get("iso"), "exposure_time": exif.get("exposure_time"), "aperture": exif.get("aperture"),
           "orientation": int(orientation or 1), "level": scene_level(analysis),
           "as_shot_temp": analysis.get("as_shot_temp"), "as_shot_tint": analysis.get("as_shot_tint"),
           "subj_level": analysis.get("subj_level")}
    summary = describe(tcrs)
    t = {"name": name, "n": 1, "files": [analysis.get("filename") or "eigener Edit"], "crs": tcrs, "ref": ref,
         "had_denoise": _denoised(crs), "source": "editor",
         "summary": [f"{s['label']} {s['value']}" for s in summary["sliders"]] +
                    [f"{m['name']}: {', '.join(m['effects'])}" for m in summary["masks"]]}
    _file(name).write_text(json.dumps(t, indent=1, ensure_ascii=False, default=float), "utf-8")
    return t

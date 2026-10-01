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
from ..lightroom.params import LOCAL_PARAMS, to_number
from ..vision.geometry import sensor_to_display
from . import masks as mk

HSL_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
GLOBAL_KEYS = ["Temperature", "Tint", "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
               "Whites2012", "Blacks2012", "Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation",
               "Sharpness", "LuminanceSmoothing", "ColorNoiseReduction", "PostCropVignetteAmount"] + \
    [f"{p}{c}" for p in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment") for c in HSL_COLORS]
LOCAL_KEYS = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
              "Texture", "Clarity2012", "Dehaze", "Saturation", "Temperature", "Tint", "Sharpness"]
MASK_KINDS = ("subject", "background", "sky", "person", "gradient", "radial")


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
                             "masks": []}
    for corr in crs.get("MaskGroupBasedCorrections") or []:
        if not isinstance(corr, dict):
            continue
        comps = [c for c in corr.get("CorrectionMasks") or [] if isinstance(c, dict)]
        kinds = [_kind(c) for c in comps]
        entry: dict[str, Any] = {"name": str(corr.get("CorrectionName") or "Maske"), "local": _local_ui(corr),
                                 "amount": _num(corr.get("CorrectionAmount"), 1.0)}
        if len(comps) == 1 and kinds[0] is not None:
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

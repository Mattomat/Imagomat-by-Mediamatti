"""Aus Vorhersagen fertige Lightroom-Einstellungen bauen.

Ablauf für einen Shoot:
1. ``predict_all``: Zielwerte pro Bild (Stilprofil oder adaptives Preset).
2. ``smooth_shoot``: Konsistenz innerhalb gleicher Lichtsituationen (Weissabgleich und
   Ausgabehelligkeit angleichen, nicht die Rohwerte).
3. ``build_settings``: crs-Dict inkl. Zuschnitt/Begradigen, Masken und Denoise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Settings
from ..io.color import mired
from ..lightroom.dialect import Dialect
from ..lightroom.params import BASE_FLAGS
from ..vision.geometry import plan_crop, to_lightroom_crop
from . import masks as mk
from .model import Prediction, Record, StyleModel
from .presets import PRESETS, apply as apply_preset, choose_preset
from .targets import DIRECT, TARGETS, decode


@dataclass
class ImageDevelop:
    image_id: int
    record: Record
    orientation: int
    width: int | None
    height: int | None
    targets: dict[str, float] = field(default_factory=dict)
    prediction: Prediction | None = None
    preset: str = ""
    confidence: float = 0.5
    subject_mask: np.ndarray | None = None
    crs: dict[str, Any] = field(default_factory=dict)
    denoise: int | None = None
    notes: list[str] = field(default_factory=list)


def predict_all(items: list[ImageDevelop], model: StyleModel | None, settings: Settings,
                preset_key: str | None = None) -> None:
    for it in items:
        if model is not None:
            p = model.predict(it.record, settings.denoise, preset_key)
            it.prediction, it.targets, it.preset, it.confidence = p, dict(p.targets), p.preset, p.confidence
        else:
            key = preset_key or choose_preset(it.record.analysis.get("scene"))
            it.preset = key
            it.targets = apply_preset(PRESETS[key], {**it.record.analysis, "iso": it.record.exif.get("iso")},
                                      settings.denoise)
            it.confidence = 0.5


def light_groups(items: list[ImageDevelop], gap_s: float = 120.0, dmired: float = 25.0,
                 dlog: float = 1.5) -> list[list[ImageDevelop]]:
    """Aufeinanderfolgende Bilder mit gleicher Lichtsituation gruppieren."""
    seq = sorted(items, key=lambda it: it.record.exif.get("capture_time") or 0.0)
    groups: list[list[ImageDevelop]] = []
    for it in seq:
        a, t = it.record.analysis, it.record.exif.get("capture_time") or 0.0
        m = mired(a.get("as_shot_temp") or 5000)
        lm = a.get("lin_log_median", -4.0)
        if groups:
            g = groups[-1]
            last = g[-1]
            gm = float(np.median([mired(x.record.analysis.get("as_shot_temp") or 5000) for x in g]))
            gl = float(np.median([x.record.analysis.get("lin_log_median", -4.0) for x in g]))
            if (t - (last.record.exif.get("capture_time") or 0.0) <= gap_s and abs(m - gm) <= dmired
                    and abs(lm - gl) <= dlog and it.preset == last.preset):
                g.append(it)
                continue
        groups.append([it])
    return groups


def smooth_shoot(items: list[ImageDevelop], alpha: float) -> None:
    if alpha <= 0:
        return
    for g in light_groups(items):
        if len(g) < 3:
            continue
        # Weissabgleich: absoluten Zielwert angleichen
        abs_m = [mired(it.record.analysis.get("as_shot_temp") or 5000) + it.targets.get("wb_dmired", 0) for it in g]
        abs_t = [(it.record.analysis.get("as_shot_tint") or 0) + it.targets.get("wb_dtint", 0) for it in g]
        med_m, med_t = float(np.median(abs_m)), float(np.median(abs_t))
        custom = np.mean([it.targets.get("wb_custom", 0) for it in g]) >= 0.5
        # Ausgabehelligkeit angleichen
        out_b = [it.record.analysis.get("lin_log_median", -4.0) + it.targets.get("Exposure2012", 0) for it in g]
        med_b = float(np.median(out_b))
        for it, m, t, b in zip(g, abs_m, abs_t, out_b):
            as_m = mired(it.record.analysis.get("as_shot_temp") or 5000)
            as_t = it.record.analysis.get("as_shot_tint") or 0
            if custom:
                it.targets["wb_custom"] = 1.0
                it.targets["wb_dmired"] = (1 - alpha) * m + alpha * med_m - as_m
                it.targets["wb_dtint"] = (1 - alpha) * t + alpha * med_t - as_t
            nb = (1 - alpha) * b + alpha * med_b
            it.targets["Exposure2012"] = float(np.clip(nb - it.record.analysis.get("lin_log_median", -4.0), -5, 5))
        # Look-Regler halb so stark angleichen
        for k in DIRECT:
            if k == "Exposure2012":
                continue
            vals = [it.targets.get(k, 0.0) for it in g]
            med = float(np.median(vals))
            for it, v in zip(g, vals):
                it.targets[k] = (1 - alpha * 0.5) * v + alpha * 0.5 * med


def _crop(it: ImageDevelop, settings: Settings) -> dict[str, Any]:
    a = it.record.analysis
    ds = settings.develop
    angle = 0.0
    if ds.auto_straighten and a.get("tilt_conf", 0) >= 0.5 and abs(a.get("tilt_angle", 0)) >= 0.3:
        angle = float(np.clip(a["tilt_angle"], -ds.max_straighten_deg, ds.max_straighten_deg))
        it.notes.append(f"begradigt {angle:+.1f}°")
    W = int(a.get("preview_w") or 3)
    H = int(a.get("preview_h") or 2)
    zoom = 1.0
    if ds.auto_crop and it.prediction is not None and it.prediction.has_crop_prob >= 0.5 \
            and it.prediction.crop_area < 0.97:
        zoom = math.sqrt(max(it.prediction.crop_area, 0.3))
    if angle == 0.0 and zoom >= 0.999:
        return {"HasCrop": False, "CropAngle": 0.0}
    subject = tuple(a["subject_bbox"]) if a.get("subject_bbox") and zoom < 0.999 else None
    mf = a.get("main_face") or {}
    face = None
    if mf and zoom < 0.999:
        b = mf.get("bbox")
        face = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    plan = plan_crop(W, H, angle, aspect=W / H, zoom=zoom, subject=subject, face=face)
    if zoom < 0.999:
        it.notes.append(f"zugeschnitten {plan.area_fraction:.0%}")
    return to_lightroom_crop(plan, it.orientation)


def _preset_masks(it: ImageDevelop, dialect: Dialect) -> list[dict[str, Any]]:
    a = it.record.analysis
    out = []
    for r in PRESETS[it.preset].masks if it.preset in PRESETS else []:
        ok = {"always": True, "has_subject": (a.get("subject_fraction") or 0) > 0.02,
              "has_sky": (a.get("sky_fraction") or 0) > 0.05, "has_face": (a.get("face_count") or 0) > 0}[r.condition]
        if not ok:
            continue
        if r.kind == "gradient_bottom":
            comp = mk.gradient_component((0.5, 0.55), (0.5, 1.0), it.orientation, r.name)
        else:
            comp = mk.ai_component(r.kind, dialect, r.name)
        out.append(mk.correction(r.name, r.local, [comp]))
    return out


def build_settings(it: ImageDevelop, settings: Settings, dialect: Dialect) -> None:
    a = it.record.analysis
    crs: dict[str, Any] = {"Version": settings.develop.camera_raw_version or dialect.crs_version,
                           "ProcessVersion": settings.develop.process_version or dialect.process_version,
                           **BASE_FLAGS}
    if it.prediction is not None:
        crs.update(it.prediction.extras)
    else:
        crs.update({"LensProfileEnable": 1, "AutoLateralCA": 1})
    crs.update(decode(it.targets, a.get("as_shot_temp"), a.get("as_shot_tint")))
    crs.update(_crop(it, settings))
    # Denoise
    amount = int(round(it.targets.get("denoise", 0)))
    if settings.denoise.always and amount < settings.denoise.min_amount:
        amount = settings.denoise.min_amount
    it.denoise = amount if amount > 0 else None
    for k in list(crs):
        if k.startswith("EnhanceDenoise"):
            crs.pop(k)
    if it.denoise and settings.denoise.mode == "lightroom":
        crs.update(dialect.denoise(it.denoise))
    # Masken
    corrections: list[dict[str, Any]] = []
    if settings.develop.write_masks:
        if it.prediction is not None and it.prediction.masks:
            for t, vals, _p in it.prediction.masks:
                c = mk.instantiate(t, vals, a, it.orientation, dialect, it.subject_mask)
                if c is not None:
                    corrections.append(c)
        else:
            corrections = _preset_masks(it, dialect)
        if not settings.develop.ai_masks:
            corrections = [_ai_to_paint(c, it) for c in corrections]
            corrections = [c for c in corrections if c]
    if corrections:
        crs["MaskGroupBasedCorrections"] = corrections
    it.crs = crs


def _ai_to_paint(corr: dict[str, Any], it: ImageDevelop) -> dict[str, Any] | None:
    """Ersetzt KI-Masken durch Pinsel aus der eigenen Segmentierung (Fallback, falls H2 scheitert)."""
    comps = []
    for m in corr.get("CorrectionMasks", []):
        if m.get("What") == "Mask/Image":
            if it.subject_mask is None:
                continue
            mask = it.subject_mask if str(m.get("MaskInverted", "false")).lower() != "true" else 1 - it.subject_mask
            comps += mk.paint_components(mask, it.orientation, str(m.get("MaskName", "Pinsel")))
        else:
            comps.append(m)
    if not comps:
        return None
    return {**corr, "CorrectionMasks": comps}


def develop_items(items: list[ImageDevelop], model: StyleModel | None, settings: Settings, dialect: Dialect,
                  preset_key: str | None = None) -> None:
    predict_all(items, model, settings, preset_key)
    smooth_shoot(items, settings.develop.shoot_consistency)
    for it in items:
        build_settings(it, settings, dialect)

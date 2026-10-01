"""Aus Vorhersagen fertige Lightroom-Einstellungen bauen.

Ablauf für einen Shoot:
1. ``predict_all``: Zielwerte pro Bild (Stilprofil oder adaptives Preset).
2. ``smooth_shoot``: Konsistenz innerhalb gleicher Lichtsituationen (Weissabgleich und
   Ausgabehelligkeit angleichen, nicht die Rohwerte).
3. ``build_settings``: crs-Dict inkl. Zuschnitt/Begradigen, Masken und Denoise.
"""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


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
    preview: str | None = None               # Kamera-Vorschau (für Weiss/Schwarz am Waveform)
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
            if PRESETS[key].wb_mode == "white":
                _white_balance(it)


def _white_balance(it: ImageDevelop) -> None:
    """Weissabgleich so, dass Weiss an den Spielern (Trikots, Hosen) neutral ist."""
    if not it.preview or it.subject_mask is None or not it.record.analysis.get("as_shot_temp"):
        return
    import cv2

    from .whitebalance import white_patch_shift

    img = cv2.imread(it.preview, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        return
    sh = white_patch_shift(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), it.subject_mask)
    if sh is None:
        return
    it.targets["wb_dmired"], it.targets["wb_dtint"], it.targets["wb_custom"] = sh[0], sh[1], 1.0
    if abs(sh[0]) >= 3 or abs(sh[1]) >= 3:
        it.notes.append(f"Weissabgleich auf neutrales Weiss: {sh[0]:+.0f} Mired, Tint {sh[1]:+.0f}")


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
    if ds.auto_straighten and a.get("tilt_conf", 0) >= 0.4 and abs(a.get("tilt_angle", 0)) >= 0.2:
        angle = float(np.clip(a["tilt_angle"], -ds.max_straighten_deg, ds.max_straighten_deg))
        it.notes.append(f"begradigt {angle:+.1f}°")
    W = int(a.get("preview_w") or 3)
    H = int(a.get("preview_h") or 2)
    zoom = 1.0
    if ds.auto_crop and it.prediction is not None and it.prediction.has_crop_prob >= 0.5 \
            and it.prediction.crop_area < 0.97:
        zoom = math.sqrt(max(it.prediction.crop_area, 0.3))
    elif ds.auto_crop and a.get("subject_bbox"):
        # Ohne gelernten Zuschnitt (Presets, oder dein Stil schneidet dieses Bild nicht): sanft enger ums
        # Geschehen, Spieler bleibt ganz im Bild, Gesicht aufs obere Drittel. Bei grossem Motiv nicht.
        sb = a["subject_bbox"]
        area = (sb[2] - sb[0]) * (sb[3] - sb[1])
        if area < 0.35:
            zoom = math.sqrt(0.82)
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


def bottom_luma(img: np.ndarray) -> tuple[float, float]:
    """Mittlere Helligkeit (0..1, sRGB) unten (untere 40 %) und in der Bildmitte."""
    g = img.mean(axis=2) / 255.0 if img.ndim == 3 else img / 255.0
    h = g.shape[0]
    return float(g[int(h * 0.6):].mean()), float(g[int(h * 0.3):int(h * 0.6)].mean())


def bottom_fade(it: ImageDevelop, settings: Settings) -> dict[str, Any] | None:
    """Dunkler, unscharfer Verlauf von unten, pro Bild angepasst:
    heller Rasen/Vordergrund -> kräftiger; unten schon dunkel -> schwächer; Verlauf beginnt unterhalb
    der Spieler-Mitte, damit Gesichter und Oberkörper hell bleiben."""
    ds = settings.develop
    if not ds.bottom_fade or ds.bottom_fade_strength <= 0:
        return None
    a = it.record.analysis
    bottom, mid = a.get("bottom_luma"), a.get("mid_luma")
    ev = -0.9
    if bottom is not None:
        rel = bottom - (mid if mid is not None else bottom)
        ev = -0.6 - 0.6 * float(np.clip((bottom - 0.2) / 0.4, 0, 1)) - 0.3 * float(np.clip(rel / 0.2, 0, 1))
        if bottom < 0.1:
            ev = -0.4                         # unten schon fast schwarz: nicht absaufen lassen
    preset_fade = PRESETS[it.preset].fade if it.prediction is None and it.preset in PRESETS else 1.0
    ev *= ds.bottom_fade_strength * preset_fade
    # Verlauf beginnt erst unter den Spielern (Beine/Füsse bleiben hell), ohne Motiv bei 60 %
    start = 0.6
    sb = a.get("subject_bbox")
    if sb:
        start = float(np.clip(sb[1] + (sb[3] - sb[1]) * 0.85, 0.55, 0.8))
    soft = float(np.clip(ds.bottom_fade_strength, 0.3, 1.5))
    preset = PRESETS.get(it.preset) if it.prediction is None else None
    if preset is not None and preset.fade_local is not None:
        local = {"Exposure2012": round(ev, 2), **{k: round(v * min(soft, 1.0)) for k, v in preset.fade_local.items()}}
    else:
        local = {"Exposure2012": round(ev, 2), "Highlights2012": -20, "Saturation": -10,
                 "Sharpness": round(-50 * soft), "Clarity2012": round(-25 * soft), "Texture": round(-35 * soft)}
    full = min(1.0, start + preset.fade_span) if preset is not None and preset.fade_span else 1.0
    comp = mk.gradient_component((0.5, start), (0.5, full), it.orientation, "Verlauf unten")
    return mk.correction("Verlauf unten", local, [comp])


def punch(crs: dict[str, Any], a: dict[str, Any], amount: float) -> list[str]:
    """Weiss- und Schwarzpunkt pro Bild setzen (wie Shift-Doppelklick in Lightroom, aber sanfter) und
    etwas mehr Kontrast. Gemittelte Stil-Vorhersagen sind sonst flacher als deine Einzelbilder."""
    from ..lightroom.params import to_number

    if amount <= 0 or a.get("lin_log_p99") is None:
        return []
    num = lambda k: float(to_number(crs.get(k)) or 0.0)  # noqa: E731
    exp = num("Exposure2012")
    whites, blacks, hl = num("Whites2012"), num("Blacks2012"), num("Highlights2012")
    # Wo landen die hellsten 1 % nach Belichtung, Weiss und Lichter? (log2, 0 = weiss)
    top = a["lin_log_p99"] + exp + whites / 60.0 + min(hl, 0.0) / 150.0
    dw = float(np.clip((-0.15 - top) * 30.0, 0.0, 45.0)) * amount
    bottom = (a.get("lin_log_p01") if a.get("lin_log_p01") is not None else -9.0) + exp + blacks / 40.0
    db = -float(np.clip((bottom + 6.5) * 4.0, 0.0, 15.0)) * amount
    notes = []
    if dw >= 1:
        crs["Whites2012"] = int(round(min(whites + dw, 70)))
        notes.append(f"Weiss +{dw:.0f}")
    if db <= -1:
        crs["Blacks2012"] = int(round(max(blacks + db, -70)))
        notes.append(f"Schwarz {db:.0f}")
    crs["Contrast2012"] = int(round(min(num("Contrast2012") + 4 * amount, 60)))
    return notes


_BACKGROUND_PUNCH = {"LocalContrast2012": 0.15, "LocalDehaze": 0.10}


def punch_background(corr: dict[str, Any], amount: float) -> None:
    """Hintergrund-Maske (Motiv invertiert): nicht nur abdunkeln, sondern Kontrast/Tiefe geben.
    Hast du selbst im Hintergrund bewusst Kontrast rausgenommen, bleibt das so."""
    from ..lightroom.params import to_number

    comps = [m for m in corr.get("CorrectionMasks", []) or [] if isinstance(m, dict)]
    if amount <= 0 or not comps or not all(
            m.get("What") == "Mask/Image" and str(m.get("MaskInverted", "false")).lower() == "true" for m in comps):
        return
    for k, v in _BACKGROUND_PUNCH.items():
        cur = float(to_number(corr.get(k)) or 0.0)
        if cur >= 0:
            corr[k] = float(np.clip(max(cur, v * amount), -1, 1))


def _is_bottom_gradient(corr: dict[str, Any], orientation: int) -> bool:
    from ..lightroom.params import to_number
    from ..vision.geometry import sensor_to_display

    comps = [m for m in corr.get("CorrectionMasks", []) or [] if isinstance(m, dict)]
    if not comps or any(m.get("What") != "Mask/Gradient" for m in comps):
        return False
    for m in comps:
        zx, zy = sensor_to_display(to_number(m.get("ZeroX")) or 0, to_number(m.get("ZeroY")) or 0, orientation)
        fx, fy = sensor_to_display(to_number(m.get("FullX")) or 0, to_number(m.get("FullY")) or 0, orientation)
        if fy > zy and fy >= 0.7 and abs(fy - zy) > abs(fx - zx):
            return True
    return False


def _preset_masks(it: ImageDevelop, dialect: Dialect) -> list[dict[str, Any]]:
    a = it.record.analysis
    out = []
    for r in PRESETS[it.preset].masks if it.preset in PRESETS else []:
        if r.kind == "gradient_bottom":
            continue                          # kommt aus bottom_fade() (pro Bild angepasst)
        ok = {"always": True, "has_subject": (a.get("subject_fraction") or 0) > 0.02,
              "has_sky": (a.get("sky_fraction") or 0) > 0.05, "has_face": (a.get("face_count") or 0) > 0}[r.condition]
        if not ok:
            continue
        if r.kind == "gradient_top":
            comp = mk.gradient_component((0.5, 0.4), (0.5, 0.0), it.orientation, r.name)
        elif r.kind == "spotlight":
            # Radialfilter um die Spieler, aussen dunkler (invertiert)
            x0, y0, x1, y1 = a.get("subject_bbox") or (0.3, 0.2, 0.7, 0.9)
            w, h = x1 - x0, y1 - y0
            box = (x0 - w * 0.45, y0 - h * 0.3, x1 + w * 0.45, y1 + h * 0.25)
            comp = mk.radial_component(box, it.orientation, feather=85, inverted=True, name=r.name)
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
    it.notes += punch(crs, a, settings.develop.punch)
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
        wants_fade = it.prediction is not None or any(
            r.kind == "gradient_bottom" for r in (PRESETS[it.preset].masks if it.preset in PRESETS else []))
        fade = bottom_fade(it, settings) if wants_fade else None
        for c in corrections:
            punch_background(c, settings.develop.punch)
        if fade is not None:
            # eigener (gelernter) Verlauf unten wird durch den angepassten, kräftigeren ersetzt
            corrections = [c for c in corrections if not _is_bottom_gradient(c, it.orientation)] + [fade]
            it.notes.append(f"Verlauf unten {fade['LocalExposure2012'] * 4:+.1f} EV")
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
                  preset_key: str | None = None, look: dict[str, Any] | None = None,
                  shoot_ref: dict[str, Any] | None = None) -> None:
    if look is not None and look.get("kind") == "template":
        from .template import develop_items as develop_template

        develop_template(items, look, settings, dialect, shoot_ref)
        return
    predict_all(items, model, settings, preset_key)
    smooth_shoot(items, settings.develop.shoot_consistency)
    for it in items:
        build_settings(it, settings, dialect)
        if look is None or not apply_look(it, look, dialect):
            fit_white_black(it, settings)


def apply_look(it: ImageDevelop, look: dict[str, Any], dialect: Dialect) -> bool:
    """Bild auf deinen Referenz-Look bringen (gemessen an deinen fertigen Bildern)."""
    if not it.preview:
        return False
    from .look import fit_look
    from .scopes import preview_linear

    try:
        lin = preview_linear(it.preview, 224)
        if lin is None:
            return False
        a = it.record.analysis
        notes = fit_look(it.crs, lin, look, it.orientation, it.subject_mask, a.get("as_shot_temp"),
                         a.get("as_shot_tint"), dialect)
    except Exception as e:  # noqa: BLE001 - dann wie bisher ohne Look-Anpassung
        log.warning("Look-Anpassung fehlgeschlagen für Bild %s: %s", it.image_id, e)
        return False
    it.notes = [n for n in it.notes if not n.startswith(("Weiss ", "Schwarz "))] + notes
    return True


def fit_white_black(it: ImageDevelop, settings: Settings) -> None:
    """Weiss/Schwarz wie am Lumetri-Waveform: oben und unten leicht anschlagen lassen."""
    if settings.develop.punch <= 0 or not it.preview:
        return
    from .scopes import fit_scopes, preview_linear

    try:
        lin = preview_linear(it.preview)
        if lin is None:
            return
        preset = PRESETS.get(it.preset) if it.prediction is None else None
        kw: dict[str, Any] = {}
        if preset is not None and preset.scope_hi is not None:
            kw["hi_target"] = preset.scope_hi
        if preset is not None and preset.scope_lo is not None:
            kw.update(lo_target=preset.scope_lo, free_blacks=True)
        if preset is not None and not preset.deepen_blacks:
            kw["deepen"] = False
        notes = fit_scopes(it.crs, lin, it.orientation, it.subject_mask, settings.develop.punch, **kw)
    except Exception as e:  # noqa: BLE001 - dann bleibt die Schätzung aus punch()
        log.debug("Waveform-Anpassung fehlgeschlagen für %s: %s", it.image_id, e)
        return
    it.notes = [n for n in it.notes if not n.startswith(("Weiss ", "Schwarz "))] + notes

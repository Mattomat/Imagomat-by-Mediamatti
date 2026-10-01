"""Vorlage aus deinen eigenen Lightroom-Bearbeitungen (XMP): 1:1 übernehmen, pro Bild nur leicht anpassen.

Statt einen Stil zu erraten, wird deine Bearbeitung so übernommen, wie sie im XMP steht: alle Regler,
Profil, Tonkurve, HSL, Schärfe und alle Masken mit denselben Werten. Pro Bild ändert Imagomat nur:

- Belichtung: andere Kamera-Einstellung (Zeit/Blende/ISO) wird ausgeglichen, dazu eine kleine Korrektur,
  wenn die Szene heller oder dunkler ist als bei der Vorlage (höchstens ±0.35 EV).
- Weissabgleich: verschiebt sich, wenn die Kamera ein anderes Licht gemessen hat (z. B. andere Flutlichter).
- Begradigen und Zuschnitt: pro Bild (Komposition ist jedes Mal anders).
- Masken: Verläufe auf die Bildausrichtung (hoch/quer) umgerechnet, KI-Masken (Motiv, Himmel …) berechnet
  Lightroom für jedes Bild neu.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import RAW_EXTENSIONS, data_dir
from ..io.color import mired
from ..lightroom.develop import refresh_sync_ids, strip_digests
from ..lightroom.params import to_number
from ..lightroom.xmp import read_xmp
from ..vision.geometry import display_to_sensor, sensor_to_display

log = logging.getLogger(__name__)

PREFIX = "tpl:"
# Zahlen-Regler, die über mehrere Vorlagen gemittelt werden (Median)
GLOBAL_KEYS = ("Temperature", "Tint", "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012",
               "Blacks2012", "Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation")
# Gehören zum einzelnen Bild, nie kopieren
DROP_KEYS = {"FilterList", "HasCrop", "CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle",
             "CropConstrainToWarp", "CropConstrainToUnitSquare", "AlreadyApplied", "RawFileName",
             "AutoToneDigest", "AutoToneDigestNoSat", "HasSettings"}
# Bildbezogene Felder in KI-Masken: ohne sie berechnet Lightroom die Maske für das neue Bild
AI_IMAGE_KEYS = {"InputDigest", "InputDigestVersion", "LocalInputDigest", "LocalInputDigestVersion", "MaskDigest",
                 "WholeImageArea", "Origin", "FullMaskSize", "ReferencePoint"}
MAX_SCENE_EV = 0.35            # Szenen-Korrektur der Belichtung (heller/dunkler als die Vorlage), ohne Spieler
MAX_SUBJECT_EV = 1.5           # mit gemessener Helligkeit der Spieler (z. B. Interview im Dunkeln)
SUBJECT_GAIN = 0.85
MAX_CAMERA_EV = 4.0
MAX_WB_MIRED, MAX_WB_TINT = 40.0, 20.0


def templates_dir() -> Path:
    p = data_dir() / "templates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _file(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in " -_äöüÄÖÜ").strip() or "Vorlage"
    return templates_dir() / f"{safe}.json"


def list_templates() -> list[dict[str, Any]]:
    out = []
    for p in sorted(templates_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text("utf-8"))
            out.append({"name": d["name"], "n": d.get("n"), "files": d.get("files", []),
                        "summary": d.get("summary", [])})
        except (OSError, ValueError, KeyError):
            continue
    return out


def load_template(name: str) -> dict[str, Any] | None:
    p = _file(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def delete_template(name: str) -> bool:
    p = _file(name)
    if p.exists():
        p.unlink()
        return True
    return False


def template_version(name: str) -> float:
    p = _file(name)
    return p.stat().st_mtime if p.exists() else 0.0


# ---------------------------------------------------------------------------
# Lernen
# ---------------------------------------------------------------------------

def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, str) and "/" in v and " " not in v:
        a, _, b = v.partition("/")
        try:
            return float(a) / float(b) if float(b) else None
        except ValueError:
            return None
    return to_number(v)


def _exif(other: dict[str, Any]) -> dict[str, float | None]:
    iso = _num(other.get("exif:ISOSpeedRatings")) or _num(other.get("exif:RecommendedExposureIndex"))
    return {"iso": iso, "exposure_time": _num(other.get("exif:ExposureTime")),
            "aperture": _num(other.get("exif:FNumber")), "orientation": _num(other.get("tiff:Orientation")) or 1}


def _clean(crs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in crs.items():
        if k in DROP_KEYS or k.startswith("Table_") or re.search(r"Digest$", k):
            continue
        out[k] = copy.deepcopy(v)
    return out


def _raw_beside(xmp: Path) -> Path | None:
    for p in xmp.parent.glob(xmp.stem + ".*"):
        if p.suffix.lower() in RAW_EXTENSIONS and not p.name.startswith("._"):
            return p
    return None


def _measure_raw(path: Path) -> dict[str, float] | None:
    """Szenenhelligkeit und Kamera-Weissabgleich der Vorlage aus ihrer RAW (falls sie daneben liegt)."""
    try:
        from ..io import raw as raw_io
        from ..vision.quality import linear_stats

        lin, info = raw_io.read_linear_any(path, max_side=1024)
        s = linear_stats(lin, info.camera_wb)
        out = {"level": scene_level(s), "as_shot_temp": info.as_shot_temp, "as_shot_tint": info.as_shot_tint}
        try:
            from ..vision.segmentation import segment

            img, _ = raw_io.load_preview(path, 1024)
            out["subj_level"] = subject_level(img, segment(img).subject)
        except Exception as e:  # noqa: BLE001
            log.info("Spieler der Vorlage nicht gemessen (%s): %s", path.name, e)
        return out
    except Exception as e:  # noqa: BLE001 - Vorlage geht auch ohne RAW
        log.info("RAW zur Vorlage nicht lesbar (%s): %s", path.name, e)
        return None


def collect_xmps(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.suffix.lower() == ".xmp" and not x.name.startswith("._"))
        elif p.suffix.lower() == ".xmp":
            out.append(p)
        elif p.with_suffix(".xmp").exists():
            out.append(p.with_suffix(".xmp"))
    seen: set[str] = set()
    return [p for p in out if not (str(p) in seen or seen.add(str(p)))]


def build_template(name: str, docs: list[tuple[str, dict[str, Any], dict[str, Any]]],
                   refs: list[dict[str, float]] | None = None) -> dict[str, Any]:
    """docs: (Dateiname, crs, exif) je XMP. Globale Regler = Median, Masken/Profil aus der Vorlage,
    die dem Median am nächsten liegt."""
    if not docs:
        raise ValueError("Keine XMP mit Bearbeitung gefunden")
    med: dict[str, float] = {}
    for k in GLOBAL_KEYS:
        vals = [v for _, crs, _ in docs if (v := _num(crs.get(k))) is not None]
        if vals:
            med[k] = float(np.median(vals))
    scale = {"Temperature": 100.0, "Exposure2012": 0.1}
    best = min(docs, key=lambda d: sum(abs((_num(d[1].get(k)) or 0.0) - v) / scale.get(k, 1.0)
                                       for k, v in med.items()))
    crs = _clean(best[1])
    _median_masks(crs, [d[1] for d in docs])
    for k, v in med.items():
        crs[k] = round(v, 2) if k == "Exposure2012" else int(round(v))
    exifs = [e for _, _, e in docs]

    def mid(key: str, src: list[dict[str, Any]]) -> float | None:
        vals = [float(v) for e in src if (v := e.get(key)) is not None]
        return float(np.median(vals)) if vals else None

    ref: dict[str, Any] = {"iso": mid("iso", exifs), "exposure_time": mid("exposure_time", exifs),
                           "aperture": mid("aperture", exifs), "orientation": int(best[2].get("orientation") or 1)}
    refs = [r for r in refs or [] if r]
    for k in ("level", "as_shot_temp", "as_shot_tint", "subj_level"):
        ref[k] = mid(k, refs)
    from .changes import describe

    summary = describe(crs)
    had_denoise = any(_denoised(c) for _, c, _ in docs)
    return {"name": name, "n": len(docs), "files": [d[0] for d in docs], "crs": crs, "ref": ref,
            "had_denoise": had_denoise,
            "summary": [f"{s['label']} {s['value']}" for s in summary["sliders"]] +
                       [f"{m['name']}: {', '.join(m['effects'])}" for m in summary["masks"]]}


def _median_masks(crs: dict[str, Any], all_crs: list[dict[str, Any]]) -> None:
    """Werte gleicher Masken (gleicher Name und gleiche Maskenart) über alle Vorlagen mitteln."""
    from .masks import signature

    def key(c: dict[str, Any]) -> tuple[str, str]:
        return str(c.get("CorrectionName", "")), signature(c)

    for corr in crs.get("MaskGroupBasedCorrections") or []:
        if not isinstance(corr, dict):
            continue
        same = [c for other in all_crs for c in other.get("MaskGroupBasedCorrections") or []
                if isinstance(c, dict) and key(c) == key(corr)]
        if len(same) < 2:
            continue
        for k in [k for k in corr if k.startswith("Local") and _num(corr[k]) is not None]:
            vals = [v for c in same if (v := _num(c.get(k))) is not None]
            if vals:
                corr[k] = round(float(np.median(vals)), 6)


def learn_template(name: str, paths: list[str | Path],
                   progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    files = collect_xmps(paths)
    docs, refs = [], []
    for i, f in enumerate(files):
        if progress:
            progress(i, len(files))
        try:
            doc = read_xmp(f)
        except Exception as e:  # noqa: BLE001
            log.warning("XMP %s nicht lesbar: %s", f, e)
            continue
        if not doc.crs or _num(doc.crs.get("Exposure2012")) is None:
            continue
        docs.append((f.name, doc.crs, _exif(doc.other)))
        raw = _raw_beside(f)
        if raw is not None:
            refs.append(_measure_raw(raw) or {})
    if not docs:
        raise ValueError("In der Auswahl ist keine XMP mit Lightroom-Bearbeitung. In Lightroom die Bilder wählen "
                         "und „Metadaten in Datei speichern“ (⌘S), dann die .xmp-Dateien wählen.")
    t = build_template(name, docs, refs)
    _file(name).write_text(json.dumps(t, indent=1, ensure_ascii=False, default=float), "utf-8")
    if progress:
        progress(len(files), len(files))
    return t


# ---------------------------------------------------------------------------
# Anwenden
# ---------------------------------------------------------------------------

def scene_level(a: dict[str, Any]) -> float | None:
    """Helligkeit der Szene im RAW (log2). Median und oberes Viertel gemischt: weniger abhängig davon,
    wie viel dunkler Himmel im Bild ist."""
    m, p75 = a.get("lin_log_median"), a.get("lin_log_p75")
    if m is None:
        return None
    return float(m) if p75 is None else 0.5 * float(m) + 0.5 * float(p75)


def subject_level(img: np.ndarray | None, subject: np.ndarray | None) -> float | None:
    """Helligkeit der Spieler (log2, linear) in der Kamera-Vorschau. Sie enthält Licht UND Kamera-Einstellung."""
    import cv2

    if img is None or subject is None:
        return None
    m = cv2.resize(subject.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
    if m.mean() < 0.005:
        return None
    x = img[m].astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    y = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    return float(np.log2(max(float(np.median(y)), 1e-5)))


def _subject_level_of(it: Any) -> float | None:
    a = it.record.analysis
    if a.get("subj_level") is not None:
        return float(a["subj_level"])
    if not getattr(it, "preview", None) or getattr(it, "subject_mask", None) is None:
        return None
    import cv2

    img = cv2.imread(it.preview, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    v = subject_level(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), it.subject_mask)
    a["subj_level"] = v
    return v


def camera_ev(exif: dict[str, Any]) -> float | None:
    """log2 der Sensor-Belichtung (Zeit · ISO / Blende²)."""
    t, iso, n = exif.get("exposure_time"), exif.get("iso"), exif.get("aperture")
    if not t or not iso or not n:
        return None
    return math.log2(float(t) * float(iso) / (float(n) ** 2))


def _move_gradient(m: dict[str, Any], src: int, dst: int) -> dict[str, Any]:
    if src == dst:
        return m
    for a, b in (("ZeroX", "ZeroY"), ("FullX", "FullY")):
        x, y = to_number(m.get(a)), to_number(m.get(b))
        if x is None or y is None:
            continue
        m[a], m[b] = (round(v, 6) for v in display_to_sensor(*sensor_to_display(x, y, src), dst))
    return m


def _move_radial(m: dict[str, Any], src: int, dst: int) -> dict[str, Any]:
    if src == dst:
        return m
    left, top, right, bottom = (to_number(m.get(k)) for k in ("Left", "Top", "Right", "Bottom"))
    if None in (left, top, right, bottom):
        return m
    p0 = display_to_sensor(*sensor_to_display(left, top, src), dst)
    p1 = display_to_sensor(*sensor_to_display(right, bottom, src), dst)
    m["Left"], m["Right"] = round(min(p0[0], p1[0]), 6), round(max(p0[0], p1[0]), 6)
    m["Top"], m["Bottom"] = round(min(p0[1], p1[1]), 6), round(max(p0[1], p1[1]), 6)
    if to_number(m.get("Angle")) and (src in (5, 6, 7, 8)) != (dst in (5, 6, 7, 8)):
        m["Angle"] = round(float(to_number(m["Angle"])) + 90.0, 4)
    return m


def masks_for(tpl: dict[str, Any], orientation: int, subject_mask: np.ndarray | None = None,
              ai_masks: bool = True) -> list[dict[str, Any]]:
    """Masken der Vorlage mit denselben Werten; Geometrie auf die Ausrichtung des Bildes umgerechnet."""
    from . import masks as mk

    src = int(tpl.get("ref", {}).get("orientation") or 1)
    out = []
    for corr in tpl["crs"].get("MaskGroupBasedCorrections") or []:
        if not isinstance(corr, dict):
            continue
        c = refresh_sync_ids(strip_digests(copy.deepcopy(corr)))
        comps = []
        for m in c.get("CorrectionMasks") or []:
            if not isinstance(m, dict):
                continue
            what = m.get("What")
            if what == "Mask/Gradient":
                comps.append(_move_gradient(m, src, orientation))
            elif what == "Mask/CircularGradient":
                comps.append(_move_radial(m, src, orientation))
            elif what == "Mask/Image":
                m = {k: v for k, v in m.items() if k not in AI_IMAGE_KEYS}
                if ai_masks:
                    comps.append(m)
                elif subject_mask is not None and str(m.get("MaskSubType")) == "1":
                    inv = str(m.get("MaskInverted", "false")).lower() == "true"
                    comps += mk.paint_components(1 - subject_mask if inv else subject_mask, orientation,
                                                 str(m.get("MaskName", "Motiv")))
            elif what == "Mask/Paint":
                # Pinselstriche gehören zum alten Bild; als Ersatz das erkannte Motiv
                if subject_mask is not None and not any(x.get("What") == "Mask/Paint" for x in comps):
                    comps += mk.paint_components(subject_mask, orientation, str(m.get("MaskName", "Pinsel")))
            else:
                comps.append(m)
        if comps:
            c["CorrectionMasks"] = comps
            out.append(c)
    return out


def adjust(tpl: dict[str, Any], analysis: dict[str, Any], exif: dict[str, Any],
           shoot_ref: dict[str, float | None] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Globale Werte der Vorlage + leichte Anpassung an dieses Bild -> (crs-Werte, Notizen)."""
    crs = tpl["crs"]
    ref = dict(tpl.get("ref") or {})
    for k, v in (shoot_ref or {}).items():          # ohne RAW der Vorlage: Shoot-Mitte als Bezug
        if ref.get(k) is None and v is not None:
            ref[k] = v
    out: dict[str, Any] = {}
    notes: list[str] = []
    exp = float(_num(crs.get("Exposure2012")) or 0.0)
    cam_now, cam_ref = camera_ev(exif), camera_ev(ref)
    d_cam = float(np.clip(cam_ref - cam_now, -MAX_CAMERA_EV, MAX_CAMERA_EV)) if cam_now is not None and \
        cam_ref is not None else 0.0
    level = scene_level(analysis)
    d_scene = 0.0
    subj, subj_ref = analysis.get("subj_level"), ref.get("subj_level")
    if subj is not None and subj_ref is not None:
        # Spieler gemessen (Kamera-Vorschau): enthält schon die Kamera-Einstellung
        d_cam = 0.0
        d_scene = float(np.clip(SUBJECT_GAIN * (float(subj_ref) - float(subj)), -MAX_SUBJECT_EV, MAX_SUBJECT_EV))
    elif level is not None and ref.get("level") is not None:
        # Szene = RAW-Helligkeit ohne den Einfluss der Kamera-Einstellung
        s_now = level - (cam_now if cam_now is not None and cam_ref is not None else 0.0)
        s_ref = float(ref["level"]) - (cam_ref if cam_now is not None and cam_ref is not None else 0.0)
        d_scene = float(np.clip(0.6 * (s_ref - s_now), -MAX_SCENE_EV, MAX_SCENE_EV))
    out["Exposure2012"] = round(float(np.clip(exp + d_cam + d_scene, -5, 5)), 2)
    if abs(d_cam) >= 0.05:
        notes.append(f"Belichtung {d_cam:+.2f} (andere Kamera-Einstellung)")
    if abs(d_scene) >= 0.05:
        what = "Spieler" if subj is not None and subj_ref is not None else "Szene"
        notes.append(f"Belichtung {d_scene:+.2f} ({what} {'dunkler' if d_scene > 0 else 'heller'} als in der Vorlage)")
    temp, tint = _num(crs.get("Temperature")), _num(crs.get("Tint"))
    if str(crs.get("WhiteBalance", "As Shot")) != "As Shot" and temp:
        a_t, a_tint = analysis.get("as_shot_temp"), analysis.get("as_shot_tint")
        dm = dt = 0.0
        if a_t and ref.get("as_shot_temp"):
            dm = float(np.clip(mired(a_t) - mired(ref["as_shot_temp"]), -MAX_WB_MIRED, MAX_WB_MIRED))
        if a_tint is not None and ref.get("as_shot_tint") is not None:
            dt = float(np.clip(a_tint - ref["as_shot_tint"], -MAX_WB_TINT, MAX_WB_TINT))
        out["WhiteBalance"] = "Custom"
        out["Temperature"] = int(round(temp)) if dm == 0 else \
            int(round(float(np.clip(1e6 / (mired(temp) + dm), 2000, 50000)) / 50) * 50)
        out["Tint"] = int(round(float(np.clip((tint or 0.0) + dt, -150, 150))))
        if abs(dm) >= 3 or abs(dt) >= 3:
            notes.append(f"Weissabgleich {out['Temperature']} K / {out['Tint']:+d} (anderes Licht)")
    return out, notes


def shoot_reference(items: list[Any]) -> dict[str, float | None]:
    """Mitte des Shoots (Szenenhelligkeit, Kamera-Weissabgleich, Einstellung): Bezug, wenn die RAW der Vorlage
    nicht gemessen werden konnte. Die Vorlage gilt dann für ein typisches Bild dieses Shoots."""
    def med(vals: list[float | None]) -> float | None:
        v = [float(x) for x in vals if x is not None]
        return float(np.median(v)) if v else None

    recs = [it.record for it in items]
    return {"level": med([scene_level(r.analysis) for r in recs]),
            "subj_level": med([r.analysis.get("subj_level") for r in recs]),
            "as_shot_temp": med([r.analysis.get("as_shot_temp") for r in recs]),
            "as_shot_tint": med([r.analysis.get("as_shot_tint") for r in recs]),
            "iso": med([r.exif.get("iso") for r in recs]),
            "exposure_time": med([r.exif.get("exposure_time") for r in recs]),
            "aperture": med([r.exif.get("aperture") for r in recs])}


def shoot_reference_db(db: Any, shoot_id: int) -> dict[str, float | None]:
    """Wie ``shoot_reference``, direkt aus der Datenbank (für die Vorschau einzelner Bilder)."""
    from types import SimpleNamespace

    rows = db.query("SELECT a.data, i.iso, i.exposure_time, i.aperture FROM images i JOIN analysis a "
                    "ON a.image_id=i.id WHERE i.shoot_id=?", (shoot_id,))
    items = [SimpleNamespace(record=SimpleNamespace(analysis=json.loads(r["data"] or "{}"), exif={
        "iso": r["iso"], "exposure_time": r["exposure_time"], "aperture": r["aperture"]})) for r in rows]
    return shoot_reference(items)


def develop_items(items: list[Any], tpl: dict[str, Any], settings: Any, dialect: Any,
                  shoot_ref: dict[str, float | None] | None = None) -> None:
    """Jedes Bild: Vorlage 1:1, dann Belichtung/Weissabgleich/Begradigen pro Bild."""
    from ..lightroom.params import BASE_FLAGS
    from .develop import _crop

    for it in items:
        _subject_level_of(it)
    ref_all = shoot_ref if shoot_ref is not None else shoot_reference(items)
    for it in items:
        a = it.record.analysis
        it.preset, it.prediction, it.confidence = "", None, 0.9
        crs: dict[str, Any] = {**BASE_FLAGS, **{k: copy.deepcopy(v) for k, v in tpl["crs"].items()
                                                 if k != "MaskGroupBasedCorrections"}}
        crs.setdefault("Version", settings.develop.camera_raw_version or dialect.crs_version)
        crs.setdefault("ProcessVersion", settings.develop.process_version or dialect.process_version)
        vals, notes = adjust(tpl, a, it.record.exif, ref_all)
        crs.update(vals)
        it.notes = [f"Vorlage „{tpl['name']}“ 1:1"] + notes
        crs.update(_crop(it, settings))
        # Entrauschen: wie in der Vorlage (Lightroom-Denoise ist an das Bild gebunden, wird neu angefordert)
        it.denoise = None
        if tpl.get("had_denoise") and settings.denoise.mode == "lightroom":
            it.denoise = int(settings.denoise.default_amount)
            crs.update(dialect.denoise(it.denoise))
        if settings.develop.write_masks:
            corr = masks_for(tpl, it.orientation, it.subject_mask, settings.develop.ai_masks)
            if corr:
                crs["MaskGroupBasedCorrections"] = corr
        it.crs = crs


def _denoised(crs: dict[str, Any]) -> bool:
    """Hat die Vorlage entrauscht (Lightroom-Denoise oder Regler Luminanz)?"""
    if any(k.startswith("EnhanceDenoise") for k in crs) or (_num(crs.get("LuminanceSmoothing")) or 0) > 0:
        return True
    return "Denoise" in json.dumps(crs.get("FilterList") or {})

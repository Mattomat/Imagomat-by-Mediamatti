"""Jobs: Stilprofil trainieren, Shoot entwickeln, aus Korrekturen lernen, Profile teilen."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import load_masks
from ..config import load_settings, profiles_dir
from ..db import Database, blob_to_f32, dumps
from ..jobs import JobContext, job
from ..lightroom.dialect import Dialect, learn_dialect
from ..lightroom.xmp import read_xmp
from .develop import ImageDevelop, develop_items
from .features import image_record
from .model import Record, StyleModel
from .sources import TrainingSample, filter_samples, from_catalog, from_folder, load_preset

log = logging.getLogger(__name__)


def samples_file(profile: str) -> Path:
    d = profiles_dir() / profile
    d.mkdir(parents=True, exist_ok=True)
    return d / "samples.jsonl"


def _append_samples(profile: str, recs: list[Record]) -> None:
    with open(samples_file(profile), "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"path": r.path, "analysis": r.analysis, "exif": r.exif,
                                "embedding": None if r.embedding is None else [float(x) for x in r.embedding],
                                "crs": r.crs, "weight": r.weight}, default=float, ensure_ascii=False) + "\n")


def load_samples(profile: str) -> list[Record]:
    p = samples_file(profile)
    if not p.exists():
        return []
    by_path: dict[str, Record] = {}
    for line in p.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        emb = np.asarray(d["embedding"], dtype=np.float32) if d.get("embedding") else None
        # Neuere Einträge (Korrekturen) ersetzen ältere desselben Bildes
        by_path[d["path"]] = Record(d["analysis"], d["exif"], emb, d["crs"], float(d.get("weight", 1.0)), d["path"])
    return list(by_path.values())


@job("train_profile")
def train_profile(ctx: JobContext, name: str, catalog: str | None = None, folders: list[str] | None = None,
                  presets: list[str] | None = None, base_preset: str | None = None, min_rating: int = 0,
                  only_picked: bool = False, append: bool = False) -> None:
    samples: list[TrainingSample] = []
    labels: list[str] = []
    ctx.progress(0, "Trainingsdaten sammeln")
    if catalog:
        cat = list(from_catalog(Path(catalog)))
        labels += [s.label for s in cat if s.label]
        samples += cat
    for f in folders or []:
        fs = list(from_folder(Path(f)))
        labels += [s.label for s in fs if s.label]
        samples += fs
    # Dialekt aus ALLEN Einstellungen lernen (auch ungefilterte)
    dialect = learn_dialect([s.crs for s in samples], labels, Dialect.load())
    dialect.save()
    good = filter_samples(samples, min_rating=min_rating, only_picked=only_picked)
    ctx.set_total(len(good) + 1)
    recs: list[Record] = []
    for i, s in enumerate(good):
        ctx.check()
        try:
            r = image_record(s.path)
            a = {**r["analysis"], "orientation": r.get("orientation") or 1}
            recs.append(Record(a, r["exif"], np.asarray(r["embedding"], np.float32), s.crs, s.weight, str(s.path)))
        except Exception as e:  # noqa: BLE001 - einzelne defekte Bilder überspringen
            log.warning("Merkmale für %s fehlgeschlagen: %s", s.path, e)
        ctx.progress(i + 1, f"Merkmale {i + 1}/{len(good)}")
    if not append:
        samples_file(name).unlink(missing_ok=True)
    _append_samples(name, recs)
    all_recs = load_samples(name)
    ctx.progress(len(good), f"Trainiere Modell mit {len(all_recs)} Bildern")
    model = StyleModel(name, base_preset).fit(all_recs, dialect.denoise_amount_key)
    for p in presets or []:
        model.look_override.update({k: v for k, v in load_preset(Path(p)).items()
                                    if k in ("CameraProfile", "Look", "ProfileName")})
    d = model.save()
    (d / "sources.json").write_text(json.dumps({"catalog": catalog, "folders": folders, "presets": presets,
                                                "skipped": len(samples) - len(good)}, indent=2), "utf-8")
    ctx.progress(len(good) + 1, f"Profil '{name}' trainiert ({model.n} Bilder)")


def shoot_records(db: Database, shoot_id: int, only_keep: bool = True) -> list[ImageDevelop]:
    sql = ("SELECT i.*, a.data, a.embedding FROM images i JOIN analysis a ON a.image_id=i.id "
           "LEFT JOIN culling c ON c.image_id=i.id WHERE i.shoot_id=?")
    if only_keep:
        sql += " AND (c.decision IS NULL OR c.decision='keep')"
    out = []
    for r in db.query(sql + " ORDER BY i.capture_time", (shoot_id,)):
        a = json.loads(r["data"])
        exif = {"capture_time": r["capture_time"], "iso": r["iso"], "exposure_time": r["exposure_time"],
                "aperture": r["aperture"], "focal_length": r["focal_length"], "camera": r["camera"],
                "lens": r["lens"]}
        m = load_masks(r["id"])
        out.append(ImageDevelop(r["id"], Record(a, exif, blob_to_f32(r["embedding"]), path=r["path"]),
                                int(r["orientation"] or 1), r["width"], r["height"],
                                subject_mask=m[0] if m else None))
    return out


@job("develop")
def develop_shoot(ctx: JobContext, shoot_id: int, profile: str | None = None, preset: str | None = None,
                  only_keep: bool = True) -> None:
    db = ctx.db
    settings = load_settings()
    shoot = db.one("SELECT * FROM shoots WHERE id=?", (shoot_id,))
    profile = profile or (shoot["profile"] if shoot else None) or settings.default_profile
    model = StyleModel.load(profile) if profile else None
    items = shoot_records(db, shoot_id, only_keep)
    ctx.set_total(len(items))
    ctx.progress(0, f"Entwickle {len(items)} Bilder mit {'Profil ' + profile if model else 'Preset'}")
    develop_items(items, model, settings, Dialect.load(), preset)
    import time

    with db.tx() as c:
        for it in items:
            masks = it.crs.get("MaskGroupBasedCorrections")
            c.execute("INSERT OR REPLACE INTO edits(image_id, profile, params, masks, confidence, denoise,"
                      " user_params, updated_at) VALUES(?,?,?,?,?,?,"
                      " (SELECT user_params FROM edits WHERE image_id=?),?)",
                      (it.image_id, profile or f"preset:{it.preset}",
                       dumps({k: v for k, v in it.crs.items() if k != "MaskGroupBasedCorrections"}),
                       dumps(masks) if masks else None, it.confidence, it.denoise, it.image_id, time.time()))
    for it in items:
        db.update_analysis(it.image_id, {"develop_notes": it.notes, "preset": it.preset})
    ctx.progress(len(items), "Entwicklung fertig")


def user_changed(predicted: dict[str, Any], final: dict[str, Any]) -> bool:
    """Hat die Person in Lightroom etwas an unseren Werten geändert?"""
    from ..lightroom.params import PARAMS, to_number

    for k, p in PARAMS.items():
        a, b = to_number(predicted.get(k)), to_number(final.get(k))
        if a is None and b is None:
            continue
        tol = 0.05 if p.decimals else 1.0
        if abs((a if a is not None else p.default) - (b if b is not None else p.default)) > tol:
            return True
    return str(predicted.get("ToneCurvePV2012")) != str(final.get("ToneCurvePV2012")) and \
        final.get("ToneCurvePV2012") is not None


@job("feedback")
def feedback(ctx: JobContext, shoot_id: int, profile: str, folder: str | None = None, weight: float = 2.0) -> None:
    """Deine Korrekturen aus Lightroom (XMP) zurücklesen und das Profil nachtrainieren."""
    from .sources import embedded_xmp

    db = ctx.db
    if folder is None:
        row = db.one("SELECT settings FROM shoots WHERE id=?", (shoot_id,))
        folder = (json.loads(row["settings"] or "{}") if row else {}).get("last_export")
    items = shoot_records(db, shoot_id, only_keep=False)
    ctx.set_total(len(items) + 1)
    recs = []
    for i, it in enumerate(items):
        src = Path(it.record.path)
        base = Path(folder) if folder else src.parent
        doc = None
        side = (base / src.name).with_suffix(".xmp")
        dn = base / f"{src.stem}-DN.dng"
        if side.exists():
            doc = read_xmp(side)
        elif dn.exists():
            doc = embedded_xmp(dn)
        if doc is None or not doc.crs:
            continue
        pred = db.one("SELECT params FROM edits WHERE image_id=?", (it.image_id,))
        if pred and not user_changed(json.loads(pred["params"]), doc.crs):
            continue  # unverändert übernommen: kein neues Lernsignal
        it.record.crs = doc.crs
        it.record.weight = weight
        recs.append(it.record)
        with db.tx() as c:
            c.execute("UPDATE edits SET user_params=? WHERE image_id=?", (dumps(doc.crs), it.image_id))
        ctx.progress(i + 1, "Korrekturen lesen")
    if not recs:
        ctx.progress(len(items) + 1, "Keine Korrekturen gefunden")
        return
    _append_samples(profile, recs)
    old = StyleModel.load(profile)
    model = StyleModel(profile, old.base_preset if old else None)
    if old:
        model.look_override = old.look_override
    model.fit(load_samples(profile), Dialect.load().denoise_amount_key)
    model.save()
    ctx.progress(len(items) + 1, f"{len(recs)} Korrekturen gelernt, Profil aktualisiert ({model.n} Bilder)")


def export_profile(name: str, target: Path) -> Path:
    src = profiles_dir() / name
    target = Path(target).with_suffix(".imagomat-profile")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in src.iterdir():
            if f.is_file():
                z.write(f, f.name)
    return target


def import_profile(archive: Path, name: str | None = None) -> str:
    with zipfile.ZipFile(archive) as z:
        meta = json.loads(z.read("meta.json"))
        name = name or meta["name"]
        d = profiles_dir() / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        z.extractall(d)
    m = StyleModel.load(name)
    if m and m.name != name:
        m.name = name
        m.save()
    return name


def profile_info(name: str) -> dict[str, Any]:
    p = profiles_dir() / name / "meta.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}

"""Export eines Shoots.

Varianten (kombinierbar):
- "xmp":     Ordner mit RAW + XMP-Sidecars (bzw. DNG mit eingebettetem XMP), bereit für den
             Lightroom-Import. Originale werden kopiert (oder per Hardlink verknüpft), nie verändert.
             Modus "inplace" schreibt nur .xmp neben die Original-RAWs (die RAW-Datei bleibt unberührt).
- "catalog": EXPERIMENTELL: zusätzlich ein fertiger Lightroom-Katalog (.lrcat) aus einer Vorlage.
- "jpeg":    Galerie-JPEGs aus der eigenen Render-Pipeline (Annäherung, nicht Lightroom-Qualität).

Immer: Export-Log (CSV + JSON) mit Entscheidung und Gründen je Bild und ein Manifest
(imagomat.json) für das Lightroom-Plugin.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis import load_masks
from ..config import load_settings
from ..culling.engine import REASONS_DE
from ..vision.action import MOMENTS_DE
from ..db import Database
from ..io import raw as raw_io
from ..jobs import JobContext, job
from ..lightroom.catalog_writer import CatalogPhoto, write_catalog
from ..lightroom.dialect import Dialect
from ..lightroom.xmp import FaceRegion, XmpDoc, serialize, write_sidecar
from ..people.registry import image_people
from ..render.pipeline import render
from ..vision.geometry import display_to_sensor

log = logging.getLogger(__name__)
LOW_CONFIDENCE = 0.35


@dataclass
class ExportItem:
    image_id: int
    src: Path
    decision: str
    rating: int | None
    label: str | None
    reasons: list[str]
    series: int | None
    is_best: bool
    score: float | None
    crs: dict[str, Any]
    confidence: float | None
    denoise: int | None
    keywords: list[str] = field(default_factory=list)
    regions: list[FaceRegion] = field(default_factory=list)
    region_dims: tuple[int, int] | None = None
    orientation: int = 1
    capture_time: float | None = None
    exif: dict[str, Any] = field(default_factory=dict)
    people: list[str] = field(default_factory=list)
    out: Path | None = None
    notes: list[str] = field(default_factory=list)
    preset: str | None = None
    dims: tuple[int, int] | None = None


def _items(db: Database, shoot_id: int, include_rejected: bool) -> list[ExportItem]:
    s = load_settings()
    kw = s.keywords
    dialect = Dialect.load()
    rows = db.query(
        "SELECT i.*, c.decision, c.rating, c.label, c.reasons, c.series_id, c.is_series_best, c.score, "
        "e.params, e.masks, e.confidence, e.denoise, a.data FROM images i "
        "LEFT JOIN culling c ON c.image_id=i.id LEFT JOIN edits e ON e.image_id=i.id "
        "LEFT JOIN analysis a ON a.image_id=i.id WHERE i.shoot_id=? ORDER BY i.capture_time, i.filename",
        (shoot_id,))
    out = []
    for r in rows:
        decision = r["decision"] or "keep"
        if decision == "reject" and not include_rejected:
            continue
        a = json.loads(r["data"]) if r["data"] else {}
        crs = json.loads(r["params"]) if r["params"] else {}
        if r["masks"]:
            crs["MaskGroupBasedCorrections"] = json.loads(r["masks"])
        reasons = json.loads(r["reasons"]) if r["reasons"] else []
        it = ExportItem(
            r["id"], Path(r["path"]), decision, r["rating"], r["label"], reasons, r["series_id"],
            bool(r["is_series_best"]), r["score"], crs, r["confidence"], r["denoise"],
            orientation=int(r["orientation"] or 1), capture_time=r["capture_time"],
            exif={"iso": r["iso"], "exposure_time": r["exposure_time"], "aperture": r["aperture"],
                  "focal_length": r["focal_length"], "camera": r["camera"]},
            notes=a.get("develop_notes", []), preset=a.get("preset"),
            dims=(int(r["width"]), int(r["height"])) if r["width"] and r["height"] else None,
        )
        # Stichwörter
        for p in image_people(db, r["id"]):
            it.keywords.append(p.keyword)
            it.people.append(p.name)
        if decision == "reject":
            for rs in reasons[:2] or ["strenge"]:
                it.keywords.append(f"{kw.culling_root}|Aussortiert|{REASONS_DE.get(rs, rs)}")
        else:
            it.keywords.append(f"{kw.culling_root}|Behalten")
            if it.is_best:
                it.keywords.append(f"{kw.culling_root}|Bestes der Serie")
            if a.get("moment") in MOMENTS_DE:
                it.keywords.append(f"Imagomat|Moment|{MOMENTS_DE[a['moment']]}")
        if it.denoise:
            it.keywords.append(kw.denoise_keyword)
            if s.denoise.mode == "mark" and not it.label:
                it.label = dialect.label("purple")
        if it.confidence is not None and it.confidence < LOW_CONFIDENCE and decision == "keep":
            it.keywords.append(kw.review_keyword)
            if not it.label and s.culling.label_review:
                it.label = dialect.label("yellow")
        if it.label:
            it.label = dialect.label(it.label)
        # Gesichtsregionen (MWG) für benannte Personen
        if kw.write_face_regions:
            W, H = r["width"], r["height"]
            for f in db.query("SELECT f.bbox, p.name FROM faces f JOIN persons p ON p.id=f.person_id "
                              "WHERE f.image_id=?", (r["id"],)):
                x0, y0, x1, y1 = json.loads(f["bbox"])
                (sx0, sy0), (sx1, sy1) = (display_to_sensor(x0, y0, it.orientation),
                                          display_to_sensor(x1, y1, it.orientation))
                lx, rx, ty, by = min(sx0, sx1), max(sx0, sx1), min(sy0, sy1), max(sy0, sy1)
                it.regions.append(FaceRegion(f["name"], (lx + rx) / 2, (ty + by) / 2, rx - lx, by - ty))
            if it.regions and W and H:
                it.region_dims = (int(W), int(H))
        out.append(it)
    return out


def _place(src: Path, dst: Path, mode: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return dst
        except OSError:
            pass
    shutil.copy2(src, dst)
    return dst


def _doc(it: ExportItem, with_develop: bool) -> XmpDoc:
    return XmpDoc(rating=it.rating, label=it.label, keywords=it.keywords,
                  crs=it.crs if with_develop else {}, regions=it.regions, region_dims=it.region_dims)


def export_xmp(items: list[ExportItem], target: Path, mode: str, ctx: JobContext | None = None) -> None:
    s = load_settings()
    from ..denoise.local import denoise_to_dng

    for i, it in enumerate(items):
        if ctx:
            ctx.check()
        with_develop = it.decision == "keep" and bool(it.crs)
        local_dn = s.denoise.mode == "local" and it.denoise and raw_io.is_raw(it.src) and with_develop
        if local_dn:
            crs = {k: v for k, v in it.crs.items() if not k.startswith("Enhance")}
            doc = XmpDoc(rating=it.rating, label=it.label, keywords=it.keywords, crs=crs, regions=it.regions,
                         region_dims=it.region_dims)
            dst = target / f"{it.src.stem}-DN.dng"
            if not dst.exists():
                res = denoise_to_dng(it.src, dst, int(it.denoise or 50), s.denoise.local_model,
                                     xmp=serialize(doc, include_parent_keywords=s.keywords.write_parent_keywords),
                                     exif=it.exif)
                it.notes.append(f"lokal entrauscht ({res['method']})")
            it.out = dst
        elif mode == "inplace":
            if not raw_io.is_raw(it.src) or it.src.suffix.lower() == ".dng":
                # DNG/JPEG: Lightroom liest nur eingebettetes XMP -> nie das Original verändern
                it.out = _place(it.src, target / it.src.name, "copy")
                _embed_or_sidecar(it, _doc(it, with_develop))
            else:
                it.out = it.src
                write_sidecar(it.src, _doc(it, with_develop),
                              include_parent_keywords=s.keywords.write_parent_keywords)
        else:
            it.out = _place(it.src, target / it.src.name, mode)
            _embed_or_sidecar(it, _doc(it, with_develop))
        if ctx:
            ctx.progress(advance=1, message=f"XMP {i + 1}/{len(items)}")


def _embed_or_sidecar(it: ExportItem, doc: XmpDoc) -> None:
    """Sidecar für proprietäre RAWs; für Kopien von DNG/JPEG zusätzlich ein Sidecar
    (Lightroom: DNG/JPEG -> 'Metadaten aus Datei lesen' nutzt eingebettetes XMP; wir schreiben
    ein Sidecar und vermerken es, eingebettetes Schreiben übernimmt ExifTool, falls vorhanden)."""
    s = load_settings()
    assert it.out is not None
    side = write_sidecar(it.out, doc, include_parent_keywords=s.keywords.write_parent_keywords)
    if it.out.suffix.lower() in (".dng", ".jpg", ".jpeg", ".tif", ".tiff") and shutil.which("exiftool"):
        import subprocess

        subprocess.run(["exiftool", "-q", "-overwrite_original", "-tagsfromfile", str(side), "-xmp:all",
                        str(it.out)], check=False)


def export_jpeg(items: list[ExportItem], target: Path, long_side: int = 2048, ctx: JobContext | None = None) -> None:
    d = target / "JPEG (Vorschau, nicht Lightroom-Qualität)"
    d.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        if ctx:
            ctx.check()
        if it.decision != "keep":
            continue
        try:
            if raw_io.is_raw(it.src):
                lin, info = raw_io.decode_any(it.src, half_size=True)
                masks = load_masks(it.image_id)
                seg = {"subject": masks[0], "sky": masks[1]} if masks else {}
                img = render(lin, info.xyz_to_cam, info.camera_wb, it.crs, info.orientation, seg, long_side)
            else:
                img, _ = raw_io.load_preview(it.src, long_side)
            cv2.imwrite(str(d / f"{it.src.stem}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:  # noqa: BLE001
            log.warning("JPEG-Export fehlgeschlagen für %s: %s", it.src.name, e)
        if ctx:
            ctx.progress(advance=1, message=f"JPEG {i + 1}/{len(items)}")


def _collections(shoot_name: str, it: ExportItem) -> list[str]:
    base = f"Imagomat|{shoot_name}"
    c = [f"{base}|{'Behalten' if it.decision == 'keep' else 'Aussortiert'}"]
    if it.decision == "reject" and it.reasons:
        c.append(f"{base}|Aussortiert nach Grund|{REASONS_DE.get(it.reasons[0], it.reasons[0])}")
    for p in it.people:
        c.append(f"{base}|Personen|{p}")
    if it.denoise:
        c.append(f"{base}|Denoise")
    if load_settings().keywords.review_keyword in it.keywords:
        c.append(f"{base}|Prüfen")
    return c


def export_catalog(items: list[ExportItem], target: Path, template: Path, shoot_name: str) -> dict[str, Any]:
    photos = []
    for it in items:
        if it.out is None:
            continue
        w, h = it.dims or it.region_dims or (6000, 4000)
        photos.append(CatalogPhoto(
            path=it.out, width=w, height=h, orientation=it.orientation,
            capture_time=it.capture_time, rating=it.rating, pick=1 if it.decision == "keep" else -1,
            color_label=it.label, keywords=it.keywords,
            develop=it.crs if it.decision == "keep" and not it.out.name.endswith("-DN.dng") else {},
            xmp=serialize(_doc(it, it.decision == "keep")).decode("utf-8"),
            collections=_collections(shoot_name, it), stack=it.series, stack_position=0 if it.is_best else 1,
            iso=it.exif.get("iso"), aperture=it.exif.get("aperture"), shutter=it.exif.get("exposure_time"),
            focal_length=it.exif.get("focal_length")))
    cat = target / f"{shoot_name} (Imagomat).lrcat"
    if cat.exists():
        cat.unlink()
    return write_catalog(template, cat, target, photos)


def write_log(items: list[ExportItem], target: Path, meta: dict[str, Any]) -> None:
    rows = []
    for it in items:
        rows.append({
            "datei": it.src.name, "entscheidung": "behalten" if it.decision == "keep" else "aussortiert",
            "gruende": "; ".join(REASONS_DE.get(r, r) for r in it.reasons), "sterne": it.rating,
            "farblabel": it.label or "", "serie": it.series, "bestes_der_serie": it.is_best,
            "score": round(it.score, 3) if it.score is not None else "", "personen": "; ".join(it.people),
            "denoise": it.denoise or "", "vertrauen": round(it.confidence, 2) if it.confidence is not None else "",
            "preset": it.preset or "", "hinweise": "; ".join(it.notes),
            "export": str(it.out.name) if it.out else "",
        })
    with open(target / "imagomat-log.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["datei"], delimiter=";")
        w.writeheader()
        w.writerows(rows)
    manifest = {
        **meta, "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "photos": [{"file": it.out.name if it.out else it.src.name, "source": str(it.src),
                    "pick": 1 if it.decision == "keep" else -1, "rating": it.rating, "label": it.label,
                    "collections": _collections(meta.get("shoot", "Shoot"), it),
                    "needs_ai_update": bool(it.crs.get("MaskGroupBasedCorrections")) or bool(
                        any(k.startswith("EnhanceDenoise") for k in it.crs)),
                    "series": it.series, "best": it.is_best} for it in items],
    }
    (target / "imagomat.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    (target / "imagomat-log.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), "utf-8")


@job("export")
def export_shoot(ctx: JobContext, shoot_id: int, target: str, formats: list[str] | None = None,
                 copy_mode: str = "copy", include_rejected: bool = True, template: str | None = None,
                 jpeg_side: int = 2048) -> None:
    db = ctx.db
    formats = formats or ["xmp"]
    shoot = db.one("SELECT * FROM shoots WHERE id=?", (shoot_id,))
    name = shoot["name"] if shoot else f"Shoot {shoot_id}"
    tgt = Path(target).expanduser()
    tgt.mkdir(parents=True, exist_ok=True)
    items = _items(db, shoot_id, include_rejected)
    ctx.set_total(len(items) * (("xmp" in formats) + ("jpeg" in formats)) + 1)
    if "xmp" in formats or "catalog" in formats:
        export_xmp(items, tgt, copy_mode, ctx)
    meta: dict[str, Any] = {"shoot": name, "formats": formats, "copy_mode": copy_mode}
    if "catalog" in formats:
        if not template:
            raise ValueError("Für den Katalog-Export wird ein leerer Vorlagen-Katalog aus Lightroom benötigt")
        meta["catalog"] = export_catalog(items, tgt, Path(template), name)
    if "jpeg" in formats:
        export_jpeg(items, tgt, jpeg_side, ctx)
    write_log(items, tgt, meta)
    row = db.one("SELECT settings FROM shoots WHERE id=?", (shoot_id,))
    st = json.loads(row["settings"] or "{}") if row else {}
    st["last_export"] = str(tgt)
    with db.tx() as c:
        c.execute("UPDATE shoots SET settings=? WHERE id=?", (json.dumps(st), shoot_id))
    kept = sum(1 for it in items if it.decision == "keep")
    ctx.progress(message=f"Export fertig: {kept} behalten, {len(items) - kept} aussortiert -> {tgt}", advance=1)

"""Nur Personen: Bilder mit Namen (Stichwörter + Gesichtsbereiche) wieder ausgeben.

Für fertige JPGs (z. B. aus Lightroom exportiert): Imagomat erkennt die Personen und schreibt
die Namen direkt in die JPEG-Datei (Lightroom liest bei JPEGs keine Sidecars). Für RAWs entsteht
ein XMP-Sidecar nur mit Stichwörtern, ohne Entwicklung. Die Originale bleiben unverändert:
geschrieben wird in einen Zielordner.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import load_settings
from ..io import jpegxmp
from ..jobs import JobContext, job
from ..lightroom.xmp import FaceRegion, XmpDoc, parse_xmp, serialize
from ..people.registry import image_people

log = logging.getLogger(__name__)
JPEG_EXT = {".jpg", ".jpeg"}


def _face_regions(db, image_id: int) -> list[FaceRegion]:
    rows = db.query("SELECT f.bbox, p.name FROM faces f JOIN persons p ON p.id=f.person_id WHERE f.image_id=?",
                    (image_id,))
    out = []
    for r in rows:
        x0, y0, x1, y1 = json.loads(r["bbox"])
        out.append(FaceRegion(r["name"], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def tagged_doc(db, image_id: int, existing: bytes | None, dims: tuple[int, int] | None,
               with_regions: bool) -> XmpDoc:
    """Bestehende Stichwörter/Gesichter behalten, erkannte Personen ergänzen."""
    from ..vision.tags import labels

    old = parse_xmp(existing) if existing else XmpDoc()
    people = [p.keyword for p in image_people(db, image_id)]
    mode = load_settings().keywords.content_keywords
    content = labels(db.get_analysis(image_id).get("tags")) if mode == "always" or (
        mode == "no_person" and not people) else []
    kws = list(dict.fromkeys([*old.keywords, *people, *content]))
    regions = list(old.regions)
    if with_regions:
        known = {r.name for r in regions}
        regions += [r for r in _face_regions(db, image_id) if r.name not in known]
    return XmpDoc(keywords=kws, regions=regions, region_dims=dims or old.region_dims)


def unique_target(folder: Path, name: str) -> Path:
    p = folder / name
    i = 2
    while p.exists():
        p = folder / f"{Path(name).stem}-{i}{Path(name).suffix}"
        i += 1
    return p


@job("tag_export")
def tag_export(ctx: JobContext, shoot_id: int, target: str, only_with_people: bool = False,
               inplace: bool = False) -> dict[str, Any]:
    """Namen schreiben. inplace: RAWs bekommen eine XMP-Datei direkt daneben (Original unverändert, vorhandene
    Einstellungen bleiben); JPG/DNG/TIFF werden nie verändert und landen mit Namen im Zielordner."""
    db = ctx.db
    tgt = Path(target).expanduser()
    rows = db.query("SELECT id, path, filename, orientation, width, height FROM images WHERE shoot_id=? "
                    "ORDER BY capture_time, filename", (shoot_id,))
    kw = load_settings().keywords
    if kw.content_keywords != "off":
        from ..vision.tags import ensure_tags

        ctx.progress(0, "Bildinhalte erkennen (Fans, Team, Jubel …)")
        ensure_tags(db, [r["id"] for r in rows])
    ctx.set_total(len(rows))
    written = named = 0
    in_place = 0
    for i, r in enumerate(rows):
        ctx.check()
        src = Path(r["path"])
        people = image_people(db, r["id"])
        if only_with_people and not people:
            ctx.progress(i + 1)
            continue
        try:
            if inplace and src.suffix.lower() not in JPEG_EXT and src.suffix.lower() not in (".dng", ".tif", ".tiff"):
                side = src.with_suffix(".xmp")
                existing = side.read_bytes() if side.exists() else None
                doc = tagged_doc(db, r["id"], existing, None, with_regions=False)
                side.write_bytes(serialize(doc, existing, include_parent_keywords=kw.write_parent_keywords,
                                           replace_develop=False))
                written += 1
                in_place += 1
                named += bool(people)
                ctx.progress(i + 1, f"Namen schreiben {i + 1}/{len(rows)}")
                continue
            tgt.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() in JPEG_EXT:
                with Image.open(src) as im:
                    dims = im.size
                    orientation = im.getexif().get(0x0112, 1)
                existing = jpegxmp.read_jpeg_xmp(src)
                doc = tagged_doc(db, r["id"], existing, dims, with_regions=kw.write_face_regions and orientation == 1)
                packet = serialize(doc, existing, include_parent_keywords=kw.write_parent_keywords,
                                   replace_develop=False)
                out = unique_target(tgt, src.name)
                shutil.copy2(src, out)
                jpegxmp.embed_xmp(out, packet)
            else:
                # RAW/TIFF: Datei kopieren + Sidecar nur mit Stichwörtern (keine Entwicklung)
                out = unique_target(tgt, src.name)
                shutil.copy2(src, out)
                side_src = src.with_suffix(".xmp")
                existing = side_src.read_bytes() if side_src.exists() else None
                doc = tagged_doc(db, r["id"], existing, None, with_regions=False)
                out.with_suffix(".xmp").write_bytes(
                    serialize(doc, existing, include_parent_keywords=kw.write_parent_keywords, replace_develop=False))
            written += 1
            named += bool(people)
        except Exception as e:  # noqa: BLE001 - eine Datei darf den Export nicht stoppen
            log.warning("Namen schreiben fehlgeschlagen für %s: %s", src.name, e, exc_info=True)
        ctx.progress(i + 1, f"Namen schreiben {i + 1}/{len(rows)}")
    db.update_shoot_settings(shoot_id, last_tag_export=str(tgt))
    where = f"{in_place} direkt neben den RAWs" + (f", {written - in_place} in {tgt.name}" if written > in_place else "") \
        if inplace else str(tgt)
    ctx.progress(len(rows), f"{written} Bilder gespeichert, davon {named} mit Namen ({where}) -> {tgt}")
    return {"target": str(tgt), "written": written, "named": named, "in_place": in_place}


def safe_folder_name(name: str) -> str:
    return re.sub(r"[/:\\]", "-", name).strip() or "Imagomat"


@job("lr_pull")
def lr_pull(ctx: JobContext, shoot_id: int, catalog: str) -> dict[str, Any]:
    """Namen aus Lightroom übernehmen."""
    from ..lightroom.sync import pull

    ctx.progress(0, "Lightroom-Katalog lesen …")
    team = (ctx.db.shoot_settings(shoot_id).get("teams") or [None])[0]
    r = pull(ctx.db, shoot_id, Path(catalog).expanduser(), team)
    ctx.progress(1, f"Aus Lightroom: {r['faces']} Gesichter und {r['image_level']} weitere Namen in {r['images']} Bildern")
    return r


@job("lr_push")
def lr_push(ctx: JobContext, shoot_id: int, catalog: str) -> dict[str, Any]:
    """Namen in den Lightroom-Katalog schreiben (Lightroom geschlossen, Sicherung vorher)."""
    from ..lightroom.sync import push

    ctx.progress(0, "Namen in den Lightroom-Katalog schreiben …")
    r = push(ctx.db, shoot_id, Path(catalog).expanduser())
    ctx.progress(1, f"In Lightroom: {r['keywords']} Namen in {r['images']} Bildern (Sicherung: {Path(r['backup']).name})")
    return r

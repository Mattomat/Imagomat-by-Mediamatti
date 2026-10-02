"""Mit einem Lightroom-Katalog synchronisieren (Personen/Gesichter).

- Übernehmen: In Lightroom benannte Gesichter auf die Bilder dieses Shoots übertragen (gleiche Datei; Gesicht an
  derselben Stelle -> dieses Gesicht, sonst die Person fürs ganze Bild).
- Schreiben: Namen als Personen-Stichwörter in den Katalog schreiben. Nur bei geschlossenem Lightroom; vorher wird
  der Katalog gesichert (…-vor-Tagmatti-JJJJ-MM-TT-hhmm.lrcat). Die Gesichtsbereiche bekommt Lightroom zusätzlich
  über die XMP-Dateien ("Metadaten aus Datei lesen").
"""

from __future__ import annotations

import datetime as _dt
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from ..db import Database
from ..people.registry import upsert_person


def catalog_open(lrcat: Path) -> bool:
    """Ist der Katalog gerade in Lightroom geöffnet (Sperrdatei)?"""
    return any(Path(str(lrcat) + suf).exists() for suf in (".lock", "-lock")) or \
        lrcat.with_suffix(".lrcat.lock").exists()


def _key(p: Path | str) -> str:
    return str(p).replace("\\", "/").lower()


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def pull(db: Database, shoot_id: int, lrcat: Path, team: str | None = None) -> dict[str, Any]:
    import json

    from .catalog import CatalogReader

    from ..config import load_settings
    from ..keywords import get_or_create, set_state

    with CatalogReader(lrcat) as cat:
        named = cat.named_faces()
        lr_kws = {_key(i.path): i.keywords for i in cat.images(include_virtual_copies=False)}
        person_kw = cat.person_keywords()
    kw_by_name: dict[str, list[list[str]]] = {}
    for p, v in lr_kws.items():
        kw_by_name.setdefault(Path(p).name, []).append(v)
    people_names = {n.lower() for v in named.values() for n, _ in v}
    root = load_settings().keywords.people_root
    skip_roots = {root.lower(), "tagmatti"}
    kw_added = 0
    by_path = {_key(p): v for p, v in named.items()}
    by_name: dict[str, list[Any]] = {}
    for p, v in named.items():
        by_name.setdefault(p.name.lower(), []).append(v)
    faces_set = images_hit = image_level = 0
    for img in db.query("SELECT id, path FROM images WHERE shoot_id=?", (shoot_id,)):
        # Stichwörter (keine Personen, keine Tagmatti-Arbeitsstichwörter) übernehmen
        lk = lr_kws.get(_key(img["path"]))
        if lk is None:
            cand_k = kw_by_name.get(Path(img["path"]).name.lower()) or []
            lk = cand_k[0] if len(cand_k) == 1 else []
        for path in lk:
            parts = [x for x in path.split("|") if x]
            if not parts or parts[0].lower() in skip_roots or path in person_kw or parts[-1].lower() in people_names:
                continue
            kid = get_or_create(db, parts[-1], theme="lightroom")
            if not db.one("SELECT 1 FROM image_keywords WHERE image_id=? AND keyword_id=?", (img["id"], kid)):
                set_state(db, [img["id"]], kid, "lightroom")
                kw_added += 1
        lr = by_path.get(_key(img["path"]))
        if lr is None:
            cand = by_name.get(Path(img["path"]).name.lower()) or []
            lr = cand[0] if len(cand) == 1 else None          # gleicher Dateiname, eindeutig
        if not lr:
            continue
        images_hit += 1
        ours = [(f["id"], tuple(json.loads(f["bbox"]))) for f in
                db.query("SELECT id, bbox FROM faces WHERE image_id=?", (img["id"],))]
        for name, box in lr:
            pid = upsert_person(db, name, team, None)
            best = max(ours, key=lambda f: _iou(f[1], box), default=None)
            with db.tx() as c:
                if best is not None and (_iou(best[1], box) > 0.2 or (len(ours) == 1 and len(lr) == 1)):
                    c.execute("UPDATE faces SET person_id=?, assigned_by='lightroom' WHERE id=?", (pid, best[0]))
                    faces_set += 1
                else:
                    if not db.one("SELECT 1 FROM numbers WHERE image_id=? AND person_id=?", (img["id"], pid)):
                        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox, person_id) VALUES(?,?,?,?,?)",
                                  (img["id"], "#lr", 1.0, json.dumps(list(box)), pid))
                    image_level += 1
    return {"images": images_hit, "faces": faces_set, "image_level": image_level, "keywords": kw_added,
            "names": len({n for v in named.values() for n, _ in v})}


def push(db: Database, shoot_id: int, lrcat: Path) -> dict[str, Any]:
    from ..keywords import image_keywords
    from ..people.registry import image_people
    from .catalog import CatalogReader
    from .catalog_writer import CatalogWriter

    if catalog_open(lrcat):
        raise RuntimeError("Lightroom hat diesen Katalog gerade geöffnet – bitte Lightroom beenden und nochmals versuchen")
    with CatalogReader(lrcat) as cat:
        ids = {_key(i.path): i.image_id for i in cat.images(include_virtual_copies=False)}
    stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    backup = lrcat.with_name(f"{lrcat.stem}-vor-Tagmatti-{stamp}.lrcat")
    shutil.copy2(lrcat, backup)
    w = CatalogWriter.existing(lrcat)
    added = images = 0
    try:
        has_type = "keywordType" in w.cols("AgLibraryKeyword")
        for img in db.query("SELECT id, path FROM images WHERE shoot_id=?", (shoot_id,)):
            lr_id = ids.get(_key(img["path"]))
            people = image_people(db, img["id"])
            own = image_keywords(db, img["id"])
            if lr_id is None or not (people or own):
                continue
            images += 1
            for name in own:
                kid = w.keyword(name)
                if kid is not None and not w.conn.execute(
                        "SELECT 1 FROM AgLibraryKeywordImage WHERE image=? AND tag=?", (lr_id, kid)).fetchone():
                    w.insert("AgLibraryKeywordImage", {"image": lr_id, "tag": kid})
                    added += 1
            for p in people:
                kid = w.keyword(p.name)
                if kid is None:
                    continue
                if has_type:
                    w.conn.execute("UPDATE AgLibraryKeyword SET keywordType='person' WHERE id_local=? "
                                   "AND keywordType IS NULL", (kid,))
                if not w.conn.execute("SELECT 1 FROM AgLibraryKeywordImage WHERE image=? AND tag=?",
                                      (lr_id, kid)).fetchone():
                    w.insert("AgLibraryKeywordImage", {"image": lr_id, "tag": kid})
                    added += 1
        res = w.finish()
    except (sqlite3.DatabaseError, Exception):
        w.conn.close()
        shutil.copy2(backup, lrcat)                    # bei Fehler: Katalog unverändert zurück
        raise
    return {"images": images, "keywords": added, "backup": str(backup), "warnings": res.get("warnings", [])}

"""Personen aus bereits bearbeiteten Bildern lernen ("Referenzbilder").

Du ziehst einen Ordner mit fertigen JPGs (oder RAW + XMP) rein, in denen die Personen
schon benannt sind. Die Namen kommen aus:
1. MWG-Gesichtsregionen (Lightroom schreibt sie für benannte Gesichter): Name + Position,
2. Personen-Stichwörtern (z. B. "Personen|FC Winterthur|Max Muster" oder "Max Muster"),
3. dem Dateinamen ("Max Muster.jpg", "Max_Muster_03.jpg"), wenn sonst nichts da ist.

Zuordnung Name -> Gesicht:
- mit Region: das Gesicht, das am besten zur Region passt,
- sonst ein Name + ein (klar grösstes) Gesicht,
- sonst mehrere Namen: über bereits gelernte Gesichter eindeutig auflösen, Rest bleibt offen.
Gelernte Gesichter gelten als bestätigt und werden für die Wiedererkennung genutzt.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import IMAGE_EXTENSIONS, load_settings
from ..db import Database, dumps, f32_to_blob
from ..io import raw as raw_io
from ..jobs import JobContext, job
from ..lightroom.xmp import XmpDoc, read_xmp, sidecar_path
from ..style.sources import embedded_xmp
from ..vision.faces import Face, detect_faces, get_backend
from .registry import MATCH_THRESHOLD, exemplars, match, persons, upsert_person

log = logging.getLogger(__name__)
REFERENCE_SHOOT = "__Referenzbilder__"
_NAME_RE = re.compile(r"^[A-ZÄÖÜÀ-Ý][\w'’\-\.]+(?: [A-ZÄÖÜÀ-Ý][\w'’\-\.]+){1,3}$")
_SKIP_WORDS = {"tagmatti", "imagomat", "culling", "behalten", "aussortiert", "denoise", "prüfen", "fc", "sport", "fussball",
               "match", "konzert", "training"}


def _doc_for(path: Path) -> XmpDoc | None:
    side = sidecar_path(path)
    if side.exists():
        try:
            return read_xmp(side)
        except Exception:  # noqa: BLE001
            pass
    return embedded_xmp(path)


def _team_from_keyword(kw: str, root: str) -> tuple[str, str | None]:
    parts = [p for p in kw.split("|") if p]
    if parts and parts[0] == root:
        parts = parts[1:]
    name = parts[-1]
    team = parts[-2] if len(parts) >= 2 else None
    return name, team


def names_from_doc(doc: XmpDoc | None, known: set[str]) -> tuple[list[tuple[str, str | None]], list[Region]]:
    """-> ([(Name, Team)], [(Name, Box)]) aus XMP."""
    if doc is None:
        return [], []
    root = load_settings().keywords.people_root
    out: dict[str, str | None] = {}
    for r in doc.regions:
        if r.name and r.type.lower() in ("face", ""):
            out.setdefault(r.name.strip(), None)
    for kw in doc.keywords:
        name, team = _team_from_keyword(kw, root)
        is_people = kw.startswith(root + "|")
        looks = bool(_NAME_RE.match(name)) and name.split()[0].lower() not in _SKIP_WORDS
        if is_people or name in known or (looks and "|" not in kw):
            out[name] = team if team else out.get(name)
    return list(out.items()), [region_from_mwg(r) for r in doc.regions if r.name]


def name_from_filename(path: Path) -> str | None:
    stem = re.sub(r"[_\-]+", " ", path.stem)
    stem = re.sub(r"\s*\d+\s*$", "", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    if _NAME_RE.match(stem) and stem.split()[0].lower() not in _SKIP_WORDS:
        return stem
    return None


Region = tuple[str, tuple[float, float, float, float]]


def region_from_mwg(r: Any) -> Region:
    return r.name, (r.x - r.w / 2, r.y - r.h / 2, r.x + r.w / 2, r.y + r.h / 2)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign(faces: list[Face], names: list[str], regions: list[Region], ex: np.ndarray, ids: np.ndarray,
           name_to_pid: dict[str, int], thr: float) -> dict[int, str]:
    """Gesichtsindex -> Name. regions: [(Name, Box)] in Anzeige-Koordinaten."""
    out: dict[int, str] = {}
    if not faces or not names:
        return out
    # 1. Regionen
    for name, box in regions:
        best, score = None, 0.0
        for i, f in enumerate(faces):
            if i in out:
                continue
            s = _iou(box, f.bbox)
            if s > score:
                best, score = i, s
        if best is not None and score > 0.2:
            out[best] = name
    rest_names = [n for n in names if n not in out.values()]
    rest_faces = [i for i in range(len(faces)) if i not in out]
    # 2. Ein Name, ein klar grösstes Gesicht
    if len(rest_names) == 1 and rest_faces:
        sizes = sorted(((faces[i].height, i) for i in rest_faces), reverse=True)
        if len(sizes) == 1 or sizes[0][0] > 1.8 * sizes[1][0]:
            out[sizes[0][1]] = rest_names[0]
            return out
    # 3. Mehrere Namen: bereits bekannte Gesichter nutzen
    if len(rest_names) >= 1 and len(ids):
        allowed = {name_to_pid[n]: n for n in rest_names if n in name_to_pid}
        for i in rest_faces:
            if faces[i].embedding is None:
                continue
            pid, _ = match(faces[i].embedding, ex, ids, thr * 0.9)
            if pid in allowed and allowed[pid] not in out.values():
                out[i] = allowed[pid]
        rest_names = [n for n in names if n not in out.values()]
        rest_faces = [i for i in range(len(faces)) if i not in out]
        if len(rest_names) == 1 and len(rest_faces) == 1:
            out[rest_faces[0]] = rest_names[0]
    return out


def reference_shoot(db: Database) -> int:
    return db.upsert_shoot(REFERENCE_SHOOT, "__referenzbilder__")


def _orient_box(box: tuple[float, float, float, float], orientation: int) -> tuple[float, float, float, float]:
    from ..vision.geometry import sensor_to_display

    (ax, ay), (bx, by) = sensor_to_display(box[0], box[1], orientation), sensor_to_display(box[2], box[3], orientation)
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def work_from_folder(paths: list[Path], known: set[str]) -> list[tuple[Path, list[tuple[str, str | None]], list[Region], bool]]:
    items = []
    for p in paths:
        pairs, regions = names_from_doc(_doc_for(p), known)
        if not pairs:
            fn = name_from_filename(p)
            if fn:
                pairs = [(fn, None)]
        if pairs:
            items.append((p, pairs, regions, False))
    return items


def work_from_catalog(catalog: Path, per_person: int = 25) -> list[tuple[Path, list[tuple[str, str | None]], list[Region], bool]]:
    """Benannte Gesichter aus dem Lightroom-Katalog (max. per_person Bilder je Person)."""
    from ..lightroom.catalog import CatalogReader

    with CatalogReader(catalog) as r:
        faces = r.named_faces()
    count: dict[str, int] = {}
    items = []
    # Bilder mit wenigen Gesichtern zuerst: eindeutiger
    for path, regs in sorted(faces.items(), key=lambda kv: len(kv[1])):
        if not path.exists():
            continue
        need = [n for n, _ in regs if count.get(n, 0) < per_person]
        if not need:
            continue
        for n, _ in regs:
            count[n] = count.get(n, 0) + 1
        items.append((path, [(n, None) for n, _ in regs], list(regs), True))
    return items


@job("learn_people")
def learn_people(ctx: JobContext, folder: str | None = None, files: list[str] | None = None,
                 team: str | None = None, catalog: str | None = None) -> None:
    """Personen in einem Durchgang lernen: aus einem Ordner (JPG/RAW+XMP) und/oder dem Lightroom-Katalog."""
    db = ctx.db
    known = {p.name for p in persons(db)}
    paths: list[Path] = [Path(f) for f in files or []]
    if folder:
        paths += [p for p in sorted(Path(folder).expanduser().rglob("*"))
                  if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")]
    ctx.progress(0, "Namen suchen …")
    work = work_from_folder(paths, known)
    if catalog:
        work += work_from_catalog(Path(catalog).expanduser())
    ctx.set_total(len(work))
    sid = reference_shoot(db)
    thr = MATCH_THRESHOLD.get(get_backend().name, 0.45)
    learned, images_ok, unclear = 0, 0, 0
    people_seen: set[str] = set()
    ex, ids = exemplars(db, get_backend().name)
    for i, (p, pairs, regions, sensor_regions) in enumerate(work):
        ctx.check()
        ctx.progress(i, f"Personen lernen: {p.name}")
        try:
            img, orientation = raw_io.load_preview(p, 2048)
        except Exception as e:  # noqa: BLE001
            log.warning("Bild nicht lesbar: %s (%s)", p.name, e)
            continue
        faces = detect_faces(img)
        name_to_pid = {n: upsert_person(db, n, t or team) for n, t in pairs}
        known |= set(name_to_pid)
        if i % 25 == 0 or not len(ids):      # alle Beispiele laden ist teuer: nicht bei jedem Bild
            ex, ids = exemplars(db, get_backend().name)
        names = list(dict.fromkeys(n for n, _ in pairs))
        mapping = assign(faces, names, regions, ex, ids, name_to_pid, thr)
        if sensor_regions and len(mapping) < len(regions) and orientation != 1:
            # Katalog-Koordinaten evtl. in Sensor-Orientierung: gedreht erneut versuchen
            alt = assign(faces, names, [(n, _orient_box(b, orientation)) for n, b in regions], ex, ids,
                         name_to_pid, thr)
            if len(alt) > len(mapping):
                mapping = alt
        if not mapping:
            unclear += 1
            continue
        images_ok += 1
        image_id = db.upsert_image(sid, {"path": str(p), "filename": p.name, "orientation": orientation,
                                         "capture_time": time.time()})
        with db.tx() as c:
            c.execute("DELETE FROM faces WHERE image_id=?", (image_id,))
            for fi, name in mapping.items():
                f = faces[fi]
                c.execute(
                    "INSERT INTO faces(image_id, bbox, det_score, embedding, eyes_open, sharpness, yaw, person_id,"
                    " assigned_by) VALUES(?,?,?,?,?,?,?,?, 'confirmed')",
                    (image_id, dumps(list(f.bbox)), f.score, f32_to_blob(f.embedding), f.eyes_open, f.eye_sharpness,
                     f.yaw, name_to_pid[name]))
                learned += 1
                people_seen.add(name)
    if not work:
        msg = "Keine benannten Personen gefunden (Stichwörter, Gesichter oder Dateinamen)"
    else:
        msg = (f"{len(people_seen)} Personen gelernt ({learned} Gesichter)"
               + (f", {unclear} Bilder ohne eindeutige Zuordnung" if unclear else ""))
    ctx.progress(len(work), msg)

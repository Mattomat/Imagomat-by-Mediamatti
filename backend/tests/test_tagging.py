"""Nur Personen: fertige JPGs benennen und mit eingebetteten Namen wieder ausgeben."""

from pathlib import Path

import numpy as np
from PIL import Image

from imagomat.analysis import import_folder
from imagomat.db import Database
from imagomat.io import jpegxmp
from imagomat.jobs import JobManager
from imagomat.lightroom.xmp import XmpDoc, parse_xmp, serialize
from imagomat.people.registry import upsert_person
from imagomat import pipeline  # noqa: F401 (registriert Jobs)


def _jpgs(folder: Path, n: int = 3) -> list[Path]:
    folder.mkdir(parents=True)
    out = []
    rng = np.random.default_rng(1)
    for i in range(n):
        p = folder / f"FCW_{i:03d}.jpg"
        Image.fromarray(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)).save(p, quality=90)
        out.append(p)
    return out


def test_embed_xmp_keeps_pixels(tmp_path: Path):
    p = _jpgs(tmp_path / "a", 1)[0]
    before = np.asarray(Image.open(p))
    pkt = serialize(XmpDoc(keywords=["Personen|FCW Herren|Elias Maluvunu"]))
    jpegxmp.embed_xmp(p, pkt)
    assert np.array_equal(before, np.asarray(Image.open(p)))
    doc = parse_xmp(jpegxmp.read_jpeg_xmp(p))
    assert "Personen|FCW Herren|Elias Maluvunu" in doc.keywords
    # Ersetzen statt verdoppeln
    jpegxmp.embed_xmp(p, serialize(XmpDoc(keywords=["X"])))
    assert p.read_bytes().count(jpegxmp.XMP_HEADER) == 1


def test_people_only_pipeline_and_tag_export(tmp_path: Path):
    src = _jpgs(tmp_path / "fertig")
    originals = {p: p.read_bytes() for p in src}
    db = Database(tmp_path / "t.db")
    sid = import_folder(db, tmp_path / "fertig")
    jm = JobManager(db)
    j = jm.run_sync(db.create_job("pipeline", sid, {"mode": "people"}))
    assert j["status"] == "done", j["error"]
    assert all(r["decision"] == "keep" for r in db.query("SELECT decision FROM culling"))
    pid = upsert_person(db, "Elias Maluvunu", "FCW Herren", "37")
    img0 = db.images(sid)[0]["id"]
    with db.tx() as c:
        c.execute("INSERT INTO numbers(image_id, text, bbox, person_id) VALUES(?,?,?,?)",
                  (img0, "37", "[0.4,0.4,0.5,0.5]", pid))
    out = tmp_path / "mit Namen"
    j = jm.run_sync(db.create_job("tag_export", sid, {"target": str(out)}))
    assert j["status"] == "done", j["error"]
    files = sorted(out.glob("*.jpg"))
    assert len(files) == 3
    kws = parse_xmp(jpegxmp.read_jpeg_xmp(files[0])).keywords
    assert kws == ["Elias Maluvunu"]
    assert all(p.read_bytes() == b for p, b in originals.items())      # Originale unverändert

import csv
import json
from pathlib import Path

from tagmatti.analysis import import_folder
from tagmatti.db import Database
from tagmatti.export import exporter  # noqa: F401 (registriert Job)
from tagmatti.jobs import JobManager
from tagmatti.lightroom.catalog import CatalogReader
from tagmatti.lightroom.xmp import read_xmp
from tagmatti.people.registry import upsert_person
from tagmatti.style import jobs  # noqa: F401

from .lrcat_fixture import make_template
from .synth import write_shoot


def test_full_pipeline_export(tmp_path: Path):
    folder = tmp_path / "Match FCW"
    src = write_shoot(folder, n=8)
    originals = {p: p.read_bytes() for p in src}
    db = Database(tmp_path / "e.db")
    sid = import_folder(db, folder)
    jm = JobManager(db)
    for kind, params in (("analyze", {}), ("cull", {"keep_ratio": 0.5}), ("develop", {})):
        j = jm.run_sync(db.create_job(kind, sid, params))
        assert j["status"] == "done", (kind, j["error"])
    # Eine Person manuell einem Bild zuordnen (ohne Gesichtserkennung im Test)
    pid = upsert_person(db, "Max Muster", "FC Winterthur", "7")
    img0 = db.images(sid)[0]["id"]
    with db.tx() as c:
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox, person_id) VALUES(?,?,?,?,?)",
                  (img0, "7", 0.9, "[0.3,0.4,0.35,0.5]", pid))
    out = tmp_path / "export"
    tpl = make_template(tmp_path / "tpl.lrcat")
    j = jm.run_sync(db.create_job("export", sid, {"target": str(out), "formats": ["xmp", "catalog", "jpeg"],
                                                   "template": str(tpl)}))
    assert j["status"] == "done", j["error"]
    # Originale unverändert
    assert all(p.read_bytes() == b for p, b in originals.items())
    xmps = sorted(out.glob("*.xmp"))
    assert len(xmps) == 8
    kept = [x for x in xmps if read_xmp(x).crs]              # behaltene Bilder tragen die Entwicklung
    n_keep = db.one("SELECT COUNT(*) FROM culling WHERE decision='keep'")[0]
    assert len(kept) == n_keep >= 2          # Strenge ist Obergrenze: Unscharfes bleibt draussen
    d = read_xmp(kept[0])
    assert d.crs.get("HasSettings") == "True" and "Exposure2012" in d.crs
    assert d.rating and d.rating >= 2
    rejected = [x for x in xmps if x not in kept]
    assert all(read_xmp(x).rating == 1 and not read_xmp(x).crs for x in rejected)
    first = read_xmp(out / (src[0].stem + ".xmp"))
    assert first.keywords == ["Max Muster"]                 # nur der Name, keine Arbeits-Stichwörter
    rows = list(csv.DictReader(open(out / "tagmatti-log.csv", encoding="utf-8-sig"), delimiter=";"))
    assert len(rows) == 8 and any(r["gruende"] for r in rows if r["entscheidung"] == "aussortiert")
    manifest = json.loads((out / "tagmatti.json").read_text())
    assert {p["pick"] for p in manifest["photos"]} == {1, -1}
    jpgs = list((out / "JPEG (Vorschau, nicht Lightroom-Qualität)").glob("*.jpg"))
    assert len(jpgs) == n_keep
    cats = list(out.glob("*.lrcat"))
    assert len(cats) == 1
    with CatalogReader(cats[0]) as r:
        imgs = list(r.images())
        assert len(imgs) == 8 and sum(1 for i in imgs if i.pick == 1) == n_keep

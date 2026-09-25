from pathlib import Path

from imagomat.analysis import import_folder
from imagomat.culling.engine import load_items
from imagomat.db import Database
from imagomat.jobs import JobManager

from .synth import write_shoot


def test_import_analyze_cull(tmp_path: Path):
    folder = tmp_path / "shoot"
    write_shoot(folder, n=12)
    db = Database(tmp_path / "t.db")
    shoot_id = import_folder(db, folder)
    rows = db.images(shoot_id)
    assert len(rows) == 12
    assert rows[0]["iso"] == 6400
    jm = JobManager(db)
    j = jm.run_sync(db.create_job("analyze", shoot_id, {}))
    assert j["status"] == "done", j["error"]
    a = db.get_analysis(rows[0]["id"])
    assert a["as_shot_temp"] > 1000 and a["noise_sigma_mid"] > 0
    assert Path(db.one("SELECT preview_path FROM images WHERE id=?", (rows[0]["id"],))[0]).exists()
    # Neustart des Jobs verarbeitet nichts doppelt (wiederaufnehmbar)
    j = jm.run_sync(db.create_job("analyze", shoot_id, {}))
    assert j["status"] == "done"
    j = jm.run_sync(db.create_job("cull", shoot_id, {"keep_ratio": 0.25}))
    assert j["status"] == "done", j["error"]
    cul = {r["image_id"]: dict(r) for r in db.query("SELECT * FROM culling")}
    kept = [i for i, c in cul.items() if c["decision"] == "keep"]
    assert len(kept) == 3
    series = {c["series_id"] for c in cul.values()}
    assert len(series) == 3
    # Die unscharfen / verwackelten Varianten dürfen nicht die Besten ihrer Serie sein
    by_name = {r["filename"]: r["id"] for r in rows}
    for blurred in ("DSC00001.dng", "DSC00002.dng", "DSC00005.dng", "DSC00006.dng"):
        assert cul[by_name[blurred]]["decision"] == "reject", blurred
    items = load_items(db, shoot_id)
    assert all(it.a for it in items)


def test_manual_override_survives(tmp_path: Path):
    folder = tmp_path / "shoot"
    write_shoot(folder, n=4)
    db = Database(tmp_path / "t.db")
    sid = import_folder(db, folder)
    jm = JobManager(db)
    jm.run_sync(db.create_job("analyze", sid, {}))
    jm.run_sync(db.create_job("cull", sid, {}))
    img = db.images(sid)[1]["id"]
    db.set_culling(img, score=0, decision="keep", rating=5, label=None, reasons=[], series_id=1, is_best=False,
                   manual=True)
    jm.run_sync(db.create_job("cull", sid, {}))
    row = db.one("SELECT decision, rating, manual FROM culling WHERE image_id=?", (img,))
    assert row["decision"] == "keep" and row["rating"] == 5 and row["manual"] == 1

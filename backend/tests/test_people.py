import json
from pathlib import Path

import numpy as np

from imagomat.db import Database, dumps, f32_to_blob
from imagomat.jobs import JobManager
from imagomat.people import clustering  # noqa: F401  (registriert den Job)
from imagomat.people.registry import assign_face, image_people, upsert_person
from imagomat.people.roster import parse_csv, parse_html


def _setup(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    sid = db.upsert_shoot("Match", str(tmp_path / "m"))
    with db.tx() as c:
        c.execute("UPDATE shoots SET settings=? WHERE id=?", (json.dumps({"teams": ["FCW"]}), sid))
    ids = [db.upsert_image(sid, {"path": str(tmp_path / f"{i}.ARW"), "filename": f"{i}.ARW"}) for i in range(8)]
    return db, sid, ids


def test_roster_csv_and_html():
    rows = parse_csv("Name;Nummer;Team\nMax Muster;7;FCW\nLuca Beispiel;10;FCW\n")
    assert [(r.name, r.number) for r in rows] == [("Max Muster", "7"), ("Luca Beispiel", "10")]
    html = "<ul><li><span>7</span> <b>Max Muster</b> Stürmer</li><li>Luca Beispiel #10</li></ul>"
    got = {r.name: r.number for r in parse_html(html, "FCW")}
    assert got["Max Muster"] == "7" and got["Luca Beispiel"] == "10"


def test_recognition_numbers_and_clusters(tmp_path: Path):
    db, sid, ids = _setup(tmp_path)
    rng = np.random.default_rng(0)
    a, b, c = (v / np.linalg.norm(v) for v in rng.normal(size=(3, 512)))
    max_id = upsert_person(db, "Max Muster", "FCW", "7")
    upsert_person(db, "Luca Beispiel", "FCW", "10")

    def face(img, base, box=(0.4, 0.1, 0.5, 0.25)):
        e = base + rng.normal(scale=0.02, size=512)
        with db.tx() as cx:
            return cx.execute("INSERT INTO faces(image_id, bbox, det_score, embedding) VALUES(?,?,?,?)",
                              (img, dumps(list(box)), 0.9, f32_to_blob(e / np.linalg.norm(e)))).lastrowid

    f0 = face(ids[0], a)
    assign_face(db, f0, max_id, "manual")              # bestätigtes Beispiel für Max
    face(ids[1], a)                                   # soll als Max erkannt werden
    f_num = face(ids[2], b)                           # unbekannt, aber Nummer 10 auf dem Trikot
    with db.tx() as cx:
        cx.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                   (ids[2], "10", 0.9, dumps([0.43, 0.35, 0.47, 0.42])))
        cx.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                   (ids[3], "7", 0.8, dumps([0.1, 0.5, 0.15, 0.6])))  # Rückenansicht ohne Gesicht
    for i in range(4, 8):
        face(ids[i], c)                               # unbekannte Person -> Cluster
    from imagomat.vision import faces as fmod

    fmod.get_backend.cache_clear()
    j = JobManager(db).run_sync(db.create_job("people", sid, {"threshold": 0.75}))
    assert j["status"] == "done", j["error"]
    assert [p.name for p in image_people(db, ids[1])] == ["Max Muster"]
    assert db.one("SELECT person_id, assigned_by FROM faces WHERE id=?", (f_num,))["assigned_by"] == "number"
    assert [p.name for p in image_people(db, ids[2])] == ["Luca Beispiel"]
    assert [p.name for p in image_people(db, ids[3])] == ["Max Muster"]
    clusters = {r["cluster_id"] for r in db.query("SELECT cluster_id FROM faces WHERE image_id >= ?", (ids[4],))}
    assert len(clusters) == 1 and None not in clusters

"""Rückennummern und Trikotnamen schlagen die Gesichtserkennung."""

import json
from pathlib import Path

import numpy as np

from imagomat.db import Database, f32_to_blob
from imagomat.jobs import JobManager
from imagomat.people import clustering  # noqa: F401 (registriert Job)
from imagomat.people.registry import upsert_person
from imagomat.vision.ocr import parse_shirt_text


def test_parse_shirt_text():
    hits = parse_shirt_text([("37", 0.9, (0.2, 0.3, 0.3, 0.4)), ("MALUVUNU", 0.8, (0.2, 0.42, 0.35, 0.46)),
                             ("SWISS STEEL", 0.8, (0.2, 0.28, 0.3, 0.3)), ("Bier", 0.9, (0.1, 0.1, 0.3, 0.2))])
    texts = [h.text for h in hits]
    assert "37" in texts and "@MALUVUNU" in texts and "@SWISS" in texts and "@Bier" not in texts


def _setup(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    maluvunu = upsert_person(db, "Elias Maluvunu", "FCW Herren", "37")
    kehrer = upsert_person(db, "Emilio Kehrer", "FCW Herren", "11")
    upsert_person(db, "Jemand Anders", "FCW Frauen", "37")          # gleiche Nummer, anderes Team
    sid = db.upsert_shoot("Match", str(tmp_path), None)
    iid = db.upsert_image(sid, {"path": str(tmp_path / "a.arw"), "filename": "a.arw"})
    emb = np.ones(512, np.float32)
    with db.tx() as c:
        # Referenzgesicht von Kehrer (bestätigt) -> die neue Rückenansicht ähnelt ihm
        ref = db.upsert_image(sid, {"path": str(tmp_path / "ref.arw"), "filename": "ref.arw"})
        c.execute("INSERT INTO faces(image_id, bbox, embedding, person_id, assigned_by, yaw) VALUES(?,?,?,?,?,?)",
                  (ref, "[0.4,0.1,0.5,0.25]", f32_to_blob(emb), kehrer, "confirmed", 0.0))
        c.execute("INSERT INTO faces(image_id, bbox, embedding, yaw) VALUES(?,?,?,?)",
                  (iid, "[0.2,0.2,0.3,0.3]", f32_to_blob(emb), 70.0))
    return db, sid, iid, maluvunu, kehrer


def test_number_beats_face_and_side_face_not_auto(tmp_path: Path):
    db, sid, iid, maluvunu, kehrer = _setup(tmp_path)
    with db.tx() as c:
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "37", 0.9, "[0.22,0.36,0.28,0.44]"))
        c.execute("INSERT INTO steps(image_id, step, version, done_at) VALUES(?,?,?,?)", (iid, "ocr", 2, 0))
    db.update_shoot_settings(sid, teams=["FCW Herren"])
    jm = JobManager(db)
    j = jm.run_sync(db.create_job("people", sid, {}))
    assert j["status"] == "done", j["error"]
    face = db.one("SELECT person_id, assigned_by FROM faces WHERE image_id=?", (iid,))
    assert face["person_id"] == maluvunu and face["assigned_by"] == "number"
    assert db.one("SELECT person_id FROM numbers WHERE image_id=?", (iid,))[0] == maluvunu


def test_frontal_face_wins_over_number(tmp_path: Path):
    db, sid, iid, maluvunu, kehrer = _setup(tmp_path)
    with db.tx() as c:
        c.execute("UPDATE faces SET yaw=5, bbox='[0.2,0.2,0.3,0.3]' WHERE image_id=?", (iid,))
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "37", 0.9, "[0.22,0.36,0.28,0.44]"))
    db.update_shoot_settings(sid, teams=["FCW Herren"])
    assert JobManager(db).run_sync(db.create_job("people", sid, {}))["status"] == "done"
    face = db.one("SELECT person_id, assigned_by FROM faces WHERE image_id=?", (iid,))
    assert face["person_id"] == kehrer and face["assigned_by"] == "auto"
    check = db.get_analysis(iid)["people_check"][0]            # Widerspruch wird zum Prüfen gemeldet
    assert check["face_person_id"] == kehrer and check["shirt_person_id"] == maluvunu
    assert check["chosen"] == "face"


def test_sponsor_words_are_not_names(tmp_path: Path):
    from imagomat.people.clustering import plausible_back_names

    rows, rid = [], 0

    def add(img, text, box):
        nonlocal rid
        rid += 1
        rows.append({"id": rid, "image_id": img, "text": text, "bbox": json.dumps(box)})
        return rid

    ok = add(1, "@MALUVUNU", [0.2, 0.30, 0.3, 0.33]); add(1, "37", [0.2, 0.34, 0.3, 0.44])
    front = add(2, "@SCHMID", [0.5, 0.30, 0.6, 0.33])                      # ohne Nummer (Brust)
    spons = [add(i, "@BAUMANN", [0.2, 0.30, 0.3, 0.33]) for i in (3, 4, 5)]
    for i, num in zip((3, 4, 5), ("7", "9", "11")):
        add(i, num, [0.2, 0.34, 0.3, 0.44])                                 # gleiches Wort, 3 Nummern
    ign = add(6, "@KELLER", [0.2, 0.30, 0.3, 0.33]); add(6, "20", [0.2, 0.34, 0.3, 0.44])
    got = plausible_back_names(rows, {"KELLER"})
    assert ok in got and front not in got and ign not in got and not set(spons) & got


def test_face_beats_number_and_name(tmp_path: Path):
    db, sid, iid, maluvunu, kehrer = _setup(tmp_path)
    with db.tx() as c:
        c.execute("UPDATE faces SET yaw=5 WHERE image_id=?", (iid,))
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "37", 0.9, "[0.22,0.36,0.28,0.44]"))
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "@MALUVUNU", 0.9, "[0.22,0.45,0.29,0.48]"))
    db.update_shoot_settings(sid, teams=["FCW Herren"])
    assert JobManager(db).run_sync(db.create_job("people", sid, {}))["status"] == "done"
    face = db.one("SELECT person_id FROM faces WHERE image_id=?", (iid,))
    assert face["person_id"] == kehrer                     # Gesicht ist stärker als das Trikot
    assert db.get_analysis(iid)["people_check"][0]["chosen"] == "face"


def test_shirt_name_resolves_without_team(tmp_path: Path):
    db, sid, iid, maluvunu, _ = _setup(tmp_path)
    with db.tx() as c:
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "@MALUVUNU", 0.8, "[0.22,0.45,0.29,0.48]"))
        c.execute("INSERT INTO numbers(image_id, text, confidence, bbox) VALUES(?,?,?,?)",
                  (iid, "37", 0.9, "[0.22,0.36,0.28,0.44]"))       # ohne Team mehrdeutig (Herren/Frauen)
    jm = JobManager(db)
    assert jm.run_sync(db.create_job("people", sid, {}))["status"] == "done"
    rows = {r["text"]: r["person_id"] for r in db.query("SELECT text, person_id FROM numbers")}
    assert rows["@MALUVUNU"] == maluvunu
    face = db.one("SELECT person_id FROM faces WHERE image_id=?", (iid,))
    assert face["person_id"] == maluvunu

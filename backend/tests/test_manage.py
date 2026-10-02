"""Alben und Stile löschen, Stile vergleichen, Kacheln."""

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tagmatti.db import Database, dumps, f32_to_blob
from tagmatti.manage import delete_profile, delete_shoot, rename_profile
from tagmatti.people.registry import upsert_person


def test_delete_shoot_keeps_originals_and_named_faces(tmp_path: Path):
    db = Database(tmp_path / "d.db")
    folder = tmp_path / "Match"
    folder.mkdir()
    files = [folder / f"{i}.ARW" for i in range(3)]
    for f in files:
        f.write_bytes(b"raw")
    sid = db.upsert_shoot("Match", str(folder))
    ids = [db.upsert_image(sid, {"path": str(f), "filename": f.name}) for f in files]
    pid = upsert_person(db, "Max Muster", "FCW", "7")
    with db.tx() as c:
        c.execute("INSERT INTO faces(image_id, bbox, embedding, person_id, assigned_by) VALUES(?,?,?,?,?)",
                  (ids[0], dumps([0.1, 0.1, 0.2, 0.2]), f32_to_blob(np.ones(8, np.float32)), pid, "manual"))
        c.execute("INSERT INTO faces(image_id, bbox, person_id, assigned_by) VALUES(?,?,?,?)",
                  (ids[1], dumps([0.1, 0.1, 0.2, 0.2]), pid, "auto"))
    res = delete_shoot(db, sid)
    assert res == {"deleted": 2, "kept_for_people": 1}
    assert all(f.read_bytes() == b"raw" for f in files)                    # Originale unangetastet
    assert db.one("SELECT COUNT(*) FROM shoots WHERE id=?", (sid,))[0] == 0
    # benanntes Gesicht bleibt als Beispiel für die Erkennung erhalten
    assert db.one("SELECT COUNT(*) FROM faces WHERE assigned_by='manual' AND person_id=?", (pid,))[0] == 1


def test_delete_and_rename_profile(tmp_path: Path):
    from tagmatti.config import load_settings, profiles_dir, save_settings

    d = profiles_dir() / "Alt"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"name": "Alt"}))
    s = load_settings()
    s.default_profile = "Alt"
    save_settings(s)
    rename_profile("Alt", "FCW Nacht")
    assert (profiles_dir() / "FCW Nacht" / "meta.json").exists() and load_settings().default_profile == "FCW Nacht"
    delete_profile("FCW Nacht")
    assert not (profiles_dir() / "FCW Nacht").exists() and load_settings().default_profile is None


def test_api_delete_and_thumb(tmp_path: Path):
    import cv2

    from tagmatti.server.app import create_app

    app = create_app(str(tmp_path / "a.db"))
    with TestClient(app) as client:
        db = Database(tmp_path / "a.db")
        sid = db.upsert_shoot("X", str(tmp_path / "x"))
        iid = db.upsert_image(sid, {"path": str(tmp_path / "x" / "a.jpg"), "filename": "a.jpg"})
        pv = tmp_path / "pv.jpg"
        cv2.imwrite(str(pv), np.full((1200, 1800, 3), 128, np.uint8))
        with db.tx() as c:
            c.execute("UPDATE images SET preview_path=? WHERE id=?", (str(pv), iid))
        r = client.get(f"/api/images/{iid}/thumb")
        assert r.status_code == 200
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) <= 480
        assert client.get(f"/api/shoots/{sid}").json()["n"] == 1
        assert client.delete(f"/api/shoots/{sid}").json()["deleted"] == 1
        assert client.get(f"/api/shoots/{sid}").status_code == 404


def test_reset_keeps_only_people(tmp_path: Path):
    import json

    from tagmatti.config import load_settings, profiles_dir, save_settings
    from tagmatti.manage import reset_keep_people

    db = Database(tmp_path / "r.db")
    folder = tmp_path / "Match"
    folder.mkdir()
    sid = db.upsert_shoot("Match", str(folder))
    ids = [db.upsert_image(sid, {"path": str(folder / f"{i}.ARW"), "filename": f"{i}.ARW"}) for i in range(3)]
    pid = upsert_person(db, "Max Muster", "FCW", "7")
    upsert_person(db, "Ohne Bild", "FCW", "9")                  # Kader-Eintrag ohne Gesicht bleibt
    emb = f32_to_blob(np.ones(8, np.float32))
    with db.tx() as c:
        c.execute("INSERT INTO faces(image_id, bbox, embedding, person_id, assigned_by) VALUES(?,?,?,?,?)",
                  (ids[0], dumps([0.1, 0.1, 0.2, 0.2]), emb, pid, "manual"))
        c.execute("INSERT INTO faces(image_id, bbox, embedding, cluster_id) VALUES(?,?,?,?)",
                  (ids[0], dumps([0.5, 0.1, 0.6, 0.2]), emb, 3))      # "Wer ist das?" -> weg
        c.execute("INSERT INTO faces(image_id, bbox, embedding, cluster_id) VALUES(?,?,?,?)",
                  (ids[1], dumps([0.5, 0.1, 0.6, 0.2]), emb, 3))
    d = profiles_dir() / "Mein Stil"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"name": "Mein Stil"}))
    s = load_settings()
    s.default_profile = "Mein Stil"
    save_settings(s)
    res = reset_keep_people(db)
    assert res["shoots"] == 1 and res["persons"] == 2 and res["faces"] == 1 and res["styles"] == 1
    assert not any(profiles_dir().iterdir()) and load_settings().default_profile is None
    assert db.one("SELECT person_id FROM faces")[0] == pid
    visible = db.query("SELECT id FROM shoots WHERE name NOT LIKE '\\_\\_%' ESCAPE '\\'")
    assert visible == []


def test_try_styles_on_one_image(tmp_path: Path):
    """Stil-Leiste: dasselbe Bild in mehreren Stilen (mit Masken/Entrauschen), dann nur für dieses Bild übernehmen."""
    from tagmatti.analysis import import_folder
    from tagmatti.jobs import JobManager
    from tagmatti.server.app import create_app

    from .synth import write_shoot

    write_shoot(tmp_path / "s", n=3)
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    jm = JobManager(db)
    for k in ("analyze", "cull", "develop"):
        assert jm.run_sync(db.create_job(k, sid, {}))["status"] == "done"
    iid = db.images(sid)[0]["id"]
    with TestClient(create_app(str(tmp_path / "a.db"))) as c:
        styles = [s["key"] for s in c.get("/api/styles").json()]
        assert "preset:fb_night" in styles and "preset:auto" in styles
        sizes = set()
        for st in ("preset:fb_night", "preset:fb_bw"):
            r = c.get(f"/api/images/{iid}/styled", params={"style": st, "size": 400})
            assert r.status_code == 200
            sizes.add(len(r.content))
        assert len(sizes) == 2                                   # verschiedene Stile -> verschiedene Bilder
        assert c.post(f"/api/images/{iid}/style", json={"style": "preset:fb_bw"}).json()["ok"]
    assert db.one("SELECT profile FROM edits WHERE image_id=?", (iid,))[0] == "preset:fb_bw"

"""Gewähltes Team ist bindend: bei "FCW Herren" wird nie jemand aus "FCW Frauen" erkannt."""

from pathlib import Path

import numpy as np


def test_exemplars_only_from_chosen_team(tmp_path: Path):
    from tagmatti.db import Database, f32_to_blob
    from tagmatti.people.registry import exemplars, match, upsert_person

    db = Database(tmp_path / "a.db")
    sid = db.upsert_shoot("s", str(tmp_path), None)
    iid = db.upsert_image(sid, {"path": str(tmp_path / "a.ARW"), "filename": "a.ARW", "orientation": 1, "exif": {}})
    herr = upsert_person(db, "Max Muster", "FCW Herren", "7")
    frau = upsert_person(db, "Maria Muster", "FCW Frauen", "7")
    staff = upsert_person(db, "Trainer Staff", None, None)
    rng = np.random.default_rng(0)
    base = rng.normal(size=128).astype(np.float32)
    with db.tx() as c:
        for pid, e in ((frau, base), (herr, base + rng.normal(size=128).astype(np.float32) * 0.9),
                       (staff, rng.normal(size=128).astype(np.float32))):
            c.execute("INSERT INTO faces(image_id, bbox, embedding, person_id, assigned_by) VALUES(?,?,?,?,?)",
                      (iid, "[0,0,1,1]", f32_to_blob(e), pid, "manual"))
    ex, ids = exemplars(db, None, ["FCW Herren"])
    assert set(ids.tolist()) == {herr, staff}
    # Gesicht, das der Spielerin am ähnlichsten ist, wird im Herren-Shoot nie ihr zugeordnet
    pid, _ = match(base, ex, ids, 0.0)
    assert pid != frau
    ex_all, ids_all = exemplars(db, None)
    assert frau in set(ids_all.tolist())

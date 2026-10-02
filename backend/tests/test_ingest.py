"""Import an den Ablageort: RAWs kopieren (Originale unverändert), dann wie gewohnt verarbeiten."""

from pathlib import Path


def test_import_copies_to_library(tmp_path: Path):
    from fastapi.testclient import TestClient

    import tagmatti.pipeline  # noqa: F401
    from tagmatti.db import Database
    from tagmatti.server.app import create_app

    from .synth import write_shoot

    card = tmp_path / "card" / "DCIM"
    files = write_shoot(card, n=3)
    before = {p.name: p.stat().st_mtime for p in files}
    dest = tmp_path / "Bilder" / "2026" / "2026-05-01 Test"
    dbp = tmp_path / "a.db"
    app = create_app(str(dbp))
    with TestClient(app) as c:
        info = c.get("/api/import/info", params={"folder": str(card)}).json()
        assert info["count"] == 3 and info["date"]
        r = c.post("/api/shoots/import", json={"folder": str(card), "copy_to": str(dest), "name": "Test"}).json()
        db = Database(dbp)
        import time

        for _ in range(600):                     # der Auftrag läuft im Hintergrund der App
            st = db.job(r["job_id"])["status"]
            if st in ("done", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert st == "done", db.job(r["job_id"])["error"]
        shoot = c.get(f"/api/shoots/{r['shoot_id']}").json()
        assert Path(shoot["folder"]) == dest.resolve() and shoot["n"] == 3
        assert sorted(p.name for p in dest.iterdir() if not p.name.endswith(".part")) >= sorted(before)
        assert all(Path(row["path"]).parent == dest.resolve() for row in db.images(r["shoot_id"]))
    assert {p.name: p.stat().st_mtime for p in files} == before          # Karte unverändert

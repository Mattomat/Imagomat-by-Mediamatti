"""0.8: neutral nach dem Import, eigene Lightroom-Vorgaben, Objekt-Maske, KI-Entrauschen-Vorschau, Export-Formate."""

import json
from pathlib import Path

PRESET_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
 crs:PresetType="Normal" crs:Cluster="" crs:UUID="ABC" crs:SupportsAmount="True" crs:Version="15.0"
 crs:Exposure2012="+0.40" crs:Contrast2012="+25" crs:Vibrance="+18" crs:HasSettings="True">
 <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">Mein Flutlicht</rdf:li></rdf:Alt></crs:Name>
</rdf:Description></rdf:RDF></x:xmpmeta>"""


def _shoot(tmp_path: Path):
    from imagomat.analysis import import_folder
    from imagomat.db import Database
    from imagomat.jobs import JobManager
    import imagomat.pipeline  # noqa: F401
    from imagomat.culling import engine as _cull  # noqa: F401

    from .synth import write_shoot

    write_shoot(tmp_path / "s", n=3)
    dbp = tmp_path / "a.db"
    db = Database(dbp)
    sid = import_folder(db, tmp_path / "s", profile="neutral")
    jm = JobManager(db)
    for k in ("analyze", "cull", "develop"):
        assert jm.run_sync(db.create_job(k, sid, {"only_keep": False} if k == "develop" else {}))["status"] == "done"
    return db, dbp, sid, jm


def test_neutral_presets_object_denoise_export(tmp_path: Path):
    from fastapi.testclient import TestClient
    from PIL import Image

    from imagomat.server.app import create_app

    db, dbp, sid, jm = _shoot(tmp_path)
    ids = [r["id"] for r in db.images(sid)]
    # neutral: alle Regler 0, Weissabgleich wie Aufnahme, keine Masken
    e = db.one("SELECT profile, params, masks FROM edits WHERE image_id=?", (ids[0],))
    crs = json.loads(e["params"])
    assert e["profile"] == "neutral" and e["masks"] is None
    assert crs["WhiteBalance"] == "As Shot" and crs["Exposure2012"] == 0 and crs["HasSettings"] == "True"
    (tmp_path / "p.xmp").write_text(PRESET_XMP)
    app = create_app(str(dbp))
    with TestClient(app) as c:
        r = c.post("/api/presets/import", json={"paths": [str(tmp_path / "p.xmp")]}).json()
        assert r["imported"] == ["Mein Flutlicht"]
        st = c.get("/api/styles").json()
        assert any(s["key"] == "up:Mein Flutlicht" and s.get("user") for s in st)
        assert any(s["key"] == "neutral" for s in st)
        # Vorgabe auf ein Bild: nur ihre Regler, Rest bleibt
        assert c.post(f"/api/images/{ids[0]}/style", json={"style": "up:Mein Flutlicht"}).json()["ok"]
        crs = json.loads(db.one("SELECT params FROM edits WHERE image_id=?", (ids[0],))["params"])
        assert float(crs["Exposure2012"]) == 0.4 and float(crs["Vibrance"]) == 18 and crs["WhiteBalance"] == "As Shot"
        assert "UUID" not in crs and "Name" not in crs
        # Zurücksetzen = neutral
        assert c.delete(f"/api/images/{ids[0]}/editor").json()["ok"]
        assert json.loads(db.one("SELECT params FROM edits WHERE image_id=?", (ids[0],))["params"])["Exposure2012"] == 0
        # Objekt auswählen -> Pinsel-Tupfer
        dabs = c.post(f"/api/images/{ids[0]}/editor/object", json={"box": [0.15, 0.1, 0.6, 0.95]}).json()["dabs"]
        assert len(dabs) > 0 and all(0 <= d[0] <= 1 and 0 <= d[1] <= 1 for d in dabs)
        # KI-Entrauschen: 1:1-Vorschau und Stärke speichern
        r = c.post(f"/api/images/{ids[0]}/denoise/preview", json={"x": 0.5, "y": 0.5, "amount": 60, "size": 128})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        model = c.get(f"/api/images/{ids[0]}/editor").json()["model"]
        model["denoise"] = 60
        assert c.put(f"/api/images/{ids[0]}/editor", json={"model": model}).json()["ok"]
        assert c.get(f"/api/images/{ids[0]}/editor").json()["model"]["denoise"] == 60
        # Vorgabe auf den ganzen Shoot (von Hand bearbeitetes Bild bleibt)
        j = c.post(f"/api/shoots/{sid}/run/develop", json={"profile": "up:Mein Flutlicht"}).json()
        from imagomat.jobs import JobManager

        assert JobManager(db).run_sync(j["job_id"])["status"] == "done"
        assert db.one("SELECT profile FROM edits WHERE image_id=?", (ids[0],))["profile"] == "manual"
        # Export: RAW + XMP mit KI-Entrauschen (DNG) und JPEG/WebP/PNG in gewählter Grösse
        out = tmp_path / "exp"
        j = c.post(f"/api/shoots/{sid}/export", json={"target": str(out), "formats": ["xmp", "jpeg", "webp", "png"],
                                                       "long_side": 300, "quality": 80, "naming": "custom",
                                                       "name_base": "Match", "denoise": "edit",
                                                       "include_rejected": False}).json()
        assert JobManager(db).run_sync(j["job_id"])["status"] == "done"
        tgt = Path(j["target"])
        assert list(tgt.glob("*-Enhanced-NR.dng")), list(tgt.iterdir())
        jpgs = sorted(tgt.glob("Match-*.jpg"))
        assert jpgs and sorted(tgt.glob("Match-*.webp")) and sorted(tgt.glob("Match-*.png"))
        assert max(Image.open(jpgs[0]).size) == 300

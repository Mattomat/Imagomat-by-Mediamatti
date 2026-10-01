"""Eigener Edit: Editor-Modell <-> Lightroom-Einstellungen, Speichern, Vorschau, auf alle übertragen."""

from pathlib import Path


def test_model_roundtrip_keeps_masks_and_unknown_keys():
    from imagomat.style.editor import crs_to_model, model_to_crs

    base = {"CameraProfile": "Adobe Standard", "Look": {"Name": "Adobe Color"}, "Exposure2012": 0.5,
            "HasCrop": "True", "CropAngle": 2.0}
    model = {"global": {"Exposure2012": 1.2, "Shadows2012": 30, "Temperature": 3600, "Tint": 35,
                        "SaturationAdjustmentGreen": -30},
             "wb_custom": True,
             "masks": [{"name": "Unten", "kind": "gradient", "zero": [0.5, 0.6], "full": [0.5, 1.0],
                        "local": {"Exposure2012": -1.5, "Sharpness": -100}},
                       {"name": "Spieler", "kind": "subject", "local": {"Whites2012": 80, "Shadows2012": 30}},
                       {"name": "Spot", "kind": "radial", "box": [0.2, 0.2, 0.6, 0.8], "invert": True, "feather": 70,
                        "local": {"Exposure2012": -0.4}}]}
    for orient in (1, 8):
        crs = model_to_crs(model, base, orient)
        assert crs["CameraProfile"] == "Adobe Standard" and crs["Look"] == {"Name": "Adobe Color"}
        assert crs["HasCrop"] == "True" and crs["WhiteBalance"] == "Custom" and crs["Exposure2012"] == 1.2
        g, s, r = crs["MaskGroupBasedCorrections"]
        assert abs(g["LocalExposure2012"] - (-1.5 / 4)) < 1e-9 and g["LocalSharpness"] == -1.0
        assert s["CorrectionMasks"][0]["What"] == "Mask/Image" and abs(s["LocalWhites2012"] - 0.8) < 1e-9
        back = crs_to_model(crs, orient)
        assert [m["kind"] for m in back["masks"]] == ["gradient", "subject", "radial"]
        assert all(abs(a - b) < 1e-6 for a, b in zip(back["masks"][0]["zero"], [0.5, 0.6]))
        assert all(abs(a - b) < 1e-6 for a, b in zip(back["masks"][2]["box"], [0.2, 0.2, 0.6, 0.8]))
        assert back["masks"][2]["invert"] and back["masks"][0]["local"]["Exposure2012"] == -1.5
        assert back["global"]["SaturationAdjustmentGreen"] == -30


def test_editor_endpoints_and_sync(tmp_path: Path):
    import json

    from fastapi.testclient import TestClient

    from imagomat.analysis import import_folder
    from imagomat.db import Database
    from imagomat.jobs import JobManager
    import imagomat.pipeline  # noqa: F401
    from imagomat.culling import engine as _cull  # noqa: F401
    from imagomat.server.app import create_app

    from .synth import write_shoot

    write_shoot(tmp_path / "s", n=3)
    dbp = tmp_path / "a.db"
    db = Database(dbp)
    sid = import_folder(db, tmp_path / "s")
    jm = JobManager(db)
    for k in ("analyze", "cull", "develop"):
        assert jm.run_sync(db.create_job(k, sid, {"only_keep": False} if k == "develop" else {}))["status"] == "done"
    app = create_app(str(dbp))
    with TestClient(app) as c:
        iid, other = [r["id"] for r in db.images(sid)][:2]
        info = c.get(f"/api/images/{iid}/editor").json()
        model = info["model"]
        model["global"]["Exposure2012"] = 1.7
        model["masks"].append({"name": "Unten", "kind": "gradient", "zero": [0.5, 0.6], "full": [0.5, 1.0],
                               "local": {"Exposure2012": -1.0}})
        r = c.post(f"/api/images/{iid}/editor/preview", json={"model": model, "size": 500,
                                                               "overlay": len(model["masks"]) - 1})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and len(r.content) > 1000
        assert c.put(f"/api/images/{iid}/editor", json={"model": model}).json()["ok"]
        row = db.one("SELECT profile, params, masks FROM edits WHERE image_id=?", (iid,))
        assert row["profile"] == "manual" and json.loads(row["params"])["Exposure2012"] == 1.7
        assert c.get(f"/api/images/{iid}/editor").json()["manual"]
        job = c.post(f"/api/images/{iid}/editor/sync", json={"model": model, "name": "Mein Edit"}).json()
        assert job["template"] == "Mein Edit"
        jm2 = JobManager(db)
        assert jm2.run_sync(job["job_id"])["status"] == "done"
        # eigenes Bild unverändert, anderes Bild mit Verlauf "Unten" aus dem Edit
        assert json.loads(db.one("SELECT params FROM edits WHERE image_id=?", (iid,))["params"])["Exposure2012"] == 1.7
        o = db.one("SELECT profile, masks FROM edits WHERE image_id=?", (other,))
        assert o["profile"] == "tpl:Mein Edit" and "Unten" in [m["CorrectionName"] for m in json.loads(o["masks"])]
        assert c.delete(f"/api/images/{iid}/editor").json()["ok"]
        assert db.one("SELECT profile FROM edits WHERE image_id=?", (iid,))["profile"] != "manual"

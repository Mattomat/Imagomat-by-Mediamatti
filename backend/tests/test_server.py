from pathlib import Path

from fastapi.testclient import TestClient

from tagmatti.server.app import create_app

from .synth import write_shoot


def test_api_flow(tmp_path: Path):
    folder = tmp_path / "shoot"
    write_shoot(folder, n=4)
    app = create_app(str(tmp_path / "api.db"))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        r = client.post("/api/shoots/import", json={"folder": str(folder), "run": False})
        sid = r.json()["shoot_id"]
        jm = app.state.jobs
        jm.run_sync(app.state.db.create_job("pipeline", sid, {}))
        imgs = client.get(f"/api/shoots/{sid}/images").json()
        assert len(imgs) == 4 and all(i["decision"] in ("keep", "reject") for i in imgs)
        iid = imgs[0]["id"]
        assert client.get(f"/api/images/{iid}/preview").status_code == 200
        det = client.get(f"/api/images/{iid}").json()
        assert det["image"]["filename"] == imgs[0]["filename"]
        kept = [i for i in imgs if i["decision"] == "keep"][0]["id"]
        rr = client.get(f"/api/images/{kept}/render?size=800")
        assert rr.status_code == 200 and rr.headers["content-type"] == "image/jpeg"
        p = client.patch(f"/api/images/{iid}/culling", json={"decision": "keep", "rating": 5}).json()
        assert p["manual"] == 1 and p["rating"] == 5
        assert len(client.get("/api/presets").json()) >= 8
        pr = client.post("/api/roster", json={"team": "FCW", "csv": "Max Muster;7\n", "save": True}).json()
        assert pr["entries"][0]["number"] == "7"
        assert client.get("/api/persons").json()[0]["name"] == "Max Muster"
        assert client.get(f"/api/shoots/{sid}/clusters").status_code == 200

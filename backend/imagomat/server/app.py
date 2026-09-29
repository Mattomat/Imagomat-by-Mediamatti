"""Lokale HTTP-API für die Oberfläche (FastAPI, nur 127.0.0.1).

Die Tauri-App startet diesen Server als Sidecar. Im Browser-Modus liefert er zusätzlich das
gebaute Frontend aus (frontend/dist).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from .. import pipeline  # noqa: F401  (registriert alle Jobs)
from ..analysis import import_folder, load_cached_preview, load_masks
from ..config import Settings, cache_dir, load_settings, save_settings
from ..culling.engine import REASONS_DE
from ..vision.action import MOMENTS_DE
from ..db import Database
from ..io import raw as raw_io
from ..jobs import JobManager
from ..lightroom.dialect import Dialect, learn_dialect
from ..people import roster
from ..people.registry import assign_cluster, assign_face, persons, upsert_person
from ..render.pipeline import render
from ..style.jobs import export_profile, import_profile
from ..style.model import list_profiles
from ..style.presets import PRESETS
from ..style.sources import from_folder
from ..vision import ocr
from ..vision.models import license_report

log = logging.getLogger(__name__)


class ImportReq(BaseModel):
    folder: str
    name: str | None = None
    profile: str | None = None
    preset: str | None = None
    teams: list[str] = []
    keep_ratio: float | None = None
    highlights: bool | None = None
    max_keep: int | None = None
    run: bool = True


class CullingPatch(BaseModel):
    decision: str | None = None
    rating: int | None = None
    label: str | None = None


class ShootPatch(BaseModel):
    name: str | None = None
    profile: str | None = None
    teams: list[str] | None = None


class AssignReq(BaseModel):
    person_id: int | None = None
    name: str | None = None
    team: str | None = None
    number: str | None = None


class TrainReq(BaseModel):
    name: str
    catalog: str | None = None
    folders: list[str] = []
    presets: list[str] = []
    base_preset: str | None = None
    min_rating: int = 0
    only_picked: bool = False
    append: bool = False
    learn_people: bool = True


class ExportReq(BaseModel):
    target: str
    formats: list[str] = ["xmp"]
    copy_mode: str = "copy"
    include_rejected: bool = True
    template: str | None = None
    jpeg_side: int = 2048


class RosterReq(BaseModel):
    team: str
    url: str | None = None
    csv: str | None = None
    path: str | None = None
    save: bool = False


class JobReq(BaseModel):
    kind: str
    shoot_id: int | None = None
    params: dict[str, Any] = {}


def find_catalogs(limit: int = 10) -> list[str]:
    """Lightroom-Kataloge an den üblichen Orten finden (neueste zuerst)."""
    home = Path.home()
    found: list[Path] = []
    for base in (home / "Pictures", home / "Bilder", home / "Documents", home / "Desktop"):
        if not base.exists():
            continue
        for depth in ("*.lrcat", "*/*.lrcat", "*/*/*.lrcat"):
            try:
                found += [p for p in base.glob(depth) if not p.name.startswith(".")]
            except OSError:
                continue
    found = sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in found[:limit]]


def create_app(db_path: str | None = None) -> FastAPI:
    db = Database(db_path)
    jobs = JobManager(db)
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        jobs.start()
        yield
        jobs.stop()

    app = FastAPI(title="Imagomat", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:1420", "http://127.0.0.1:1420",
                                                      "tauri://localhost", "http://tauri.localhost"],
                       allow_methods=["*"], allow_headers=["*"])
    sockets: set[WebSocket] = set()

    def on_event(evt: dict[str, Any]) -> None:
        loop = loop_holder.get("loop")
        if not loop:
            return
        msg = json.dumps({"type": "job", **evt})
        for ws in list(sockets):
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)

    jobs.listeners.append(on_event)

    # ------------------------------------------------------------------ Allgemein
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from ..vision.embeddings import get_embedder
        from ..vision.faces import get_backend
        from ..vision.models import torch_device

        # Das grosse CLIP-Modell wird erst bei der ersten Analyse geladen, nicht hier (sonst hängt der Start)
        embedder = get_embedder().name if get_embedder.cache_info().currsize else "lazy"
        return {"version": __version__, "device": torch_device(), "ocr": ocr.available(),
                "face_backend": get_backend().name, "embedder": embedder}

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return load_settings().to_dict()

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any]) -> dict[str, Any]:
        s = Settings.from_dict({**load_settings().to_dict(), **body})
        save_settings(s)
        return s.to_dict()

    @app.get("/api/logs")
    def logs() -> dict[str, Any]:
        """Protokoll für Einstellungen › Protokoll (zum Kopieren bei Problemen)."""
        from ..logs import log_file, logs_dir, tail

        failed = [dict(r) for r in db.query(
            "SELECT id, kind, shoot_id, error, message, updated_at FROM jobs WHERE status='failed' "
            "AND updated_at > ? ORDER BY updated_at DESC LIMIT 10", (time.time() - 86400,))]
        for f in failed:
            f["error"] = (f["error"] or "")[-4000:]
        return {"dir": str(logs_dir()), "version": __version__, "platform": sys.platform,
                "failed_jobs": failed, "log": tail(log_file(), 400),
                "setup": tail(logs_dir() / "setup.log", 60), "backend": tail(logs_dir() / "backend.log", 80)}

    @app.get("/api/licenses")
    def licenses() -> list[dict[str, Any]]:
        return license_report()

    @app.get("/api/dialect")
    def dialect() -> dict[str, Any]:
        return Dialect.load().__dict__

    @app.post("/api/dialect/learn")
    def dialect_learn(body: dict[str, str]) -> dict[str, Any]:
        """Referenz-XMPs aus Lightroom einlesen (Ordner)."""
        samples = list(from_folder(Path(body["folder"])))
        d = learn_dialect([s.crs for s in samples], [s.label for s in samples if s.label], Dialect.load())
        d.save()
        return {**d.__dict__, "files": len(samples)}

    # ------------------------------------------------------------------ Jobs
    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [dict(r) for r in db.query("SELECT * FROM jobs ORDER BY id DESC LIMIT 50")]

    @app.post("/api/jobs")
    def submit_job(req: JobReq) -> dict[str, Any]:
        return {"job_id": jobs.submit(req.kind, req.shoot_id, **req.params)}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int) -> dict[str, Any]:
        jobs.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: int) -> dict[str, Any]:
        jobs.resume(job_id)
        return {"ok": True}

    @app.websocket("/api/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        sockets.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            sockets.discard(ws)

    # ------------------------------------------------------------------ Shoots
    @app.get("/api/shoots")
    def shoots() -> list[dict[str, Any]]:
        rows = db.query(
            "SELECT s.*, COUNT(i.id) n, SUM(CASE WHEN c.decision='keep' THEN 1 ELSE 0 END) kept, "
            "MIN(CASE WHEN c.is_series_best=1 AND c.decision='keep' THEN i.id END) cover, MIN(i.id) first_image "
            "FROM shoots s LEFT JOIN images i ON i.shoot_id=s.id LEFT JOIN culling c ON c.image_id=i.id "
            "WHERE s.name NOT LIKE 'Kalibrierung:%' AND s.name != '__Referenzbilder__' "
            "GROUP BY s.id ORDER BY s.created_at DESC")
        out = []
        for r in rows:
            d = dict(r)
            d["cover"] = d["cover"] or d["first_image"]
            job = db.one("SELECT id, kind, status, progress, total, message FROM jobs WHERE shoot_id=? "
                         "ORDER BY id DESC LIMIT 1", (d["id"],))
            d["job"] = dict(job) if job else None
            out.append(d)
        return out

    @app.get("/api/shoots/{sid}")
    def shoot(sid: int) -> dict[str, Any]:
        r = next((x for x in shoots() if x["id"] == sid), None)
        if r is None:
            raise HTTPException(404)
        return r

    @app.post("/api/reveal")
    def reveal(body: dict[str, str]) -> dict[str, Any]:
        """Ordner im Finder zeigen (nur lokal)."""
        import subprocess
        import sys

        path = Path(body["path"]).expanduser()
        if not path.exists():
            raise HTTPException(404, "Ordner nicht gefunden")
        cmd = ["open", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path)]
        subprocess.Popen(cmd)
        return {"ok": True}

    @app.post("/api/shoots/import")
    def import_shoot(req: ImportReq) -> dict[str, Any]:
        folder = Path(req.folder).expanduser()
        if not folder.is_dir():
            raise HTTPException(400, f"Ordner nicht gefunden: {folder}")
        sid = import_folder(db, folder, req.name, req.profile)
        if req.teams:
            db.update_shoot_settings(sid, teams=req.teams)
        job_id = jobs.submit("pipeline", sid, keep_ratio=req.keep_ratio, profile=req.profile,
                             preset=req.preset, highlights=req.highlights,
                             max_keep=req.max_keep) if req.run else None
        return {"shoot_id": sid, "job_id": job_id}

    @app.patch("/api/shoots/{sid}")
    def patch_shoot(sid: int, req: ShootPatch) -> dict[str, Any]:
        with db.tx() as c:
            if req.name:
                c.execute("UPDATE shoots SET name=? WHERE id=?", (req.name, sid))
            if req.profile is not None:
                c.execute("UPDATE shoots SET profile=? WHERE id=?", (req.profile or None, sid))
        if req.teams is not None:
            db.update_shoot_settings(sid, teams=req.teams)
        return dict(db.one("SELECT * FROM shoots WHERE id=?", (sid,)))

    @app.post("/api/shoots/{sid}/run/{kind}")
    def run_stage(sid: int, kind: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        if kind not in ("analyze", "cull", "people", "develop", "pipeline"):
            raise HTTPException(400, "unbekannter Schritt")
        return {"job_id": jobs.submit(kind, sid, **(body or {}))}

    @app.get("/api/shoots/{sid}/images")
    def shoot_images(sid: int) -> list[dict[str, Any]]:
        rows = db.query(
            "SELECT i.id, i.filename, i.capture_time, i.iso, i.orientation, c.decision, c.rating, c.label, "
            "c.reasons, c.series_id, c.is_series_best, c.score, c.manual, e.confidence, e.denoise, a.data "
            "FROM images i LEFT JOIN culling c ON c.image_id=i.id LEFT JOIN edits e ON e.image_id=i.id "
            "LEFT JOIN analysis a ON a.image_id=i.id WHERE i.shoot_id=? ORDER BY i.capture_time, i.filename",
            (sid,))
        people: dict[int, list[str]] = {}
        for r in db.query(
                "SELECT DISTINCT x.image_id, p.name FROM (SELECT image_id, person_id FROM faces UNION "
                "SELECT image_id, person_id FROM numbers) x JOIN persons p ON p.id=x.person_id "
                "JOIN images i ON i.id=x.image_id WHERE i.shoot_id=?", (sid,)):
            people.setdefault(r["image_id"], []).append(r["name"])
        out = []
        for r in rows:
            a = json.loads(r["data"]) if r["data"] else {}
            reasons = json.loads(r["reasons"]) if r["reasons"] else []
            out.append({
                "id": r["id"], "filename": r["filename"], "capture_time": r["capture_time"], "iso": r["iso"],
                "decision": r["decision"], "rating": r["rating"], "label": r["label"],
                "reasons": [REASONS_DE.get(x, x) for x in reasons], "reason_keys": reasons,
                "series": r["series_id"], "best": bool(r["is_series_best"]), "score": r["score"],
                "manual": bool(r["manual"]), "confidence": r["confidence"], "denoise": r["denoise"],
                "people": people.get(r["id"], []), "notes": a.get("develop_notes", []), "preset": a.get("preset"),
                "faces": a.get("face_count", 0),
                "moment": MOMENTS_DE.get(a.get("moment") or ""), "action": a.get("action"),
            })
        return out

    @app.post("/api/shoots/{sid}/export")
    def export(sid: int, req: ExportReq) -> dict[str, Any]:
        return {"job_id": jobs.submit("export", sid, **req.model_dump())}

    @app.post("/api/shoots/{sid}/feedback")
    def feedback(sid: int, body: dict[str, Any]) -> dict[str, Any]:
        return {"job_id": jobs.submit("feedback", sid, **body)}

    # ------------------------------------------------------------------ Bilder
    @app.get("/api/images/{iid}")
    def image(iid: int) -> dict[str, Any]:
        r = db.one("SELECT * FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        e = db.one("SELECT * FROM edits WHERE image_id=?", (iid,))
        c = db.one("SELECT * FROM culling WHERE image_id=?", (iid,))
        faces = [dict(f) | {"embedding": None, "bbox": json.loads(f["bbox"])} for f in
                 db.query("SELECT f.*, p.name person FROM faces f LEFT JOIN persons p ON p.id=f.person_id "
                          "WHERE f.image_id=?", (iid,))]
        return {"image": {k: r[k] for k in r.keys() if k != "exif"} | {"exif": json.loads(r["exif"] or "{}")},
                "analysis": {k: v for k, v in db.get_analysis(iid).items() if k not in ("hist", "lin_hist")},
                "edit": ({**dict(e), "params": json.loads(e["params"]),
                          "masks": json.loads(e["masks"]) if e["masks"] else None} if e else None),
                "culling": ({**dict(c), "reasons": [REASONS_DE.get(x, x) for x in json.loads(c["reasons"] or "[]")]}
                            if c else None),
                "faces": faces}

    @app.get("/api/images/{iid}/preview")
    def preview(iid: int) -> FileResponse:
        r = db.one("SELECT preview_path FROM images WHERE id=?", (iid,))
        if not r or not r["preview_path"] or not Path(r["preview_path"]).exists():
            raise HTTPException(404)
        return FileResponse(r["preview_path"], media_type="image/jpeg")

    @app.get("/api/images/{iid}/render")
    def render_after(iid: int, size: int = 1600) -> Response:
        r = db.one("SELECT i.path, e.params, e.masks, e.updated_at FROM images i LEFT JOIN edits e "
                   "ON e.image_id=i.id WHERE i.id=?", (iid,))
        if not r:
            raise HTTPException(404)
        cache = cache_dir() / "renders" / f"{iid}_{size}_{int((r['updated_at'] or 0) * 1000)}.jpg"
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg")
        crs = json.loads(r["params"]) if r["params"] else {}
        if r["masks"]:
            crs["MaskGroupBasedCorrections"] = json.loads(r["masks"])
        path = Path(r["path"])
        if not raw_io.is_raw(path):
            raise HTTPException(415, "Rendering nur für RAW-Dateien")
        lin_cache = cache_dir() / "linear" / f"{iid}.npz"
        lin_cache.parent.mkdir(parents=True, exist_ok=True)
        if lin_cache.exists():
            z = np.load(lin_cache)
            lin, xyz, wb, orient = z["lin"].astype(np.float32), z["xyz"], z["wb"], int(z["orient"])
        else:
            lin, info = raw_io.decode_any(path, half_size=True)
            h, w = lin.shape[:2]
            s = 2048 / max(h, w)
            if s < 1:
                lin = cv2.resize(lin, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            xyz, wb, orient = info.xyz_to_cam, info.camera_wb, info.orientation
            np.savez(lin_cache, lin=lin.astype(np.float16), xyz=xyz, wb=wb, orient=orient)
        m = load_masks(iid)
        seg = {"subject": m[0], "sky": m[1]} if m else {}
        img = render(lin, xyz, wb, crs, orient, seg, size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return FileResponse(cache, media_type="image/jpeg")

    @app.patch("/api/images/{iid}/culling")
    def patch_culling(iid: int, req: CullingPatch) -> dict[str, Any]:
        c = db.one("SELECT * FROM culling WHERE image_id=?", (iid,))
        decision = req.decision or (c["decision"] if c else "keep")
        rating = req.rating if req.rating is not None else (c["rating"] if c else None)
        if req.decision == "reject" and req.rating is None:
            rating = load_settings().culling.reject_rating
        label = req.label if req.label is not None else (c["label"] if c else None)
        db.set_culling(iid, score=c["score"] if c else 0.0, decision=decision, rating=rating or 0,
                       label=label or None, reasons=json.loads(c["reasons"]) if c and decision == "reject" else [],
                       series_id=c["series_id"] if c else None, is_best=bool(c["is_series_best"]) if c else False,
                       manual=True)
        return dict(db.one("SELECT * FROM culling WHERE image_id=?", (iid,)))

    # ------------------------------------------------------------------ Personen
    @app.get("/api/persons")
    def get_persons() -> list[dict[str, Any]]:
        return [p.__dict__ for p in persons(db)]

    @app.post("/api/persons")
    def add_person(req: AssignReq) -> dict[str, Any]:
        if not req.name:
            raise HTTPException(400, "Name fehlt")
        return {"id": upsert_person(db, req.name, req.team, req.number)}

    @app.delete("/api/persons/{pid}")
    def delete_person(pid: int) -> dict[str, Any]:
        with db.tx() as c:
            c.execute("DELETE FROM persons WHERE id=?", (pid,))
        return {"ok": True}

    @app.get("/api/shoots/{sid}/clusters")
    def clusters(sid: int) -> list[dict[str, Any]]:
        rows = db.query(
            "SELECT f.id, f.image_id, f.cluster_id, f.person_id, f.assigned_by, p.name FROM faces f "
            "JOIN images i ON i.id=f.image_id LEFT JOIN persons p ON p.id=f.person_id WHERE i.shoot_id=? "
            "ORDER BY f.det_score DESC", (sid,))
        groups: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = f"p{r['person_id']}" if r["person_id"] else (f"c{r['cluster_id']}" if r["cluster_id"] is not None
                                                              else "unknown")
            g = groups.setdefault(key, {"key": key, "person_id": r["person_id"], "name": r["name"],
                                        "cluster_id": r["cluster_id"], "faces": [], "count": 0})
            g["count"] += 1
            if len(g["faces"]) < 12:
                g["faces"].append({"id": r["id"], "image_id": r["image_id"], "assigned_by": r["assigned_by"]})
        return sorted(groups.values(), key=lambda g: (g["name"] is None, -g["count"]))

    @app.get("/api/faces/{fid}/crop")
    def face_crop(fid: int) -> Response:
        f = db.one("SELECT f.bbox, i.* FROM faces f JOIN images i ON i.id=f.image_id WHERE f.id=?", (fid,))
        if not f:
            raise HTTPException(404)
        img = load_cached_preview(f)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = json.loads(f["bbox"])
        pad = 0.35
        bw, bh = x1 - x0, y1 - y0
        crop = img[max(0, int((y0 - pad * bh) * h)):int((y1 + pad * bh) * h),
                   max(0, int((x0 - pad * bw) * w)):int((x1 + pad * bw) * w)]
        crop = cv2.resize(crop, (160, int(160 * crop.shape[0] / max(crop.shape[1], 1))))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return Response(buf.tobytes(), media_type="image/jpeg")

    def _person_from(req: AssignReq) -> int | None:
        if req.person_id:
            return req.person_id
        if req.name:
            return upsert_person(db, req.name, req.team, req.number)
        return None

    @app.post("/api/shoots/{sid}/clusters/{cid}/assign")
    def assign_c(sid: int, cid: int, req: AssignReq) -> dict[str, Any]:
        pid = _person_from(req)
        if pid is None:
            raise HTTPException(400, "Person fehlt")
        return {"assigned": assign_cluster(db, sid, cid, pid), "person_id": pid}

    @app.post("/api/faces/{fid}/assign")
    def assign_f(fid: int, req: AssignReq) -> dict[str, Any]:
        pid = _person_from(req)
        assign_face(db, fid, pid, "manual")
        return {"person_id": pid}

    @app.post("/api/roster")
    def roster_import(req: RosterReq) -> dict[str, Any]:
        if req.path:
            entries = roster.parse_csv(Path(req.path).expanduser().read_text("utf-8-sig"), req.team)
        elif req.csv:
            entries = roster.parse_csv(req.csv, req.team)
        else:
            url = req.url or roster.DEFAULT_SOURCES.get(req.team)
            if not url:
                raise HTTPException(400, "URL oder CSV angeben")
            try:
                entries = roster.fetch(url, req.team)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"Kader konnte nicht geladen werden: {e}") from e
        if req.save:
            roster.save(db, entries)
        return {"entries": [e.__dict__ for e in entries], "saved": req.save}

    @app.post("/api/people/learn")
    def people_learn(body: dict[str, Any]) -> dict[str, Any]:
        """Referenzbilder (bereits benannte JPGs) einlesen."""
        return {"job_id": jobs.submit("learn_people", None, **body)}

    @app.get("/api/persons/{pid}/face")
    def person_face(pid: int) -> Response:
        f = db.one("SELECT id FROM faces WHERE person_id=? ORDER BY (assigned_by IN ('manual','confirmed')) DESC, "
                   "sharpness DESC LIMIT 1", (pid,))
        if not f:
            raise HTTPException(404)
        return face_crop(f["id"])

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        n_persons = db.one("SELECT COUNT(*) FROM persons")[0]
        n_known = db.one("SELECT COUNT(DISTINCT person_id) FROM faces WHERE assigned_by IN "
                         "('manual','confirmed','number')")[0]
        teams = [r[0] for r in db.query("SELECT DISTINCT team FROM persons WHERE team IS NOT NULL ORDER BY team")]
        return {"persons": n_persons, "persons_with_face": n_known, "teams": teams,
                "profiles": list_profiles(), "catalogs": find_catalogs()}

    @app.get("/api/roster/sources")
    def roster_sources() -> dict[str, str]:
        return roster.DEFAULT_SOURCES

    # ------------------------------------------------------------------ Profile
    @app.get("/api/profiles")
    def profiles() -> list[dict[str, Any]]:
        return list_profiles()

    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        return [{"key": p.key, "name": p.name, "description": p.description} for p in PRESETS.values()]

    @app.post("/api/profiles/train")
    def train(req: TrainReq) -> dict[str, Any]:
        return {"job_id": jobs.submit("train_profile", None, **req.model_dump())}

    @app.post("/api/profiles/{name}/export")
    def prof_export(name: str, body: dict[str, str]) -> dict[str, Any]:
        return {"file": str(export_profile(name, Path(body["target"])))}

    @app.post("/api/profiles/import")
    def prof_import(body: dict[str, str]) -> dict[str, Any]:
        return {"name": import_profile(Path(body["file"]), body.get("name"))}

    @app.post("/api/calibrate")
    def calibrate(body: dict[str, Any]) -> dict[str, Any]:
        return {"job_id": jobs.submit("calibrate_culling", None, **body)}

    # ------------------------------------------------------------------ Frontend
    dist = Path(os.environ.get("IMAGOMAT_FRONTEND", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")

    app.state.db = db
    app.state.jobs = jobs
    return app


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    from ..logs import setup_logging

    setup_logging()
    uvicorn.run(create_app(), host=host, port=port, log_level="info", access_log=False)

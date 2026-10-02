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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from .. import pipeline  # noqa: F401  (registriert alle Jobs)
from ..analysis import import_folder, load_cached_preview, load_masks
from ..config import Settings, cache_dir, image_key, load_settings, save_settings
from ..culling.engine import REASONS_DE
from ..vision.action import MOMENTS_DE
from ..vision.tags import labels as tag_labels
from ..db import Database
from ..io import raw as raw_io
from ..jobs import JobManager
from ..lightroom.dialect import Dialect, learn_dialect
from ..people import roster
from ..people.registry import assign_cluster, assign_face, persons, upsert_person
from ..render.pipeline import RENDER_VERSION, render, render_hybrid
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
    burst_keep: int | None = None
    mode: str | None = None          # "full" (Standard) | "people" (nur Personen, z. B. fertige JPGs)
    run: bool = True
    copy_to: str | None = None       # RAWs zuerst hierhin kopieren (Ablageort/Shoot-Ordner)


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
    long_side: int | None = None      # 0 = Originalgrösse (JPEG/WebP/PNG/TIFF in voller Qualität)
    quality: int = 90
    naming: str = "original"          # "original" | "custom" (name_base-001 …)
    name_base: str = ""
    denoise: str | None = None        # "off" | "edit" | "all": KI-Entrauschen als DNG / im fertigen Bild
    min_rating: int = 0
    render_target: str | None = None  # Ordner für JPEG/WebP/PNG/TIFF (sonst wie target)


class SocialReq(BaseModel):
    format: str = "story"
    selection: str = "top"
    ids: list[int] | None = None
    target: str | None = None


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
    jobs = JobManager(db, recover=True)
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        jobs.start()
        # Unterbrochenes Lernen (z. B. durch ein App-Update) automatisch fortsetzen; bereits
        # berechnete Merkmale liegen im Cache, es geht also schnell weiter.
        from ..jobs import LEARN_KINDS

        latest: dict[tuple[str, str], int] = {}
        for j in db.query("SELECT id, kind, params FROM jobs WHERE status='paused' AND updated_at > ? ORDER BY id",
                          (time.time() - 7 * 86400,)):
            if j["kind"] in LEARN_KINDS:
                name = str(json.loads(j["params"] or "{}").get("name", ""))
                latest[(j["kind"], name)] = j["id"]          # pro Stil nur den neuesten Auftrag
        for jid in latest.values():
            jobs.resume(jid)
        yield
        jobs.stop()

    app = FastAPI(title="Imagomat", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:1420", "http://127.0.0.1:1420",
                                                      "tauri://localhost", "http://tauri.localhost"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def no_stale_images(request: Any, call_next: Any) -> Any:
        """Bilder nie ungeprüft aus dem Browser-Cache zeigen: nach Löschen/Neuimport bekommt ein anderes Bild
        dieselbe Nummer, und die Vorschau-Engine ändert sich mit neuen Versionen."""
        resp = await call_next(request)
        if request.url.path.startswith("/api/images/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

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
        sample = next((r["path"] for r in db.query(
            "SELECT i.path FROM images i JOIN shoots s ON s.id=i.shoot_id ORDER BY s.created_at DESC, i.id DESC "
            "LIMIT 200") if raw_io.is_raw(Path(r["path"])) and Path(r["path"]).exists()), None)
        try:
            raw = raw_io.raw_status(sample)
        except Exception as e:  # noqa: BLE001
            raw = {"error": str(e)}
        return {"dir": str(logs_dir()), "version": __version__, "platform": sys.platform, "raw": raw,
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

    @app.post("/api/jobs/{job_id}/finish")
    def finish_job(job_id: int) -> dict[str, Any]:
        """Stil-Lernen jetzt abschliessen: mit allen bisher analysierten Bildern das Modell bauen."""
        j = db.job(job_id)
        if not j:
            raise HTTPException(404, "Auftrag nicht gefunden")
        if j["status"] == "running" and jobs.finish(job_id):
            return {"job_id": job_id, "finishing": True}
        if j["kind"] != "train_profile":
            raise HTTPException(400, "Nur Stil-Lernen kann vorzeitig abgeschlossen werden")
        jobs.cancel(job_id)
        params = {**json.loads(j["params"] or "{}"), "cached_only": True}
        return {"job_id": jobs.submit("train_profile", None, **params), "finishing": True}

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
        r = db.one(
            "SELECT s.*, COUNT(i.id) n, SUM(CASE WHEN c.decision='keep' THEN 1 ELSE 0 END) kept, "
            "MIN(CASE WHEN c.is_series_best=1 AND c.decision='keep' THEN i.id END) cover, MIN(i.id) first_image "
            "FROM shoots s LEFT JOIN images i ON i.shoot_id=s.id LEFT JOIN culling c ON c.image_id=i.id "
            "WHERE s.id=? GROUP BY s.id", (sid,))
        if r is None:
            raise HTTPException(404)
        d = dict(r)
        d["cover"] = d["cover"] or d["first_image"]
        job = db.one("SELECT id, kind, status, progress, total, message FROM jobs WHERE shoot_id=? "
                     "ORDER BY id DESC LIMIT 1", (sid,))
        d["job"] = dict(job) if job else None
        # Wie wurden die RAWs gelesen? (LibRaw = echte Sensordaten; Vorschau = nur eingebettetes JPEG)
        src: dict[str, int] = {}
        for a in db.query("SELECT json_extract(a.data, '$.raw_source') s FROM analysis a JOIN images i "
                          "ON i.id=a.image_id WHERE i.shoot_id=? AND lower(i.path) NOT LIKE '%.jpg' "
                          "AND lower(i.path) NOT LIKE '%.jpeg'", (sid,)):
            key = a["s"] or "unbekannt"
            src[key] = src.get(key, 0) + 1
        d["raw_sources"] = src
        return d

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

    @app.get("/api/import/info")
    def import_info(folder: str) -> dict[str, Any]:
        """Für den Import-Dialog: Anzahl, Grösse und Aufnahmedatum der Bilder im Quellordner."""
        import datetime as _dt

        from ..analysis import scan_folder
        from ..io import exif as exif_io

        f = Path(folder).expanduser()
        if not f.is_dir():
            raise HTTPException(400, f"Ordner nicht gefunden: {f}")
        files = scan_folder(f)
        date = None
        if files:
            m = exif_io.read_exif(files[:1]).get(str(files[0]), {})
            ts = m.get("capture_time") or files[0].stat().st_mtime
            date = _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
        return {"count": len(files), "bytes": sum(p.stat().st_size for p in files), "date": date,
                "home": str(Path.home())}

    @app.post("/api/shoots/import")
    def import_shoot(req: ImportReq) -> dict[str, Any]:
        folder = Path(req.folder).expanduser()
        if not folder.is_dir():
            raise HTTPException(400, f"Ordner nicht gefunden: {folder}")
        copy_from = None
        if not req.profile and not req.preset:
            req.profile = "neutral"           # wie Lightroom: frisch importiert, ohne Vorgabe darüber
        if req.copy_to and Path(req.copy_to).expanduser().resolve() != folder.resolve():
            dest = Path(req.copy_to).expanduser().resolve()
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise HTTPException(400, f"Ablageort nicht beschreibbar: {dest} ({e})") from e
            if folder.resolve() in dest.parents:
                raise HTTPException(400, "Der Ablageort darf nicht im Quellordner liegen")
            sid = db.upsert_shoot(req.name or dest.name, str(dest), req.profile)
            copy_from = str(folder)
        else:
            sid = import_folder(db, folder, req.name, req.profile)
        if req.teams:
            db.update_shoot_settings(sid, teams=req.teams)
        job_id = jobs.submit("pipeline", sid, keep_ratio=req.keep_ratio, profile=req.profile,
                             preset=req.preset, highlights=req.highlights,
                             max_keep=req.max_keep, mode=req.mode,
                             burst_keep=req.burst_keep, copy_from=copy_from) if req.run or copy_from else None
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

    @app.get("/api/shoots/{sid}/style")
    def shoot_style(sid: int) -> dict[str, Any]:
        """Womit wurde der Shoot bearbeitet? (eigener Stil oder mitgeliefertes Preset)"""
        rows = db.query("SELECT e.profile, COUNT(*) n FROM edits e JOIN images i ON i.id=e.image_id "
                        "WHERE i.shoot_id=? GROUP BY e.profile ORDER BY n DESC", (sid,))
        used = rows[0]["profile"] if rows else None
        from ..style.model import list_profiles

        from ..style import compare as cmp
        from ..style.presets import PRESETS

        chosen = bool(used and used.startswith("preset:") and PRESETS.get(used[7:]) and PRESETS[used[7:]].group)
        return {"used": used, "label": cmp.label(used),
                "is_preset": bool(used and used.startswith("preset:")) and not chosen,
                "profiles": [p["name"] for p in list_profiles()], "default": load_settings().default_profile}

    @app.get("/api/shoots/{sid}/images")
    def shoot_images(sid: int) -> list[dict[str, Any]]:
        rows = db.query(
            "SELECT i.id, i.filename, i.path, i.capture_time, i.iso, i.orientation, c.decision, c.rating, c.label, "
            "c.reasons, c.series_id, c.is_series_best, c.score, c.manual, e.confidence, e.denoise, e.updated_at, "
            "e.profile AS edit_profile, a.data "
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
                "people_check": a.get("people_check", []),
                "tags": tag_labels(a.get("tags")),
                "edit_v": int((r["updated_at"] or 0) * 1000) % 10_000_000,
                "rendered": _rendered_any(r["id"], r["path"], r["updated_at"]) is not None,
                "hand_edited": r["edit_profile"] == "manual",
                # Original, solange nichts bewusst bearbeitet wurde (neutral = wie importiert)
                "edited": r["edit_profile"] not in (None, "neutral"),
            })
        _prerender([o["id"] for o in out if o["decision"] == "keep" and o["edited"]])
        return out

    @app.post("/api/shoots/{sid}/export")
    def export(sid: int, req: ExportReq) -> dict[str, Any]:
        params = req.model_dump()
        if req.copy_mode != "inplace":
            # Nie in einen Ordner mit einem früheren Export schreiben: Lightroom würde sonst auch die
            # alten (evtl. aussortierten) Bilder wieder importieren
            tgt = Path(req.target).expanduser()
            base, i = tgt, 2
            while tgt.exists() and any(tgt.iterdir()):
                tgt = base.with_name(f"{base.name} ({i})")
                i += 1
            params["target"] = str(tgt)
        return {"job_id": jobs.submit("export", sid, **params), "target": params["target"]}

    @app.post("/api/shoots/{sid}/tag-export")
    def tag_export(sid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Nur Personen: Bilder mit eingebetteten Namen in einen Ordner schreiben."""
        if not body.get("target"):
            raise HTTPException(400, "Zielordner fehlt")
        return {"job_id": jobs.submit("tag_export", sid, target=body["target"],
                                      only_with_people=bool(body.get("only_with_people"))),
                "target": body["target"]}

    @app.post("/api/lightroom/open")
    def lightroom_open(body: dict[str, str]) -> dict[str, Any]:
        """Ordner an Lightroom Classic übergeben: öffnet den Import-Dialog mit diesem Ordner."""
        import subprocess

        path = Path(body["path"]).expanduser()
        if not path.exists():
            raise HTTPException(404, "Ordner nicht gefunden")
        if sys.platform != "darwin":
            raise HTTPException(400, "Nur auf dem Mac verfügbar")
        for cmd in (["open", "-b", "com.adobe.LightroomClassicCC7", str(path)],
                    ["open", "-a", "Adobe Lightroom Classic", str(path)]):
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                return {"ok": True}
        raise HTTPException(404, "Lightroom Classic nicht gefunden")

    @app.post("/api/shoots/{sid}/social")
    def social(sid: int, req: SocialReq) -> dict[str, Any]:
        """Bilder für Instagram & Co. speichern (Story, Post, Quadrat, Original)."""
        from ..export.social import default_target

        shoot = db.one("SELECT name FROM shoots WHERE id=?", (sid,))
        if not shoot:
            raise HTTPException(404, "Shoot nicht gefunden")
        target = req.target or str(default_target(shoot["name"], req.format))
        return {"job_id": jobs.submit("social", sid, **{**req.model_dump(), "target": target}), "target": target}

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
        r = db.one("SELECT * FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        if not r["preview_path"] or not Path(r["preview_path"]).exists():
            # z. B. Referenzbilder aus Lightroom: Vorschau bei Bedarf erzeugen und merken
            from ..analysis import preview_path as _pp

            try:
                img = load_cached_preview(r)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(404, f"Vorschau nicht verfügbar: {e}") from e
            pp = _pp(iid)
            cv2.imwrite(str(pp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            with db.tx() as c:
                c.execute("UPDATE images SET preview_path=? WHERE id=?", (str(pp), iid))
            return FileResponse(str(pp), media_type="image/jpeg")
        return FileResponse(r["preview_path"], media_type="image/jpeg")

    import threading

    _decode_locks: dict[int, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def _linear(iid: int, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Lineare Bilddaten fürs Rendern (einmal dekodieren, dann aus dem Cache). Mehrere gleichzeitige
        Vorschauen desselben Bildes (Stil-Leiste) dekodieren die RAW nur einmal."""
        with _locks_guard:
            lock = _decode_locks.setdefault(iid, threading.Lock())
        with lock:
            return _linear_once(iid, path)

    def _linear_once(iid: int, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        lin_cache = cache_dir() / "linear" / f"{image_key(iid, path)}.npz"
        lin_cache.parent.mkdir(parents=True, exist_ok=True)
        if lin_cache.exists():
            z = np.load(lin_cache)
            if "source" in z.files:
                _sources[iid] = (str(z["source"]), float(z["as_temp"]), float(z["as_tint"]))
                return z["lin"].astype(np.float32), z["xyz"], z["wb"], int(z["orient"])
        lin, info = raw_io.decode_any(path, half_size=True)
        h, w = lin.shape[:2]
        s = 2048 / max(h, w)
        if s < 1:
            lin = cv2.resize(lin, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        xyz, wb, orient = info.xyz_to_cam, info.camera_wb, info.orientation
        source = "preview" if info.extra.get("from_preview") else str(info.extra.get("source", "libraw"))
        as_temp, as_tint = float(info.as_shot_temp or 0.0), float(info.as_shot_tint or 0.0)
        if source != "libraw" and not as_temp:
            a = db.one("SELECT data FROM analysis WHERE image_id=?", (iid,))
            d = json.loads(a["data"]) if a and a["data"] else {}
            as_temp, as_tint = float(d.get("as_shot_temp") or 0.0), float(d.get("as_shot_tint") or 0.0)
        _sources[iid] = (source, as_temp, as_tint)
        np.savez(lin_cache, lin=lin.astype(np.float16), xyz=xyz, wb=wb, orient=orient, source=source,
                 as_temp=as_temp, as_tint=as_tint)
        return lin, xyz, wb, orient

    _sources: dict[int, tuple[str, float, float]] = {}
    _hi: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    _full: dict[str, Any] = {}
    _full_lock = threading.Lock()
    _hi_lock = threading.Lock()

    def _linear_hi(iid: int, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Lineare Daten in hoher Auflösung (halbes RAW, z. B. 3504 px bei 33 MP) für Lupe und Editor auf grossen
        Bildschirmen. Nur im Speicher (die letzten Bilder), damit der Zwischenspeicher auf der Platte klein bleibt."""
        key = image_key(iid, path)
        with _hi_lock:
            hit = _hi.get(key)
            if hit is not None:
                _hi[key] = _hi.pop(key)
                return hit
        _linear(iid, path)                     # Quelle / Aufnahme-Weissabgleich ermitteln (aus dem Cache schnell)
        lin, info = raw_io.decode_any(path, half_size=True)
        if info.extra.get("from_preview"):
            return _linear(iid, path)
        h, w = lin.shape[:2]
        s = 4096 / max(h, w)
        if s < 1:
            lin = cv2.resize(lin, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out = (lin.astype(np.float32), info.xyz_to_cam, info.camera_wb, info.orientation)
        with _hi_lock:
            _hi[key] = out
            while len(_hi) > 3:
                _hi.pop(next(iter(_hi)))
        return out

    def _retouch_ops(iid: int) -> list[dict[str, Any]]:
        r = db.one("SELECT ops FROM retouch WHERE image_id=?", (iid,))
        return json.loads(r["ops"]) if r and r["ops"] else []

    def _full_lin(iid: int, path: Path) -> tuple[np.ndarray, Any]:
        """Volle Auflösung (Anzeigeorientierung) für Retusche und 1:1-Ansichten; nur das letzte Bild im Speicher."""
        with _full_lock:
            if _full.get("key") != image_key(iid, path):
                lin, info = raw_io.decode_any(path, half_size=False)
                _full.clear()
                _full.update(key=image_key(iid, path), lin=lin, info=info)
            return _full["lin"], _full["info"]

    def _patches(iid: int, path: Path, ops: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        from ..retouch import ensure_patches

        ops = _retouch_ops(iid) if ops is None else ops
        if not ops:
            return []

        def full() -> tuple[np.ndarray, np.ndarray]:
            lin, _xyz, wb, _o = _linear_hi(iid, path)
            return lin, np.asarray(wb, np.float32)
        return ensure_patches(image_key(iid, path), ops, full)

    def _cam_curve(iid: int) -> list[list[float]] | None:
        """Profil "Kamera": Kurven, mit denen die RAW-Entwicklung bei 0 genau wie das Kamera-Original aussieht."""
        from ..render.pipeline import camera_curve, render
        from ..style.userpresets import neutral_crs

        a = db.get_analysis(iid) or {}
        if a.get("camera_curve_v") == 1 and a.get("camera_curve"):
            return a["camera_curve"]
        r = db.one("SELECT * FROM images WHERE id=?", (iid,))
        if not r or not raw_io.is_raw(Path(r["path"])):
            return None
        try:
            lin, xyz, wb, orient = _linear(iid, Path(r["path"]))
            crs = {**neutral_crs(), "CameraProfile": "Adobe Color"}
            img = render(lin, xyz, wb, _with_as_shot(iid, crs, cam=False), orient, {}, 640)
            cam = load_cached_preview(r)
            if abs(np.log((cam.shape[1] / cam.shape[0]) / (img.shape[1] / img.shape[0]))) > 0.03:
                return None
            cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            cc = camera_curve(img.astype(np.float32) / 255, cam.astype(np.float32) / 255)
        except Exception:  # noqa: BLE001
            log.debug("Kamera-Profil für %s nicht berechnet", iid, exc_info=True)
            return None
        db.update_analysis(iid, {"camera_curve": cc, "camera_curve_v": 1})
        return cc

    def _with_as_shot(iid: int, crs: dict[str, Any], cam: bool = True) -> dict[str, Any]:
        """Bei schon weissabgeglichenen Daten (Apple RAW-Engine, Vorschau) den Aufnahme-Weissabgleich mitgeben;
        beim Profil "Kamera" die Kurven dazu."""
        from ..render.pipeline import CAM_CURVE, uses_camera_profile

        if cam and uses_camera_profile(crs) and CAM_CURVE not in crs:
            cc = _cam_curve(iid)
            if cc:
                crs = {**crs, CAM_CURVE: cc}
        src = _sources.get(iid)
        if not src or src[0] == "libraw":
            return crs
        from ..render.pipeline import AS_SHOT_TEMP, AS_SHOT_TINT

        if not src[1]:                       # Aufnahme-Wert unbekannt: Weissabgleich nicht verschieben
            return {**crs, AS_SHOT_TEMP: crs.get("Temperature"), AS_SHOT_TINT: crs.get("Tint") or 0}
        return {**crs, AS_SHOT_TEMP: src[1], AS_SHOT_TINT: src[2]}

    def _render_file(iid: int, path: Path, crs: dict[str, Any], size: int, cache: Path) -> Response:
        if not raw_io.is_raw(path):
            return preview(iid)            # fertige JPGs: nichts zu entwickeln, Vorschau zeigen
        from ..retouch import apply_patches

        m = load_masks(iid)
        seg = {"subject": m[0], "sky": m[1]} if m else {}
        patches = _patches(iid, path)
        if size > 2048 or patches:
            # grosse Vorschau: direkt aus dem RAW in Bildschirmauflösung (mit Schärfen wie Lightroom); mit Retusche
            # nie aus der Kamera-Vorschau (dort wäre das Entfernte noch zu sehen)
            lin, xyz, wb, orient = _linear_hi(iid, path) if size > 2048 else _linear(iid, path)
            lin = apply_patches(lin, patches)
            img = render(lin, xyz, wb, _with_as_shot(iid, crs), orient, seg, size)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            return FileResponse(cache, media_type="image/jpeg")
        lin, xyz, wb, orient = _linear(iid, path)
        pr = db.one("SELECT preview_path FROM images WHERE id=?", (iid,))
        detail = None
        if pr and pr["preview_path"] and Path(pr["preview_path"]).exists():
            d = cv2.imread(str(pr["preview_path"]), cv2.IMREAD_COLOR)
            detail = cv2.cvtColor(d, cv2.COLOR_BGR2RGB) if d is not None else None
        img = render_hybrid(lin, xyz, wb, _with_as_shot(iid, crs), detail, orient, seg, size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if "_cmp_" not in cache.name:
            for old in cache.parent.glob(f"{image_key(iid, str(path))}_{size}_*.jpg"):
                if old != cache:
                    old.unlink(missing_ok=True)          # ältere Bearbeitungsstände dieses Bildes
        return FileResponse(cache, media_type="image/jpeg")

    _render_gate = asyncio.Semaphore(2)
    _busy = {"n": 0, "last": 0.0}
    _pre: dict[str, Any] = {"queue": [], "thread": None, "size": 2880}
    _pre_lock = threading.Lock()

    def _prerender(ids: list[int]) -> None:
        """Bearbeitete Lupen-Vorschauen im Hintergrund vorbereiten (wie "Vorschauen erstellen" in Lightroom):
        nur wenn gerade nichts anderes gerechnet wird, das aktuelle Blättern hat immer Vorrang."""
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("IMAGOMAT_NO_PRERENDER"):
            return
        with _pre_lock:
            _pre["queue"] = list(ids)
            t = _pre["thread"]
            if t is None or not t.is_alive():
                t = threading.Thread(target=_prerender_loop, name="prerender", daemon=True)
                _pre["thread"] = t
                t.start()

    def _prerender_loop() -> None:
        while True:
            with _pre_lock:
                if not _pre["queue"]:
                    _pre["thread"] = None
                    return
                iid = _pre["queue"].pop(0)
            # warten, bis keine Vorschau angefragt wird und kein Auftrag (Analyse, Entwickeln) läuft
            while _busy["n"] > 0 or time.time() - _busy["last"] < 1.5 or jobs.active():
                time.sleep(0.3)
            try:
                hit = _render_cache(iid, _pre["size"])
                if hit and not hit[1].exists():
                    render_after(iid, _pre["size"])
            except Exception:  # noqa: BLE001
                log.debug("Vorberechnen %s fehlgeschlagen", iid, exc_info=True)

    async def _gated(request: Request, fn: Any) -> Response:
        """Höchstens zwei Vorschauen gleichzeitig rechnen; beim schnellen Blättern verworfene Anfragen (Browser hat
        die Verbindung schon geschlossen) gar nicht erst rechnen: kein Rückstau, das aktuelle Bild kommt sofort."""
        _busy["n"] += 1
        try:
            async with _render_gate:
                if await request.is_disconnected():
                    return Response(status_code=204)
                return await run_in_threadpool(fn)
        finally:
            _busy["n"] -= 1
            _busy["last"] = time.time()

    @app.get("/api/images/{iid}/render")
    async def render_route(iid: int, request: Request, size: int = 1600) -> Response:
        if size in LOUPE_SIZES:
            _pre["size"] = size                      # Grösse der Lupe auf diesem Bildschirm: so vorberechnen
        hit = _render_cache(iid, size)
        if hit and hit[1].exists():                  # schon berechnet: sofort, ohne anzustehen
            return FileResponse(hit[1], media_type="image/jpeg")
        return await _gated(request, lambda: render_after(iid, size))

    def _neutral_profile(profile: str | None, crs: dict[str, Any]) -> dict[str, Any]:
        """Ältere neutrale Bearbeitungen ohne Profil: wie das Kamera-Original zeigen."""
        if profile == "neutral" and not crs.get("CameraProfile"):
            return {**crs, "CameraProfile": "Camera Standard"}
        return crs

    def _render_path(iid: int, path: str, updated_at: float | None, size: int) -> Path:
        return cache_dir() / "renders" / f"{image_key(iid, path)}_{size}_{int((updated_at or 0) * 1000)}_{RENDER_VERSION}.jpg"

    LOUPE_SIZES = (2048, 2880, 3600)

    def _rendered_any(iid: int, path: str, updated_at: float | None) -> Path | None:
        for sz in (_pre["size"], *LOUPE_SIZES):
            c = _render_path(iid, path, updated_at, sz)
            if c.exists():
                return c
        return None

    def _render_cache(iid: int, size: int) -> tuple[Any, Path] | None:
        r = db.one("SELECT i.path, e.params, e.masks, e.updated_at, e.profile FROM images i LEFT JOIN edits e "
                   "ON e.image_id=i.id WHERE i.id=?", (iid,))
        if not r:
            return None
        return r, _render_path(iid, r["path"], r["updated_at"], size)

    def render_after(iid: int, size: int = 1600) -> Response:
        hit = _render_cache(iid, size)
        if not hit:
            raise HTTPException(404)
        r, cache = hit
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg")
        crs = _neutral_profile(r["profile"], json.loads(r["params"]) if r["params"] else {})
        if r["masks"]:
            crs["MaskGroupBasedCorrections"] = json.loads(r["masks"])
        return _render_file(iid, Path(r["path"]), crs, size, cache)

    # ------------------------------------------------------------------ Eigener Edit (Editor in der Lupe)
    def _edit_row(iid: int) -> tuple[Any, dict[str, Any], int]:
        r = db.one("SELECT i.*, e.params, e.masks, e.profile, e.denoise AS edit_denoise FROM images i "
                   "LEFT JOIN edits e ON e.image_id=i.id WHERE i.id=?", (iid,))
        if not r:
            raise HTTPException(404)
        crs = _neutral_profile(r["profile"], json.loads(r["params"]) if r["params"] else {})
        if r["masks"]:
            crs["MaskGroupBasedCorrections"] = json.loads(r["masks"])
        return r, crs, int(r["orientation"] or 1)

    @app.get("/api/images/{iid}/editor")
    def editor_get(iid: int) -> dict[str, Any]:
        from ..style.editor import LOCAL_KEYS, crs_to_model

        r, crs, orient = _edit_row(iid)
        a = db.get_analysis(iid) or {}
        model = crs_to_model(crs, orient)
        model["denoise"] = r["edit_denoise"]
        model["retouch"] = _retouch_ops(iid)
        return {"model": model, "manual": r["profile"] == "manual", "orientation": orient,
                "has_subject": load_masks(iid) is not None, "local_keys": LOCAL_KEYS,
                "as_shot": [a.get("as_shot_temp"), a.get("as_shot_tint")], "profile": r["profile"],
                "is_raw": raw_io.is_raw(Path(r["path"])),
                "exif": {"iso": r["iso"], "exposure_time": r["exposure_time"], "aperture": r["aperture"],
                         "focal_length": r["focal_length"], "camera": r["camera"], "lens": r["lens"]}}

    def _wb_table(xyz: np.ndarray, wb: np.ndarray, iid: int) -> dict[str, Any]:
        """Weissabgleich-Faktoren für ein Raster aus Temperatur (Mired) und Tönung: der Editor rechnet in der
        Grafikkarte genau mit den Faktoren, die auch die Vorschau hier verwendet."""
        from ..render.pipeline import _wb_mult

        mireds = [float(m) for m in np.arange(20, 501, 5)]
        tints = [float(t) for t in np.arange(-150, 151, 10)]
        key = (xyz.tobytes(), np.asarray(wb, np.float32).tobytes(), _sources.get(iid))
        hit = _wb_cache.get(key)
        if hit is None:
            mult = []
            for m in mireds:
                for t in tints:
                    c = _with_as_shot(iid, {"WhiteBalance": "Custom", "Temperature": 1e6 / m, "Tint": t})
                    mult.extend(float(v) for v in _wb_mult(xyz, wb, c))
            base = _wb_mult(xyz, wb, {})
            hit = {"mireds": mireds, "tints": tints, "mult": mult, "base": [float(v) for v in base]}
            if len(_wb_cache) > 16:
                _wb_cache.clear()
            _wb_cache[key] = hit
        return hit

    _wb_cache: dict[Any, dict[str, Any]] = {}

    @app.get("/api/images/{iid}/editor/source")
    def editor_source(iid: int, size: int = 1600) -> Response:
        """Lineare RAW-Daten für den Editor in der Grafikkarte: Kopf (JSON) + RGB als Halbfloat.
        Format: 4 Byte Länge des Kopfes (little endian), Kopf, Daten (auf 4 Byte ausgerichtet)."""
        import struct

        from ..io.color import multipliers_to_temp_tint
        from ..render.pipeline import BASE_GAIN, _black_floor, cam_to_srgb

        r = db.one("SELECT path FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        path = Path(r["path"])
        if not raw_io.is_raw(path):
            raise HTTPException(400, "Nur RAW-Dateien lassen sich bearbeiten")
        size = int(min(max(size, 400), 4096))
        lin, xyz, wb, orient = _linear_hi(iid, path) if size > 2048 else _linear(iid, path)
        h, w = lin.shape[:2]
        s = size / max(h, w)
        if s < 1:
            lin = cv2.resize(lin, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
        lin = lin.astype(np.float32)
        floored = _black_floor(lin)
        floor = (lin - floored).reshape(-1, 3)[0]
        src = _sources.get(iid) or ("libraw", 0.0, 0.0)
        a = db.get_analysis(iid) or {}
        as_temp, as_tint = a.get("as_shot_temp"), a.get("as_shot_tint")
        if not as_temp:
            as_temp, as_tint = (src[1], src[2]) if src[1] else multipliers_to_temp_tint(xyz, np.asarray(wb[:3]))
        head = {"w": int(floored.shape[1]), "h": int(floored.shape[0]), "floor": [float(v) for v in floor],
                "m": [float(v) for v in cam_to_srgb(xyz).astype(np.float32).ravel()], "gain": float(BASE_GAIN),
                "wb": _wb_table(xyz, wb, iid), "as_shot": [float(as_temp or 5500), float(as_tint or 0)],
                "source": src[0], "orientation": int(orient), "cam_curve": _cam_curve(iid)}
        hb = json.dumps(head).encode()
        hb += b" " * ((-len(hb)) % 4)
        data = np.ascontiguousarray(floored.astype("<f2")).tobytes()
        return Response(struct.pack("<I", len(hb)) + hb + data, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/images/{iid}/denoise/preview")
    def denoise_preview(iid: int, body: dict[str, Any]) -> Response:
        """1:1-Ausschnitt vorher | nachher fürs KI-Entrauschen (wie das Detail-Fenster in Lightroom)."""
        from ..denoise.local import denoise_linear
        from ..render.pipeline import _wb_mult, base_curve, BASE_GAIN, cam_to_srgb, _black_floor

        r = db.one("SELECT path FROM images WHERE id=?", (iid,))
        if not r or not raw_io.is_raw(Path(r["path"])):
            raise HTTPException(400, "Nur RAW-Dateien")
        path = Path(r["path"])
        lin, info = _full_lin(iid, path)
        with _full_lock:
            if "noise" not in _full:
                _full["noise"] = raw_io.estimate_noise(path)
            noise = _full["noise"]
        from ..retouch import apply_patches

        lin = apply_patches(lin, _patches(iid, path))
        h, w = lin.shape[:2]
        side, pad = int(min(max(int(body.get("size") or 360), 128), 720)), 32
        cx, cy = float(body.get("x", 0.5)) * w, float(body.get("y", 0.5)) * h
        x0 = int(min(max(cx - side / 2, 0), max(w - side, 0)))
        y0 = int(min(max(cy - side / 2, 0), max(h - side, 0)))
        xa, ya, xb, yb = max(x0 - pad, 0), max(y0 - pad, 0), min(x0 + side + pad, w), min(y0 + side + pad, h)
        crop = lin[ya:yb, xa:xb].astype(np.float32)
        amount = int(min(max(int(body.get("amount") or 50), 1), 100))
        den, used = denoise_linear(crop, info.camera_wb, noise["sigma_mid"], amount,
                                   load_settings().denoise.local_model)
        _r, crs, _o = _edit_row(iid)

        def show(x: np.ndarray) -> np.ndarray:
            x = _black_floor(x) * _wb_mult(info.xyz_to_cam, info.camera_wb, _with_as_shot(iid, crs))[None, None, :]
            x = np.clip(x @ cam_to_srgb(info.xyz_to_cam).T.astype(np.float32), 0, None)
            from ..render.pipeline import _n

            x = x * (BASE_GAIN * 2.0 ** _n(crs, "Exposure2012"))
            d = base_curve(x, _n(crs, "Contrast2012"))
            return (np.clip(d, 0, 1) * 255 + 0.5).astype(np.uint8)[y0 - ya:y0 - ya + side, x0 - xa:x0 - xa + side]

        a, b = show(crop), show(den)
        img = np.concatenate([a, np.full((a.shape[0], 4, 3), 30, np.uint8), b], 1)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Denoise-Method": used,
                                                                         "Cache-Control": "no-store"})

    @app.post("/api/images/{iid}/editor/retouch")
    def editor_retouch(iid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Entfernen (KI) / Reparieren: einmal in voller Auflösung rechnen und zwischenspeichern."""
        import uuid

        from ..retouch import compute_patch, save_patch

        r = db.one("SELECT path FROM images WHERE id=?", (iid,))
        if not r or not raw_io.is_raw(Path(r["path"])):
            raise HTTPException(400, "Nur RAW-Dateien")
        path = Path(r["path"])
        op = {"id": uuid.uuid4().hex[:10], "mode": "remove" if body.get("mode") == "remove" else "heal",
              "dabs": body.get("dabs") or [], "feather": float(body.get("feather", 40)), "variant": int(body.get("variant", 0))}
        if not op["dabs"]:
            raise HTTPException(400, "Nichts markiert")
        # Arbeitsauflösung (halbes RAW, ~3500 px, schon im Speicher): schnell; beim Export hochgerechnet
        lin, _xyz, wb, _o = _linear_hi(iid, path)
        # frühere Retuschen zuerst, damit sich Reparaturen aufeinander beziehen können
        from ..retouch import apply_patches

        base = apply_patches(lin, _patches(iid, path, body.get("before") or []))
        p = compute_patch(base, np.asarray(wb, np.float32), op)
        save_patch(image_key(iid, path), op["id"], p)
        return {"op": op, "method": p.get("method", "heal")}

    @app.post("/api/images/{iid}/editor/retouch/warmup")
    def editor_retouch_warmup(iid: int) -> dict[str, Any]:
        """Werkzeug gewählt: Bilddaten und KI-Modell im Hintergrund vorbereiten."""
        r = db.one("SELECT path FROM images WHERE id=?", (iid,))

        def work() -> None:
            from ..retouch import warmup

            try:
                if r and raw_io.is_raw(Path(r["path"])):
                    _linear_hi(iid, Path(r["path"]))
                warmup()
            except Exception:  # noqa: BLE001
                log.debug("Vorbereiten fehlgeschlagen", exc_info=True)
        threading.Thread(target=work, daemon=True).start()
        return {"ok": True}

    @app.get("/api/images/{iid}/editor/retouch/{op_id}")
    def editor_retouch_patch(iid: int, op_id: str, w: int, h: int, floor: str = "0,0,0") -> Response:
        """Retusche-Ausschnitt in der Grösse der Editor-Quelle: Kopf + RGB (Halbfloat, ohne Sockel) + Deckkraft."""
        import struct

        from ..retouch import load_patch

        r = db.one("SELECT path FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        p = load_patch(image_key(iid, Path(r["path"])), op_id)
        if p is None:
            ops = [o for o in _retouch_ops(iid) if str(o.get("id")) == op_id]
            if not ops:
                raise HTTPException(404, "Retusche nicht gefunden")
            p = _patches(iid, Path(r["path"]), ops)[0]
        pl = p["lin"]
        sx, sy = w / float(p["W"]), h / float(p["H"])
        x0, y0 = int(np.floor(p["x"] * sx)), int(np.floor(p["y"] * sy))
        x1, y1 = min(w, int(np.ceil((p["x"] + pl.shape[1]) * sx))), min(h, int(np.ceil((p["y"] + pl.shape[0]) * sy)))
        tw, th = max(1, x1 - x0), max(1, y1 - y0)
        interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
        fl = np.array([float(v) for v in floor.split(",")[:3]], np.float32)
        rgb = cv2.resize(pl.astype(np.float32), (tw, th), interpolation=interp) - fl[None, None]
        a = cv2.resize(p["alpha"], (tw, th), interpolation=interp)
        head = json.dumps({"x": x0, "y": y0, "w": tw, "h": th}).encode()
        head += b" " * ((-len(head)) % 4)
        return Response(struct.pack("<I", len(head)) + head + np.ascontiguousarray(rgb.astype("<f2")).tobytes()
                        + np.ascontiguousarray(a.astype(np.uint8)).tobytes(), media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/images/{iid}/editor/object")
    def editor_object(iid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Objekt auswählen: Rahmen um ein Objekt -> Maske an seinen Kanten (GrabCut), als Pinsel-Tupfer, die
        Lightroom genauso übernimmt."""
        from ..style.masks import mask_to_dabs

        r = db.one("SELECT * FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        img = load_cached_preview(r)
        h0, w0 = img.shape[:2]
        s = 1400 / max(h0, w0)
        small = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        h, w = small.shape[:2]
        bx = [float(v) for v in (body.get("box") or [0.3, 0.3, 0.7, 0.7])]
        x0, y0 = int(max(0, min(bx[0], bx[2])) * w), int(max(0, min(bx[1], bx[3])) * h)
        x1, y1 = int(min(1, max(bx[0], bx[2])) * w), int(min(1, max(bx[1], bx[3])) * h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            raise HTTPException(400, "Rahmen zu klein")
        m = None
        try:
            # KI (BiRefNet, wie die Motiv-Maske) nur im gezogenen Rahmen: genaue Kanten
            from ..vision.segmentation import _birefnet

            net = _birefnet()
            if net is not None:
                H0, W0 = img.shape[:2]
                mx, my = (bx[2] - bx[0]) * 0.08, (bx[3] - bx[1]) * 0.08
                cx0, cy0 = int(max(0, min(bx[0], bx[2]) - mx) * W0), int(max(0, min(bx[1], bx[3]) - my) * H0)
                cx1, cy1 = int(min(1, max(bx[0], bx[2]) + mx) * W0), int(min(1, max(bx[1], bx[3]) + my) * H0)
                crop = img[cy0:cy1, cx0:cx1]
                x = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                t = net.torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(net.device)
                if net.device in ("mps", "cuda"):
                    t = t.half()
                with net.torch.no_grad():
                    pred = net.model(t)[-1].sigmoid().float().cpu().numpy()[0, 0]
                full = np.zeros((H0, W0), np.float32)
                full[cy0:cy1, cx0:cx1] = cv2.resize(pred, (cx1 - cx0, cy1 - cy0))
                m = (cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
                if m.sum() < 20:
                    m = None
        except Exception:  # noqa: BLE001
            log.debug("Objekt-KI nicht verfügbar", exc_info=True)
            m = None
        if m is None:
            mask = np.zeros((h, w), np.uint8)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 8,
                            cv2.GC_INIT_WITH_RECT)
                m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
            except cv2.error:
                m = np.zeros((h, w), np.uint8)
                m[y0:y1, x0:x1] = 1
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dabs = mask_to_dabs(m, max_dabs=2500, coverage=0.97)
        return {"dabs": [[round(x, 5), round(y, 5), round(rr, 5)] for x, y, rr in dabs]}

    @app.post("/api/images/{iid}/editor/mask")
    def editor_mask(iid: int, body: dict[str, Any]) -> Response:
        """Maske als Graustufenbild (KI-Masken, Masken aus Lightroom) für den Editor in der Grafikkarte."""
        from ..render.pipeline import correction_mask
        from ..style.editor import model_to_crs

        r, base, orient = _edit_row(iid)
        m = load_masks(iid)
        seg = {"subject": m[0], "sky": m[1]} if m else {}
        size = int(min(max(int(body.get("size") or 768), 64), 1600))
        ref = seg.get("subject")
        aspect = float(body.get("aspect") or ((ref.shape[1] / ref.shape[0]) if ref is not None else 1.5))
        shape = (int(round(size / aspect)), size) if aspect >= 1 else (size, int(round(size * aspect)))
        crs = model_to_crs({"global": {}, "wb_custom": True, "masks": [body.get("mask") or {}]}, {}, orient)
        corrs = crs.get("MaskGroupBasedCorrections") or []
        out = np.zeros(shape, np.float32)
        if corrs:
            out = correction_mask({**corrs[0], "CorrectionAmount": 1}, shape, orient, seg)
        ok, png = cv2.imencode(".png", (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        return Response(png.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/images/{iid}/editor/preview")
    def editor_preview(iid: int, body: dict[str, Any]) -> Response:
        """Live-Vorschau für den Editor (ohne Zuschnitt, damit Verläufe auf dem ganzen Bild liegen)."""
        from ..render.pipeline import correction_mask
        from ..style.editor import model_to_crs

        r, base, orient = _edit_row(iid)
        if not raw_io.is_raw(Path(r["path"])):
            raise HTTPException(400, "Nur RAW-Dateien lassen sich bearbeiten")
        crs = model_to_crs(body.get("model") or {}, base, orient)
        flat = {**crs, "HasCrop": "False", "CropAngle": 0}
        lin, xyz, wb, lorient = _linear(iid, Path(r["path"]))
        m = load_masks(iid)
        seg = {"subject": m[0], "sky": m[1]} if m else {}
        detail = None
        if r["preview_path"] and Path(r["preview_path"]).exists():
            d = cv2.imread(str(r["preview_path"]), cv2.IMREAD_COLOR)
            detail = cv2.cvtColor(d, cv2.COLOR_BGR2RGB) if d is not None else None
        size = int(min(max(int(body.get("size") or 1400), 400), 2400))
        img = render_hybrid(lin, xyz, wb, _with_as_shot(iid, flat), detail, lorient, seg, size)
        ov = body.get("overlay")
        corrs = flat.get("MaskGroupBasedCorrections") or []
        if isinstance(ov, int) and 0 <= ov < len(corrs):
            mk = correction_mask({**corrs[ov], "CorrectionAmount": 1}, img.shape[:2], lorient, seg)[..., None]
            red = np.array([235, 40, 60], np.float32)
            img = (img.astype(np.float32) * (1 - 0.55 * mk) + red * 0.55 * mk).clip(0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg")

    def _save_edit(iid: int, model: dict[str, Any]) -> dict[str, Any]:
        from ..db import dumps
        from ..style.editor import model_to_crs

        _r, base, orient = _edit_row(iid)
        crs = model_to_crs(model, base, orient)
        masks = crs.get("MaskGroupBasedCorrections")
        old = db.one("SELECT denoise FROM edits WHERE image_id=?", (iid,))
        dn = model.get("denoise") if "denoise" in model else (old["denoise"] if old else None)
        dn = int(dn) if dn else None
        with db.tx() as c:
            c.execute("INSERT OR REPLACE INTO edits(image_id, profile, params, masks, confidence, denoise, user_params,"
                      " updated_at) VALUES(?, 'manual', ?, ?, 1.0, ?, (SELECT user_params FROM edits WHERE image_id=?), ?)",
                      (iid, dumps({k: v for k, v in crs.items() if k != "MaskGroupBasedCorrections"}),
                       dumps(masks) if masks else None, dn, iid, time.time()))
        if isinstance(model.get("retouch"), list):
            ops = [{"id": str(o.get("id")), "mode": "remove" if o.get("mode") == "remove" else "heal",
                    "dabs": o.get("dabs") or [], "feather": float(o.get("feather", 40)),
                    "variant": int(o.get("variant", 0))} for o in model["retouch"] if isinstance(o, dict) and o.get("id")]
            with db.tx() as c:
                if ops:
                    c.execute("INSERT OR REPLACE INTO retouch(image_id, ops) VALUES(?, ?)", (iid, json.dumps(ops)))
                else:
                    c.execute("DELETE FROM retouch WHERE image_id=?", (iid,))
        db.update_analysis(iid, {"develop_notes": ["von Hand bearbeitet"]})
        return crs

    @app.put("/api/images/{iid}/editor")
    def editor_save(iid: int, body: dict[str, Any]) -> dict[str, Any]:
        _save_edit(iid, body.get("model") or {})
        return {"ok": True}

    @app.delete("/api/images/{iid}/editor")
    def editor_reset(iid: int) -> dict[str, Any]:
        """Eigenen Edit verwerfen: wieder automatisch mit dem Stil des Shoots."""
        from ..style import compare as cmp

        r = db.one("SELECT i.shoot_id, s.profile FROM images i JOIN shoots s ON s.id=i.shoot_id WHERE i.id=?", (iid,))
        if not r:
            raise HTTPException(404)
        style = "neutral"                     # wie in Lightroom: Zurücksetzen = alle Regler auf 0
        try:
            cmp.apply_to_image(db, r["shoot_id"], iid, style)
        except KeyError as e:
            raise HTTPException(404, "Stil nicht gefunden") from e
        return {"ok": True}

    @app.post("/api/images/{iid}/editor/sync")
    def editor_sync(iid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Diesen Edit auf alle Bilder des Shoots übertragen: als Vorlage (1:1, pro Bild Belichtung, Weissabgleich,
        Begradigen angepasst). Von Hand bearbeitete Bilder bleiben, wie sie sind."""
        from ..style.editor import template_from_edit
        from ..style.template import subject_level

        crs = _save_edit(iid, body.get("model") or {})
        r, _crs, orient = _edit_row(iid)
        a = dict(db.get_analysis(iid) or {})
        if a.get("subj_level") is None and r["preview_path"] and Path(r["preview_path"]).exists():
            m = load_masks(iid)
            img = cv2.imread(str(r["preview_path"]), cv2.IMREAD_REDUCED_COLOR_4)
            if m is not None and img is not None:
                a["subj_level"] = subject_level(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), m[0])
        a["filename"] = r["filename"]
        name = str(body.get("name") or "").strip() or f"Edit {r['filename']}"
        exif = {"iso": r["iso"], "exposure_time": r["exposure_time"], "aperture": r["aperture"]}
        template_from_edit(name, crs, a, exif, orient)
        key = f"tpl:{name}"
        with db.tx() as c:
            c.execute("UPDATE shoots SET profile=? WHERE id=?", (key, r["shoot_id"]))
        job_id = jobs.submit("develop", r["shoot_id"], profile=key, only_keep=False,
                             overwrite_manual=bool(body.get("overwrite_manual")), keep=[iid])
        return {"job_id": job_id, "template": name}

    @app.get("/api/images/{iid}/thumb")
    def thumb(iid: int) -> Response:
        """Kleine Kachel fürs Raster: die Bearbeitung, sobald sie vorberechnet ist (wie in Lightroom), sonst die
        Kamera-Vorschau. Lädt viel schneller als die grosse Vorschau."""
        r = db.one("SELECT i.path, i.preview_path, e.updated_at, e.profile, c.decision FROM images i LEFT JOIN edits e "
                   "ON e.image_id=i.id LEFT JOIN culling c ON c.image_id=i.id WHERE i.id=?", (iid,))
        if not r:
            raise HTTPException(404)
        is_edited = r["profile"] not in (None, "neutral")
        edited = _rendered_any(iid, r["path"], r["updated_at"]) if r["decision"] == "keep" and is_edited else None
        src = edited if edited is not None else (Path(r["preview_path"]) if r["preview_path"] else None)
        if src is None or not src.exists():
            return preview(iid)
        tag = "e" + str(int((r["updated_at"] or 0) * 1000)) if src == edited else "c"
        t = cache_dir() / "thumbs" / f"{image_key(iid, r['path'])}_{tag}.jpg"
        if not t.exists() or t.stat().st_mtime < src.stat().st_mtime:
            img = cv2.imread(str(src), cv2.IMREAD_REDUCED_COLOR_2)
            if img is None:
                return FileResponse(src, media_type="image/jpeg")
            h, w = img.shape[:2]
            s = 480 / max(h, w)
            if s < 1:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            t.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(t), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            for old in t.parent.glob(f"{image_key(iid, r['path'])}_e*.jpg"):
                if old != t:
                    old.unlink(missing_ok=True)          # alte Bearbeitungsstände
        return FileResponse(t, media_type="image/jpeg")

    # ------------------------------------------------------------------ Waveform (Lumetri) und Änderungen
    def _jpeg_for(iid: int, src: str) -> Path:
        if src == "before":
            resp = preview(iid)
        elif src.startswith("style:"):
            resp = styled(iid, src[6:], 900)
        else:
            resp = render_after(iid, 900)
        p = getattr(resp, "path", None)
        if not p:
            raise HTTPException(404, "Bild nicht verfügbar")
        return Path(p)

    @app.get("/api/images/{iid}/waveform")
    def waveform_png(iid: int, src: str = "after", v: int = 0) -> Response:
        """Helligkeits-Waveform wie in Premiere (Lumetri): vorher (Kamera) oder nachher (Bearbeitung)."""
        from ..style.scopes import scopes as _scopes, waveform

        img = cv2.imread(str(_jpeg_for(iid, src)), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(404, "Bild nicht lesbar")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ok, png = cv2.imencode(".png", cv2.cvtColor(waveform(rgb), cv2.COLOR_RGB2BGR))
        sc = _scopes(rgb)
        return Response(png.tobytes(), media_type="image/png",
                        headers={"X-White-Clip": f"{sc['white_clip']:.4f}", "X-Black-Clip": f"{sc['black_clip']:.4f}",
                                 "Cache-Control": "no-store"})

    @app.get("/api/images/{iid}/changes")
    def changes(iid: int, style: str | None = None) -> dict[str, Any]:
        """Was die Bearbeitung konkret ändert (Regler, Masken, Hinweise) – für die Anzeige neben der Waveform."""
        from ..style import compare as cmp
        from ..style.changes import describe

        r = db.one("SELECT i.shoot_id, e.params, e.masks FROM images i LEFT JOIN edits e ON e.image_id=i.id "
                   "WHERE i.id=?", (iid,))
        if not r:
            raise HTTPException(404)
        a = db.get_analysis(iid)
        if style:
            try:
                crs = cmp.develop_preview(db, r["shoot_id"], iid, style)
            except KeyError as e:
                raise HTTPException(404, "Bild ist noch nicht analysiert") from e
            notes: list[str] = []
        else:
            crs = json.loads(r["params"]) if r["params"] else {}
            if r["masks"]:
                crs["MaskGroupBasedCorrections"] = json.loads(r["masks"])
            notes = list(a.get("develop_notes") or [])
        return {**describe(crs, a.get("as_shot_temp"), a.get("as_shot_tint")), "notes": notes}

    # ------------------------------------------------------------------ Stile vergleichen
    @app.get("/api/shoots/{sid}/compare")
    def compare(sid: int, n: int = 3) -> dict[str, Any]:
        from ..style import compare as cmp

        return {"images": cmp.sample_images(db, sid, n), "styles": cmp.styles()}

    @app.get("/api/images/{iid}/styled")
    async def styled_route(iid: int, style: str, request: Request, size: int = 900) -> Response:
        return await _gated(request, lambda: styled(iid, style, size))

    def styled(iid: int, style: str, size: int = 900) -> Response:
        """Bild mit einem bestimmten Stil entwickelt (nur Vorschau, nichts wird gespeichert)."""
        import hashlib

        from ..style import compare as cmp

        r = db.one("SELECT path, shoot_id FROM images WHERE id=?", (iid,))
        if not r:
            raise HTTPException(404)
        tag = hashlib.sha1(f"{style}|{cmp.model_version(style)}|{settings_version()}|{RENDER_VERSION}|"
                           f"{r['path']}".encode()).hexdigest()[:12]
        cache = cache_dir() / "renders" / f"{iid}_cmp_{tag}_{size}.jpg"
        if cache.exists():
            return FileResponse(cache, media_type="image/jpeg")
        try:
            crs = cmp.develop_preview(db, r["shoot_id"], iid, style)
        except KeyError as e:
            raise HTTPException(404, "Bild ist noch nicht analysiert") from e
        return _render_file(iid, Path(r["path"]), crs, size, cache)

    @app.post("/api/images/{iid}/style")
    def apply_image_style(iid: int, body: dict[str, str]) -> dict[str, Any]:
        """Stil nur für dieses eine Bild übernehmen."""
        from ..style import compare as cmp

        r = db.one("SELECT shoot_id FROM images WHERE id=?", (iid,))
        if not r or not body.get("style"):
            raise HTTPException(404)
        try:
            cmp.apply_to_image(db, r["shoot_id"], iid, body["style"])
        except KeyError as e:
            raise HTTPException(404, "Bild ist noch nicht analysiert") from e
        return {"ok": True}

    @app.get("/api/styles")
    def all_styles() -> list[dict[str, Any]]:
        from ..style import compare as cmp

        return cmp.styles()

    def settings_version() -> float:
        from ..config import settings_path

        p = settings_path()
        return p.stat().st_mtime if p.exists() else 0.0

    # ------------------------------------------------------------------ Löschen
    @app.delete("/api/shoots/{sid}")
    def delete_shoot(sid: int) -> dict[str, Any]:
        from ..manage import delete_shoot as _delete

        for j in db.query("SELECT id FROM jobs WHERE shoot_id=? AND status IN ('running','queued')", (sid,)):
            jobs.cancel(int(j["id"]))
        try:
            return _delete(db, sid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/reset")
    def reset(body: dict[str, Any]) -> dict[str, Any]:
        """Neu anfangen: alles löschen ausser den Personen (und ihren benannten Gesichtern)."""
        from ..manage import reset_keep_people

        if body.get("confirm") != "personen-behalten":
            raise HTTPException(400, "Bestätigung fehlt")
        for j in db.query("SELECT id FROM jobs WHERE status IN ('running','queued')"):
            jobs.cancel(int(j["id"]))
        return reset_keep_people(db)

    @app.get("/api/looks")
    def looks() -> list[dict[str, Any]]:
        from ..style.look import list_looks

        return list_looks()

    @app.delete("/api/looks/{name}")
    def delete_look(name: str) -> dict[str, Any]:
        from ..style.look import delete_look as _delete

        if not _delete(name):
            raise HTTPException(404, "Look nicht gefunden")
        s = load_settings()
        if s.default_profile == f"look:{name}":
            s.default_profile = None
            save_settings(s)
        return {"ok": True}

    @app.get("/api/templates")
    def templates() -> list[dict[str, Any]]:
        from ..style.template import list_templates

        return list_templates()

    @app.delete("/api/templates/{name}")
    def delete_template(name: str) -> dict[str, Any]:
        from ..style.template import delete_template as _delete

        if not _delete(name):
            raise HTTPException(404, "Vorlage nicht gefunden")
        s = load_settings()
        if s.default_profile == f"tpl:{name}":
            s.default_profile = None
            save_settings(s)
        return {"ok": True}

    @app.delete("/api/profiles/{name}")
    def delete_profile(name: str) -> dict[str, Any]:
        from ..manage import delete_profile as _delete

        try:
            _delete(name)
        except KeyError as e:
            raise HTTPException(404, "Stil nicht gefunden") from e
        return {"ok": True}

    @app.patch("/api/profiles/{name}")
    def rename_profile(name: str, body: dict[str, str]) -> dict[str, Any]:
        from ..manage import rename_profile as _rename

        try:
            _rename(name, body.get("name", ""))
        except KeyError as e:
            raise HTTPException(404, "Stil nicht gefunden") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

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
            "AND IFNULL(f.assigned_by,'') <> 'ignored' ORDER BY f.det_score DESC", (sid,))
        groups: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = f"p{r['person_id']}" if r["person_id"] else (f"c{r['cluster_id']}" if r["cluster_id"] is not None
                                                              else "unknown")
            g = groups.setdefault(key, {"key": key, "person_id": r["person_id"], "name": r["name"],
                                        "cluster_id": r["cluster_id"], "faces": [], "count": 0})
            g["count"] += 1
            if len(g["faces"]) < (60 if key == "unknown" else 12):
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
        from ..people.registry import find_person

        if req.person_id:
            return req.person_id
        if req.name:
            # Bekannte Person (auch mit anderer Schreibweise/ohne Team) wiederverwenden statt Duplikat
            return find_person(db, req.name) or upsert_person(db, req.name, req.team, req.number)
        return None

    @app.post("/api/faces/{fid}/uncluster")
    def uncluster_face(fid: int) -> dict[str, Any]:
        """Gesicht aus einer Gruppe "Wer ist das?" nehmen (gehört nicht dazu)."""
        with db.tx() as c:
            c.execute("UPDATE faces SET cluster_id=NULL WHERE id=?", (fid,))
        return {"ok": True}

    @app.post("/api/shoots/{sid}/clusters/{cid}/ignore")
    def ignore_cluster(sid: int, cid: int) -> dict[str, Any]:
        """Gruppe ausblenden (z. B. Gegner, Zuschauer): wird nicht mehr gefragt."""
        with db.tx() as c:
            cur = c.execute("UPDATE faces SET assigned_by='ignored', cluster_id=NULL, person_id=NULL "
                            "WHERE cluster_id=? AND image_id IN (SELECT id FROM images WHERE shoot_id=?)", (cid, sid))
        return {"ignored": cur.rowcount}

    @app.get("/api/images/{iid}/faces")
    def image_faces(iid: int) -> dict[str, Any]:
        """Gesichter eines Bildes (mit Namen) und Personen ohne Gesicht (Nummer/manuell)."""
        faces = [{"id": r["id"], "bbox": json.loads(r["bbox"]), "person_id": r["person_id"], "name": r["name"],
                  "assigned_by": r["assigned_by"]}
                 for r in db.query("SELECT f.id, f.bbox, f.person_id, f.assigned_by, p.name FROM faces f "
                                   "LEFT JOIN persons p ON p.id=f.person_id WHERE f.image_id=? "
                                   "AND IFNULL(f.assigned_by,'') <> 'ignored'", (iid,))]
        with_face = {f["person_id"] for f in faces if f["person_id"]}
        others = [{"person_id": r["person_id"], "name": r["name"], "via": r["text"]}
                  for r in db.query("SELECT DISTINCT n.person_id, n.text, p.name FROM numbers n JOIN persons p "
                                    "ON p.id=n.person_id WHERE n.image_id=?", (iid,))
                  if r["person_id"] not in with_face]
        return {"faces": faces, "others": others}

    @app.post("/api/images/{iid}/people-check")
    def resolve_people_check(iid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Widerspruch Gesicht/Trikot entscheiden: gewählte Person wird fest (manuell) zugeordnet."""
        fid, pid = int(body["face_id"]), int(body["person_id"])
        assign_face(db, fid, pid, "manual")
        a = db.get_analysis(iid)
        db.update_analysis(iid, {"people_check": [c for c in a.get("people_check", []) if c.get("face_id") != fid]})
        return {"ok": True}

    @app.post("/api/images/{iid}/persons")
    def add_image_person(iid: int, req: AssignReq) -> dict[str, Any]:
        """Person manuell zu einem Bild hinzufügen (z. B. von hinten, ohne erkanntes Gesicht)."""
        pid = _person_from(req)
        if pid is None:
            raise HTTPException(400, "Name fehlt")
        with db.tx() as c:
            c.execute("INSERT INTO numbers(image_id, text, confidence, bbox, person_id) VALUES(?,?,?,?,?)",
                      (iid, "#manuell", 1.0, None, pid))
            c.execute("DELETE FROM person_rejects WHERE image_id=? AND person_id=?", (iid, pid))
        return {"person_id": pid}

    @app.delete("/api/images/{iid}/persons/{pid}")
    def remove_image_person(iid: int, pid: int) -> dict[str, Any]:
        """Person aus dem Bild entfernen und merken (wird nicht wieder automatisch zugeordnet)."""
        from ..people.registry import reject

        reject(db, iid, pid)
        return {"ok": True}

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
        if pid is not None:
            with db.tx() as c:          # du hast es selbst so benannt: frühere Ablehnung aufheben
                c.execute("DELETE FROM person_rejects WHERE person_id=? AND image_id=(SELECT image_id FROM faces "
                          "WHERE id=?)", (pid, fid))
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

    @app.patch("/api/persons/{pid}")
    def patch_person(pid: int, body: dict[str, Any]) -> dict[str, Any]:
        """Name, Team oder Nummer ändern (führt bei gleichem Namen automatisch zusammen)."""
        from ..people.registry import update_person

        kw = {k: body[k] for k in ("team", "number") if k in body}
        try:
            new_id = update_person(db, pid, name=body.get("name"), **kw)
        except KeyError as e:
            raise HTTPException(404, "Person nicht gefunden") from e
        p = db.one("SELECT * FROM persons WHERE id=?", (new_id,))
        return {**dict(p), "merged": new_id != pid}

    @app.get("/api/persons/{pid}/images")
    def person_images(pid: int) -> dict[str, Any]:
        """Alle Bilder einer Person (Gesicht oder Rückennummer), neueste zuerst."""
        p = db.one("SELECT * FROM persons WHERE id=?", (pid,))
        if not p:
            raise HTTPException(404, "Person nicht gefunden")
        rows = db.query(
            "SELECT i.id AS image_id, i.filename, i.capture_time, s.name AS shoot, s.id AS shoot_id, "
            "f.id AS face_id, f.bbox, f.assigned_by, 'face' AS via FROM faces f JOIN images i ON i.id=f.image_id "
            "JOIN shoots s ON s.id=i.shoot_id WHERE f.person_id=? "
            "UNION ALL SELECT i.id, i.filename, i.capture_time, s.name, s.id, NULL, n.bbox, 'number', 'number' "
            "FROM numbers n JOIN images i ON i.id=n.image_id JOIN shoots s ON s.id=i.shoot_id WHERE n.person_id=? "
            "ORDER BY capture_time DESC LIMIT 2000", (pid, pid))
        seen: dict[int, dict[str, Any]] = {}
        for r in rows:
            d = seen.setdefault(r["image_id"], {**dict(r), "bbox": json.loads(r["bbox"]) if r["bbox"] else None})
            if d["face_id"] is None and r["face_id"] is not None:
                d.update(face_id=r["face_id"], bbox=json.loads(r["bbox"]) if r["bbox"] else None)
        items = list(seen.values())
        for it in items:
            it["reference"] = it["shoot"].startswith("__")
            if it["reference"]:
                it["shoot"] = "Referenzbilder"
        return {"person": dict(p), "images": items}

    @app.post("/api/faces/{fid}/unassign")
    def unassign_face(fid: int) -> dict[str, Any]:
        """Falsch erkanntes Gesicht von der Person lösen (und merken, damit es nicht wiederkommt)."""
        from ..people.registry import reject

        f = db.one("SELECT image_id, person_id FROM faces WHERE id=?", (fid,))
        if f and f["person_id"]:
            reject(db, int(f["image_id"]), int(f["person_id"]))
        else:
            assign_face(db, fid, None)
        return {"ok": True}

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

    @app.get("/api/presets/user")
    def user_presets() -> list[dict[str, Any]]:
        from ..style.userpresets import list_presets

        return list_presets()

    @app.post("/api/presets/import")
    def import_user_presets(body: dict[str, Any]) -> dict[str, Any]:
        """Lightroom-Presets (.xmp, .lrtemplate oder ein Ordner damit) als eigene Vorgaben übernehmen."""
        from ..style.userpresets import import_presets

        paths = [Path(str(x)).expanduser() for x in body.get("paths") or []]
        if not paths:
            raise HTTPException(400, "Keine Dateien angegeben")
        done, errors = import_presets(paths)
        if not done:
            raise HTTPException(400, errors[0] if errors else "Keine Lightroom-Vorgaben gefunden")
        return {"imported": done, "errors": errors}

    @app.delete("/api/presets/user/{name}")
    def delete_user_preset(name: str) -> dict[str, Any]:
        from ..style.userpresets import delete_preset

        return {"ok": delete_preset(name)}

    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        return [{"key": p.key, "name": p.name, "description": p.description, "group": p.group}
                for p in PRESETS.values()]

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

"""Lokale SQLite-Datenbank (Bilder, Analysen, Personen, Profile, Jobs).

Die Personen-Tabelle ist global, damit die App Personen über Shoots hinweg wiedererkennt.
Embeddings werden als float32-Blobs gespeichert.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import data_dir

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS shoots (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  folder TEXT NOT NULL UNIQUE,
  profile TEXT,
  created_at REAL NOT NULL,
  settings TEXT
);

CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  shoot_id INTEGER NOT NULL REFERENCES shoots(id) ON DELETE CASCADE,
  path TEXT NOT NULL UNIQUE,
  filename TEXT NOT NULL,
  capture_time REAL,
  camera TEXT,
  lens TEXT,
  iso REAL,
  exposure_time REAL,
  aperture REAL,
  focal_length REAL,
  width INTEGER,
  height INTEGER,
  orientation INTEGER DEFAULT 1,
  preview_path TEXT,
  exif TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_shoot ON images(shoot_id, capture_time);

CREATE TABLE IF NOT EXISTS analysis (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  data TEXT NOT NULL,              -- JSON mit allen Metriken
  embedding BLOB,
  phash TEXT,
  updated_at REAL
);

CREATE TABLE IF NOT EXISTS steps (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  step TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  done_at REAL NOT NULL,
  PRIMARY KEY (image_id, step)
);

CREATE TABLE IF NOT EXISTS culling (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  score REAL,
  decision TEXT,                   -- keep | reject
  rating INTEGER,
  label TEXT,
  reasons TEXT,                    -- JSON-Liste
  series_id INTEGER,
  is_series_best INTEGER DEFAULT 0,
  manual INTEGER DEFAULT 0         -- 1 = manuell übersteuert, wird nie überschrieben
);

CREATE TABLE IF NOT EXISTS persons (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  keyword TEXT,                    -- z. B. "Personen|FC Winterthur|Vorname Nachname"
  team TEXT,
  number TEXT,
  created_at REAL NOT NULL,
  UNIQUE(name, team)
);

CREATE TABLE IF NOT EXISTS faces (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  bbox TEXT NOT NULL,              -- JSON [x0,y0,x1,y1] normiert, Anzeige-Orientierung
  landmarks TEXT,
  det_score REAL,
  embedding BLOB,
  eyes_open REAL,
  sharpness REAL,
  yaw REAL,
  cluster_id INTEGER,
  person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  assigned_by TEXT                 -- auto | manual | number
);
CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

CREATE TABLE IF NOT EXISTS numbers (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  confidence REAL,
  bbox TEXT,
  person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL
);

-- "Das ist nicht X": nie wieder automatisch zuordnen (Gesicht oder ganzes Bild)
CREATE TABLE IF NOT EXISTS person_rejects (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  face_id INTEGER,
  created_at REAL,
  PRIMARY KEY (image_id, person_id)
);

CREATE TABLE IF NOT EXISTS edits (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  profile TEXT,
  params TEXT NOT NULL,            -- JSON: crs-Parameter (vorhergesagt)
  masks TEXT,                      -- JSON: MaskGroupBasedCorrections
  confidence REAL,
  denoise INTEGER,                 -- Denoise-Stärke, NULL = kein Denoise
  user_params TEXT,                -- JSON: zurückgelesene Korrektur aus Lightroom
  updated_at REAL
);

CREATE TABLE IF NOT EXISTS retouch (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  ops TEXT NOT NULL                -- JSON: Retusche (Entfernen/Reparieren) in Anzeige-Koordinaten
);

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  shoot_id INTEGER,
  status TEXT NOT NULL,            -- queued | running | paused | done | failed | cancelled
  progress INTEGER DEFAULT 0,
  total INTEGER DEFAULT 0,
  message TEXT,
  params TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
"""


def f32_to_blob(v: np.ndarray | None) -> bytes | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_f32(b: bytes | None) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32).copy()


class Database:
    """Dünne Hülle um sqlite3 mit einer Verbindung pro Thread."""

    def __init__(self, path: Path | str | None = None):
        self.path = str(path or (data_dir() / "imagomat.db"))
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.tx() as c:
            c.executescript(SCHEMA)
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def tx(self):
        db = self

        class _Tx:
            def __enter__(self_inner):
                db._write_lock.acquire()
                return db.conn

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    if exc_type is None:
                        db.conn.commit()
                    else:
                        db.conn.rollback()
                finally:
                    db._write_lock.release()
                return False

        return _Tx()

    def query(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(args)).fetchall()

    def one(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(args)).fetchone()

    # ---- Shoots / Bilder -------------------------------------------------
    def upsert_shoot(self, name: str, folder: str, profile: str | None = None) -> int:
        with self.tx() as c:
            row = c.execute("SELECT id FROM shoots WHERE folder=?", (folder,)).fetchone()
            if row:
                if profile:
                    c.execute("UPDATE shoots SET profile=? WHERE id=?", (profile, row["id"]))
                return int(row["id"])
            cur = c.execute(
                "INSERT INTO shoots(name, folder, profile, created_at) VALUES(?,?,?,?)",
                (name, folder, profile, time.time()),
            )
            return int(cur.lastrowid)

    def upsert_image(self, shoot_id: int, rec: dict[str, Any]) -> int:
        cols = ["path", "filename", "capture_time", "camera", "lens", "iso", "exposure_time", "aperture",
                "focal_length", "width", "height", "orientation", "preview_path", "exif"]
        vals = [rec.get(k) for k in cols]
        vals[cols.index("exif")] = json.dumps(rec.get("exif") or {}, default=str)
        with self.tx() as c:
            row = c.execute("SELECT id FROM images WHERE path=?", (rec["path"],)).fetchone()
            if row:
                sets = ", ".join(f"{k}=?" for k in cols[1:])
                c.execute(f"UPDATE images SET {sets} WHERE id=?", (*vals[1:], row["id"]))
                return int(row["id"])
            cur = c.execute(
                f"INSERT INTO images(shoot_id, {', '.join(cols)}) VALUES(?{', ?' * len(cols)})",
                (shoot_id, *vals),
            )
            return int(cur.lastrowid)

    def images(self, shoot_id: int) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM images WHERE shoot_id=? ORDER BY capture_time, filename", (shoot_id,))

    # ---- Analyse ---------------------------------------------------------
    def shoot_settings(self, shoot_id: int) -> dict[str, Any]:
        row = self.one("SELECT settings FROM shoots WHERE id=?", (shoot_id,))
        try:
            return json.loads(row["settings"] or "{}") if row else {}
        except json.JSONDecodeError:
            return {}

    def set_shoot_settings(self, shoot_id: int, settings: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute("UPDATE shoots SET settings=? WHERE id=?", (json.dumps(settings, ensure_ascii=False), shoot_id))

    def update_shoot_settings(self, shoot_id: int, **values: Any) -> None:
        self.set_shoot_settings(shoot_id, {**self.shoot_settings(shoot_id), **values})

    def get_analysis(self, image_id: int) -> dict[str, Any]:
        row = self.one("SELECT data FROM analysis WHERE image_id=?", (image_id,))
        return json.loads(row["data"]) if row else {}

    def update_analysis(self, image_id: int, data: dict[str, Any], embedding: np.ndarray | None = None,
                        phash: str | None = None) -> None:
        with self.tx() as c:
            row = c.execute("SELECT data, embedding, phash FROM analysis WHERE image_id=?", (image_id,)).fetchone()
            merged = {**(json.loads(row["data"]) if row else {}), **data}
            emb = f32_to_blob(embedding) if embedding is not None else (row["embedding"] if row else None)
            ph = phash if phash is not None else (row["phash"] if row else None)
            c.execute(
                "INSERT OR REPLACE INTO analysis(image_id, data, embedding, phash, updated_at) VALUES(?,?,?,?,?)",
                (image_id, json.dumps(merged, default=_json_default), emb, ph, time.time()),
            )

    def embeddings(self, image_ids: list[int]) -> dict[int, np.ndarray]:
        if not image_ids:
            return {}
        q = ",".join("?" * len(image_ids))
        rows = self.query(f"SELECT image_id, embedding FROM analysis WHERE image_id IN ({q})", image_ids)
        return {r["image_id"]: blob_to_f32(r["embedding"]) for r in rows if r["embedding"] is not None}

    def step_done(self, image_id: int, step: str, version: int = 1) -> bool:
        row = self.one("SELECT version FROM steps WHERE image_id=? AND step=?", (image_id, step))
        return bool(row and row["version"] >= version)

    def mark_step(self, image_id: int, step: str, version: int = 1) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO steps(image_id, step, version, done_at) VALUES(?,?,?,?)",
                      (image_id, step, version, time.time()))

    # ---- Culling ---------------------------------------------------------
    def set_culling(self, image_id: int, *, score: float, decision: str, rating: int, label: str | None,
                    reasons: list[str], series_id: int | None, is_best: bool, manual: bool = False) -> None:
        with self.tx() as c:
            row = c.execute("SELECT manual FROM culling WHERE image_id=?", (image_id,)).fetchone()
            if row and row["manual"] and not manual:
                # Manuelle Entscheidungen bleiben; nur Score/Serie aktualisieren.
                c.execute("UPDATE culling SET score=?, series_id=?, is_series_best=? WHERE image_id=?",
                          (score, series_id, int(is_best), image_id))
                return
            c.execute(
                "INSERT OR REPLACE INTO culling(image_id, score, decision, rating, label, reasons, series_id,"
                " is_series_best, manual) VALUES(?,?,?,?,?,?,?,?,?)",
                (image_id, score, decision, rating, label, json.dumps(reasons), series_id, int(is_best), int(manual)),
            )

    # ---- Jobs ------------------------------------------------------------
    def create_job(self, kind: str, shoot_id: int | None, params: dict[str, Any]) -> int:
        now = time.time()
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO jobs(kind, shoot_id, status, params, created_at, updated_at) VALUES(?,?,?,?,?,?)",
                (kind, shoot_id, "queued", json.dumps(params), now, now),
            )
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def job(self, job_id: int) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return dict(row) if row else None


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def dumps(o: Any) -> str:
    return json.dumps(o, default=_json_default, ensure_ascii=False)

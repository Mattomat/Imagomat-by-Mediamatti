"""Lightroom-Classic-Katalog (.lrcat) lesen.

Der Katalog wird **nie direkt geöffnet**: Wir kopieren ihn in den Cache und lesen nur die
Kopie (read-only). So kann Lightroom parallel laufen, und das Original bleibt unberührt.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .develop import lua_to_crs
from .lua import LuaError, parse_lua


@dataclass
class CatalogImage:
    image_id: int
    path: Path
    rating: int | None
    pick: int
    color_label: str | None
    capture_time: str | None
    orientation: str | None
    develop: dict[str, Any]           # crs-Dict
    process_version: str | None
    history_steps: int
    keywords: list[str] = field(default_factory=list)
    iso: float | None = None
    aperture: float | None = None
    shutter: float | None = None
    focal_length: float | None = None
    camera: str | None = None
    lens: str | None = None
    is_virtual_copy: bool = False

    @property
    def exists(self) -> bool:
        return self.path.exists()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.DatabaseError:
        return set()


def _apex_to_fnumber(v: float | None) -> float | None:
    return None if v is None else round(2 ** (v / 2), 1)


def _apex_to_seconds(v: float | None) -> float | None:
    return None if v is None else 2 ** (-v)


class CatalogReader:
    def __init__(self, lrcat: str | Path, copy: bool = True):
        self.source = Path(lrcat)
        if not self.source.exists():
            raise FileNotFoundError(self.source)
        if copy:
            self._tmp = Path(tempfile.mkdtemp(prefix="imagomat-lrcat-"))
            self.path = self._tmp / self.source.name
            shutil.copy2(self.source, self.path)
            for ext in ("-wal", "-shm"):
                side = Path(str(self.source) + ext)
                if side.exists():
                    shutil.copy2(side, Path(str(self.path) + ext))
        else:
            self._tmp = None
            self.path = self.source
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> "CatalogReader":
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    def variables(self) -> dict[str, str]:
        if "name" not in _columns(self.conn, "Adobe_variablesTable"):
            return {}
        return {r["name"]: r["value"] for r in self.conn.execute("SELECT name, value FROM Adobe_variablesTable")}

    def color_labels(self) -> dict[str, int]:
        """Welche Farblabel-Texte kommen wie oft vor (für die Label-Zuordnung)."""
        rows = self.conn.execute(
            "SELECT colorLabels, COUNT(*) n FROM Adobe_images WHERE colorLabels IS NOT NULL AND colorLabels != ''"
            " GROUP BY colorLabels ORDER BY n DESC")
        return {r["colorLabels"]: r["n"] for r in rows}

    def _keyword_paths(self) -> dict[int, str]:
        cols = _columns(self.conn, "AgLibraryKeyword")
        if not cols:
            return {}
        rows = {r["id_local"]: (r["name"], r["parent"]) for r in
                self.conn.execute("SELECT id_local, name, parent FROM AgLibraryKeyword")}
        cache: dict[int, str] = {}

        def path(i: int, depth: int = 0) -> str:
            if i in cache:
                return cache[i]
            name, parent = rows.get(i, (None, None))
            if name is None or depth > 32:
                return ""
            pp = path(parent, depth + 1) if parent in rows else ""
            cache[i] = f"{pp}|{name}" if pp else name
            return cache[i]

        return {i: path(i) for i in rows}

    def _image_keywords(self) -> dict[int, list[str]]:
        if not _columns(self.conn, "AgLibraryKeywordImage"):
            return {}
        paths = self._keyword_paths()
        out: dict[int, list[str]] = {}
        for r in self.conn.execute("SELECT image, tag FROM AgLibraryKeywordImage"):
            p = paths.get(r["tag"])
            if p:
                out.setdefault(r["image"], []).append(p)
        return out

    def _history_counts(self) -> dict[int, int]:
        if not _columns(self.conn, "Adobe_libraryImageDevelopHistoryStep"):
            return {}
        return {r["image"]: r["n"] for r in self.conn.execute(
            "SELECT image, COUNT(*) n FROM Adobe_libraryImageDevelopHistoryStep GROUP BY image")}

    def _exif(self) -> dict[int, dict[str, Any]]:
        cols = _columns(self.conn, "AgHarvestedExifMetadata")
        if not cols:
            return {}
        cams = {}
        lenses = {}
        if _columns(self.conn, "AgInternedExifCameraModel"):
            cams = {r[0]: r[1] for r in self.conn.execute("SELECT id_local, value FROM AgInternedExifCameraModel")}
        if _columns(self.conn, "AgInternedExifLens"):
            lenses = {r[0]: r[1] for r in self.conn.execute("SELECT id_local, value FROM AgInternedExifLens")}
        want = [c for c in ("image", "isoSpeedRating", "aperture", "shutterSpeed", "focalLength",
                            "cameraModelRef", "lensRef") if c in cols]
        out = {}
        for r in self.conn.execute(f"SELECT {', '.join(want)} FROM AgHarvestedExifMetadata"):
            d = dict(r)
            out[d["image"]] = {
                "iso": d.get("isoSpeedRating"),
                "aperture": _apex_to_fnumber(d.get("aperture")),
                "shutter": _apex_to_seconds(d.get("shutterSpeed")),
                "focal_length": d.get("focalLength"),
                "camera": cams.get(d.get("cameraModelRef")),
                "lens": lenses.get(d.get("lensRef")),
            }
        return out

    def named_faces(self) -> dict[Path, list[tuple[str, tuple[float, float, float, float]]]]:
        """Alle in Lightroom benannten Gesichter: Bildpfad -> [(Name, Box x0 y0 x1 y1 normiert)].

        Lightroom speichert Gesichter in AgLibraryFace (Eckpunkte tl/br) und die Zuordnung zur
        Personen-Stichwort in AgLibraryKeywordFace. Das Schema wird zur Laufzeit geprüft.
        """
        fcols = _columns(self.conn, "AgLibraryFace")
        kcols = _columns(self.conn, "AgLibraryKeywordFace")
        if not fcols or not kcols or not {"tl_x", "tl_y", "br_x", "br_y", "image"} <= fcols:
            return {}
        tag_col = "tag" if "tag" in kcols else ("keyword" if "keyword" in kcols else None)
        if tag_col is None:
            return {}
        conds = []
        if "userReject" in kcols:
            conds.append("IFNULL(kf.userReject, 0) = 0")
        if "userPick" in kcols:
            conds.append("IFNULL(kf.userPick, 1) != 0")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        sql = (
            "SELECT f.image, f.tl_x, f.tl_y, f.br_x, f.br_y, k.name, fo.pathFromRoot, rf.absolutePath, "
            "fi.baseName, fi.extension FROM AgLibraryFace f "
            f"JOIN AgLibraryKeywordFace kf ON kf.face = f.id_local JOIN AgLibraryKeyword k ON k.id_local = kf.{tag_col} "
            "JOIN Adobe_images i ON i.id_local = f.image JOIN AgLibraryFile fi ON i.rootFile = fi.id_local "
            "JOIN AgLibraryFolder fo ON fi.folder = fo.id_local JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local"
            + where
        )
        out: dict[Path, list[tuple[str, tuple[float, float, float, float]]]] = {}
        try:
            rows = self.conn.execute(sql).fetchall()
        except sqlite3.DatabaseError:
            return {}
        for r in rows:
            if not r["name"]:
                continue
            path = Path(r["absolutePath"] or "") / (r["pathFromRoot"] or "") / f"{r['baseName']}.{r['extension']}"
            box = (min(r["tl_x"], r["br_x"]), min(r["tl_y"], r["br_y"]), max(r["tl_x"], r["br_x"]),
                   max(r["tl_y"], r["br_y"]))
            out.setdefault(path, []).append((r["name"], tuple(float(v) for v in box)))
        return out

    def images(self, include_virtual_copies: bool = False) -> Iterator[CatalogImage]:
        icols = _columns(self.conn, "Adobe_images")
        dcols = _columns(self.conn, "Adobe_imageDevelopSettings")
        sel = ["i.id_local AS image_id", "f.baseName", "f.extension", "fo.pathFromRoot", "rf.absolutePath"]
        for c in ("rating", "pick", "colorLabels", "captureTime", "orientation", "masterImage"):
            if c in icols:
                sel.append(f"i.{c}")
        if "text" in dcols:
            sel.append("ds.text AS develop_text")
        if "processVersion" in dcols:
            sel.append("ds.processVersion")
        sql = (
            f"SELECT {', '.join(sel)} FROM Adobe_images i "
            "JOIN AgLibraryFile f ON i.rootFile = f.id_local "
            "JOIN AgLibraryFolder fo ON f.folder = fo.id_local "
            "JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local "
            "LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local"
        )
        kws = self._image_keywords()
        hist = self._history_counts()
        exif = self._exif()
        for r in self.conn.execute(sql):
            d = dict(r)
            vc = bool(d.get("masterImage"))
            if vc and not include_virtual_copies:
                continue
            path = Path(d["absolutePath"] or "") / (d["pathFromRoot"] or "") / f"{d['baseName']}.{d['extension']}"
            develop: dict[str, Any] = {}
            if d.get("develop_text"):
                try:
                    parsed = parse_lua(d["develop_text"])
                    if isinstance(parsed, dict):
                        develop = lua_to_crs(parsed)
                except LuaError:
                    develop = {}
            e = exif.get(d["image_id"], {})
            yield CatalogImage(
                image_id=d["image_id"], path=path, rating=d.get("rating"), pick=int(d.get("pick") or 0),
                color_label=d.get("colorLabels") or None, capture_time=d.get("captureTime"),
                orientation=d.get("orientation"), develop=develop, process_version=d.get("processVersion"),
                history_steps=hist.get(d["image_id"], 0), keywords=kws.get(d["image_id"], []),
                is_virtual_copy=vc, **e,
            )


def lr_time(ts: float | None) -> str:
    """Zeitstempel im Lightroom-Format (ISO ohne Zeitzone)."""
    if ts is None:
        ts = _dt.datetime.now().timestamp()
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]

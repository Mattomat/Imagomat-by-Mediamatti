"""EXPERIMENTELL: Lightroom-Katalog (.lrcat) auf Basis eines leeren Vorlagen-Katalogs erzeugen.

Achtung: Das Katalogformat ist nicht dokumentiert. Dieser Writer
- arbeitet nur auf einer **Kopie** eines leeren, von *deiner* Lightroom-Version erzeugten
  Katalogs (Datei > Neuer Katalog, Lightroom danach schliessen),
- liest das Schema der Vorlage zur Laufzeit aus und füllt nur Spalten, die dort existieren,
- setzt alle Pflichtspalten (NOT NULL ohne Default) mit neutralen Werten,
- schreibt alles in einer Transaktion und prüft danach ``PRAGMA integrity_check``.

Zusätzlich schreibt der Export immer XMP-Sidecars. Falls Lightroom einen Wert aus dem
Katalog nicht übernimmt, hilft *Metadaten > Metadaten aus Dateien lesen*.
KI-Daten (Denoise, KI-Masken) stehen nicht im Katalog; Lightroom berechnet sie über
*Foto > Entwicklungseinstellungen > KI-Einstellungen aktualisieren*.
"""

from __future__ import annotations

import datetime as _dt
import math
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import lr_time
from .develop import crs_to_lua
from .lua import dump_lua

_COCOA_EPOCH = _dt.datetime(2001, 1, 1, tzinfo=_dt.timezone.utc).timestamp()

# EXIF-Orientierung -> Lightroom-Code (AB = normal)
_ORIENT = {1: "AB", 3: "CD", 6: "BC", 8: "DA", 2: "BA", 4: "DC", 5: "CB", 7: "AD"}


def cocoa_time(ts: float | None = None) -> float:
    return (ts if ts is not None else _dt.datetime.now().timestamp()) - _COCOA_EPOCH


def _gid() -> str:
    return str(uuid.uuid4()).upper()


@dataclass
class CatalogPhoto:
    path: Path
    width: int
    height: int
    orientation: int = 1
    capture_time: float | None = None
    rating: int | None = None
    pick: int = 0                      # 1 = ausgewählt, -1 = abgelehnt
    color_label: str | None = None
    keywords: list[str] = field(default_factory=list)
    develop: dict[str, Any] = field(default_factory=dict)   # crs-Dict
    xmp: str | None = None
    collections: list[str] = field(default_factory=list)    # "Imagomat|Shoot|Personen|Name"
    stack: int | None = None           # Serien-ID, Bilder mit gleicher ID werden gestapelt
    stack_position: int = 0
    iso: float | None = None
    aperture: float | None = None
    shutter: float | None = None
    focal_length: float | None = None


class CatalogWriteError(RuntimeError):
    pass


class CatalogWriter:
    def __init__(self, template: str | Path, target: str | Path):
        template, target = Path(template), Path(target)
        if not template.exists():
            raise FileNotFoundError(f"Vorlagen-Katalog fehlt: {template}")
        if target.exists():
            raise FileExistsError(f"Ziel existiert bereits: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, target)
        self.target = target
        self.conn = sqlite3.connect(str(target))
        self.conn.row_factory = sqlite3.Row
        self._schema: dict[str, dict[str, sqlite3.Row]] = {}
        self._next_id = self._max_id() + 1
        self._keyword_ids: dict[str, int] = {}
        self._collection_ids: dict[str, int] = {}
        self._folders: dict[tuple[int, str], int] = {}
        self._roots: dict[str, int] = {}
        self._stacks: dict[int, int] = {}
        self._interned: dict[tuple[str, str], int] = {}
        self.warnings: list[str] = []

    @classmethod
    def existing(cls, catalog: str | Path) -> "CatalogWriter":
        """Bestehenden Katalog direkt bearbeiten (Aufrufer sichert ihn vorher)."""
        self = cls.__new__(cls)
        self.target = Path(catalog)
        self.conn = sqlite3.connect(str(self.target))
        self.conn.row_factory = sqlite3.Row
        self._schema = {}
        self._next_id = self._max_id() + 1
        self._keyword_ids = {}
        self._collection_ids = {}
        self._folders = {}
        self._roots = {}
        self._stacks = {}
        self._interned = {}
        self.warnings = []
        return self

    # ---- Schema-Helfer ----------------------------------------------
    def tables(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def cols(self, table: str) -> dict[str, sqlite3.Row]:
        if table not in self._schema:
            self._schema[table] = {r["name"]: r for r in self.conn.execute(f"PRAGMA table_info({table})")}
        return self._schema[table]

    def _max_id(self) -> int:
        m = 0
        for t in self.tables():
            if "id_local" in {r[1] for r in self.conn.execute(f"PRAGMA table_info({t})")}:
                v = self.conn.execute(f"SELECT MAX(id_local) FROM {t}").fetchone()[0]
                if isinstance(v, (int, float)):
                    m = max(m, int(v))
        return m

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def insert(self, table: str, values: dict[str, Any]) -> int | None:
        cols = self.cols(table)
        if not cols:
            self.warnings.append(f"Tabelle {table} fehlt in der Vorlage, übersprungen")
            return None
        row = {k: v for k, v in values.items() if k in cols}
        if "id_local" in cols and "id_local" not in row:
            row["id_local"] = self.new_id()
        if "id_global" in cols and "id_global" not in row:
            row["id_global"] = _gid()
        for name, info in cols.items():
            if name in row or info["pk"] or not info["notnull"] or info["dflt_value"] is not None:
                continue
            typ = (info["type"] or "").upper()
            row[name] = 0 if any(t in typ for t in ("INT", "REAL", "NUM")) else ""
        names = list(row)
        self.conn.execute(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                          [row[n] for n in names])
        return row.get("id_local")

    # ---- Ordner ------------------------------------------------------
    def _root(self, folder: Path) -> int:
        key = str(folder)
        if key not in self._roots:
            rel = None
            try:
                rel = str(Path(folder).relative_to(self.target.parent)) + "/"
            except ValueError:
                pass
            self._roots[key] = self.insert("AgLibraryRootFolder", {
                "absolutePath": key.rstrip("/") + "/", "name": folder.name, "relativePathFromCatalog": rel,
            })
        return self._roots[key]

    def _folder(self, root_id: int, rel: str) -> int:
        key = (root_id, rel)
        if key not in self._folders:
            self._folders[key] = self.insert("AgLibraryFolder", {
                "pathFromRoot": rel, "rootFolder": root_id, "visibility": None,
            })
        return self._folders[key]

    # ---- Stichwörter -------------------------------------------------
    def _keyword_root(self) -> int | None:
        if "AgLibraryKeyword" not in self.tables():
            return None
        vars_ = {}
        if "Adobe_variablesTable" in self.tables():
            vars_ = {r["name"]: r["value"] for r in self.conn.execute("SELECT name, value FROM Adobe_variablesTable")}
        rid = vars_.get("AgLibraryKeyword_rootTagID")
        if rid:
            return int(float(rid))
        r = self.conn.execute("SELECT id_local FROM AgLibraryKeyword WHERE parent IS NULL LIMIT 1").fetchone()
        return int(r[0]) if r else None

    def keyword(self, path: str) -> int | None:
        if path in self._keyword_ids:
            return self._keyword_ids[path]
        root = self._keyword_root()
        if root is None:
            self.warnings.append("Kein Stichwort-Stamm in der Vorlage gefunden")
            return None
        parent, genealogy, parent_path = root, "", ""
        for part in [p for p in path.split("|") if p]:
            parent_path = f"{parent_path}|{part}" if parent_path else part
            if parent_path in self._keyword_ids:
                parent = self._keyword_ids[parent_path]
                genealogy = self._genealogy(parent)
                continue
            existing = self.conn.execute(
                "SELECT id_local, genealogy FROM AgLibraryKeyword WHERE parent=? AND name=?", (parent, part)).fetchone()
            if existing:
                kid, genealogy = existing[0], existing[1] or ""
            else:
                kid = self.new_id()
                genealogy = f"{genealogy}/{len(str(kid))}{kid}"
                self.insert("AgLibraryKeyword", {
                    "id_local": kid, "dateCreated": cocoa_time(), "genealogy": genealogy,
                    "includeOnExport": 1, "includeParents": 1, "includeSynonyms": 1,
                    "keywordType": None, "lc_name": part.lower(), "name": part, "parent": parent,
                })
            self._keyword_ids[parent_path] = kid
            parent = kid
        return parent

    def _genealogy(self, kid: int) -> str:
        r = self.conn.execute("SELECT genealogy FROM AgLibraryKeyword WHERE id_local=?", (kid,)).fetchone()
        return (r[0] if r else "") or ""

    # ---- Sammlungen --------------------------------------------------
    def collection(self, path: str) -> int | None:
        if "AgLibraryCollection" not in self.tables():
            return None
        parts = [p for p in path.split("|") if p]
        parent, genealogy, acc = None, "", ""
        for i, part in enumerate(parts):
            acc = f"{acc}|{part}" if acc else part
            is_leaf = i == len(parts) - 1
            if acc not in self._collection_ids:
                cid = self.new_id()
                genealogy = f"{genealogy}/{len(str(cid))}{cid}"
                self.insert("AgLibraryCollection", {
                    "id_local": cid, "creationId": "com.adobe.ag.library.collection" if is_leaf
                    else "com.adobe.ag.library.group", "genealogy": genealogy, "imageCount": None,
                    "name": part, "parent": parent, "systemOnly": 0,
                })
                self._collection_ids[acc] = cid
            else:
                genealogy = self.conn.execute("SELECT genealogy FROM AgLibraryCollection WHERE id_local=?",
                                              (self._collection_ids[acc],)).fetchone()[0] or genealogy
            parent = self._collection_ids[acc]
        return parent

    def _interned(self, table: str, value: str | None) -> int | None:
        if not value or table not in self.tables():
            return None
        key = (table, value)
        if key not in self._interned:
            r = self.conn.execute(f"SELECT id_local FROM {table} WHERE value=?", (value,)).fetchone()
            self._interned[key] = r[0] if r else self.insert(table, {"value": value, "searchIndex": value.lower()})
        return self._interned[key]

    # ---- Bilder ------------------------------------------------------
    def add_photo(self, p: CatalogPhoto, root_folder: Path) -> int:
        root_id = self._root(root_folder)
        rel = str(p.path.parent.relative_to(root_folder)).replace("\\", "/")
        rel = "" if rel == "." else rel.rstrip("/") + "/"
        folder_id = self._folder(root_id, rel)
        stat = p.path.stat() if p.path.exists() else None
        ext = p.path.suffix.lstrip(".")
        sidecar = p.path.with_suffix(".xmp").exists()
        file_id = self.insert("AgLibraryFile", {
            "baseName": p.path.stem, "extension": ext, "folder": folder_id,
            "idx_filename": p.path.name, "lc_idx_filename": p.path.name.lower(),
            "lc_idx_filenameExtension": ext.lower(), "originalFilename": p.path.name,
            "importHash": f"{p.path.name}-{stat.st_size if stat else 0}",
            "modTime": cocoa_time(stat.st_mtime) if stat else cocoa_time(),
            "externalModTime": cocoa_time(stat.st_mtime) if stat else None,
            "sidecarExtensions": "xmp" if sidecar else "",
        })
        image_id = self.new_id()
        dev_id = self.new_id() if p.develop else None
        is_raw = p.path.suffix.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".heic")
        self.insert("Adobe_images", {
            "id_local": image_id, "rootFile": file_id, "captureTime": lr_time(p.capture_time),
            "fileFormat": "RAW" if is_raw else p.path.suffix.lstrip(".").upper().replace("JPG", "JPEG"),
            "fileWidth": p.width, "fileHeight": p.height, "orientation": _ORIENT.get(p.orientation, "AB"),
            "rating": p.rating if p.rating else None, "pick": p.pick, "colorLabels": p.color_label or "",
            "touchCount": 0, "touchTime": cocoa_time(), "developSettingsIDCache": dev_id,
            "hasMissingSidecars": 0, "sidecarStatus": 0, "masterImage": None, "bitDepth": 16 if is_raw else 8,
            "colorChannels": 3, "colorMode": -1, "aspectRatioCache": (p.width / p.height) if p.height else 1.5,
            "originalCaptureTime": lr_time(p.capture_time), "copyReason": None,
        })
        if p.develop:
            text = dump_lua(crs_to_lua(p.develop))
            self.insert("Adobe_imageDevelopSettings", {
                "id_local": dev_id, "image": image_id, "text": text, "digest": None, "grayscale": 0,
                "hasDevelopAdjustments": 1, "hasDevelopAdjustmentsEx": 1,
                "processVersion": str(p.develop.get("ProcessVersion", "11.0")),
                "whiteBalance": str(p.develop.get("WhiteBalance", "As Shot")),
                "fileWidth": p.width, "fileHeight": p.height,
                "croppedWidth": p.width, "croppedHeight": p.height,
            })
        if p.xmp:
            self.insert("Adobe_AdditionalMetadata", {
                "image": image_id, "xmp": p.xmp, "isRawFile": int(is_raw), "embeddedXmp": 0,
                "externalXmpIsDirty": 0, "monochrome": 0, "additionalInfoSet": 0,
            })
        cap = _dt.datetime.fromtimestamp(p.capture_time) if p.capture_time else None
        self.insert("AgHarvestedExifMetadata", {
            "image": image_id, "isoSpeedRating": p.iso,
            "aperture": (2 * math.log2(p.aperture)) if p.aperture else None,
            "shutterSpeed": (-math.log2(p.shutter)) if p.shutter else None,
            "focalLength": p.focal_length, "dateYear": cap.year if cap else None,
            "dateMonth": cap.month if cap else None, "dateDay": cap.day if cap else None,
            "hasGPS": 0,
        })
        for kw in p.keywords:
            tag = self.keyword(kw)
            if tag is not None:
                self.insert("AgLibraryKeywordImage", {"image": image_id, "tag": tag})
        for c in p.collections:
            cid = self.collection(c)
            if cid is not None:
                self.insert("AgLibraryCollectionImage", {"collection": cid, "image": image_id, "pick": 0,
                                                         "positionInCollection": None})
        if p.stack is not None and "AgLibraryFolderStack" in self.tables():
            sid = self._stacks.get(p.stack)
            if sid is None:
                sid = self.insert("AgLibraryFolderStack", {"collapsed": 1, "text": ""})
                self._stacks[p.stack] = sid
            self.insert("AgLibraryFolderStackImage", {
                "collapsed": 1, "image": image_id, "position": p.stack_position + 1, "stack": sid,
            })
        return image_id

    def finish(self) -> dict[str, Any]:
        if "Adobe_variablesTable" in self.tables():
            row = self.conn.execute("SELECT value FROM Adobe_variablesTable WHERE name='Adobe_entityIDCounter'"
                                    ).fetchone()
            if row is not None:
                self.conn.execute("UPDATE Adobe_variablesTable SET value=? WHERE name='Adobe_entityIDCounter'",
                                  (str(self._next_id),))
        self.conn.commit()
        ok = self.conn.execute("PRAGMA integrity_check").fetchone()[0]
        self.conn.close()
        if ok != "ok":
            raise CatalogWriteError(f"Integritätsprüfung fehlgeschlagen: {ok}")
        return {"catalog": str(self.target), "warnings": self.warnings}


def write_catalog(template: Path, target: Path, root_folder: Path, photos: list[CatalogPhoto]) -> dict[str, Any]:
    w = CatalogWriter(template, target)
    try:
        for p in photos:
            w.add_photo(p, root_folder)
        return w.finish()
    except Exception:
        w.conn.close()
        target.unlink(missing_ok=True)
        raise

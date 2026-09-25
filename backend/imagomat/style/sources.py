"""Trainingsquellen für Stilprofile.

1. Lightroom-Katalog (.lrcat): Entwicklungseinstellungen + Bewertungen, nur Kopie gelesen.
2. Ordner mit Bildern + XMP: jede Person kann ihre eigenen, in Lightroom/ACR bearbeiteten
   Bilder laden (RAW + .xmp-Sidecar oder DNG/JPEG mit eingebettetem XMP).
3. Lightroom-/ACR-Presets (.xmp ohne Bild): werden als fester Look übernommen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..config import IMAGE_EXTENSIONS
from ..lightroom.catalog import CatalogReader
from ..lightroom.params import PARAMS, to_number
from ..lightroom.xmp import XmpDoc, parse_xmp, read_xmp, sidecar_path

_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)


@dataclass
class TrainingSample:
    path: Path
    crs: dict[str, Any]
    rating: int | None = None
    pick: int = 0
    label: str | None = None
    keywords: list[str] = field(default_factory=list)
    source: str = "xmp"
    weight: float = 1.0


def process_version_ok(crs: dict[str, Any]) -> bool:
    pv = to_number(str(crs.get("ProcessVersion", "11.0")).split()[0])
    return pv is None or pv >= 6.7


def is_edited(crs: dict[str, Any]) -> bool:
    """Wurde das Bild wirklich bearbeitet (nicht nur importiert)?"""
    if not crs:
        return False
    changed = 0
    for k, p in PARAMS.items():
        v = to_number(crs.get(k))
        if v is not None and abs(v - p.default) > 1e-6 and k not in ("Temperature", "Tint", "Sharpness",
                                                                       "ColorNoiseReduction"):
            changed += 1
    return changed >= 2 or bool(crs.get("MaskGroupBasedCorrections"))


def embedded_xmp(path: Path, max_bytes: int = 64 << 20) -> XmpDoc | None:
    """XMP-Paket aus DNG/JPEG/TIFF lesen (einfache Suche nach dem Paket)."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return None
    m = _XMP_PACKET.search(data)
    if not m:
        return None
    try:
        return parse_xmp(m.group(0))
    except Exception:  # noqa: BLE001 - defekte Pakete ignorieren
        return None


def from_folder(folder: Path, recursive: bool = True) -> Iterator[TrainingSample]:
    folder = Path(folder)
    files = folder.rglob("*") if recursive else folder.glob("*")
    for p in sorted(files):
        if p.suffix.lower() not in IMAGE_EXTENSIONS or p.name.startswith("._"):
            continue
        doc = None
        side = sidecar_path(p)
        if side.exists():
            try:
                doc = read_xmp(side)
            except Exception:  # noqa: BLE001
                doc = None
        if doc is None or not doc.crs:
            doc = embedded_xmp(p) if p.suffix.lower() in (".dng", ".jpg", ".jpeg", ".tif", ".tiff") else doc
        if doc is None or not doc.crs:
            continue
        yield TrainingSample(p, doc.crs, doc.rating, 0, doc.label, doc.keywords, "xmp")


def from_catalog(lrcat: Path) -> Iterator[TrainingSample]:
    with CatalogReader(lrcat) as r:
        for im in r.images():
            yield TrainingSample(im.path, im.develop, im.rating, im.pick, im.color_label, im.keywords, "catalog")


def load_preset(path: Path) -> dict[str, Any]:
    """Lightroom-/ACR-Preset (.xmp) -> crs-Dict (nur Look-Parameter)."""
    doc = read_xmp(path)
    return doc.crs


def filter_samples(samples: list[TrainingSample], min_rating: int = 0, only_picked: bool = False,
                   require_file: bool = True) -> list[TrainingSample]:
    out = []
    seen = set()
    for s in samples:
        key = str(s.path).lower()
        if key in seen:
            continue
        seen.add(key)
        if require_file and not s.path.exists():
            continue
        if not process_version_ok(s.crs) or not is_edited(s.crs):
            continue
        if s.pick < 0:
            continue
        if only_picked and s.pick <= 0:
            continue
        if min_rating and (s.rating or 0) < min_rating and s.pick <= 0:
            continue
        out.append(s)
    return out

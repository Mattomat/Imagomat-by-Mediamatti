"""EXIF lesen: bevorzugt ExifTool (alle Formate inkl. CR3, Sub-Sekunden), sonst exifread.

Sub-Sekunden sind für Sport-Serien wichtig (10 bis 30 Bilder/s): Ohne sie fallen alle
Bilder einer Sekunde auf denselben Zeitstempel.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

_TAGS = ["DateTimeOriginal", "SubSecTimeOriginal", "OffsetTimeOriginal", "Make", "Model", "LensModel",
         "LensID", "ISO", "ExposureTime", "FNumber", "FocalLength", "ImageWidth", "ImageHeight",
         "Orientation", "Flash", "SerialNumber", "ShutterCount"]


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _parse_time(dt: str | None, subsec: Any = None) -> float | None:
    if not dt:
        return None
    s = str(dt).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            t = _dt.datetime.strptime(s[:19], fmt[:17] if len(fmt) > 17 else fmt).timestamp()
            break
        except ValueError:
            continue
    else:
        return None
    if subsec not in (None, ""):
        digits = "".join(c for c in str(subsec) if c.isdigit())
        if digits:
            t += float("0." + digits)
    return t


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).split()[0]
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize(d: dict[str, Any]) -> dict[str, Any]:
    make = str(d.get("Make") or "").strip()
    model = str(d.get("Model") or "").strip()
    camera = model if model.lower().startswith(make.lower()) else f"{make} {model}".strip()
    return {
        "capture_time": _parse_time(d.get("DateTimeOriginal"), d.get("SubSecTimeOriginal")),
        "camera": camera or None,
        "lens": (d.get("LensModel") or d.get("LensID") or None),
        "iso": _num(d.get("ISO")),
        "exposure_time": _num(d.get("ExposureTime")),
        "aperture": _num(d.get("FNumber")),
        "focal_length": _num(d.get("FocalLength")),
        "orientation": int(_num(d.get("Orientation")) or 1),
        "flash": d.get("Flash"),
        "raw": {k: d.get(k) for k in _TAGS if d.get(k) is not None},
    }


def read_exif_exiftool(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(paths), 200):
        chunk = paths[i:i + 200]
        cmd = ["exiftool", "-json", "-n", "-fast2", *[f"-{t}" for t in _TAGS], *map(str, chunk)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        try:
            rows = json.loads(res.stdout or "[]")
        except json.JSONDecodeError:
            rows = []
        for row in rows:
            out[str(Path(row.get("SourceFile", "")))] = _normalize(row)
    return out


def read_exif_fallback(path: Path) -> dict[str, Any]:
    d: dict[str, Any] = {}
    try:
        import exifread

        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        m = {
            "DateTimeOriginal": "EXIF DateTimeOriginal", "SubSecTimeOriginal": "EXIF SubSecTimeOriginal",
            "Make": "Image Make", "Model": "Image Model", "LensModel": "EXIF LensModel",
            "ISO": "EXIF ISOSpeedRatings", "ExposureTime": "EXIF ExposureTime", "FNumber": "EXIF FNumber",
            "FocalLength": "EXIF FocalLength", "Orientation": "Image Orientation",
        }
        for k, t in m.items():
            v = tags.get(t) or tags.get(t.replace("EXIF ", "Image "))
            if v is not None:
                vals = getattr(v, "values", None)
                if k == "Orientation" and vals:
                    d[k] = vals[0]
                elif vals and not isinstance(vals, str) and hasattr(vals[0], "num"):
                    d[k] = vals[0].num / vals[0].den if vals[0].den else None
                elif vals and not isinstance(vals, str):
                    d[k] = vals[0]
                else:
                    d[k] = str(v)
    except Exception:  # noqa: BLE001 - exifread wirft bei unbekannten Formaten beliebige Fehler
        pass
    n = _normalize(d)
    if n["capture_time"] is None:
        n["capture_time"] = path.stat().st_mtime
    return n


def read_exif(paths: list[Path]) -> dict[str, dict[str, Any]]:
    paths = [Path(p) for p in paths]
    out: dict[str, dict[str, Any]] = {}
    if exiftool_available():
        out = read_exif_exiftool(paths)
    for p in paths:
        if str(p) not in out or out[str(p)].get("capture_time") is None:
            fb = read_exif_fallback(p)
            out[str(p)] = {**fb, **{k: v for k, v in out.get(str(p), {}).items() if v not in (None, {}, "")}}
    return out

"""EXIF direkt aus TIFF-basierten Dateien lesen (ARW, NEF, DNG, CR2, ORF, RW2, PEF, TIFF, JPEG-APP1).

Ohne externe Abhängigkeiten, damit Aufnahmezeit (inkl. Sub-Sekunden) und ISO immer verfügbar
sind – auch ohne ExifTool und unabhängig von Bibliotheksversionen.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

_TAGS = {
    0x010F: "Make", 0x0110: "Model", 0x0112: "Orientation", 0x0132: "DateTime",
    0x829A: "ExposureTime", 0x829D: "FNumber", 0x8827: "ISO", 0x9003: "DateTimeOriginal",
    0x9291: "SubSecTimeOriginal", 0x920A: "FocalLength", 0xA434: "LensModel", 0x9209: "Flash",
}
_EXIF_IFD = 0x8769
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8}


def _read_value(data: bytes, bo: str, typ: int, count: int, raw: bytes) -> Any:
    size = _SIZES.get(typ, 1) * count
    if size <= 4:
        buf = raw[:size]
    else:
        off = struct.unpack(bo + "I", raw)[0]
        buf = data[off:off + size]
    if len(buf) < size:
        return None
    if typ == 2:
        return buf.split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
    if typ == 3:
        vals = struct.unpack(bo + f"{count}H", buf)
    elif typ in (4, 13):
        vals = struct.unpack(bo + f"{count}I", buf)
    elif typ == 9:
        vals = struct.unpack(bo + f"{count}i", buf)
    elif typ in (5, 10):
        fmt = "I" if typ == 5 else "i"
        nums = struct.unpack(bo + f"{2 * count}{fmt}", buf)
        vals = tuple(nums[i] / nums[i + 1] if nums[i + 1] else 0.0 for i in range(0, len(nums), 2))
    elif typ in (1, 7):
        vals = tuple(buf)
    else:
        return None
    return vals[0] if count == 1 else vals


def _parse_ifd(data: bytes, bo: str, off: int, out: dict[str, Any], depth: int = 0) -> None:
    if depth > 3 or off <= 0 or off + 2 > len(data):
        return
    (n,) = struct.unpack(bo + "H", data[off:off + 2])
    if n > 1000:
        return
    for i in range(n):
        e = off + 2 + 12 * i
        if e + 12 > len(data):
            break
        tag, typ, count = struct.unpack(bo + "HHI", data[e:e + 8])
        raw = data[e + 8:e + 12]
        if tag == _EXIF_IFD:
            sub = struct.unpack(bo + "I", raw)[0]
            _parse_ifd(data, bo, sub, out, depth + 1)
        elif tag in _TAGS and _TAGS[tag] not in out:
            try:
                v = _read_value(data, bo, typ, count, raw)
            except struct.error:
                v = None
            if v is not None and v != "":
                out[_TAGS[tag]] = v


def read_tiff_exif(path: str | Path, max_bytes: int = 4 << 20) -> dict[str, Any]:
    """Liefert Rohwerte wie ExifTool-Namen (DateTimeOriginal, ISO, ...). Leer, wenn kein TIFF/EXIF."""
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return {}
    start = 0
    if data[:2] == b"\xff\xd8":                      # JPEG: APP1 "Exif\0\0"
        i = data.find(b"Exif\x00\x00")
        if i < 0:
            return {}
        start = i + 6
    head = data[start:start + 8]
    if head[:2] == b"II":
        bo = "<"
    elif head[:2] == b"MM":
        bo = ">"
    else:
        return {}
    tiff = data[start:]
    ifd0 = struct.unpack(bo + "I", tiff[4:8])[0]
    out: dict[str, Any] = {}
    _parse_ifd(tiff, bo, ifd0, out)
    return out

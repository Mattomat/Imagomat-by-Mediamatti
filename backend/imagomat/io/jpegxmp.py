"""XMP direkt in JPEG-Dateien lesen und schreiben (APP1-Segment, ohne ExifTool).

Lightroom liest bei JPEGs keine .xmp-Sidecars, deshalb müssen Stichwörter und Gesichter
eingebettet werden. Pixel und EXIF bleiben unverändert; nur das XMP-Segment wird ersetzt.
"""

from __future__ import annotations

import struct
from pathlib import Path

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
MAX_SEGMENT = 65533


def _segments(data: bytes) -> list[tuple[int, bytes]]:
    """(Marker, Segment inkl. Marker+Länge) bis SOS; der Rest (Bilddaten) folgt als Marker 0."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("keine JPEG-Datei")
    out: list[tuple[int, bytes]] = []
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("defekte JPEG-Struktur")
        marker = data[i + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out.append((marker, data[i:i + 2]))
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        seg = data[i:i + 2 + length]
        out.append((marker, seg))
        i += 2 + length
        if marker == 0xDA:                     # Start of Scan: Rest sind Bilddaten
            out.append((0, data[i:]))
            break
    return out


def read_jpeg_xmp(path: str | Path) -> bytes | None:
    data = Path(path).read_bytes()
    for marker, seg in _segments(data):
        if marker == 0xE1 and seg[4:4 + len(XMP_HEADER)] == XMP_HEADER:
            return seg[4 + len(XMP_HEADER):]
    return None


def embed_xmp(src: str | Path, packet: bytes, dst: str | Path | None = None) -> Path:
    """Schreibt ``packet`` als XMP in die JPEG (ersetzt ein vorhandenes XMP-Segment)."""
    src = Path(src)
    dst = Path(dst) if dst else src
    payload = XMP_HEADER + packet
    if len(payload) > MAX_SEGMENT:
        raise ValueError("XMP zu gross für ein JPEG-Segment")
    new_seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    segs = _segments(src.read_bytes())
    out = [b"\xff\xd8"]
    inserted = False
    for idx, (marker, seg) in enumerate(segs):
        if marker == 0xE1 and seg[4:4 + len(XMP_HEADER)] == XMP_HEADER:
            continue                                   # altes XMP entfernen
        if not inserted and marker not in (0xE0, 0xE1):
            out.append(new_seg)                        # nach JFIF/EXIF, vor allem anderen
            inserted = True
        out.append(seg)
    if not inserted:
        out.append(new_seg)
    tmp = dst.with_suffix(dst.suffix + ".imagomat-tmp")
    tmp.write_bytes(b"".join(out))
    tmp.replace(dst)
    return dst

"""Ersatzweg, wenn LibRaw eine RAW-Datei nicht öffnen kann (eingebettete JPEG-Vorschau)."""

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from imagomat.io import raw as raw_io


def _fake_raw(path: Path, jpeg: bytes, orientation: int = 1) -> None:
    """Minimales TIFF (wie ARW-IFD0) mit JPEG-Vorschau über JPEGInterchangeFormat, ohne Rohdaten."""
    entries = [(0x010F, 2, 5, b"SONY"), (0x0112, 3, 1, orientation), (0x0201, 4, 1, 0), (0x0202, 4, 1, len(jpeg))]
    ifd_off = 8
    ifd_size = 2 + 12 * len(entries) + 4
    data_off = ifd_off + ifd_size
    make = b"SONY\0"
    jpeg_off = data_off + len(make)
    out = bytearray(b"II*\0" + struct.pack("<I", ifd_off))
    out += struct.pack("<H", len(entries))
    for tag, typ, cnt, val in entries:
        if tag == 0x010F:
            out += struct.pack("<HHII", tag, typ, cnt, data_off)
        elif tag == 0x0201:
            out += struct.pack("<HHII", tag, typ, cnt, jpeg_off)
        elif typ == 3:
            out += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        else:
            out += struct.pack("<HHII", tag, typ, cnt, val)
    out += struct.pack("<I", 0) + make + jpeg
    path.write_bytes(bytes(out))


def test_embedded_jpeg_fallback(tmp_path: Path):
    img = np.zeros((1080, 1616, 3), np.uint8)
    img[:, :800] = (255, 0, 0)            # links rot (BGR -> blau im RGB? wir prüfen nur die Form)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    p = tmp_path / "DSC01234.ARW"
    _fake_raw(p, buf.tobytes(), orientation=6)
    prev, orientation = raw_io.load_preview(p, 2048)
    assert orientation == 6
    assert prev.shape[:2] == (1616, 1080)     # hochkant gedreht


def test_unreadable_raises_clear_error(tmp_path: Path):
    p = tmp_path / "kaputt.ARW"
    p.write_bytes(b"not a raw file at all" * 100)
    with pytest.raises(raw_io.RawReadError, match="kaputt.ARW"):
        raw_io.load_preview(p)

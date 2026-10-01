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


def test_refresh_raw_metrics_after_reader_update(tmp_path):
    """Früher nur über die Vorschau gemessen (Kamera unbekannt) -> sobald lesbar, echte RAW-Daten."""
    from imagomat.analysis import import_folder, refresh_raw_metrics
    from imagomat.db import Database
    from imagomat.jobs import JobContext, JobManager

    from .synth import write_shoot

    write_shoot(tmp_path / "s", n=2)
    db = Database(tmp_path / "r.db")
    sid = import_folder(db, tmp_path / "s")
    assert JobManager(db).run_sync(db.create_job("analyze", sid, {}))["status"] == "done"
    iid = db.images(sid)[0]["id"]
    db.update_analysis(iid, {"raw_source": "preview", "raw_error": "Unsupported file format"})
    assert refresh_raw_metrics(JobContext(db, 0), sid) == 1
    a = db.get_analysis(iid)
    assert a["raw_source"] == "libraw" and not a.get("raw_error")


def test_stale_training_features_are_recomputed(tmp_path):
    """Lern-Merkmale, die früher nur über die Vorschau entstanden, werden neu gemessen, sobald lesbar."""
    import json

    from imagomat.io import raw as raw_io
    from imagomat.style import features

    from .synth import write_shoot

    p = write_shoot(tmp_path / "s", n=1)[0]
    rec = features.image_record(p)
    assert rec["analysis"]["raw_source"] == "libraw" and not features.is_stale(rec)
    cf = features._cache_file(features._cache_key(p))
    rec["analysis"].update(raw_source="preview", raw_error="Unsupported file format")
    cf.write_text(json.dumps(rec))
    assert features.is_stale(json.loads(cf.read_text()))
    again = features.image_record(p)
    assert again["analysis"]["raw_source"] == "libraw"
    st = raw_io.raw_status(p)
    assert st["libraw_ok"] and st["sample"] == p.name

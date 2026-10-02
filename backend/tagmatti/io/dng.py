"""Minimaler DNG-Writer (CFA-Bayer oder LinearRaw), eigener TIFF-Schreiber ohne Abhängigkeiten.

Verwendung:
- Lokales Denoise: entrauschte, demosaikte Daten als *lineare DNG* ausgeben. Lightroom
  behandelt sie wie eine RAW-Datei (Weissabgleich, Profile, Masken funktionieren).
- Tests: synthetische Bayer-DNGs erzeugen, die rawpy/LibRaw wie echte RAWs lesen.
"""

from __future__ import annotations

import datetime as _dt
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np

# sRGB(D65)->XYZ; ColorMatrix in DNG ist XYZ->Kamera
SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])


def _srational(values: np.ndarray, denom: int = 10000) -> list[int]:
    out: list[int] = []
    for v in np.asarray(values, dtype=float).ravel():
        out += [int(round(v * denom)), denom]
    return out


def _rational(values: np.ndarray | list[float], denom: int = 1000000) -> list[int]:
    out: list[int] = []
    for v in np.asarray(values, dtype=float).ravel():
        f = Fraction(float(v)).limit_denominator(denom)
        out += [f.numerator, f.denominator]
    return out


# TIFF-Typen: Code -> (struct-Format pro Wert, Bytes pro Wert)
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _UNDEF, _SRATIONAL = 1, 2, 3, 4, 5, 7, 10


def _encode(typ: int, values) -> tuple[bytes, int]:
    """-> (Bytes, Anzahl) für einen IFD-Eintrag."""
    if typ == _ASCII:
        b = str(values).encode("latin-1", errors="replace") + b"\0"
        return b, len(b)
    if typ in (_BYTE, _UNDEF):
        b = bytes(values)
        return b, len(b)
    vals = list(values) if isinstance(values, (list, tuple)) else [values]
    if typ == _SHORT:
        return struct.pack(f"<{len(vals)}H", *[int(v) for v in vals]), len(vals)
    if typ == _LONG:
        return struct.pack(f"<{len(vals)}I", *[int(v) for v in vals]), len(vals)
    if typ == _RATIONAL:
        return struct.pack(f"<{len(vals)}I", *[int(v) for v in vals]), len(vals) // 2
    if typ == _SRATIONAL:
        return struct.pack(f"<{len(vals)}i", *[int(v) for v in vals]), len(vals) // 2
    raise ValueError(typ)


class _TiffBuilder:
    """Minimaler Little-Endian-TIFF-Schreiber (unkomprimiert, ein Streifen pro Bild).

    Eigenes Format statt tifffile, damit die DNG unabhängig von Bibliotheksversionen
    immer gleich aussieht (LibRaw ist bei DNG-Details empfindlich)."""

    def __init__(self) -> None:
        self.buf = bytearray(b"II*\0\0\0\0\0")

    def _align(self) -> None:
        if len(self.buf) % 2:
            self.buf += b"\0"

    def add_data(self, data: bytes) -> int:
        self._align()
        off = len(self.buf)
        self.buf += data
        return off

    def add_ifd(self, entries: dict[int, tuple[int, object]], next_ifd: int = 0) -> int:
        self._align()
        ifd_off = len(self.buf)
        tags = sorted(entries)
        size = 2 + 12 * len(tags) + 4
        extra = bytearray()
        body = bytearray(struct.pack("<H", len(tags)))
        for tag in tags:
            typ, values = entries[tag]
            data, count = _encode(typ, values)
            if len(data) <= 4:
                body += struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0")
            else:
                off = ifd_off + size + len(extra)
                body += struct.pack("<HHII", tag, typ, count, off)
                extra += data
                if len(extra) % 2:
                    extra += b"\0"
        body += struct.pack("<I", next_ifd)
        self.buf += body + extra
        return ifd_off

    def finish(self, first_ifd: int) -> bytes:
        self.buf[4:8] = struct.pack("<I", first_ifd)
        return bytes(self.buf)


def _image_entries(arr: np.ndarray, offset: int, photometric: int, subfile: int) -> dict[int, tuple[int, object]]:
    h, w = arr.shape[:2]
    spp = 1 if arr.ndim == 2 else arr.shape[2]
    bits = arr.dtype.itemsize * 8
    return {
        254: (_LONG, subfile), 256: (_LONG, w), 257: (_LONG, h), 258: (_SHORT, [bits] * spp),
        259: (_SHORT, 1), 262: (_SHORT, photometric), 273: (_LONG, offset), 277: (_SHORT, spp),
        278: (_LONG, h), 279: (_LONG, arr.nbytes), 284: (_SHORT, 1),
    }


def write_dng(path: str | Path, data: np.ndarray, *, cfa: bool, color_matrix: np.ndarray | None = None,
              as_shot_neutral: tuple[float, float, float] = (0.5, 1.0, 0.7), black: int = 0,
              white: int = 65535, make: str = "Tagmatti", model: str = "Synthetic",
              unique_model: str | None = None, iso: int | None = None, exposure_time: float | None = None,
              fnumber: float | None = None, focal_length: float | None = None,
              capture_time: _dt.datetime | None = None, orientation: int = 1,
              baseline_exposure: float = 0.0, preview: np.ndarray | None = None,
              xmp: bytes | None = None) -> Path:
    """Schreibt eine DNG 1.4.

    data: uint16, bei cfa=True (H, W) im RGGB-Muster, sonst (H, W, 3) linear in Kamerafarben.
    color_matrix: XYZ(D65)->Kamera (3x3). Standard: Kamera = linear sRGB.
    Mit Vorschau: IFD0 = Vorschau, Rohdaten in einer SubIFD (wie von Adobe geschrieben).
    """
    path = Path(path)
    raw = np.ascontiguousarray(data.astype("<u2"))
    cm = color_matrix if color_matrix is not None else np.linalg.inv(SRGB_TO_XYZ)
    dt = (capture_time or _dt.datetime.now()).strftime("%Y:%m:%d %H:%M:%S")
    meta: dict[int, tuple[int, object]] = {
        271: (_ASCII, make), 272: (_ASCII, model), 274: (_SHORT, int(orientation)), 306: (_ASCII, dt),
        36867: (_ASCII, dt),
        50706: (_BYTE, (1, 4, 0, 0)), 50707: (_BYTE, (1, 1, 0, 0)),
        50708: (_ASCII, unique_model or f"{make} {model}"),
        50721: (_SRATIONAL, _srational(cm)), 50778: (_SHORT, 21),
        50728: (_RATIONAL, _rational(as_shot_neutral)),
        50730: (_SRATIONAL, _srational([baseline_exposure], 100)),
    }
    if iso:
        meta[34855] = (_SHORT, min(int(iso), 65535))
    if exposure_time:
        meta[33434] = (_RATIONAL, _rational([exposure_time]))
    if fnumber:
        meta[33437] = (_RATIONAL, _rational([fnumber]))
    if focal_length:
        meta[37386] = (_RATIONAL, _rational([focal_length]))
    if xmp:
        meta[700] = (_BYTE, xmp)
    raw_tags: dict[int, tuple[int, object]] = {50714: (_SHORT, int(black)), 50717: (_SHORT, int(white))}
    if cfa:
        raw_tags.update({33421: (_SHORT, (2, 2)), 33422: (_BYTE, (0, 1, 1, 2)), 50710: (_BYTE, (0, 1, 2)),
                         50711: (_SHORT, 1)})
        photometric = 32803
    else:
        photometric = 34892                                        # LinearRaw
    tb = _TiffBuilder()
    raw_off = tb.add_data(raw.tobytes())
    if preview is not None:
        prev = np.ascontiguousarray(preview.astype(np.uint8))
        prev_off = tb.add_data(prev.tobytes())
        sub = tb.add_ifd({**_image_entries(raw, raw_off, photometric, 0), **raw_tags})
        ifd0 = tb.add_ifd({**_image_entries(prev, prev_off, 2, 1), **meta, 330: (_LONG, sub)})
    else:
        ifd0 = tb.add_ifd({**_image_entries(raw, raw_off, photometric, 0), **raw_tags, **meta})
    path.write_bytes(tb.finish(ifd0))
    return path


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> Bayer-Mosaik RGGB (H, W)."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=rgb.dtype)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out

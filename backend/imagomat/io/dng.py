"""Minimaler DNG-Writer (CFA-Bayer oder LinearRaw) auf Basis von tifffile.

Verwendung:
- Lokales Denoise: entrauschte, demosaikte Daten als *lineare DNG* ausgeben. Lightroom
  behandelt sie wie eine RAW-Datei (Weissabgleich, Profile, Masken funktionieren).
- Tests: synthetische Bayer-DNGs erzeugen, die rawpy/LibRaw wie echte RAWs lesen.
"""

from __future__ import annotations

import datetime as _dt
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

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


def write_dng(path: str | Path, data: np.ndarray, *, cfa: bool, color_matrix: np.ndarray | None = None,
              as_shot_neutral: tuple[float, float, float] = (0.5, 1.0, 0.7), black: int = 0,
              white: int = 65535, make: str = "Imagomat", model: str = "Synthetic",
              unique_model: str | None = None, iso: int | None = None, exposure_time: float | None = None,
              fnumber: float | None = None, focal_length: float | None = None,
              capture_time: _dt.datetime | None = None, orientation: int = 1,
              baseline_exposure: float = 0.0, preview: np.ndarray | None = None) -> Path:
    """Schreibt eine DNG.

    data: uint16, bei cfa=True (H, W) im RGGB-Muster, sonst (H, W, 3) linear in Kamerafarben.
    color_matrix: XYZ(D65)->Kamera (3x3). Standard: Kamera = linear sRGB.
    """
    path = Path(path)
    data = np.ascontiguousarray(data.astype(np.uint16))
    cm = color_matrix if color_matrix is not None else np.linalg.inv(SRGB_TO_XYZ)
    extratags: list[tuple] = [
        (50706, "B", 4, (1, 4, 0, 0), True),                      # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), True),                      # DNGBackwardVersion
        (50708, "s", 0, unique_model or f"{make} {model}", True),  # UniqueCameraModel
        (50721, "q" if False else "2i", 9, _srational(cm), True),  # ColorMatrix1 (SRATIONAL)
        (50778, "H", 1, 21, True),                                # CalibrationIlluminant1 = D65
        (50728, "2I", 3, _rational(as_shot_neutral), True),       # AsShotNeutral
        (50714, "H" if cfa else "H", 1, int(black), True),        # BlackLevel
        (50717, "H", 1, int(white), True),                        # WhiteLevel
        (50730, "2i", 1, _srational([baseline_exposure], 100), True),  # BaselineExposure
        (271, "s", 0, make, True),
        (272, "s", 0, model, True),
        (274, "H", 1, int(orientation), True),
    ]
    if iso:
        extratags.append((34855, "H", 1, int(iso), True))
    if exposure_time:
        extratags.append((33434, "2I", 1, _rational([exposure_time]), True))
    if fnumber:
        extratags.append((33437, "2I", 1, _rational([fnumber]), True))
    if focal_length:
        extratags.append((37386, "2I", 1, _rational([focal_length]), True))
    dt = (capture_time or _dt.datetime.now()).strftime("%Y:%m:%d %H:%M:%S")
    extratags.append((36867, "s", 0, dt, True))                   # DateTimeOriginal
    extratags.append((306, "s", 0, dt, True))
    if cfa:
        extratags += [
            (33421, "H", 2, (2, 2), True),                        # CFARepeatPatternDim
            (33422, "B", 4, (0, 1, 1, 2), True),                  # CFAPattern RGGB
            (50710, "B", 3, (0, 1, 2), True),                     # CFAPlaneColor
            (50711, "H", 1, 1, True),                             # CFALayout
        ]
        photometric = 32803
    else:
        photometric = 34892                                       # LinearRaw
    with tifffile.TiffWriter(path, bigtiff=False) as tw:
        if preview is not None:
            # IFD0: kleine Vorschau (NewSubFileType=1), Rohdaten im zweiten IFD
            tw.write(np.ascontiguousarray(preview.astype(np.uint8)), photometric="rgb", subfiletype=1,
                     extratags=extratags, metadata=None)
            tw.write(data, photometric=photometric, subfiletype=0, metadata=None,
                     extratags=[t for t in extratags if t[0] in (33421, 33422, 50710, 50711, 50714, 50717)])
        else:
            tw.write(data, photometric=photometric, subfiletype=0, extratags=extratags, metadata=None)
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

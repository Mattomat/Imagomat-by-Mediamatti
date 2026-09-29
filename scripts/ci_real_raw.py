"""Testet echte Kamera-RAWs (z. B. von raw.pixls.us): LibRaw direkt, Vorschau mit Ersatzweg, lineare Daten.

Aufruf: python scripts/ci_real_raw.py DATEI [DATEI ...]
Exit 1, wenn für eine Datei gar keine Vorschau entsteht.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import rawpy

from imagomat.io import raw as raw_io

logging.basicConfig(level=logging.INFO, format="  %(levelname)s %(name)s: %(message)s")
print("rawpy", rawpy.__version__, "LibRaw", rawpy.libraw_version)
bad = 0
for f in sys.argv[1:]:
    p = Path(f)
    print(f"\n{p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    try:
        with rawpy.imread(str(p)) as r:
            print("  LibRaw: ok", r.sizes.raw_width, "x", r.sizes.raw_height)
    except Exception as e:  # noqa: BLE001
        print("  LibRaw: FEHLER", e)
    t = time.time()
    try:
        img, orientation = raw_io.load_preview(p)
        print(f"  Vorschau: {img.shape} Orientierung {orientation} ({time.time() - t:.2f} s)")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print("  Vorschau: FEHLER", e)
    try:
        img, _ = raw_io.fallback_preview(p)
        print("  Ersatzweg allein:", img.shape)
    except Exception as e:  # noqa: BLE001
        print("  Ersatzweg allein: FEHLER", e)
    try:
        lin, info = raw_io.read_linear(p, 1024)
        print(f"  Linear: {lin.shape} WB {info.as_shot_temp} K")
    except Exception as e:  # noqa: BLE001
        print("  Linear: FEHLER", e)
sys.exit(1 if bad else 0)

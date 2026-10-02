"""Testet echte Kamera-RAWs (z. B. von raw.pixls.us): LibRaw direkt, Vorschau mit Ersatzweg, lineare Daten.

Aufruf: python scripts/ci_real_raw.py DATEI [DATEI ...]
Exit 1, wenn für eine Datei gar keine Vorschau entsteht.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import rawpy

from tagmatti.io import raw as raw_io

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
    ref_med = None
    try:
        lin, info = raw_io.read_linear(p, 1024)
        ref_med = float(np.median(np.log2(np.clip((lin * info.camera_wb).mean(axis=2), 1e-6, None))))
        print(f"  Linear (LibRaw): {lin.shape} WB {info.as_shot_temp:.0f} K / {info.as_shot_tint:+.0f}  "
              f"log2-Median {ref_med:.2f}")
    except Exception as e:  # noqa: BLE001
        print("  Linear (LibRaw): FEHLER", e)
    try:
        from tagmatti.io import decoders

        prev, _ = raw_io.load_preview(p, 1024)
        res = decoders.coreimage_linear(p, 1024)
        if res is None:
            print("  Apple RAW-Engine: nicht verfügbar/nicht lesbar")
        else:
            cl, meta = res
            cl = decoders.align_to(cl, prev)
            med = float(np.median(np.log2(np.clip(cl.mean(axis=2), 1e-6, None))))
            diff = f" (Abweichung zu LibRaw {med - ref_med:+.2f} EV)" if ref_med is not None else ""
            print(f"  Apple RAW-Engine: {cl.shape} WB {meta['as_shot_temp']:.0f} K / {meta['as_shot_tint']:+.0f} "
                  f"BE {meta['baseline_exposure']:+.2f} log2-Median {med:.2f}{diff}")
        lin_any, info_any = raw_io.read_linear_any(p, 1024, reference=prev)
        print(f"  Kette: Quelle {info_any.extra.get('source', 'libraw')}, {lin_any.shape}")
    except Exception as e:  # noqa: BLE001
        print("  Apple RAW-Engine: FEHLER", e)
sys.exit(1 if bad else 0)

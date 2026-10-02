"""Ersatz-Decoder für Kameras, die LibRaw (noch) nicht kennt.

Reihenfolge (siehe ``raw.read_linear_any``):
1. LibRaw (rawpy)
2. macOS: Apples RAW-Engine (Core Image, ``CIRAWFilter``). Unterstützt praktisch jede aktuelle
   Kamera und bekommt neue Modelle mit macOS-Updates. Liefert lineare Daten und den
   Weissabgleich der Aufnahme (Temperatur/Tönung).
3. Adobe DNG Converter, falls installiert: wandelt temporär in eine DNG, die LibRaw immer lesen kann.
4. Eingebettete JPEG-Vorschau (in ``raw.load_preview`` bzw. ``analysis.preview_linear_stats``).

Alles hier ist optional: fehlt ein Weg, wird der nächste versucht.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DNG_CONVERTER_PATHS = (
    "/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter",
    r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe",
)


# ---------------------------------------------------------------------------
# Apple Core Image
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _quartz():
    if sys.platform != "darwin":
        return None
    try:
        import objc  # noqa: F401
        import Quartz  # noqa: F401
        from Foundation import NSURL  # noqa: F401

        return True
    except ImportError:
        log.info("pyobjc (Quartz) nicht installiert: Apples RAW-Engine nicht verfügbar")
        return None


def coreimage_linear(path: str | Path, max_side: int = 1600) -> tuple[np.ndarray, dict] | None:
    """Lineares sRGB (Weissabgleich der Aufnahme angewendet, Anzeigeorientierung) + Metadaten.

    Werte sind auf die RAW-Skala zurückgerechnet (Baseline-Exposure abgezogen), damit sie mit
    LibRaw-Daten vergleichbar sind: Sensorsättigung ~ 1.0."""
    if not _quartz():
        return None
    import objc
    import Quartz
    from Foundation import NSURL

    try:
        CIRAWFilter = objc.lookUpClass("CIRAWFilter")
    except objc.error:
        return None
    try:
        f = CIRAWFilter.filterWithImageURL_(NSURL.fileURLWithPath_(str(path)))
    except Exception as e:  # noqa: BLE001
        log.debug("CIRAWFilter: %s", e)
        return None
    if f is None:
        return None
    try:
        meta = {"as_shot_temp": float(f.neutralTemperature()), "as_shot_tint": float(f.neutralTint()),
                "baseline_exposure": float(f.baselineExposure())}
        native = f.nativeSize()
        nw, nh = float(native.width), float(native.height)
        if nw <= 0 or nh <= 0:
            return None
        f.setBoostAmount_(0.0)            # keine Tonkurve -> linear
        f.setExposure_(0.0)
        f.setScaleFactor_(min(1.0, max_side / max(nw, nh)))
        img = f.outputImage()
        if img is None:
            return None
        cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceLinearSRGB)
        ctx = Quartz.CIContext.contextWithOptions_({Quartz.kCIContextWorkingColorSpace: cs,
                                                    Quartz.kCIContextOutputColorSpace: cs})
        ext = img.extent()
        w, h = int(ext.size.width), int(ext.size.height)
        if w <= 0 or h <= 0:
            return None
        buf = bytearray(w * h * 16)
        ctx.render_toBitmap_rowBytes_bounds_format_colorSpace_(img, buf, w * 16, ext, Quartz.kCIFormatRGBAf, cs)
        arr = np.frombuffer(bytes(buf), dtype=np.float32).reshape(h, w, 4)[..., :3]
        lin = np.clip(arr / (2.0 ** meta["baseline_exposure"]), 0, None).astype(np.float32)
        meta.update({"width": int(nw), "height": int(nh), "source": "coreimage"})
        return np.ascontiguousarray(lin), meta
    except Exception as e:  # noqa: BLE001 - jede Core-Image-Panne -> nächster Weg
        log.warning("Apples RAW-Engine konnte %s nicht lesen: %s", Path(path).name, e)
        return None


def align_to(lin: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Ausrichtung an einer korrekt gedrehten Referenz (eingebettete Vorschau) angleichen.

    Core Image hat den Ursprung unten links; je nach Version und Kamera kommt das Bild
    gespiegelt oder ungedreht zurück. Statt das zu raten, vergleichen wir mit der Vorschau."""
    import cv2

    def small(x: np.ndarray, gamma: bool) -> np.ndarray:
        g = x.mean(axis=2) if x.ndim == 3 else x
        g = g.astype(np.float32)
        if gamma:
            g = np.power(np.clip(g / max(float(np.percentile(g, 99)), 1e-6), 0, 1), 1 / 2.2)
        else:
            g = g / 255.0
        g = cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA)
        return (g - g.mean()) / (g.std() + 1e-6)

    ref = small(reference, False)
    cands = {
        "id": lin, "flipud": lin[::-1], "rot180": lin[::-1, ::-1],
        "rot90": np.rot90(lin, -1), "rot270": np.rot90(lin, 1), "rot90f": np.rot90(lin[::-1], -1),
    }
    ref_portrait = reference.shape[0] > reference.shape[1]
    best, best_s = "id", -9.0
    for k, c in cands.items():
        if (c.shape[0] > c.shape[1]) != ref_portrait:
            continue
        s = float((small(c, True) * ref).mean())
        if s > best_s:
            best, best_s = k, s
    return np.ascontiguousarray(cands[best])


# ---------------------------------------------------------------------------
# Adobe DNG Converter
# ---------------------------------------------------------------------------

def dng_converter() -> str | None:
    for p in DNG_CONVERTER_PATHS:
        if Path(p).exists():
            return p
    return shutil.which("dngconverter")


def convert_to_dng(path: str | Path, out_dir: Path) -> Path | None:
    exe = dng_converter()
    if not exe:
        return None
    path = Path(path)
    try:
        subprocess.run([exe, "-c", "-p0", "-d", str(out_dir), str(path)], capture_output=True, timeout=180,
                       check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Adobe DNG Converter fehlgeschlagen für %s: %s", path.name, e)
        return None
    out = out_dir / (path.stem + ".dng")
    return out if out.exists() else None


class TempDng:
    """Kontextmanager: RAW -> temporäre DNG (wird danach gelöscht)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path | None:
        self._dir = tempfile.TemporaryDirectory(prefix="tagmatti-dng-")
        return convert_to_dng(self.path, Path(self._dir.name))

    def __exit__(self, *exc) -> None:
        if self._dir:
            self._dir.cleanup()


def available() -> dict[str, bool]:
    """Welche Ersatzwege es auf diesem Rechner gibt (für Diagnose/Einstellungen)."""
    return {"coreimage": bool(_quartz()), "dng_converter": dng_converter() is not None}

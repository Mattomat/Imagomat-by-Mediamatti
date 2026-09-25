"""RAW-Zugriff über rawpy/LibRaw.

- ``load_preview``: eingebettete JPEG-Vorschau (schnell, fürs Culling), korrekt gedreht.
- ``read_linear``: lineare Kamera-RGB-Daten durch 2x2-Binning des Bayer-Musters
  (ohne Demosaicing, sehr schnell; für Statistik, Merkmale und Rauschmessung).
- ``decode``: vollständiges Demosaicing (für Vorschau-Rendering und lokales Denoise).
- ``estimate_noise``: Rauschen direkt aus den Sensordaten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

from ..config import RAW_EXTENSIONS
from .color import SRGB_TO_XYZ, multipliers_to_temp_tint

_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}


@dataclass
class RawInfo:
    width: int
    height: int
    orientation: int                     # EXIF-Orientierung
    xyz_to_cam: np.ndarray               # 3x3
    camera_wb: np.ndarray                # Multiplikatoren R,G,B (G=1)
    black: float
    white: float
    as_shot_temp: float | None = None
    as_shot_tint: float | None = None
    pattern: str = "RGGB"
    extra: dict = field(default_factory=dict)


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def orient(img: np.ndarray, orientation: int) -> np.ndarray:
    """Wendet eine EXIF-Orientierung an (Ergebnis = Anzeigeorientierung)."""
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == 2:
        return img[:, ::-1]
    if orientation == 4:
        return img[::-1]
    if orientation == 5:
        return np.ascontiguousarray(np.transpose(img, (1, 0, 2) if img.ndim == 3 else (1, 0)))
    if orientation == 7:
        return cv2.rotate(np.ascontiguousarray(np.transpose(img, (1, 0, 2) if img.ndim == 3 else (1, 0))),
                          cv2.ROTATE_180)
    return img


def _xyz_to_cam(r: rawpy.RawPy) -> np.ndarray:
    m = np.asarray(r.rgb_xyz_matrix, dtype=float)[:3, :3]
    if np.abs(m).sum() > 1e-6:
        return m
    # DNG: LibRaw liefert stattdessen color_matrix (Kamera -> sRGB)
    cm = np.asarray(r.color_matrix, dtype=float)[:3, :3]
    if np.abs(cm).sum() > 1e-6:
        return np.linalg.inv(cm) @ np.linalg.inv(SRGB_TO_XYZ)
    return np.linalg.inv(SRGB_TO_XYZ)


def raw_info(r: rawpy.RawPy) -> RawInfo:
    s = r.sizes
    wb = np.asarray(r.camera_whitebalance[:3], dtype=float)
    if wb[1] <= 0 or not np.all(np.isfinite(wb)) or wb.sum() <= 0:
        wb = np.asarray(r.daylight_whitebalance[:3], dtype=float)
    wb = wb / wb[1] if wb[1] > 0 else np.array([2.0, 1.0, 1.5])
    xyz_to_cam = _xyz_to_cam(r)
    try:
        temp, tint = multipliers_to_temp_tint(xyz_to_cam, wb)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        temp, tint = None, None
    blacks = np.asarray(r.black_level_per_channel, dtype=float)
    desc = r.color_desc.decode(errors="ignore")
    pat = "".join(desc[i] for i in np.asarray(r.raw_pattern).ravel()) if r.raw_pattern is not None else ""
    return RawInfo(
        width=s.width, height=s.height, orientation=_FLIP_TO_EXIF.get(s.flip, 1), xyz_to_cam=xyz_to_cam,
        camera_wb=wb, black=float(blacks.mean()), white=float(r.white_level), as_shot_temp=temp,
        as_shot_tint=tint, pattern=pat,
    )


def load_preview(path: str | Path, max_side: int = 2048) -> tuple[np.ndarray, int]:
    """Eingebettete Vorschau als RGB uint8 in Anzeigeorientierung und EXIF-Orientierung.

    Fällt auf ein halbes RAW-Decoding zurück, wenn keine Vorschau eingebettet ist.
    """
    path = Path(path)
    if not is_raw(path):
        with Image.open(path) as im:
            orientation = im.getexif().get(0x0112, 1)
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            return np.asarray(im), orientation
    with rawpy.imread(str(path)) as r:
        orientation = _FLIP_TO_EXIF.get(r.sizes.flip, 1)
        img = None
        try:
            th = r.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                arr = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    # Vorschau ist in Sensororientierung gespeichert, wenn ihr Seitenverhältnis dem Sensor entspricht
                    if (img.shape[1] >= img.shape[0]) == (r.sizes.width >= r.sizes.height):
                        img = orient(img, orientation)
            elif th.format == rawpy.ThumbFormat.BITMAP:
                img = orient(np.asarray(th.data), orientation)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            img = None
        if img is None:
            img = r.postprocess(half_size=True, use_camera_wb=True, output_bps=8, no_auto_bright=False)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img), orientation


def _binned(r: rawpy.RawPy) -> np.ndarray:
    """2x2-Binning des Bayer-Musters -> lineares Kamera-RGB (H/2, W/2, 3), 0..1."""
    raw = r.raw_image_visible.astype(np.float32)
    colors = r.raw_colors_visible
    blacks = np.asarray(r.black_level_per_channel, dtype=np.float32)
    white = float(r.white_level)
    h, w = raw.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    raw, colors = raw[:h2, :w2], colors[:h2, :w2]
    raw = raw - blacks[colors]
    raw /= max(white - float(blacks.mean()), 1.0)
    out = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    cnt = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    desc = r.color_desc.decode(errors="ignore")
    for dy in (0, 1):
        for dx in (0, 1):
            ci = int(colors[dy, dx])
            ch = {"R": 0, "G": 1, "B": 2}.get(desc[ci] if ci < len(desc) else "G", 1)
            out[..., ch] += raw[dy::2, dx::2]
            cnt[..., ch] += 1
    return out / np.maximum(cnt, 1)


def read_linear(path: str | Path, max_side: int | None = 1600) -> tuple[np.ndarray, RawInfo]:
    """Lineares Kamera-RGB (nicht weissabgeglichen) in Anzeigeorientierung + RawInfo."""
    with rawpy.imread(str(path)) as r:
        info = raw_info(r)
        lin = _binned(r)
    lin = orient(lin, info.orientation)
    if max_side:
        h, w = lin.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            lin = cv2.resize(lin, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(lin), info


def decode(path: str | Path, half_size: bool = True) -> tuple[np.ndarray, RawInfo]:
    """Demosaictes, lineares Kamera-RGB als float32 (ohne Weissabgleich), Anzeigeorientierung."""
    with rawpy.imread(str(path)) as r:
        info = raw_info(r)
        rgb = r.postprocess(half_size=half_size, use_camera_wb=False, user_wb=[1, 1, 1, 1],
                            output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
                            output_bps=16, user_flip=0, highlight_mode=rawpy.HighlightMode.Clip)
    lin = rgb.astype(np.float32) / 65535.0
    return orient(lin, info.orientation), info


def estimate_noise(path: str | Path) -> dict[str, float]:
    """Rauschschätzung auf einem Grünkanal der Bayer-Daten.

    Modell (Poisson-Gauss): var(x) = a * x + b. Schätzung über robuste lokale Varianzen
    (Hochpass, MAD) in flachen Bereichen, gruppiert nach Helligkeit.
    Rückgabe: sigma bei 18 % Grau und im Schatten (5 %), relativ zum Weisspunkt.
    """
    with rawpy.imread(str(path)) as r:
        raw = r.raw_image_visible.astype(np.float32)
        colors = r.raw_colors_visible
        blacks = np.asarray(r.black_level_per_channel, dtype=np.float32)
        white = float(r.white_level)
        desc = r.color_desc.decode(errors="ignore")
        # ersten Grünkanal im 2x2-Muster finden
        gy, gx = 0, 1
        for dy in (0, 1):
            for dx in (0, 1):
                if desc[int(colors[dy, dx])] == "G":
                    gy, gx = dy, dx
                    break
            else:
                continue
            break
    g = (raw[gy::2, gx::2] - blacks.mean()) / max(white - float(blacks.mean()), 1.0)
    return noise_from_plane(g)


def noise_from_plane(g: np.ndarray) -> dict[str, float]:
    g = g.astype(np.float32)
    # Hochpass: Differenz zu 3x3-Mittel, skaliert auf Einzelpixel-Sigma
    hp = (g - cv2.blur(g, (3, 3))) * (3.0 / np.sqrt(8.0))
    local_mean = cv2.blur(g, (7, 7))
    grad = np.abs(cv2.Sobel(local_mean, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(local_mean, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, 40)
    bins = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.95])
    xs, vs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = flat & (local_mean >= lo) & (local_mean < hi)
        if m.sum() < 500:
            continue
        mad = np.median(np.abs(hp[m] - np.median(hp[m])))
        xs.append(float(np.median(local_mean[m])))
        vs.append(float((1.4826 * mad) ** 2))
    if len(xs) >= 2:
        a, b = np.polyfit(xs, vs, 1)
        a, b = max(a, 1e-9), max(b, 1e-12)
    elif xs:
        a, b = vs[0] / max(xs[0], 1e-3), 1e-8
    else:
        a, b = 1e-5, 1e-8
    sig = lambda x: float(np.sqrt(a * x + b))  # noqa: E731
    return {"a": float(a), "b": float(b), "sigma_mid": sig(0.18), "sigma_shadow": sig(0.05),
            "snr_mid": 0.18 / max(sig(0.18), 1e-9)}

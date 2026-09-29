"""Merkmale fürs Stilmodell.

Für Shoot-Bilder kommen sie aus der Analyse-Datenbank, für Trainingsbilder (Katalog,
XMP-Ordner) berechnen wir dieselben Werte mit denselben Funktionen und cachen sie
(Schlüssel: Pfad + Grösse + Änderungszeit), damit ein erneutes Training schnell ist.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import cache_dir
from ..io.color import mired
from ..vision.embeddings import SCENE_PROMPTS, get_embedder, heuristic_scene

SCENES = list(SCENE_PROMPTS)
LIN_HIST_BINS = 32

NUMERIC_KEYS = [
    "log_iso", "log_shutter", "aperture", "log_focal", "hour_sin", "hour_cos",
    "as_shot_mired", "as_shot_tint", "wb_rg", "wb_bg", "gray_rg", "gray_bg",
    "lin_log_p01", "lin_log_p05", "lin_log_p25", "lin_log_median", "lin_log_p75", "lin_log_p95", "lin_log_p99",
    "lin_log_mean", "raw_clip", "log_noise",
    "median", "p05", "p95", "clip_hi", "clip_lo", "mean", "sat_mean", "sat_p95", "lab_a", "lab_b", "neon_fraction",
    "face_count", "max_face_height", "face_luma", "subject_fraction", "sky_fraction", "subject_luma",
    "background_luma", "sky_luma", "subject_cx", "subject_cy", "tilt_angle", "tilt_conf", "portrait_orientation",
] + [f"scene_{s}" for s in SCENES] + [f"lin_hist_{i}" for i in range(LIN_HIST_BINS)]


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def numeric_features(a: dict[str, Any], exif: dict[str, Any]) -> dict[str, float]:
    """Analyse-Dict + EXIF -> flaches Merkmals-Dict (alle Schlüssel aus NUMERIC_KEYS)."""
    t = exif.get("capture_time")
    hour = _dt.datetime.fromtimestamp(t).hour + _dt.datetime.fromtimestamp(t).minute / 60 if t else 12.0
    wb = a.get("camera_wb") or [2.0, 1.0, 1.5]
    rgb = a.get("lin_rgb_mean") or [0.1, 0.1, 0.1]
    g = max(_f(rgb[1], 1e-4), 1e-4)
    f: dict[str, float] = {
        "log_iso": math.log2(max(_f(exif.get("iso"), 100), 1)),
        "log_shutter": math.log2(max(_f(exif.get("exposure_time"), 1 / 250), 1e-5)),
        "aperture": _f(exif.get("aperture"), 4.0),
        "log_focal": math.log2(max(_f(exif.get("focal_length"), 50), 1)),
        "hour_sin": math.sin(hour / 24 * 2 * math.pi), "hour_cos": math.cos(hour / 24 * 2 * math.pi),
        "as_shot_mired": mired(_f(a.get("as_shot_temp"), 5000)), "as_shot_tint": _f(a.get("as_shot_tint")),
        "wb_rg": _f(wb[0], 2.0) / max(_f(wb[1], 1.0), 1e-3), "wb_bg": _f(wb[2], 1.5) / max(_f(wb[1], 1.0), 1e-3),
        "gray_rg": _f(rgb[0]) / g, "gray_bg": _f(rgb[2]) / g,
        "log_noise": math.log2(max(_f(a.get("noise_sigma_mid"), 1e-3), 1e-5)),
        "portrait_orientation": float(_f(a.get("preview_h"), 2) > _f(a.get("preview_w"), 3)),
    }
    for k in NUMERIC_KEYS:
        if k in f or k.startswith(("scene_", "lin_hist_")):
            continue
        default = {"lin_log_median": -4.0, "median": 0.4, "subject_cx": 0.5, "subject_cy": 0.5,
                   "face_luma": _f(a.get("subject_luma"), 0.4)}.get(k, 0.0)
        f[k] = _f(a.get(k), default)
    scene = a.get("scene") or heuristic_scene({**a, "iso": exif.get("iso")})
    for s in SCENES:
        f[f"scene_{s}"] = _f(scene.get(s))
    hist = a.get("lin_hist") or [0.0] * LIN_HIST_BINS
    for i in range(LIN_HIST_BINS):
        f[f"lin_hist_{i}"] = _f(hist[i]) if i < len(hist) else 0.0
    return f


def vectorize(feats: list[dict[str, float]]) -> np.ndarray:
    return np.array([[f.get(k, 0.0) for k in NUMERIC_KEYS] for f in feats], dtype=np.float64)


# ---------------------------------------------------------------------------
# Trainingsbilder
# ---------------------------------------------------------------------------

def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}|v2".encode()).hexdigest()


def _cache_file(key: str) -> Path:
    d = cache_dir() / "features" / key[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def image_record(path: Path, exif: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse + Embedding für eine beliebige Bilddatei (gecacht)."""
    from ..analysis import compute_content, compute_metrics
    from ..io.exif import read_exif

    path = Path(path)
    key = _cache_key(path)
    cf = _cache_file(key)
    if cf.exists():
        try:
            return json.loads(cf.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    img, data, orientation, width, height, ph = compute_metrics(path)
    _, content, _ = compute_content(img)
    data.update(content)
    emb = get_embedder()
    small = cv2.resize(img, (448, int(448 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    vec = emb.embed([small])
    if hasattr(emb, "scenes_from_embeddings"):
        data["scene"] = emb.scenes_from_embeddings(vec)[0]
    ex = exif if exif is not None else read_exif([path]).get(str(path), {})
    data.pop("hist", None)
    rec = {"path": str(path), "analysis": data, "embedding": vec[0].tolist(), "embedder": emb.name,
           "exif": {k: v for k, v in ex.items() if k != "raw"}, "orientation": orientation,
           "width": width, "height": height}
    cf.write_text(json.dumps(rec, default=float), "utf-8")
    return rec

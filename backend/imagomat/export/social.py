"""Bilder direkt für Social Media speichern: Story (9:16), Post (4:5), Quadrat oder Original.

Das Bild wird mit der Imagomat-Entwicklung gerendert (Vorschau-Qualität, nicht Lightroom) und
automatisch um die Personen herum zugeschnitten.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis import load_masks
from ..io import raw as raw_io
from ..jobs import JobContext, job
from ..render.pipeline import render
from .exporter import ExportItem, _items

log = logging.getLogger(__name__)

FORMATS: dict[str, tuple[str, int, int]] = {
    "story": ("Story", 1080, 1920),
    "post": ("Post", 1080, 1350),
    "square": ("Quadrat", 1080, 1080),
    "original": ("Original", 0, 0),
}


def render_item(it: ExportItem, long_side: int) -> np.ndarray:
    """Entwickeltes Bild (RGB uint8, Anzeigeorientierung)."""
    if raw_io.is_raw(it.src):
        lin, info = raw_io.decode_any(it.src, half_size=long_side <= 3000)
        masks = load_masks(it.image_id)
        seg = {"subject": masks[0], "sky": masks[1]} if masks else {}
        return render(lin, info.xyz_to_cam, info.camera_wb, it.crs, info.orientation, seg, long_side)
    img, _ = raw_io.load_preview(it.src, long_side)
    return img


def focus_point(img: np.ndarray) -> tuple[float, float]:
    """Wohin der Zuschnitt zielt: Gesichter (gewichtet nach Grösse), sonst Bildmitte."""
    from ..vision.faces import detect_faces

    try:
        faces = detect_faces(img, min_score=0.6)
    except Exception:  # noqa: BLE001
        faces = []
    if not faces:
        return 0.5, 0.45
    w = np.array([f.height ** 2 for f in faces])
    cx = float(np.average([f.center[0] for f in faces], weights=w))
    cy = float(np.average([f.center[1] for f in faces], weights=w))
    return cx, cy


def smart_crop(img: np.ndarray, aspect: float, focus: tuple[float, float]) -> np.ndarray:
    """Grösstmöglicher Ausschnitt mit Seitenverhältnis ``aspect`` (Breite/Höhe) um den Fokuspunkt.
    Gesichter landen etwas über der Mitte (Drittel-Regel), damit Körper und Ball Platz haben."""
    h, w = img.shape[:2]
    if w / h > aspect:
        cw, ch = int(round(h * aspect)), h
    else:
        cw, ch = w, int(round(w / aspect))
    cx, cy = focus[0] * w, focus[1] * h
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    y0 = int(np.clip(cy - ch * 0.38, 0, h - ch))
    return img[y0:y0 + ch, x0:x0 + cw]


def make_social(img: np.ndarray, fmt: str) -> np.ndarray:
    _, tw, th = FORMATS[fmt]
    if fmt == "original":
        return img
    out = smart_crop(img, tw / th, focus_point(img))
    return cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA if out.shape[1] > tw else cv2.INTER_CUBIC)


def default_target(shoot_name: str, fmt: str) -> Path:
    safe = re.sub(r"[/:\\]", "-", shoot_name).strip() or "Shoot"
    return Path.home() / "Downloads" / "Imagomat" / safe / FORMATS[fmt][0]


def select_items(items: list[ExportItem], selection: str, ids: list[int] | None) -> list[ExportItem]:
    if ids:
        wanted = set(ids)
        return [it for it in items if it.image_id in wanted]
    kept = [it for it in items if it.decision == "keep"]
    if selection == "top":
        top = [it for it in kept if it.is_best or (it.rating or 0) >= 4]
        return top or sorted(kept, key=lambda x: -(x.score or 0))[:10]
    return kept


@job("social")
def social_export(ctx: JobContext, shoot_id: int, format: str = "story", selection: str = "top",
                  ids: list[int] | None = None, target: str | None = None) -> dict[str, Any]:
    if format not in FORMATS:
        raise ValueError(f"Unbekanntes Format: {format}")
    db = ctx.db
    shoot = db.one("SELECT name FROM shoots WHERE id=?", (shoot_id,))
    name = shoot["name"] if shoot else f"Shoot {shoot_id}"
    tgt = Path(target).expanduser() if target else default_target(name, format)
    tgt.mkdir(parents=True, exist_ok=True)
    items = select_items(_items(db, shoot_id, include_rejected=bool(ids)), selection, ids)
    ctx.set_total(len(items))
    long_side = 2048 if format == "original" else 2400
    saved = 0
    for i, it in enumerate(items):
        ctx.check()
        try:
            img = make_social(render_item(it, long_side), format)
            suffix = "" if format == "original" else f"_{format}"
            cv2.imwrite(str(tgt / f"{it.src.stem}{suffix}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Social-Export fehlgeschlagen für %s: %s", it.src.name, e, exc_info=True)
        ctx.progress(i + 1, f"{FORMATS[format][0]}: {i + 1}/{len(items)}")
    db.update_shoot_settings(shoot_id, last_social=str(tgt))
    ctx.progress(len(items), f"{saved} Bilder für {FORMATS[format][0]} gespeichert -> {tgt}")
    return {"target": str(tgt), "saved": saved}

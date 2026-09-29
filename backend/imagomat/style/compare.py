"""Stile vergleichen: dieselben Bilder mit verschiedenen Stilen bearbeiten, ohne etwas zu speichern."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..config import load_settings, profiles_dir
from ..db import Database
from ..lightroom.dialect import Dialect
from .develop import develop_items
from .model import StyleModel, list_profiles

AUTO = "preset:auto"


@lru_cache(maxsize=8)
def _model(name: str, mtime: float) -> StyleModel | None:   # mtime: neu laden, wenn neu gelernt
    return StyleModel.load(name)


def model_version(name: str) -> float:
    p = profiles_dir() / name / "model.pkl"
    return p.stat().st_mtime if p.exists() else 0.0


def load_model(name: str) -> StyleModel | None:
    return _model(name, model_version(name))


def styles() -> list[dict[str, Any]]:
    out = [{"key": p["name"], "label": p["name"], "n": p.get("n")} for p in list_profiles()]
    out.append({"key": AUTO, "label": "Standard (ohne eigenen Stil)", "n": None})
    return out


def sample_images(db: Database, shoot_id: int, n: int = 3) -> list[int]:
    """Beispielbilder: die besten behaltenen, möglichst aus verschiedenen Szenen/Lichtsituationen."""
    rows = db.query(
        "SELECT i.id, i.capture_time, c.score FROM images i JOIN culling c ON c.image_id=i.id "
        "JOIN analysis a ON a.image_id=i.id WHERE i.shoot_id=? AND c.decision='keep' "
        "ORDER BY c.is_series_best DESC, c.score DESC LIMIT 60", (shoot_id,))
    if not rows:
        rows = db.query("SELECT i.id, i.capture_time, 0 score FROM images i JOIN analysis a ON a.image_id=i.id "
                        "WHERE i.shoot_id=? LIMIT 60", (shoot_id,))
    picked: list[Any] = []
    for r in rows:                         # zeitlich verteilt: nicht dreimal dieselbe Szene
        if all(abs((r["capture_time"] or 0) - (p["capture_time"] or 0)) > 60 for p in picked):
            picked.append(r)
        if len(picked) == n:
            break
    for r in rows:
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)
    return [int(r["id"]) for r in picked]


def develop_preview(db: Database, shoot_id: int, image_id: int, style: str) -> dict[str, Any]:
    """Entwicklungseinstellungen (crs) für ein Bild mit einem Stil, ohne sie zu speichern."""
    from .jobs import shoot_records

    items = shoot_records(db, shoot_id, only_keep=False, image_ids=[image_id])
    if not items:
        raise KeyError(image_id)
    model = None if style.startswith("preset:") else load_model(style)
    preset = style.split(":", 1)[1] if style.startswith("preset:") and style != AUTO else None
    develop_items(items, model, load_settings(), Dialect.load(), preset)
    return items[0].crs

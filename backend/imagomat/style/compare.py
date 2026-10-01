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
LOOK = "look:"
TPL = "tpl:"


def resolve(style: str | None) -> tuple[StyleModel | None, str | None, dict[str, Any] | None]:
    """Stil-Schlüssel -> (gelerntes Profil, Preset, Referenz-Look)."""
    from .presets import PRESETS

    if not style or style == AUTO:
        return None, None, None
    if style.startswith(TPL):
        from .template import load_template

        tpl = load_template(style[len(TPL):])
        if tpl is None:
            raise KeyError(style)
        return None, None, {**tpl, "kind": "template"}
    if style.startswith(LOOK):
        from .look import load_look

        look = load_look(style[len(LOOK):])
        if look is None:
            raise KeyError(style)
        base = look.get("base")
        return None, (base if base in PRESETS else None), look
    if style.startswith("preset:"):
        key = style.split(":", 1)[1]
        return None, (key if key in PRESETS else None), None
    return load_model(style), None, None


@lru_cache(maxsize=8)
def _model(name: str, mtime: float) -> StyleModel | None:   # mtime: neu laden, wenn neu gelernt
    return StyleModel.load(name)


def model_version(name: str) -> float:
    if name.startswith(TPL):
        from .template import template_version

        return template_version(name[len(TPL):])
    if name.startswith(LOOK):
        from .look import look_version

        return look_version(name[len(LOOK):])
    p = profiles_dir() / name / "model.pkl"
    return p.stat().st_mtime if p.exists() else 0.0


def load_model(name: str) -> StyleModel | None:
    return _model(name, model_version(name))


def styles() -> list[dict[str, Any]]:
    from .presets import PRESETS

    from .look import list_looks
    from .template import list_templates

    out = [{"key": f"{TPL}{t['name']}", "label": f"Vorlage: {t['name']}", "n": t.get("n"), "group": "Deine Stile",
            "description": "Deine Lightroom-Bearbeitung 1:1, pro Bild nur Belichtung, Weissabgleich und "
                           "Begradigen angepasst."} for t in list_templates()]
    out += [{"key": f"{LOOK}{lk['name']}", "label": f"Look: {lk['name']}", "n": lk.get("n"), "group": "Deine Stile",
            "description": "Jedes Bild wird auf deine fertigen Referenzbilder abgeglichen."} for lk in list_looks()]
    out += [{"key": p["name"], "label": p["name"], "n": p.get("n"), "group": "Deine Stile"} for p in list_profiles()]
    out += [{"key": f"preset:{p.key}", "label": p.name, "n": None, "group": p.group, "description": p.description}
            for p in PRESETS.values() if p.group]
    out.append({"key": AUTO, "label": "Standard (automatisch)", "n": None, "group": "Standard"})
    return out


def label(key: str | None) -> str | None:
    from .presets import PRESETS

    if key and key.startswith("preset:") and key[7:] in PRESETS:
        p = PRESETS[key[7:]]
        return f"{p.group}: {p.name}" if p.group else p.name
    if key and key.startswith(LOOK):
        return f"Look: {key[len(LOOK):]}"
    if key and key.startswith(TPL):
        return f"Vorlage: {key[len(TPL):]}"
    return key


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


def _shoot_ref(db: Database, shoot_id: int, look: dict[str, Any] | None) -> dict[str, Any] | None:
    if look is None or look.get("kind") != "template":
        return None
    from .template import shoot_reference_db

    return shoot_reference_db(db, shoot_id)


def develop_preview(db: Database, shoot_id: int, image_id: int, style: str) -> dict[str, Any]:
    """Entwicklungseinstellungen (crs) für ein Bild mit einem Stil, ohne sie zu speichern."""
    from .jobs import shoot_records

    items = shoot_records(db, shoot_id, only_keep=False, image_ids=[image_id])
    if not items:
        raise KeyError(image_id)
    model, preset, look = resolve(style)
    develop_items(items, model, load_settings(), Dialect.load(), preset, look, _shoot_ref(db, shoot_id, look))
    return items[0].crs


def apply_to_image(db: Database, shoot_id: int, image_id: int, style: str) -> dict[str, Any]:
    """Einen Stil nur für ein Bild übernehmen (Bearbeitung wird gespeichert, wie beim Entwickeln)."""
    import time

    from ..db import dumps
    from .jobs import shoot_records

    items = shoot_records(db, shoot_id, only_keep=False, image_ids=[image_id])
    if not items:
        raise KeyError(image_id)
    model, preset, look = resolve(style)
    develop_items(items, model, load_settings(), Dialect.load(), preset, look, _shoot_ref(db, shoot_id, look))
    it = items[0]
    masks = it.crs.get("MaskGroupBasedCorrections")
    with db.tx() as c:
        c.execute("INSERT OR REPLACE INTO edits(image_id, profile, params, masks, confidence, denoise, user_params,"
                  " updated_at) VALUES(?,?,?,?,?,?,(SELECT user_params FROM edits WHERE image_id=?),?)",
                  (image_id, style if not style.startswith("preset:") else f"preset:{it.preset}",
                   dumps({k: v for k, v in it.crs.items() if k != "MaskGroupBasedCorrections"}),
                   dumps(masks) if masks else None, it.confidence, it.denoise, image_id, time.time()))
    db.update_analysis(image_id, {"develop_notes": it.notes, "preset": it.preset})
    return it.crs

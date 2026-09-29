"""Aufräumen: Shoots (Alben) und Stile löschen. Originalbilder werden nie angefasst."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import cache_dir, load_settings, profiles_dir, save_settings
from .db import Database


def _cache_files(image_id: int) -> list[Path]:
    sub = f"{image_id // 1000:04d}"
    c = cache_dir()
    files = [c / "previews" / sub / f"{image_id}.jpg", c / "masks" / sub / f"{image_id}.npy",
             c / "linear" / f"{image_id}.npz", c / "thumbs" / f"{image_id}.jpg"]
    files += list((c / "renders").glob(f"{image_id}_*.jpg")) if (c / "renders").exists() else []
    return files


def delete_shoot(db: Database, shoot_id: int) -> dict[str, int]:
    """Shoot aus Imagomat entfernen. Bilder mit von dir benannten Gesichtern wandern in die (unsichtbaren)
    Referenzbilder, damit die Personenerkennung nichts vergisst."""
    from .people.reference import reference_shoot

    ref = reference_shoot(db)
    if shoot_id == ref:
        raise ValueError("Referenzbilder können nicht gelöscht werden")
    keep = [int(r[0]) for r in db.query(
        "SELECT DISTINCT f.image_id FROM faces f JOIN images i ON i.id=f.image_id WHERE i.shoot_id=? "
        "AND f.assigned_by IN ('manual','confirmed')", (shoot_id,))]
    gone = [int(r[0]) for r in db.query("SELECT id FROM images WHERE shoot_id=?", (shoot_id,))
            if int(r[0]) not in set(keep)]
    with db.tx() as c:
        if keep:
            q = ",".join("?" * len(keep))
            c.execute(f"UPDATE images SET shoot_id=? WHERE id IN ({q})", (ref, *keep))
            c.execute(f"DELETE FROM culling WHERE image_id IN ({q})", keep)
            c.execute(f"DELETE FROM edits WHERE image_id IN ({q})", keep)
            c.execute(f"UPDATE faces SET person_id=NULL, assigned_by=NULL WHERE image_id IN ({q}) "
                      "AND IFNULL(assigned_by,'') NOT IN ('manual','confirmed')", keep)
        c.execute("DELETE FROM images WHERE shoot_id=?", (shoot_id,))
        c.execute("DELETE FROM jobs WHERE shoot_id=? AND status NOT IN ('running','queued')", (shoot_id,))
        c.execute("DELETE FROM shoots WHERE id=?", (shoot_id,))
    for iid in gone:
        for p in _cache_files(iid):
            p.unlink(missing_ok=True)
    return {"deleted": len(gone), "kept_for_people": len(keep)}


def delete_profile(name: str) -> None:
    d = (profiles_dir() / name).resolve()
    if d.parent != profiles_dir().resolve() or not d.is_dir():
        raise KeyError(name)
    shutil.rmtree(d)
    s = load_settings()
    if s.default_profile == name:
        s.default_profile = None
        save_settings(s)


def rename_profile(old: str, new: str) -> None:
    import json

    new = new.strip()
    src, dst = profiles_dir() / old, profiles_dir() / new
    if not new or "/" in new or new.startswith(".") or not src.is_dir():
        raise KeyError(old)
    if dst.exists():
        raise ValueError(f"Es gibt schon einen Stil „{new}“")
    src.rename(dst)
    meta = dst / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text("utf-8"))
        m["name"] = new
        meta.write_text(json.dumps(m, indent=2, ensure_ascii=False), "utf-8")
    from .style.model import StyleModel

    model = StyleModel.load(new)
    if model is not None:
        model.name = new
        model.save()
    s = load_settings()
    if s.default_profile == old:
        s.default_profile = new
        save_settings(s)

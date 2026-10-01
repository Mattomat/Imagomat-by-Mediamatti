"""Kompletter Durchlauf für einen Shoot: Analyse -> Culling -> Personen -> Entwicklung (-> Export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import analysis  # noqa: F401  (registriert Jobs)
from .culling import calibrate as _cal  # noqa: F401
from .culling import engine
from .export import exporter
from .export import social  # noqa: F401  (registriert Job social)
from .export import tagging  # noqa: F401  (registriert Job tag_export)
from .jobs import JobContext, job
from .people import clustering
from .people import reference  # noqa: F401  (registriert Job learn_people)
from .style import jobs as style_jobs

STAGES = ["analyze", "cull", "people", "develop", "export"]


def ingest(ctx: JobContext, shoot_id: int, source: Path) -> int:
    """RAWs (und vorhandene XMP) von der Karte/dem Quellordner in den Shoot-Ordner am Ablageort kopieren.
    Originale bleiben unverändert; schon vorhandene gleiche Dateien werden übersprungen."""
    import shutil

    db = ctx.db
    shoot = db.one("SELECT * FROM shoots WHERE id=?", (shoot_id,))
    dest = Path(shoot["folder"])
    dest.mkdir(parents=True, exist_ok=True)
    files = analysis.scan_folder(source)
    extra = [p.with_suffix(ext) for p in files for ext in (".xmp", ".XMP") if p.with_suffix(ext).exists()]
    todo = files + extra
    need = sum(p.stat().st_size for p in todo if not (dest / p.name).exists())
    free = shutil.disk_usage(dest).free
    if need > free * 0.98:
        raise ValueError(f"Zu wenig Platz am Ablageort: {need / 1e9:.1f} GB nötig, {free / 1e9:.1f} GB frei ({dest})")
    ctx.set_total(len(todo))
    copied = 0
    for i, p in enumerate(todo):
        ctx.check()
        out = dest / p.name
        if out.exists() and out.stat().st_size == p.stat().st_size:
            continue
        if out.exists():                    # gleicher Name, anderes Bild (z. B. zweite Karte): nicht überschreiben
            out = dest / f"{p.stem}_{i}{p.suffix}"
        tmp = out.with_name(out.name + ".part")
        shutil.copy2(p, tmp)
        tmp.replace(out)
        copied += 1
        if i % 5 == 0:
            ctx.progress(i + 1, f"Kopieren {i + 1}/{len(todo)} nach {dest.name}")
    analysis.import_folder(db, dest, shoot["name"], shoot["profile"])
    db.update_shoot_settings(shoot_id, library=True, source=str(source))
    ctx.progress(len(todo), f"{copied} Dateien nach {dest} kopiert")
    return copied


@job("pipeline")
def run_pipeline(ctx: JobContext, shoot_id: int, keep_ratio: float | None = None, profile: str | None = None,
                 preset: str | None = None, export: dict[str, Any] | None = None, highlights: bool | None = None,
                 max_keep: int | None = None, mode: str | None = None,
                 burst_keep: int | None = None, copy_from: str | None = None) -> None:
    db = ctx.db
    if copy_from:
        ingest(ctx, shoot_id, Path(copy_from))
    if mode:
        db.update_shoot_settings(shoot_id, mode=mode)
    mode = mode or db.shoot_settings(shoot_id).get("mode", "full")
    if mode == "people":
        # Nur Personen (z. B. fertige JPGs): Analyse ohne Bild-KI, Personen erkennen, alles behalten
        ctx.progress(0, "1/2 Analyse")
        analysis.analyze_shoot(ctx, shoot_id, light=True)
        ctx.progress(0, "2/2 Personen")
        clustering.people_job(ctx, shoot_id)
        for r in db.images(shoot_id):
            db.set_culling(r["id"], score=0.0, decision="keep", rating=0, label=None, reasons=[],
                           series_id=None, is_best=False)
        return
    ctx.progress(0, "1/4 Analyse")
    analysis.analyze_shoot(ctx, shoot_id)
    ctx.progress(0, "2/4 Culling")
    engine.cull_shoot(ctx, shoot_id, keep_ratio, highlights, max_keep, burst_keep)
    ctx.progress(0, "3/4 Personen")
    clustering.people_job(ctx, shoot_id)
    ctx.progress(0, "4/4 Entwicklung")
    style_jobs.develop_shoot(ctx, shoot_id, profile, preset)
    if export:
        ctx.progress(0, "Export")
        exporter.export_shoot(ctx, shoot_id, **export)

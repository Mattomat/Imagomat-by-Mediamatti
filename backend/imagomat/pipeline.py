"""Kompletter Durchlauf für einen Shoot: Analyse -> Culling -> Personen -> Entwicklung (-> Export)."""

from __future__ import annotations

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


@job("pipeline")
def run_pipeline(ctx: JobContext, shoot_id: int, keep_ratio: float | None = None, profile: str | None = None,
                 preset: str | None = None, export: dict[str, Any] | None = None, highlights: bool | None = None,
                 max_keep: int | None = None, mode: str | None = None) -> None:
    db = ctx.db
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
    engine.cull_shoot(ctx, shoot_id, keep_ratio, highlights, max_keep)
    ctx.progress(0, "3/4 Personen")
    clustering.people_job(ctx, shoot_id)
    ctx.progress(0, "4/4 Entwicklung")
    style_jobs.develop_shoot(ctx, shoot_id, profile, preset)
    if export:
        ctx.progress(0, "Export")
        exporter.export_shoot(ctx, shoot_id, **export)

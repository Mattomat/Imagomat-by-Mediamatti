"""Kommandozeile.

Beispiele:
  imagomat serve
  imagomat run ~/Bilder/2026-05-01_FCW-GCZ --profile "Sport Nacht" --keep 0.2 --export ~/Export/FCW-GCZ
  imagomat train "Sport Nacht" --catalog ~/Pictures/Lightroom/Katalog.lrcat --min-rating 2
  imagomat train "Konzert" --folder ~/Bilder/Konzerte_bearbeitet
  imagomat dialect ~/Referenz-XMPs
  imagomat roster "FC Winterthur 1. Mannschaft" --csv kader.csv
  imagomat feedback 3 --profile "Sport Nacht"
  imagomat calibrate "Sport Nacht" ~/Bilder/Shoot1 ~/Bilder/Shoot2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import pipeline  # noqa: F401  (registriert alle Jobs)
from .analysis import import_folder
from .db import Database
from .jobs import JobManager


def _run(db: Database, kind: str, shoot_id: int | None = None, **params) -> dict:
    jm = JobManager(db)

    def show(evt: dict) -> None:
        tot = evt.get("total") or 0
        msg = evt.get("message") or ""
        sys.stderr.write(f"\r[{kind}] {evt.get('progress', 0)}/{tot} {msg[:70]:<70}")
        sys.stderr.flush()

    jm.listeners.append(show)
    res = jm.run_sync(db.create_job(kind, shoot_id, params))
    sys.stderr.write("\n")
    if res and res["status"] != "done":
        sys.stderr.write(f"Fehler: {res.get('error')}\n")
        sys.exit(1)
    return res or {}


def main(argv: list[str] | None = None) -> None:
    # Ruhigeres Terminal: Warnungen und Protokoll-Rauschen der KI-Bibliotheken ausblenden
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from .logs import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser(prog="imagomat", description="Lokale AI-Foto-Workflow-App")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="API + Oberfläche starten")
    s.add_argument("--port", type=int, default=8765)

    r = sub.add_parser("run", help="Ordner komplett verarbeiten (Import bis Export)")
    r.add_argument("folder")
    r.add_argument("--name")
    r.add_argument("--profile")
    r.add_argument("--preset")
    r.add_argument("--keep", type=float)
    r.add_argument("--highlights", action="store_true", help="Nur die besten Momente (ein Bild pro Spielszene)")
    r.add_argument("--max", type=int, help="Höchstens so viele Bilder behalten")
    r.add_argument("--teams", nargs="*", default=[])
    r.add_argument("--export")
    r.add_argument("--formats", nargs="*", default=["xmp"])
    r.add_argument("--copy-mode", default="copy", choices=["copy", "hardlink", "inplace"])
    r.add_argument("--template", help="leerer Lightroom-Katalog als Vorlage (für --formats catalog)")

    t = sub.add_parser("train", help="Stilprofil trainieren")
    t.add_argument("name")
    t.add_argument("--catalog")
    t.add_argument("--folder", action="append", default=[])
    t.add_argument("--preset-file", action="append", default=[])
    t.add_argument("--base-preset")
    t.add_argument("--min-rating", type=int, default=0)
    t.add_argument("--only-picked", action="store_true")
    t.add_argument("--append", action="store_true")

    f = sub.add_parser("feedback", help="Korrekturen aus Lightroom zurücklesen und Profil verbessern")
    f.add_argument("shoot_id", type=int)
    f.add_argument("--profile", required=True)
    f.add_argument("--folder", help="Ordner mit den korrigierten XMPs (Standard: neben den Originalen)")

    c = sub.add_parser("calibrate", help="Culling an deine Auswahl anpassen")
    c.add_argument("profile")
    c.add_argument("folders", nargs="+")
    c.add_argument("--catalog")
    c.add_argument("--min-rating", type=int, default=2)

    d = sub.add_parser("dialect", help="Lightroom-Dialekt aus Referenz-XMPs lernen")
    d.add_argument("folder")

    ro = sub.add_parser("roster", help="Kader importieren")
    ro.add_argument("team")
    ro.add_argument("--csv")
    ro.add_argument("--url")

    sub.add_parser("licenses", help="Lizenzen der Modelle anzeigen")
    sub.add_parser("presets", help="Mitgelieferte Presets anzeigen")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .server.app import main as serve

        serve(port=a.port)
        return
    db = Database()
    if a.cmd == "run":
        sid = import_folder(db, Path(a.folder), a.name, a.profile)
        if a.teams:
            db.update_shoot_settings(sid, teams=a.teams)
        export = None
        if a.export:
            export = {"target": a.export, "formats": a.formats, "copy_mode": a.copy_mode, "template": a.template}
        _run(db, "pipeline", sid, keep_ratio=a.keep, profile=a.profile, preset=a.preset, export=export,
             highlights=a.highlights or None, max_keep=a.max)
        print(f"Shoot {sid} verarbeitet." + (f" Export: {a.export}" if a.export else ""))
    elif a.cmd == "train":
        _run(db, "train_profile", None, name=a.name, catalog=a.catalog, folders=a.folder, presets=a.preset_file,
             base_preset=a.base_preset, min_rating=a.min_rating, only_picked=a.only_picked, append=a.append)
        from .style.jobs import profile_info

        info = profile_info(a.name)
        mae = info.get("metrics", {}).get("mae_model", {})
        print(f"Profil '{a.name}': {info.get('n')} Bilder. Mittlere Abweichung (Kreuzvalidierung):")
        for k in ("Exposure2012", "wb_dmired", "Contrast2012", "Highlights2012", "Shadows2012", "Vibrance"):
            if k in mae:
                print(f"  {k:<16} {mae[k]:.3f}")
    elif a.cmd == "feedback":
        _run(db, "feedback", a.shoot_id, profile=a.profile, folder=a.folder)
    elif a.cmd == "calibrate":
        _run(db, "calibrate_culling", None, profile=a.profile, folders=a.folders, catalog=a.catalog,
             min_rating=a.min_rating)
    elif a.cmd == "dialect":
        from .lightroom.dialect import Dialect, learn_dialect
        from .style.sources import from_folder

        samples = list(from_folder(Path(a.folder)))
        dl = learn_dialect([x.crs for x in samples], [x.label for x in samples if x.label], Dialect.load())
        dl.save()
        print(json.dumps(dl.__dict__, indent=2, ensure_ascii=False))
    elif a.cmd == "roster":
        from .people import roster

        entries = roster.parse_csv(Path(a.csv).read_text("utf-8-sig"), a.team) if a.csv else \
            roster.fetch(a.url or roster.DEFAULT_SOURCES[a.team], a.team)
        for e in entries:
            print(f"  {e.number or '-':>3}  {e.name}")
        roster.save(db, entries)
        print(f"{len(entries)} Personen gespeichert.")
    elif a.cmd == "licenses":
        from .vision.models import license_report

        for r in license_report():
            flag = "ok " if r["commercial_ok"] else "NC!"
            print(f"  [{flag}] {r['name']:<50} {r['license']}")
    elif a.cmd == "presets":
        from .style.presets import PRESETS

        for p in PRESETS.values():
            print(f"  {p.key:<18} {p.name:<16} {p.description}")


if __name__ == "__main__":
    main()

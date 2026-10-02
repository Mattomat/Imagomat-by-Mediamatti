"""Eigene Stichwörter: Objekte, Szenen, wiederkehrende Fans … zusätzlich zu den Personen.

- Bibliothek (Tabelle ``keywords``): eigene Stichwörter, optional mit Beschreibung für die Suche. Themenlisten
  (Fussball, Hochzeit, Event) fügen fertige Stichwörter mit Beschreibung hinzu.
- Zuordnung (``image_keywords``): von Hand (manual), bestätigter Vorschlag (confirmed), aus Lightroom
  (lightroom), offener Vorschlag (suggested) oder verworfen (rejected; wird nie wieder vorgeschlagen).
- Vorschläge: CLIP vergleicht die Bilder des Shoots mit der Beschreibung und/oder mit den Bildern, die das
  Stichwort schon haben (Beispielbilder). Man bestätigt oder verwirft sie.
- Beim Speichern (XMP, Lightroom-Katalog, Abgleich) gehen alle zugeordneten Stichwörter mit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .db import Database
from .jobs import JobContext, job

log = logging.getLogger(__name__)

ASSIGNED = ("manual", "confirmed", "lightroom")
MAX_SUGGEST = 300

# Themenlisten: Stichwort -> Beschreibung (Englisch: so versteht sie die Bild-KI am besten)
THEMES: dict[str, dict[str, str]] = {
    "Fussball": {
        "Fans": "football fans cheering in the stands with scarves and flags",
        "Choreo": "a big tifo choreography banner held up by fans in a stadium",
        "Pyro": "burning red flares and smoke in a football crowd",
        "Teamfoto": "a football team group photo, players standing in a line",
        "Einlaufen": "football players walking out of the tunnel onto the pitch",
        "Jubel": "football players celebrating a goal together",
        "Zweikampf": "two football players fighting for the ball",
        "Torschuss": "a football player shooting the ball at goal",
        "Kopfball": "a football player heading the ball",
        "Torhüter": "a goalkeeper diving to save the ball",
        "Trainer": "a football coach on the sideline giving instructions",
        "Schiedsrichter": "a referee with a whistle on the pitch",
        "Stadion": "a wide view of a football stadium and pitch",
        "Maskottchen": "a costumed sports mascot",
        "Kinder": "children at a football match",
    },
    "Hochzeit": {
        "Getting Ready": "a bride getting ready, makeup and dress preparation",
        "Trauung": "a wedding ceremony with the couple at the altar",
        "Ringe": "close-up of wedding rings",
        "Kuss": "the bride and groom kissing",
        "Brautpaar": "a portrait of the bride and groom together",
        "Brautstrauss": "a bridal bouquet of flowers",
        "Apéro": "wedding guests having drinks at a reception",
        "Gäste": "wedding guests smiling and talking",
        "Gruppenfoto": "a large group photo of wedding guests",
        "Torte": "a wedding cake, cutting the cake",
        "Tanz": "the couple dancing their first dance",
        "Party": "people dancing at a wedding party at night",
        "Details": "wedding decoration details, table setting and flowers",
        "Location": "a wedding venue, wide view of the location",
    },
    "Event": {
        "Bühne": "a speaker or performer on a stage with lights",
        "Publikum": "an audience sitting in rows watching a presentation",
        "Rede": "a person speaking into a microphone at a podium",
        "Networking": "people standing and talking with drinks at an event",
        "Gruppenfoto": "a group photo of people posing together",
        "Preisverleihung": "an award ceremony, a person receiving a trophy",
        "Konzert": "a band playing music on stage",
        "Essen": "food and catering at an event buffet",
        "Deko": "event decoration, table setting and flowers",
        "Location": "a wide view of an event venue",
        "Logo": "a company logo on a banner or screen",
    },
}


# ---------------------------------------------------------------------------
# Bibliothek
# ---------------------------------------------------------------------------

def get_or_create(db: Database, name: str, prompt: str | None = None, theme: str | None = None) -> int:
    name = " ".join(name.split()).strip()
    if not name:
        raise ValueError("Stichwort ist leer")
    row = db.one("SELECT id, prompt FROM keywords WHERE name=? COLLATE NOCASE", (name,))
    if row:
        if prompt and not row["prompt"]:
            with db.tx() as c:
                c.execute("UPDATE keywords SET prompt=? WHERE id=?", (prompt, row["id"]))
        return int(row["id"])
    with db.tx() as c:
        cur = c.execute("INSERT INTO keywords(name, prompt, theme, created_at) VALUES(?,?,?,?)",
                        (name, (prompt or "").strip() or None, theme, time.time()))
        return int(cur.lastrowid)


def add_theme(db: Database, theme: str) -> list[str]:
    if theme not in THEMES:
        raise ValueError(f"Unbekanntes Thema: {theme}")
    for name, prompt in THEMES[theme].items():
        get_or_create(db, name, prompt, theme)
    return list(THEMES[theme])


def library(db: Database) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT k.id, k.name, k.prompt, k.theme, "
        "SUM(CASE WHEN ik.state IN ('manual','confirmed','lightroom') THEN 1 ELSE 0 END) AS n "
        "FROM keywords k LEFT JOIN image_keywords ik ON ik.keyword_id=k.id GROUP BY k.id ORDER BY k.name COLLATE NOCASE")
    return [{"id": r["id"], "name": r["name"], "prompt": r["prompt"] or "", "theme": r["theme"], "n": r["n"] or 0}
            for r in rows]


def shoot_keywords(db: Database, shoot_id: int) -> list[dict[str, Any]]:
    """Bibliothek mit Zählern für diesen Shoot (zugeordnet / offene Vorschläge)."""
    counts = {r["keyword_id"]: (r["assigned"], r["suggested"]) for r in db.query(
        "SELECT ik.keyword_id, "
        "SUM(CASE WHEN ik.state IN ('manual','confirmed','lightroom') THEN 1 ELSE 0 END) AS assigned, "
        "SUM(CASE WHEN ik.state='suggested' THEN 1 ELSE 0 END) AS suggested "
        "FROM image_keywords ik JOIN images i ON i.id=ik.image_id WHERE i.shoot_id=? GROUP BY ik.keyword_id",
        (shoot_id,))}
    out = []
    for k in library(db):
        a, s = counts.get(k["id"], (0, 0))
        out.append({**k, "assigned": a or 0, "suggested": s or 0})
    return out


def image_keywords(db: Database, image_id: int) -> list[str]:
    """Zugeordnete Stichwörter eines Bildes (für XMP/Katalog)."""
    return [r["name"] for r in db.query(
        "SELECT k.name FROM image_keywords ik JOIN keywords k ON k.id=ik.keyword_id "
        "WHERE ik.image_id=? AND ik.state IN ('manual','confirmed','lightroom') ORDER BY k.name COLLATE NOCASE",
        (image_id,))]


def shoot_image_keywords(db: Database, shoot_id: int) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for r in db.query(
            "SELECT ik.image_id, k.name FROM image_keywords ik JOIN keywords k ON k.id=ik.keyword_id "
            "JOIN images i ON i.id=ik.image_id WHERE i.shoot_id=? AND ik.state IN ('manual','confirmed','lightroom') "
            "ORDER BY k.name COLLATE NOCASE", (shoot_id,)):
        out.setdefault(r["image_id"], []).append(r["name"])
    return out


def set_state(db: Database, image_ids: list[int], keyword_id: int, state: str | None) -> int:
    """state None = Zuordnung entfernen (ohne Sperre); 'rejected' = nie wieder vorschlagen."""
    if state not in (None, "manual", "confirmed", "lightroom", "rejected", "suggested"):
        raise ValueError(f"Unbekannter Zustand: {state}")
    with db.tx() as c:
        for iid in image_ids:
            if state is None:
                c.execute("DELETE FROM image_keywords WHERE image_id=? AND keyword_id=?", (iid, keyword_id))
            else:
                c.execute("INSERT INTO image_keywords(image_id, keyword_id, state) VALUES(?,?,?) "
                          "ON CONFLICT(image_id, keyword_id) DO UPDATE SET state=excluded.state",
                          (iid, keyword_id, state))
    return len(image_ids)


def suggestions(db: Database, shoot_id: int, keyword_id: int) -> list[dict[str, Any]]:
    return [{"image_id": r["image_id"], "score": r["score"]} for r in db.query(
        "SELECT ik.image_id, ik.score FROM image_keywords ik JOIN images i ON i.id=ik.image_id "
        "WHERE i.shoot_id=? AND ik.keyword_id=? AND ik.state='suggested' ORDER BY ik.score DESC",
        (shoot_id, keyword_id))]


# ---------------------------------------------------------------------------
# Vorschläge (CLIP)
# ---------------------------------------------------------------------------

def pick(scores: np.ndarray, floor: float, spread: float) -> np.ndarray:
    """Auffällig passende Bilder: über einer festen Schwelle UND deutlich über dem Durchschnitt des Shoots."""
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    thr = max(floor, float(scores.mean() + spread * scores.std()))
    return scores >= thr


def score_images(img: np.ndarray, text: np.ndarray | None, examples: np.ndarray | None) -> np.ndarray:
    """Ähnlichkeit jedes Bildes zur Beschreibung bzw. zu den Beispielbildern, auf 0..1 gebracht.
    Bild-Text-Ähnlichkeiten von CLIP liegen typischerweise bei 0.15–0.35, Bild-Bild bei 0.5–1."""
    parts = []
    if text is not None:
        t = (img @ text.T).max(axis=1)
        parts.append(np.clip((t - 0.17) / 0.15, 0, 1) * pick(t, 0.21, 1.2))
    if examples is not None and len(examples):
        e = (img @ examples.T).max(axis=1)
        parts.append(np.clip((e - 0.6) / 0.35, 0, 1) * pick(e, 0.78, 0.8))
    if not parts:
        return np.zeros(len(img))
    return np.max(np.stack(parts), axis=0)


@job("kw_suggest")
def kw_suggest(ctx: JobContext, shoot_id: int, keyword_id: int) -> dict[str, Any]:
    from .vision.tags import clip_embeddings, encode_text

    db = ctx.db
    kw = db.one("SELECT * FROM keywords WHERE id=?", (keyword_id,))
    if kw is None:
        raise ValueError("Stichwort nicht gefunden")
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE shoot_id=? ORDER BY capture_time, filename",
                                      (shoot_id,))]
    state = {r["image_id"]: r["state"] for r in db.query(
        "SELECT image_id, state FROM image_keywords WHERE keyword_id=?", (keyword_id,))}
    example_ids = [i for i, s in state.items() if s in ASSIGNED]
    if not kw["prompt"] and not example_ids:
        raise ValueError(f"„{kw['name']}“: zuerst eine Beschreibung eingeben oder ein paar Bilder von Hand zuordnen")
    ctx.set_total(len(ids))
    res = clip_embeddings(db, list(dict.fromkeys(ids + example_ids)),
                          progress=lambda d, n: ctx.progress(min(d, len(ids)), f"Bilder ansehen {d}/{n}"))
    if res is None:
        raise RuntimeError("Die Bild-KI (CLIP) ist nicht installiert – Vorschläge gehen nicht, von Hand zuordnen geht")
    emb, vecs = res
    cand = [i for i in ids if i in vecs and state.get(i) not in (*ASSIGNED, "rejected")]
    text = encode_text(emb, [kw["prompt"]]) if kw["prompt"] else None
    ex = np.stack([vecs[i] for i in example_ids if i in vecs]) if any(i in vecs for i in example_ids) else None
    scores = score_images(np.stack([vecs[i] for i in cand]), text, ex) if cand else np.zeros(0)
    ranked = sorted(((float(s), i) for s, i in zip(scores, cand) if s > 0), reverse=True)[:MAX_SUGGEST]
    with db.tx() as c:
        c.execute("DELETE FROM image_keywords WHERE keyword_id=? AND state='suggested' AND image_id IN "
                  "(SELECT id FROM images WHERE shoot_id=?)", (keyword_id, shoot_id))
        c.executemany("INSERT INTO image_keywords(image_id, keyword_id, state, score) VALUES(?,?,'suggested',?)",
                      [(i, keyword_id, s) for s, i in ranked])
    ctx.progress(len(ids), f"„{kw['name']}“: {len(ranked)} Vorschläge")
    return {"suggested": len(ranked)}

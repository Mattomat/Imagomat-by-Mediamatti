"""Inhalts-Stichwörter (Fans, Team, Jubel, Trainer ...) aus CLIP, ohne Training.

Werden als Stichwörter geschrieben, wenn auf einem Bild keine Person erkannt wurde (Einstellung
``keywords.content_keywords``), damit jedes Bild in Lightroom auffindbar ist.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
TAGS_VERSION = 1

# Schlüssel -> (Stichwort, Beschreibungen)
TAGS: dict[str, tuple[str, list[str]]] = {
    "fans": ("Fans", ["football fans cheering in the stands", "a crowd of supporters with scarves and flags",
                      "spectators in a stadium stand"]),
    "choreo": ("Choreo", ["fans holding up a huge choreography banner in the stands",
                          "a big tifo display in a football stadium"]),
    "pyro": ("Pyro", ["burning red flares and smoke in a stadium stand", "pyrotechnics in a football crowd"]),
    "team": ("Team", ["a football team group photo", "players standing in a line before the match",
                      "a team huddle in a circle on the pitch"]),
    "einlaufen": ("Einlaufen", ["football players walking out of the tunnel onto the pitch",
                                "players lining up with children mascots before kickoff"]),
    "jubel": ("Jubel", ["football players celebrating a goal together", "a player screaming with joy, arms raised"]),
    "zweikampf": ("Zweikampf", ["two football players fighting for the ball", "a hard tackle in a soccer match"]),
    "schuss": ("Torschuss", ["a football player shooting the ball at goal", "a striker taking a powerful shot"]),
    "kopfball": ("Kopfball", ["a football player heading the ball", "players jumping high for a header"]),
    "torhueter": ("Torhüter", ["a goalkeeper diving to save the ball", "a goalkeeper with gloves in front of the goal"]),
    "trainer": ("Trainer", ["a football coach on the sideline giving instructions",
                            "a manager in a jacket at the touchline"]),
    "schiedsrichter": ("Schiedsrichter", ["a referee showing a yellow card", "a referee with a whistle on the pitch"]),
    "stadion": ("Stadion", ["a wide view of a football stadium and pitch", "an empty stadium with floodlights"]),
    "portraet": ("Porträt", ["a close-up portrait of a football player", "a headshot of an athlete"]),
    "detail": ("Detail", ["a close-up of football boots on grass", "a close-up of a football ball",
                          "a close-up detail of a jersey"]),
    "training": ("Training", ["a football training session with cones", "players warming up before a match"]),
}
# Gegenklassen, damit nicht jedes Bild irgendein Stichwort bekommt
NEGATIVE = ["a photo", "a blurry photo", "an unclear picture"]
MIN_PROB = 0.28
MAX_TAGS = 2


def _text(embedder: Any) -> tuple[np.ndarray, list[str]]:
    cached = getattr(embedder, "_content_text", None)
    if cached is not None:
        return cached
    torch = embedder.torch
    keys, prompts = [], []
    for k, (_label, ps) in TAGS.items():
        for p in ps:
            keys.append(k)
            prompts.append(p)
    for p in NEGATIVE:
        keys.append("_neg")
        prompts.append(p)
    with torch.no_grad():
        t = embedder.model.encode_text(embedder.tokenizer(prompts).to(embedder.device)).float()
        t = t / t.norm(dim=-1, keepdim=True)
    embedder._content_text = (t.cpu().numpy(), keys)
    return embedder._content_text


def tags_from_scores(keys: list[str], logits: np.ndarray) -> list[str]:
    per: dict[str, float] = {}
    for k, v in zip(keys, logits):
        per[k] = max(per.get(k, -1e9), float(v))
    names = list(per)
    vals = np.array([per[n] for n in names])
    p = np.exp(vals - vals.max())
    p /= p.sum()
    ranked = sorted(((n, float(x)) for n, x in zip(names, p) if n != "_neg"), key=lambda kv: -kv[1])
    return [n for n, x in ranked[:MAX_TAGS] if x >= MIN_PROB]


def classify(embedder: Any, embs: np.ndarray) -> list[list[str]]:
    from .embeddings import _LOCK

    with _LOCK:
        temb, keys = _text(embedder)
    return [tags_from_scores(keys, row) for row in 100.0 * embs @ temb.T]


def labels(keys: list[str] | None) -> list[str]:
    return [TAGS[k][0] for k in keys or [] if k in TAGS]


def ensure_tags(db: Any, image_ids: list[int]) -> None:
    """Stichwörter für Bilder berechnen, die noch keine (aktuellen) haben. Braucht CLIP; ohne CLIP nichts."""
    from ..analysis import load_cached_preview
    from .embeddings import get_embedder

    todo = [i for i in image_ids if db.get_analysis(i).get("tags_v") != TAGS_VERSION]
    if not todo:
        return
    try:
        emb = get_embedder()
    except Exception as e:  # noqa: BLE001
        log.info("Inhalts-Stichwörter nicht verfügbar: %s", e)
        return
    if getattr(emb, "name", "") != "clip":
        return
    stored = db.embeddings(todo)
    missing = [i for i in todo if i not in stored or db.get_analysis(i).get("embedder") != "clip"]
    for start in range(0, len(missing), 16):                 # z. B. Nur-Personen-Modus: noch ohne Embedding
        chunk = missing[start:start + 16]
        imgs, ok = [], []
        for i in chunk:
            row = db.one("SELECT * FROM images WHERE id=?", (i,))
            try:
                imgs.append(load_cached_preview(row))
                ok.append(i)
            except Exception:  # noqa: BLE001
                continue
        if imgs:
            for i, v in zip(ok, emb.embed(imgs)):
                stored[i] = v
    ids = [i for i in todo if i in stored and len(stored[i]) == len(next(iter(stored.values())))]
    if not ids:
        return
    for i, t in zip(ids, classify(emb, np.stack([stored[i] for i in ids]))):
        db.update_analysis(i, {"tags": t, "tags_v": TAGS_VERSION})

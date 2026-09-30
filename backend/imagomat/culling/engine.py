"""AI-Culling: Bewertung, Serien/Duplikate, Auswahl nach Strenge, Sterne und Farblabels.

Scores werden innerhalb eines Shoots über Ränge normiert. So funktioniert das Culling
unabhängig von Kamera, Objektiv und Vorschaugrösse, und "behalte ca. 20 %" bedeutet wirklich
20 % *dieses* Shoots.

Ist für das Stilprofil ein kalibriertes Culling-Modell vorhanden (gelernt aus deinen Picks
bzw. Sternen), ersetzt es den heuristischen Score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import CullingSettings, load_settings
from ..db import Database
from ..jobs import JobContext, job
from ..lightroom.dialect import Dialect

log = logging.getLogger(__name__)

REASONS_DE = {
    "unscharf": "Unscharf",
    "bewegung": "Bewegungsunschärfe",
    "augen_zu": "Augen geschlossen",
    "abgewandt": "Gesicht abgewandt",
    "angeschnitten": "Person angeschnitten",
    "ueberbelichtet": "Überbelichtet",
    "unterbelichtet": "Unterbelichtet",
    "duplikat": "Duplikat",
    "serie": "Nicht das beste Bild der Serie",
    "strenge": "Unter der Auswahlgrenze",
    "kein_moment": "Kein Action-Moment",
    "unlesbar": "Datei nicht lesbar",
    "szene": "Nicht das beste Bild der Spielszene",
}

LEGACY_FEATURES = ["sharp", "motion", "exposure", "eyes_open", "yaw", "face_size", "aesthetic", "subject_size",
                   "clip_hi", "clip_lo", "cut", "has_face"]
FEATURE_NAMES = LEGACY_FEATURES + ["action"]
SPORT_SCENES = ("sport_day", "sport_floodlight", "sport_indoor", "concert_stage", "concert_club")


@dataclass
class CullItem:
    image_id: int
    t: float
    a: dict[str, Any]
    emb: np.ndarray | None
    phash: str | None
    feats: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    hard: list[str] = field(default_factory=list)
    series: int = 0
    reasons: list[str] = field(default_factory=list)
    keep: bool = False
    best: bool = False


def _rank(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) <= 1:
        return np.full(len(v), 0.5)
    order = v.argsort().argsort()
    return order / (len(v) - 1)


def _phash_dist(a: str | None, b: str | None) -> int:
    if not a or not b:
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def build_features(items: list[CullItem]) -> None:
    sharp_raw = []
    for it in items:
        a = it.a
        mf = a.get("main_face") or {}
        if mf and (mf.get("bbox", [0, 0, 0, 0])[3] - mf.get("bbox", [0, 0, 0, 0])[1]) > 0.04:
            sharp_raw.append(mf.get("eye_sharpness") or mf.get("sharpness") or 0.0)
        else:
            sharp_raw.append(a.get("subject_sharpness") or a.get("edge_sharpness") or a.get("sharpness") or 0.0)
    ranks = _rank(sharp_raw)
    med = float(np.median(sharp_raw)) if sharp_raw else 1.0
    for it, r, s in zip(items, ranks, sharp_raw):
        a = it.a
        mf = a.get("main_face") or {}
        fh = (mf.get("bbox", [0, 0, 0, 0])[3] - mf.get("bbox", [0, 0, 0, 0])[1]) if mf else 0.0
        it.feats = {
            "sharp": float(0.7 * r + 0.3 * min(1.0, s / max(med, 1e-6) / 2)),
            "sharp_abs_rel": float(s / max(med, 1e-6)),
            "motion": float(a.get("motion_anisotropy", 0.0)),
            "exposure": float(a.get("exposure_score", 0.5)),
            "eyes_open": float(mf.get("eyes_open")) if mf and mf.get("eyes_open") is not None else 0.75,
            "yaw": float(abs(mf.get("yaw") or 0.0)) if mf else 0.0,
            "face_size": float(fh),
            "aesthetic": float(a["aesthetic"]) if a.get("aesthetic") is not None else float("nan"),
            "subject_size": float(a.get("subject_fraction", 0.2)),
            "clip_hi": float(max(a.get("clip_hi", 0.0), a.get("raw_clip", 0.0))),
            "clip_lo": float(a.get("clip_lo", 0.0)),
            "cut": float(bool(a.get("faces_cut")) or bool(a.get("subject_cut"))),
            "has_face": float(bool(mf)),
            "median": float(a.get("median", 0.4)),
        }
    # Action-Moment: innerhalb des Shoots rangnormiert (robust gegen unterschiedliche Quellen)
    has_action = [it for it in items if it.a.get("action") is not None]
    if has_action:
        ar = _rank([float(it.a["action"]) for it in has_action])
        for it, r in zip(has_action, ar):
            it.feats["action"] = float(0.6 * r + 0.4 * float(it.a["action"]))
            it.feats["has_action"] = 1.0
    for it in items:
        it.feats.setdefault("action", 0.5)
        it.feats.setdefault("has_action", 0.0)
        sc = it.a.get("scene") or {}
        it.feats["sport"] = float(sum(sc.get(k, 0.0) for k in SPORT_SCENES)) if sc else 0.5
    # Ästhetik-Ersatz, wenn kein Modell: Motivgrösse + Schärfe + Belichtung
    for it in items:
        if np.isnan(it.feats["aesthetic"]):
            f = it.feats
            it.feats["aesthetic"] = float(0.4 * min(1.0, f["subject_size"] * 3) + 0.3 * f["sharp"] + 0.3 * f["exposure"])


def heuristic_score(f: dict[str, float], w: dict[str, float]) -> float:
    face_q = 0.6
    if f["has_face"]:
        face_q = f["eyes_open"] * (1.0 - min(1.0, max(0.0, f["yaw"] - 45) / 45) * 0.6)
    comp = 1.0 - 0.5 * f["cut"]
    motion_pen = max(0.0, f["motion"] - 0.5) * (1.0 - f["sharp"]) * 0.8
    s = (w.get("sharpness", 0.3) * f["sharp"] + w.get("face_quality", 0.25) * face_q
         + w.get("exposure", 0.15) * f["exposure"] + w.get("aesthetic", 0.2) * f["aesthetic"]
         + w.get("composition", 0.1) * comp - motion_pen)
    return float(s / max(sum(w.values()), 1e-6))


def hard_reasons(f: dict[str, float], cs: CullingSettings) -> list[str]:
    r = []
    if f["sharp"] < 0.12 and f["sharp_abs_rel"] < 0.45:
        r.append("unscharf")
    if f["motion"] > 0.6 and f["sharp"] < 0.3:
        r.append("bewegung")
    if f["has_face"] and f["face_size"] > 0.05 and f["eyes_open"] < 0.35:
        r.append("augen_zu")
    # Abgewandt/angeschnitten sind im Sport oft gerade die guten Bilder: kein harter Ausschluss mehr
    if f["clip_hi"] > 0.3:
        r.append("ueberbelichtet")
    if f["median"] < 0.06 and f["clip_lo"] > 0.5:
        r.append("unterbelichtet")
    return r


def detect_series(items: list[CullItem], cs: CullingSettings) -> None:
    sid = 0
    prev: CullItem | None = None
    for it in items:
        same = False
        if prev is not None and (it.t - prev.t) <= cs.series_gap_seconds:
            sim = float(it.emb @ prev.emb) if it.emb is not None and prev.emb is not None else 0.0
            same = sim >= cs.series_similarity or _phash_dist(it.phash, prev.phash) <= 22
        if not same:
            sid += 1
        it.series = sid
        prev = it


def _weaker_reason(it: CullItem, best: CullItem) -> str | None:
    """Warum ist ein Serienbild schlechter als das beste? (für ein verständliches Log)"""
    f, b = it.feats, best.feats
    if f["sharp"] < b["sharp"] - 0.25:
        return "bewegung" if f["motion"] > b["motion"] + 0.1 else "unscharf"
    if f["has_face"] and f["eyes_open"] < b["eyes_open"] - 0.3:
        return "augen_zu"
    if f["exposure"] < b["exposure"] - 0.3:
        return "ueberbelichtet" if f["clip_hi"] > b["clip_hi"] else "unterbelichtet"
    return None


def action_weight(items: list[CullItem], cs: CullingSettings) -> float:
    """Wie stark Action-Momente zählen: nur bei Sport/Konzert, im Highlight-Modus stärker."""
    if not items or not any(it.feats.get("has_action") for it in items):
        return 0.0
    base = cs.highlights_action_weight if cs.highlights else cs.action_weight
    sport = float(np.median([it.feats.get("sport", 0.5) for it in items]))
    return float(base * min(1.0, max(0.0, (sport - 0.15) / 0.35)))


def detect_scenes(items: list[CullItem], cs: CullingSettings, max_len: float = 15.0) -> None:
    """Highlight-Modus: Serien zu Spielszenen zusammenfassen (kurze Pausen, max. 15 s)."""
    sid = 0
    start = prev_t = 0.0
    prev_series: int | None = None
    for it in items:
        if prev_series is None:
            new = True
        elif it.series == prev_series:
            new = False  # eine Serie wird nie geteilt
        else:
            new = it.t - prev_t > cs.moment_gap_seconds or it.t - start > max_len
        if new:
            sid += 1
            start = it.t
        prev_t, prev_series = it.t, it.series
        it.series = sid


def near_duplicate(a: CullItem, b: CullItem, window: float = 900.0) -> bool:
    """Gleiches Motiv innerhalb von 15 Minuten (CLIP-Ähnlichkeit oder Bild-Hash)."""
    if abs(a.t - b.t) > window:
        return False
    if _phash_dist(a.phash, b.phash) <= 12:
        return True
    if a.emb is not None and b.emb is not None and len(a.emb) == len(b.emb) and len(a.emb) >= 512:
        return float(a.emb @ b.emb) >= 0.93          # nur für CLIP-Embeddings verlässlich
    return False


def select_clean(items: list[CullItem], cs: CullingSettings) -> None:
    """Nur Schlechte raus: technisch Misslungenes weg, aus jeder Serie (fast gleiches Motiv) nur die besten
    ``burst_keep`` Bilder, die sich auch wirklich unterscheiden. Keine Prozent-Quote."""
    by_series: dict[int, list[CullItem]] = {}
    for it in items:
        by_series.setdefault(it.series, []).append(it)
    kept_all: list[CullItem] = []
    for members in by_series.values():
        members.sort(key=lambda x: -x.score)
        kept: list[CullItem] = []
        for it in members:
            if it.hard:
                continue
            if not kept:
                it.best = True
                kept.append(it)
                continue
            if len(kept) >= max(1, cs.burst_keep):
                it.reasons.append(_weaker_reason(it, kept[0]) or "serie")
                continue
            same = any((it.emb is not None and k.emb is not None and float(it.emb @ k.emb) > cs.duplicate_similarity)
                       or _phash_dist(it.phash, k.phash) <= 6 for k in kept)
            if same or it.score < kept[0].score * 0.75:
                it.reasons.append(_weaker_reason(it, kept[0]) or ("duplikat" if same else "serie"))
                continue
            kept.append(it)
        kept_all += kept
    # Gleiches Bild über eine Seriengrenze hinweg (wenige Sekunden später): nur wenn fast identisch
    kept_all.sort(key=lambda x: -(x.score + (0.05 if x.best else 0.0)))
    final: list[CullItem] = []
    for it in kept_all:
        twin = next((k for k in final if abs(it.t - k.t) <= 5 and (
            _phash_dist(it.phash, k.phash) <= 6 or (it.emb is not None and k.emb is not None
                                                     and len(it.emb) == len(k.emb) and len(it.emb) >= 512
                                                     and float(it.emb @ k.emb) >= 0.97))), None)
        if twin is not None:
            it.reasons.append("duplikat")
            continue
        if cs.max_keep and len(final) >= int(cs.max_keep):
            it.reasons.append("strenge")
            continue
        it.keep = True
        final.append(it)
    for it in items:
        it.reasons = it.hard + [r for r in it.reasons if r not in it.hard]


def select(items: list[CullItem], cs: CullingSettings, w_action: float = 0.0) -> None:
    if cs.burst_keep and not cs.highlights:
        return select_clean(items, cs)
    n = len(items)
    ratio = min(cs.keep_ratio, cs.highlights_ratio) if cs.highlights else cs.keep_ratio
    target = max(1, int(round(ratio * n)))
    if cs.max_keep:
        target = max(1, min(target, int(cs.max_keep)))
    by_series: dict[int, list[CullItem]] = {}
    for it in items:
        by_series.setdefault(it.series, []).append(it)
    candidates: list[CullItem] = []
    for members in by_series.values():
        members.sort(key=lambda x: -x.score)
        kept: list[CullItem] = []
        for i, it in enumerate(members):
            if it.hard:
                continue
            if not kept:
                it.best = True
                kept.append(it)
                continue
            if cs.highlights:
                it.reasons.append(_weaker_reason(it, kept[0]) or "szene")
                continue
            dup = any((it.emb is not None and k.emb is not None and float(it.emb @ k.emb) > cs.duplicate_similarity)
                      or _phash_dist(it.phash, k.phash) <= 6 for k in kept)
            if dup or it.score < kept[0].score * 0.9:
                it.reasons.append(_weaker_reason(it, kept[0]) or ("duplikat" if dup else "serie"))
                continue
            kept.append(it)
        candidates += kept
    if cs.highlights and w_action > 0.1:
        # Nur echte Momente: Action deutlich über dem Shoot-Durchschnitt
        moment = [it for it in candidates if it.feats.get("action", 0.5) >= 0.55]
        for it in candidates:
            if it not in moment:
                it.reasons.append("kein_moment")
        candidates = moment
    # Serien-Beste zuerst, dann nach Score. Dabei keine Beinahe-Duplikate über Seriengrenzen hinweg:
    # dasselbe Motiv (Kuchen, Gruppenfoto ...) ein paar Sekunden später zählt nicht als neues Bild.
    candidates.sort(key=lambda x: (-(x.score + (0.05 if x.best else 0.0))))
    kept: list[CullItem] = []
    for it in candidates:
        if len(kept) >= target:
            it.reasons.append("strenge")
            continue
        twin = next((k for k in kept if near_duplicate(it, k)), None)
        if twin is not None:
            it.reasons.append("duplikat")
            continue
        it.keep = True
        kept.append(it)
    for it in items:
        it.reasons = it.hard + [r for r in it.reasons if r not in it.hard]


def ratings(items: list[CullItem], cs: CullingSettings) -> dict[int, int]:
    kept = sorted([it for it in items if it.keep], key=lambda x: -x.score)
    out: dict[int, int] = {}
    n = len(kept)
    for i, it in enumerate(kept):
        q = i / max(n, 1)
        stars = 5 if q < 0.10 else 4 if q < 0.35 else 3 if q < 0.70 else 2
        out[it.image_id] = max(stars, cs.min_rating_keep)
    for it in items:
        if not it.keep:
            out[it.image_id] = cs.reject_rating
    return out


def load_items(db: Database, shoot_id: int) -> list[CullItem]:
    rows = db.query(
        "SELECT i.id, i.capture_time, a.data, a.embedding, a.phash FROM images i "
        "LEFT JOIN analysis a ON a.image_id = i.id WHERE i.shoot_id=? ORDER BY i.capture_time, i.filename",
        (shoot_id,))
    items = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        if emb is not None:
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        items.append(CullItem(r["id"], r["capture_time"] or 0.0, json.loads(r["data"]) if r["data"] else {},
                              emb, r["phash"]))
    return items


def cull(items: list[CullItem], cs: CullingSettings, model: Any | None = None) -> list[CullItem]:
    build_features(items)
    w = action_weight(items, cs)
    for it in items:
        it.hard = ["unlesbar"] if it.a.get("read_error") or "sharpness" not in it.a else hard_reasons(it.feats, cs)
        if model is not None:
            tech = float(model.predict(it.feats))
        else:
            tech = heuristic_score(it.feats, cs.weights)
        it.feats["technical"] = tech
        it.score = (1.0 - w) * tech + w * it.feats["action"]
    detect_series(items, cs)
    if cs.highlights:
        detect_scenes(items, cs)
    select(items, cs, w)
    return items


@job("cull")
def cull_shoot(ctx: JobContext, shoot_id: int, keep_ratio: float | None = None, highlights: bool | None = None,
               max_keep: int | None = None, burst_keep: int | None = None) -> None:
    from .calibrate import CullingModel

    db = ctx.db
    s = load_settings()
    cs = s.culling
    shoot_settings = db.shoot_settings(shoot_id)
    # Einstellungen pro Shoot merken, damit ein erneutes Culling dieselbe Auswahl-Art nutzt
    culling_opts = {**shoot_settings.get("culling", {}),
                    **{k: v for k, v in (("keep_ratio", keep_ratio), ("highlights", highlights),
                                         ("max_keep", max_keep), ("burst_keep", burst_keep)) if v is not None}}
    if max_keep == 0:
        culling_opts.pop("max_keep", None)
    if culling_opts != shoot_settings.get("culling", {}):
        db.set_shoot_settings(shoot_id, {**shoot_settings, "culling": culling_opts})
    if culling_opts.get("keep_ratio") is not None:
        cs.keep_ratio = float(culling_opts["keep_ratio"])
    cs.highlights = bool(culling_opts.get("highlights", cs.highlights))
    if culling_opts.get("burst_keep") is not None:
        cs.burst_keep = int(culling_opts["burst_keep"])
    elif culling_opts.get("keep_ratio") is not None:
        cs.burst_keep = 0          # ältere Shoots: mit Prozent-Auswahl angelegt
    if culling_opts.get("max_keep"):
        cs.max_keep = int(culling_opts["max_keep"])
    from ..analysis import ensure_action

    ensure_action(ctx, shoot_id)
    items = load_items(db, shoot_id)
    ctx.set_total(len(items))
    shoot = db.one("SELECT profile FROM shoots WHERE id=?", (shoot_id,))
    model = CullingModel.load(shoot["profile"]) if shoot and shoot["profile"] else None
    cull(items, cs, model)
    stars = ratings(items, cs)
    dialect = Dialect.load()
    for i, it in enumerate(items):
        label = dialect.label("green") if it.best and it.keep else None
        db.set_culling(it.image_id, score=it.score, decision="keep" if it.keep else "reject",
                       rating=stars[it.image_id], label=label, reasons=it.reasons, series_id=it.series,
                       is_best=it.best)
        if i % 100 == 0:
            ctx.progress(i, "Culling")
    ctx.progress(len(items), f"{sum(it.keep for it in items)} von {len(items)} behalten")

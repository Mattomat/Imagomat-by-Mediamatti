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
}

FEATURE_NAMES = ["sharp", "motion", "exposure", "eyes_open", "yaw", "face_size", "aesthetic", "subject_size",
                 "clip_hi", "clip_lo", "cut", "has_face"]


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
    if f["has_face"] and f["yaw"] > 75 and f["face_size"] > 0.08:
        r.append("abgewandt")
    if f["cut"] and f["has_face"] and f["face_size"] > 0.08:
        r.append("angeschnitten")
    if f["clip_hi"] > 0.12:
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


def select(items: list[CullItem], cs: CullingSettings) -> None:
    n = len(items)
    target = max(1, int(round(cs.keep_ratio * n)))
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
            dup = any((it.emb is not None and k.emb is not None and float(it.emb @ k.emb) > cs.duplicate_similarity)
                      or _phash_dist(it.phash, k.phash) <= 6 for k in kept)
            if dup:
                it.reasons.append("duplikat")
                continue
            if it.score < kept[0].score * 0.9:
                it.reasons.append("serie")
                continue
            kept.append(it)
        candidates += kept
    # Serien-Beste zuerst, dann nach Score
    candidates.sort(key=lambda x: (-(x.score + (0.05 if x.best else 0.0))))
    for it in candidates[:target]:
        it.keep = True
    for it in candidates[target:]:
        it.reasons.append("strenge")
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
    for it in items:
        it.hard = hard_reasons(it.feats, cs)
        if model is not None:
            it.score = float(model.predict(it.feats))
        else:
            it.score = heuristic_score(it.feats, cs.weights)
    detect_series(items, cs)
    select(items, cs)
    return items


@job("cull")
def cull_shoot(ctx: JobContext, shoot_id: int, keep_ratio: float | None = None) -> None:
    from .calibrate import CullingModel

    db = ctx.db
    s = load_settings()
    cs = s.culling
    if keep_ratio is not None:
        cs.keep_ratio = float(keep_ratio)
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

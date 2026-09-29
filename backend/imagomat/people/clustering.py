"""Gesichter clustern und automatisch zuordnen (Job "people").

Ablauf pro Shoot:
1. Bekannte Personen wiedererkennen (Vergleich mit bestätigten Gesichtern).
2. Rückennummern: Nummer im Körperbereich eines Gesichts + Kader des Shoots -> Person.
   Nummern ohne Gesicht (Spieler von hinten) ordnen die Person trotzdem dem Bild zu.
3. Rest mit HDBSCAN clustern. In der UI gibst du Clustern einen Namen; ab dann erkennt die
   App die Person auch in künftigen Shoots.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from ..db import Database, blob_to_f32
from ..jobs import JobContext, job
from ..vision.faces import get_backend
from ..vision.segmentation import body_boxes
from .registry import MATCH_THRESHOLD, exemplars, match

log = logging.getLogger(__name__)


def cluster_embeddings(embs: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    if len(embs) < min_cluster_size:
        return np.full(len(embs), -1)
    from sklearn.cluster import HDBSCAN

    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean",
                   cluster_selection_epsilon=0.0, allow_single_cluster=True, copy=True).fit_predict(X)


def attach_noise(embs: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Rauschpunkte dem nächsten Cluster zuordnen, wenn sie ähnlich genug sind."""
    labels = labels.copy()
    ids = [c for c in np.unique(labels) if c >= 0]
    if not ids:
        return labels
    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([X[labels == c].mean(0) for c in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    for i in np.where(labels < 0)[0]:
        sims = cents @ X[i]
        j = int(np.argmax(sims))
        if sims[j] >= threshold:
            labels[i] = ids[j]
    return labels


def shoot_teams(db: Database, shoot_id: int) -> list[str]:
    row = db.one("SELECT settings FROM shoots WHERE id=?", (shoot_id,))
    try:
        return list((json.loads(row["settings"] or "{}") if row else {}).get("teams", []))
    except json.JSONDecodeError:
        return []


def roster_lookup(db: Database, teams: list[str]) -> dict[str, int]:
    """Rückennummer -> Person für die Teams des Shoots (eindeutige Nummern)."""
    if not teams:
        rows = db.query("SELECT id, number FROM persons WHERE number IS NOT NULL")
    else:
        q = ",".join("?" * len(teams))
        rows = db.query(f"SELECT id, number FROM persons WHERE number IS NOT NULL AND team IN ({q})", teams)
    seen: dict[str, list[int]] = {}
    for r in rows:
        seen.setdefault(str(r["number"]).lstrip("0") or "0", []).append(r["id"])
    return {n: ids[0] for n, ids in seen.items() if len(ids) == 1}


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def dominant_team(db: Database, shoot_id: int) -> str | None:
    """Welches Team ist in diesem Shoot am häufigsten erkannt? (für doppelte Nummern Herren/Frauen/U21)"""
    rows = db.query(
        "SELECT p.team, COUNT(*) n FROM faces f JOIN images i ON i.id=f.image_id JOIN persons p ON p.id=f.person_id "
        "WHERE i.shoot_id=? AND p.team IS NOT NULL AND f.assigned_by IN ('auto','confirmed','manual','number') "
        "GROUP BY p.team ORDER BY n DESC", (shoot_id,))
    total = sum(r["n"] for r in rows)
    if rows and total >= 3 and rows[0]["n"] / total >= 0.5:
        return rows[0]["team"]
    return None


class ShirtResolver:
    """Rückennummer oder Trikotname -> Person (eindeutig, sonst None)."""

    def __init__(self, db: Database, teams: list[str], dominant: str | None):
        from .registry import norm_name

        self.teams, self.dominant = teams, dominant
        self.by_number: dict[str, list[tuple[int, str | None]]] = {}
        self.names: list[tuple[str, int]] = []
        for r in db.query("SELECT id, name, team, number FROM persons"):
            if r["number"]:
                self.by_number.setdefault(str(r["number"]).lstrip("0") or "0", []).append((r["id"], r["team"]))
            parts = norm_name(r["name"]).split()
            if parts and len(parts[-1]) >= 4:
                self.names.append((parts[-1].upper(), r["id"]))
        self._norm = norm_name

    def number(self, text: str) -> int | None:
        cands = self.by_number.get(text.lstrip("0") or "0", [])
        if self.teams:
            cands = [c for c in cands if c[1] in self.teams]
        elif len(cands) > 1 and self.dominant:
            cands = [c for c in cands if c[1] == self.dominant] or cands
        return cands[0][0] if len(cands) == 1 else None

    def name(self, text: str) -> int | None:
        t = self._norm(text).upper().replace(" ", "")
        if len(t) < 4:
            return None
        exact = {pid for n, pid in self.names if n == t}
        if len(exact) == 1:
            return exact.pop()
        if exact:
            return None
        near = {pid for n, pid in self.names if len(t) >= 6 and _edit_distance(n, t) <= 1}
        return near.pop() if len(near) == 1 else None


MAX_AUTO_YAW = 55.0       # Gesichter von der Seite/hinten nicht automatisch benennen (unsicher)
MIN_AUTO_FACE = 0.035     # zu kleine Gesichter ebenso
FRONTAL_YAW = 35.0        # "klar von vorne": dann gewinnt das Gesicht gegen die Rückennummer


def _frontal(f) -> bool:
    bb = json.loads(f["bbox"])
    return abs(f["yaw"] or 0.0) <= FRONTAL_YAW and (bb[3] - bb[1]) >= 0.05


@job("people")
def people_job(ctx: JobContext, shoot_id: int, threshold: float | None = None) -> None:
    from ..analysis import ensure_ocr

    db = ctx.db
    ensure_ocr(ctx, shoot_id)
    thr = threshold if threshold is not None else MATCH_THRESHOLD.get(get_backend().name, 0.45)
    faces = db.query(
        "SELECT f.id, f.image_id, f.bbox, f.embedding, f.person_id, f.assigned_by, f.yaw FROM faces f "
        "JOIN images i ON i.id=f.image_id WHERE i.shoot_id=?", (shoot_id,))
    ctx.set_total(len(faces) + 2)
    ex, ids = exemplars(db)
    # 1. Wiedererkennen (nur gut sichtbare Gesichter)
    auto = 0
    for f in faces:
        if f["assigned_by"] in ("manual", "confirmed") or f["embedding"] is None:
            continue
        bb = json.loads(f["bbox"])
        clear = abs(f["yaw"] or 0.0) <= MAX_AUTO_YAW and (bb[3] - bb[1]) >= MIN_AUTO_FACE
        pid = match(blob_to_f32(f["embedding"]), ex, ids, thr)[0] if clear else None
        with db.tx() as c:
            c.execute("UPDATE faces SET person_id=?, assigned_by=? WHERE id=?",
                      (pid, "auto" if pid else None, f["id"]))
        auto += pid is not None
    ctx.progress(len(faces), f"{auto} Gesichter wiedererkannt")
    # 2. Rückennummern und Trikotnamen: stärker als das Gesicht
    resolver = ShirtResolver(db, shoot_teams(db, shoot_id), dominant_team(db, shoot_id))
    by_image: dict[int, list] = {}
    for f in db.query("SELECT f.* FROM faces f JOIN images i ON i.id=f.image_id WHERE i.shoot_id=?", (shoot_id,)):
        by_image.setdefault(f["image_id"], []).append(f)
    texts = db.query("SELECT n.* FROM numbers n JOIN images i ON i.id=n.image_id WHERE i.shoot_id=?", (shoot_id,))
    via_shirt = 0
    for n in texts:
        text = str(n["text"])
        pid = resolver.name(text[1:]) if text.startswith("@") else resolver.number(text)
        with db.tx() as c:
            c.execute("UPDATE numbers SET person_id=? WHERE id=?", (pid, n["id"]))
        if pid is None:
            continue
        nb = json.loads(n["bbox"])
        ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
        img_faces = by_image.get(n["image_id"], [])
        boxes = body_boxes([tuple(json.loads(f["bbox"])) for f in img_faces], 1.5)
        for f, (x0, y0, x1, y1) in zip(img_faces, boxes):
            if x0 <= ncx <= x1 and y0 <= ncy <= y1 and f["assigned_by"] not in ("manual", "confirmed"):
                if f["person_id"] not in (None, pid) and f["assigned_by"] == "auto" and _frontal(f):
                    # Klar von vorne erkanntes Gesicht zählt mehr als eine (evtl. verdeckte) Nummer
                    log.info("Gesicht (%s) und Trikot (%s) widersprechen sich, Gesicht gewinnt", f["person_id"], pid)
                    break
                with db.tx() as c:
                    c.execute("UPDATE faces SET person_id=?, assigned_by='number' WHERE id=?", (pid, f["id"]))
                via_shirt += 1
                break
    ctx.progress(len(faces) + 1, f"{via_shirt} Personen über Rückennummer/Trikotname")
    # 3. Clustern der unbekannten Gesichter
    rest = db.query(
        "SELECT f.id, f.embedding FROM faces f JOIN images i ON i.id=f.image_id "
        "WHERE i.shoot_id=? AND f.person_id IS NULL AND f.embedding IS NOT NULL", (shoot_id,))
    if rest:
        embs = np.stack([blob_to_f32(r["embedding"]) for r in rest])
        labels = attach_noise(embs, cluster_embeddings(embs), thr)
        with db.tx() as c:
            for r, lab in zip(rest, labels):
                c.execute("UPDATE faces SET cluster_id=? WHERE id=?", (int(lab) if lab >= 0 else None, r["id"]))
    ctx.progress(len(faces) + 2, "Personen fertig")

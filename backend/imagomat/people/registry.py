"""Personen-Datenbank (global über alle Shoots) und Zuordnung von Gesichtern.

Jede Person hat beliebig viele bestätigte Gesichter. Neue Gesichter werden über die
höchste Kosinus-Ähnlichkeit zu diesen Beispielen zugeordnet. Die Schwelle hängt vom
Embedding-Modell ab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import load_settings
from ..db import Database, blob_to_f32

# Kosinus-Schwellen pro Backend (Embeddings normiert). Konservativ gewählt:
# lieber ein Gesicht unbenannt lassen als falsch benennen.
MATCH_THRESHOLD = {"insightface": 0.42, "yunet": 0.40, "haar": 0.97}
MAX_EXEMPLARS = 60


@dataclass
class Person:
    id: int
    name: str
    team: str | None
    number: str | None
    keyword: str


def keyword_for(name: str, team: str | None) -> str:
    root = load_settings().keywords.people_root
    return f"{root}|{team}|{name}" if team else f"{root}|{name}"


def upsert_person(db: Database, name: str, team: str | None = None, number: str | None = None) -> int:
    name = name.strip()
    with db.tx() as c:
        row = c.execute("SELECT id FROM persons WHERE name=? AND IFNULL(team,'')=IFNULL(?,'')", (name, team)).fetchone()
        if row:
            if number is not None:
                c.execute("UPDATE persons SET number=? WHERE id=?", (number, row["id"]))
            return int(row["id"])
        cur = c.execute("INSERT INTO persons(name, keyword, team, number, created_at) VALUES(?,?,?,?,?)",
                        (name, keyword_for(name, team), team, number, time.time()))
        return int(cur.lastrowid)


def persons(db: Database) -> list[Person]:
    return [Person(r["id"], r["name"], r["team"], r["number"], r["keyword"] or keyword_for(r["name"], r["team"]))
            for r in db.query("SELECT * FROM persons ORDER BY team, CAST(number AS INTEGER), name")]


def exemplars(db: Database) -> tuple[np.ndarray, np.ndarray]:
    """Bestätigte Gesichter aller Personen (Embeddings, Person-IDs)."""
    rows = db.query(
        "SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL AND embedding IS NOT NULL "
        "AND assigned_by IN ('manual','number','confirmed') ORDER BY id DESC")
    per: dict[int, list[np.ndarray]] = {}
    for r in rows:
        lst = per.setdefault(r["person_id"], [])
        if len(lst) < MAX_EXEMPLARS:
            lst.append(blob_to_f32(r["embedding"]))
    embs, ids = [], []
    for pid, lst in per.items():
        for e in lst:
            embs.append(e / (np.linalg.norm(e) + 1e-9))
            ids.append(pid)
    if not embs:
        return np.zeros((0, 1), np.float32), np.zeros(0, int)
    return np.stack(embs), np.asarray(ids)


def match(emb: np.ndarray, ex: np.ndarray, ids: np.ndarray, threshold: float) -> tuple[int | None, float]:
    if len(ids) == 0 or emb is None or ex.shape[1] != emb.shape[0]:
        return None, 0.0
    sims = ex @ (emb / (np.linalg.norm(emb) + 1e-9))
    best: dict[int, float] = {}
    for pid, s in zip(ids, sims):
        best[int(pid)] = max(best.get(int(pid), -1.0), float(s))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    pid, s = ranked[0]
    # Abstand zur zweitbesten Person verlangen, sonst unsicher
    if s >= threshold and (len(ranked) == 1 or s - ranked[1][1] > 0.05):
        return pid, s
    return None, s


def assign_face(db: Database, face_id: int, person_id: int | None, how: str = "manual") -> None:
    with db.tx() as c:
        c.execute("UPDATE faces SET person_id=?, assigned_by=? WHERE id=?", (person_id, how if person_id else None,
                                                                             face_id))


def assign_cluster(db: Database, shoot_id: int, cluster_id: int, person_id: int) -> int:
    """Alle Gesichter eines Clusters einer Person zuordnen (UI: Cluster benennen)."""
    with db.tx() as c:
        cur = c.execute(
            "UPDATE faces SET person_id=?, assigned_by='manual' WHERE cluster_id=? AND image_id IN "
            "(SELECT id FROM images WHERE shoot_id=?)", (person_id, cluster_id, shoot_id))
        return cur.rowcount


def image_people(db: Database, image_id: int) -> list[Person]:
    rows = db.query(
        "SELECT DISTINCT p.* FROM persons p WHERE p.id IN ("
        " SELECT person_id FROM faces WHERE image_id=? AND person_id IS NOT NULL"
        " UNION SELECT person_id FROM numbers WHERE image_id=? AND person_id IS NOT NULL)", (image_id, image_id))
    return [Person(r["id"], r["name"], r["team"], r["number"], r["keyword"] or keyword_for(r["name"], r["team"]))
            for r in rows]

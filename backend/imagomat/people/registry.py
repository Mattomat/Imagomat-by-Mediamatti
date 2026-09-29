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
    kw = load_settings().keywords
    if kw.person_keyword_style == "name":
        return name
    return f"{kw.people_root}|{team}|{name}" if team else f"{kw.people_root}|{name}"


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
    return [Person(r["id"], r["name"], r["team"], r["number"], keyword_for(r["name"], r["team"]))
            for r in db.query("SELECT * FROM persons ORDER BY team IS NULL, team, number IS NULL, "
                                "CAST(number AS INTEGER), name")]


# Beispiele einer Person, die keinem anderen ihrer Beispiele ähneln, sind vermutlich falsch benannt
# (z. B. eine gemischte Gruppe auf einmal benannt). Sie würden sonst fremde Gesichter anziehen.
EXEMPLAR_SUPPORT = {"insightface": 0.25, "yunet": 0.25}


def _clean(lst: list[np.ndarray], min_support: float | None) -> list[np.ndarray]:
    if min_support is None or len(lst) < 4:
        return lst
    L = np.stack(lst)
    sims = L @ L.T
    np.fill_diagonal(sims, -1.0)
    k = min(3, len(lst) - 1)
    support = np.sort(sims, axis=1)[:, -k:].mean(axis=1)     # Ähnlichkeit zu den k nächsten eigenen Beispielen
    return [e for e, s in zip(lst, support) if s >= min_support]


def exemplars(db: Database, backend: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Bestätigte Gesichter aller Personen (Embeddings, Person-IDs), ohne offensichtliche Ausreisser."""
    rows = db.query(
        "SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL AND embedding IS NOT NULL "
        "AND assigned_by IN ('manual','confirmed') ORDER BY id DESC")   # nur sichere Beispiele, nie Trikot
    per: dict[int, list[np.ndarray]] = {}
    dim = None
    for r in rows:
        e = blob_to_f32(r["embedding"])
        dim = dim or e.shape[0]
        if e.shape[0] != dim:
            continue                      # Embeddings eines anderen Modells nicht mischen
        lst = per.setdefault(r["person_id"], [])
        if len(lst) < MAX_EXEMPLARS:
            lst.append(e / (np.linalg.norm(e) + 1e-9))
    embs, ids = [], []
    for pid, lst in per.items():
        for e in _clean(lst, EXEMPLAR_SUPPORT.get(backend or "")):
            embs.append(e)
            ids.append(pid)
    if not embs:
        return np.zeros((0, 1), np.float32), np.zeros(0, int)
    return np.stack(embs), np.asarray(ids)


def person_scores(emb: np.ndarray, ex: np.ndarray, ids: np.ndarray) -> dict[int, float]:
    """Ähnlichkeit eines Gesichts zu jeder Person. Robust: Mittel der zwei besten Beispiele, damit ein
    einzelnes falsches Beispiel nicht reicht."""
    if len(ids) == 0 or emb is None or ex.shape[1] != emb.shape[0]:
        return {}
    sims = ex @ (emb / (np.linalg.norm(emb) + 1e-9))
    per: dict[int, list[float]] = {}
    for pid, s in zip(ids, sims):
        per.setdefault(int(pid), []).append(float(s))
    out = {}
    for pid, lst in per.items():
        lst.sort(reverse=True)
        out[pid] = (lst[0] + lst[1]) / 2 if len(lst) >= 2 else lst[0]
    return out


def best_similarity(emb: np.ndarray, ex: np.ndarray, ids: np.ndarray, pid: int) -> float | None:
    """Höchste Ähnlichkeit zu den Beispielen einer bestimmten Person (None: keine Beispiele)."""
    mask = ids == pid
    if not mask.any() or emb is None or ex.shape[1] != emb.shape[0]:
        return None
    return float((ex[mask] @ (emb / (np.linalg.norm(emb) + 1e-9))).max())


def match(emb: np.ndarray, ex: np.ndarray, ids: np.ndarray, threshold: float) -> tuple[int | None, float]:
    scores = person_scores(emb, ex, ids)
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
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
    return [Person(r["id"], r["name"], r["team"], r["number"], keyword_for(r["name"], r["team"])) for r in rows]


def norm_name(name: str) -> str:
    """Für den Namensabgleich: ohne Akzente, Bindestriche und Gross/Klein ("Théo" == "Theo")."""
    import unicodedata

    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.replace("-", " ").casefold().split())


def find_person(db: Database, name: str, aliases: list[str] | None = None) -> int | None:
    """Bestehende Person über den Namen (oder Varianten) finden, unabhängig vom Team."""
    wanted = {norm_name(n) for n in [name, *(aliases or [])] if n}
    for r in db.query("SELECT id, name FROM persons ORDER BY id"):
        if norm_name(r["name"]) in wanted:
            return int(r["id"])
    return None


def merge_persons(db: Database, source: int, target: int) -> None:
    """Person ``source`` in ``target`` aufgehen lassen (Gesichter und Nummern wandern mit)."""
    if source == target:
        return
    with db.tx() as c:
        c.execute("UPDATE faces SET person_id=? WHERE person_id=?", (target, source))
        c.execute("UPDATE numbers SET person_id=? WHERE person_id=?", (target, source))
        src = c.execute("SELECT number, team FROM persons WHERE id=?", (source,)).fetchone()
        if src:
            c.execute("UPDATE persons SET number=COALESCE(number, ?), team=COALESCE(team, ?) WHERE id=?",
                      (src["number"], src["team"], target))
        c.execute("DELETE FROM persons WHERE id=?", (source,))


_UNSET = object()


def update_person(db: Database, pid: int, name: str | None = None, team: object = _UNSET,
                  number: object = _UNSET) -> int:
    """Name, Team oder Nummer ändern. Gibt es danach schon eine Person mit gleichem Namen und Team,
    werden beide zusammengeführt. Rückgabe: ID der (verbleibenden) Person."""
    row = db.one("SELECT * FROM persons WHERE id=?", (pid,))
    if row is None:
        raise KeyError(pid)
    new_name = (name or row["name"]).strip()
    new_team = row["team"] if team is _UNSET else ((str(team).strip() or None) if team else None)
    new_number = row["number"] if number is _UNSET else ((str(number).strip() or None) if number else None)
    other = db.one("SELECT id FROM persons WHERE name=? AND IFNULL(team,'')=IFNULL(?,'') AND id<>?",
                   (new_name, new_team, pid))
    if other is None and name and norm_name(name) != norm_name(row["name"]):
        # Umbenannt auf eine bereits bekannte Person (z. B. Tippfehler korrigiert) -> zusammenführen
        other_id = next((int(r["id"]) for r in db.query("SELECT id, name FROM persons WHERE id<>?", (pid,))
                         if norm_name(r["name"]) == norm_name(new_name)), None)
        if other_id is not None:
            other = {"id": other_id}
            tgt = db.one("SELECT team, number FROM persons WHERE id=?", (other_id,))
            new_team = new_team if team is not _UNSET else (tgt["team"] or new_team)
            new_number = new_number if number is not _UNSET else (tgt["number"] or new_number)
    if other:
        merge_persons(db, pid, int(other["id"]))
        pid = int(other["id"])
    with db.tx() as c:
        c.execute("UPDATE persons SET name=?, team=?, number=?, keyword=? WHERE id=?",
                  (new_name, new_team, new_number, keyword_for(new_name, new_team), pid))
    return pid

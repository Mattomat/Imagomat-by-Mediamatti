"""Gesichter clustern und automatisch zuordnen (Job "people").

Ablauf pro Shoot:
1. Bekannte Personen wiedererkennen (Vergleich mit bestätigten Gesichtern).
2. Rückennummer/Trikotname nur als Notlösung: nur wenn im Bild gar kein Gesicht erkannt wurde, und nur Spieler des
   Shoot-Teams (optional nur auf rot/weiss/schwarzen Trikots, Einstellung shirt_color_check). Andere Teams zählen nie.
3. Rest clustern. In der UI gibst du Clustern einen Namen; ab dann erkennt die
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
from .registry import MATCH_THRESHOLD, best_similarity, exemplars, match, rejected_pairs

log = logging.getLogger(__name__)


def cluster_embeddings(embs: np.ndarray, similarity: float, min_size: int = 2) -> np.ndarray:
    """Gesichter gruppieren, die sich wirklich ähnlich sind (lieber zu viele kleine Gruppen als eine
    grosse gemischte). Average-Linkage: ein Gesicht muss im Schnitt allen der Gruppe ähneln, nicht nur
    einem einzelnen (sonst entstehen Ketten über viele verschiedene Personen)."""
    n = len(embs)
    if n < min_size:
        return np.full(n, -1)
    from sklearn.cluster import AgglomerativeClustering

    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1.0 - similarity).fit_predict(X)
    counts = np.bincount(labels)
    return np.where(counts[labels] >= min_size, labels, -1)


# Ähnlichkeit für Gruppen "Wer ist das?": deutlich strenger als fürs Wiedererkennen bekannter Personen
CLUSTER_SIMILARITY = {"insightface": 0.55, "yunet": 0.52, "haar": 0.985}


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
    """Welches Team ist in diesem Shoot am häufigsten erkannt? (für doppelte Nummern Herren/Frauen/U21)
    Nur Gesichter zählen: Trikot-Zuordnungen würden sich sonst selbst bestätigen."""
    rows = db.query(
        "SELECT p.team, COUNT(*) n FROM faces f JOIN images i ON i.id=f.image_id JOIN persons p ON p.id=f.person_id "
        "WHERE i.shoot_id=? AND p.team IS NOT NULL AND f.assigned_by IN ('auto','confirmed','manual') "
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
            if parts and len(parts[-1]) >= 4 and (not self.scope or r["team"] in self.scope):
                self.names.append((parts[-1].upper(), r["id"]))
        self._norm = norm_name

    @property
    def scope(self) -> list[str]:
        """Teams, deren Nummern in diesem Shoot gelten (leer = unbekannt)."""
        return self.teams or ([self.dominant] if self.dominant else [])

    def number(self, text: str) -> int | None:
        cands = self.by_number.get(text.lstrip("0") or "0", [])
        if self.scope:
            # Nur das Team des Shoots. Fehlt die Nummer dort, ist es ein Gegner oder ein Lesefehler,
            # nie eine Spielerin/ein Spieler eines anderen eigenen Teams.
            cands = [c for c in cands if c[1] in self.scope]
        return cands[0][0] if len(cands) == 1 else None

    def name_exact(self, text: str) -> int | None:
        """Nur exakter Treffer auf einen Nachnamen im Team (für Namen ohne lesbare Nummer daneben)."""
        t = self._norm(text).upper().replace(" ", "")
        if len(t) < 5:
            return None
        hits = {pid for n, pid in self.names if n == t}
        return hits.pop() if len(hits) == 1 else None

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


def plausible_back_names(texts, ignored: set[str]) -> set[int]:
    """Welche gelesenen Wörter sind Spielernamen (Rücken) und keine Sponsoren?

    - der Name steht auf dem Rücken direkt über/unter der Rückennummer (Brust-Sponsoren nicht)
    - ein Wort, das bei drei oder mehr verschiedenen Nummern auftaucht, ist ein Sponsor
    - Wörter aus der Ignorier-Liste (Einstellungen) zählen nie
    Rückgabe: IDs der Zeilen in ``numbers``, die als Name gelten dürfen."""
    by_img: dict[int, list] = {}
    for n in texts:
        if n["bbox"] and not str(n["text"]).startswith("#"):
            by_img.setdefault(n["image_id"], []).append(n)
    near: dict[int, tuple[str, set[str]]] = {}
    word_numbers: dict[str, set[str]] = {}
    word_images: dict[str, set[int]] = {}
    for img, rows in by_img.items():
        nums = [(str(r["text"]), json.loads(r["bbox"])) for r in rows if not str(r["text"]).startswith("@")]
        for r in rows:
            t = str(r["text"])
            if not t.startswith("@") or t[1:].upper() in ignored:
                continue
            wb = json.loads(r["bbox"])
            close = set()
            for num, nb in nums:
                nh = max(nb[3] - nb[1], 1e-3)
                overlap = min(wb[2], nb[2]) - max(wb[0], nb[0])
                gap = max(nb[1] - wb[3], wb[1] - nb[3], 0.0)       # vertikaler Abstand
                if overlap > -0.5 * nh and gap <= 1.5 * nh:
                    close.add(num)
            if close:
                word = t[1:].upper()
                near[r["id"]] = (word, close)
                word_numbers.setdefault(word, set()).update(close)
                word_images.setdefault(word, set()).add(img)
    sponsors = {w for w, nums in word_numbers.items()
                if len(nums) >= 3 or (len(nums) >= 2 and len(word_images[w]) >= 5)}
    return {rid for rid, (word, _nums) in near.items() if word not in sponsors}


# Wie in den ersten Versionen: auch kleine Gesichter (Totale) und Halbprofile wiedererkennen.
# Seitliche Gesichter (Profil) zählen auch, brauchen aber eine deutlich höhere Ähnlichkeit.
MAX_AUTO_YAW = 90.0
PROFILE_YAW = 55.0
PROFILE_EXTRA = 0.08
# Serien: Person von einem Bild aufs nächste übertragen, wenn das Gesicht fast an derselben Stelle ist
TRACK_GAP_SECONDS = 2.5
TRACK_MIN_IOU = 0.25
MIN_AUTO_FACE = 0.012
FRONTAL_YAW = 35.0
# Trikot sagt Person X, das Gesicht ähnelt X aber überhaupt nicht -> Trikot verwerfen
SHIRT_VETO = {"insightface": 0.18, "yunet": 0.22, "haar": 0.5}
# Rückennummer: so sicher und so gross muss sie gelesen sein
SHIRT_MIN_CONF = 0.6
SHIRT_MIN_HEIGHT = 0.02
# Anteil roter, weisser oder schwarzer Pixel rund um die Nummer (FC-Winterthur-Trikots)
TEAM_COLOR_MIN = 0.6


def _frontal(f) -> bool:
    bb = json.loads(f["bbox"])
    return abs(f["yaw"] or 0.0) <= FRONTAL_YAW and (bb[3] - bb[1]) >= 0.05


def _clear(f) -> bool:
    bb = json.loads(f["bbox"])
    return abs(f["yaw"] or 0.0) <= MAX_AUTO_YAW and (bb[3] - bb[1]) >= MIN_AUTO_FACE


def team_color_share(img: np.ndarray, bbox) -> float | None:
    """Wie viel des Trikots rund um die Nummer ist rot, weiss oder schwarz? (z. B. Heim rot, auswärts
    weiss/schwarz). Blaue, gelbe, grüne ... Gegner-Trikots fallen so heraus."""
    import cv2

    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    X0, X1 = int(max(0, x0 - bw * 0.8) * w), int(min(1, x1 + bw * 0.8) * w)
    Y0, Y1 = int(max(0, y0 - bh * 0.4) * h), int(min(1, y1 + bh * 0.6) * h)
    if X1 - X0 < 6 or Y1 - Y0 < 6:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(img[Y0:Y1, X0:X1]), cv2.COLOR_RGB2HSV).astype(np.int32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((H <= 10) | (H >= 165)) & (S >= 90) & (V >= 50)
    white = (S <= 50) & (V >= 150)
    black = V <= 60
    return float((red | white | black).mean())


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def track_series(db: Database, shoot_id: int, rejected: set[tuple[int, int]] | None = None) -> int:
    """Innerhalb einer Serie (Bilder wenige Sekunden auseinander) die Person eines erkannten Gesichts auf das
    Gesicht an fast derselben Stelle im Nachbarbild übertragen, vorwärts und rückwärts. So bleiben Spieler
    benannt, wenn sie sich zur Seite oder nach hinten drehen."""
    rejected = rejected or set()
    rows = db.query(
        "SELECT f.id, f.image_id, f.bbox, f.person_id, f.assigned_by, i.capture_time FROM faces f "
        "JOIN images i ON i.id=f.image_id WHERE i.shoot_id=? ORDER BY i.capture_time, i.filename", (shoot_id,))
    frames: list[tuple[float | None, int, list[dict]]] = []
    for r in rows:
        d = dict(r)
        d["box"] = json.loads(r["bbox"])
        if frames and frames[-1][1] == r["image_id"]:
            frames[-1][2].append(d)
        else:
            frames.append((r["capture_time"], r["image_id"], [d]))
    changed: dict[int, int] = {}

    def propagate(order: list[int]) -> None:
        for a, b in zip(order, order[1:]):
            ta, _ia, fa = frames[a]
            tb, ib, fb = frames[b]
            if ta is None or tb is None or abs(tb - ta) > TRACK_GAP_SECONDS:
                continue                          # ohne Aufnahmezeit keine Serie (z. B. JPGs ohne EXIF)
            present = {f["person_id"] for f in fb if f["person_id"]}
            for f in fb:
                if f["person_id"] or f["assigned_by"] == "ignored":
                    continue
                best, best_iou = None, TRACK_MIN_IOU
                for g in fa:
                    if g["person_id"] and g["person_id"] not in present and (ib, g["person_id"]) not in rejected:
                        iou = _iou(f["box"], g["box"])
                        if iou >= best_iou:
                            best, best_iou = g, iou
                if best is not None:
                    f["person_id"], f["assigned_by"] = best["person_id"], "track"
                    present.add(best["person_id"])
                    changed[f["id"]] = best["person_id"]

    idx = list(range(len(frames)))
    propagate(idx)
    propagate(idx[::-1])
    if changed:
        with db.tx() as c:
            for fid, pid in changed.items():
                c.execute("UPDATE faces SET person_id=?, assigned_by='track' WHERE id=? AND person_id IS NULL",
                          (pid, fid))
    return len(changed)


@job("people")
def people_job(ctx: JobContext, shoot_id: int, threshold: float | None = None) -> None:
    from ..analysis import ensure_ocr, load_cached_preview

    db = ctx.db
    ensure_ocr(ctx, shoot_id)
    backend = get_backend().name
    thr = threshold if threshold is not None else MATCH_THRESHOLD.get(backend, 0.45)
    faces = db.query(
        "SELECT f.id, f.image_id, f.bbox, f.embedding, f.person_id, f.assigned_by, f.yaw FROM faces f "
        "JOIN images i ON i.id=f.image_id WHERE i.shoot_id=?", (shoot_id,))
    ctx.set_total(len(faces) + 2)
    ex, ids = exemplars(db, backend, shoot_teams(db, shoot_id))      # nur das gewählte Team
    rejected = rejected_pairs(db, shoot_id)
    not_on: dict[int, set[int]] = {}
    for img, pid in rejected:
        not_on.setdefault(img, set()).add(pid)
    # 1. Gesichter: die Hauptsache
    auto = 0
    for f in faces:
        if f["assigned_by"] in ("manual", "confirmed", "ignored") or f["embedding"] is None:
            continue
        t_face = thr + (PROFILE_EXTRA if abs(f["yaw"] or 0.0) > PROFILE_YAW else 0.0)
        pid = match(blob_to_f32(f["embedding"]), ex, ids, t_face, not_on.get(f["image_id"]))[0] if _clear(f) else None
        with db.tx() as c:
            c.execute("UPDATE faces SET person_id=?, assigned_by=? WHERE id=?",
                      (pid, "auto" if pid else None, f["id"]))
        auto += pid is not None
    ctx.progress(len(faces), f"{auto} Gesichter wiedererkannt")
    # 2. Trikot (Nummer/Name) nur als Notlösung: wenn im Bild KEIN Gesicht erkannt wurde, nur Spieler
    #    des Teams dieses Shoots und nur auf roten, weissen oder schwarzen Trikots.
    resolver = ShirtResolver(db, shoot_teams(db, shoot_id), dominant_team(db, shoot_id))
    by_image: dict[int, list] = {}
    for f in db.query("SELECT f.* FROM faces f JOIN images i ON i.id=f.image_id WHERE i.shoot_id=?", (shoot_id,)):
        by_image.setdefault(f["image_id"], []).append(f)
    texts = db.query("SELECT n.* FROM numbers n JOIN images i ON i.id=n.image_id WHERE i.shoot_id=?", (shoot_id,))
    from ..config import load_settings

    ignored = {w.strip().upper() for w in load_settings().ignored_shirt_words if w.strip()}
    back_names = plausible_back_names(texts, ignored)
    with_person = {img for img, fs in by_image.items() if any(f["person_id"] for f in fs)}
    rows_by_id = {r["id"]: r for r in db.images(shoot_id)}
    previews: dict[int, np.ndarray | None] = {}

    def preview(image_id: int) -> np.ndarray | None:
        if image_id not in previews:
            try:
                previews[image_id] = load_cached_preview(rows_by_id[image_id])
            except Exception:  # noqa: BLE001 - ohne Bild keine Farbprüfung -> kein Trikot-Treffer
                previews[image_id] = None
        return previews[image_id]

    evidence: dict[tuple[int, int], list] = {}
    if resolver.scope:
        for n in texts:
            text = str(n["text"])
            if text.startswith("#") or not n["bbox"] or n["image_id"] in with_person:
                continue
            if text.startswith("@"):
                if n["id"] in back_names:
                    pid = resolver.name(text[1:])
                elif text[1:].upper() not in ignored:
                    pid = resolver.name_exact(text[1:])      # Name klar lesbar, Nummer nicht gelesen
                else:
                    pid = None
            else:
                bb = json.loads(n["bbox"])
                ok = (n["confidence"] or 0) >= SHIRT_MIN_CONF and bb[3] - bb[1] >= SHIRT_MIN_HEIGHT
                pid = resolver.number(text) if ok else None
            if pid is not None and (n["image_id"], pid) not in rejected:
                evidence.setdefault((n["image_id"], pid), []).append(n)
    accepted: dict[int, int] = {}                 # numbers.id -> Person
    via_shirt = vetoed = 0
    color_check = load_settings().shirt_color_check
    for (image_id, pid), rows in evidence.items():
        img = preview(image_id) if color_check else None
        if color_check and (img is None or not any((team_color_share(img, json.loads(n["bbox"])) or 0)
                                                   >= TEAM_COLOR_MIN for n in rows)):
            vetoed += 1
            continue
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (json.loads(n["bbox"]) for n in rows)]
        img_faces = by_image.get(image_id, [])
        boxes = body_boxes([tuple(json.loads(f["bbox"])) for f in img_faces], 1.5)
        owner = next((f for f, (x0, y0, x1, y1) in zip(img_faces, boxes)
                      if any(x0 <= cx <= x1 and y0 <= cy <= y1 for cx, cy in centers)), None)
        if owner is not None and owner["assigned_by"] == "ignored":
            continue
        if owner is not None and owner["embedding"] is not None and _clear(owner):
            # Gesicht sichtbar, aber nicht erkannt: darf der Trikot-Person nicht klar widersprechen
            s = best_similarity(blob_to_f32(owner["embedding"]), ex, ids, pid)
            if s is not None and s < SHIRT_VETO.get(backend, 0.2):
                vetoed += 1
                continue
        accepted.update({n["id"]: pid for n in rows})
        if owner is not None:
            with db.tx() as c:
                c.execute("UPDATE faces SET person_id=?, assigned_by='number' WHERE id=?", (pid, owner["id"]))
        via_shirt += 1
    with db.tx() as c:
        for n in texts:
            if not str(n["text"]).startswith("#"):
                c.execute("UPDATE numbers SET person_id=? WHERE id=?", (accepted.get(n["id"]), n["id"]))
    for r in rows_by_id.values():
        if db.get_analysis(r["id"]).get("people_check"):
            db.update_analysis(r["id"], {"people_check": []})
    log.info("Personen Shoot %s: %d Gesichter erkannt, %d über Trikot, %d Trikot-Treffer verworfen, Team %s",
             shoot_id, auto, via_shirt, vetoed, resolver.scope or "unbekannt")
    ctx.progress(len(faces) + 1, f"{auto} Gesichter, {via_shirt} über Rückennummer/Trikotname")
    # 2b. Serien-Verfolgung: Spieler, der sich wegdreht, behält seinen Namen
    tracked = track_series(db, shoot_id, rejected)
    if tracked:
        log.info("Personen Shoot %s: %d Gesichter über Serien-Verfolgung benannt", shoot_id, tracked)
    # 3. Clustern der unbekannten Gesichter
    rest = db.query(
        "SELECT f.id, f.embedding FROM faces f JOIN images i ON i.id=f.image_id "
        "WHERE i.shoot_id=? AND f.person_id IS NULL AND f.embedding IS NOT NULL "
        "AND IFNULL(f.assigned_by,'') <> 'ignored'", (shoot_id,))
    if rest:
        embs = np.stack([blob_to_f32(r["embedding"]) for r in rest])
        sim = (threshold + 0.05) if threshold is not None else CLUSTER_SIMILARITY.get(get_backend().name, 0.55)
        labels = cluster_embeddings(embs, sim) if len(rest) <= 6000 else np.full(len(rest), -1)
        with db.tx() as c:
            for r, lab in zip(rest, labels):
                c.execute("UPDATE faces SET cluster_id=? WHERE id=?", (int(lab) if lab >= 0 else None, r["id"]))
    ctx.progress(len(faces) + 2, "Personen fertig")

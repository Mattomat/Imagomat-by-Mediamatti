"""Action-Momente (Pose-Merkmale) und Highlight-Culling."""

import numpy as np

from imagomat.config import CullingSettings
from imagomat.culling.engine import CullItem, cull
from imagomat.vision.action import combine, pose_features


def _person(x0, y0, x1, y1, feet=(0.95, 0.95), wrists_up=False):
    """Box + 17 COCO-Keypoints in Pixeln (Bild 1000x1000), Füsse relativ zur Boxhöhe."""
    h = y1 - y0
    k = np.zeros((17, 3))
    cx = (x0 + x1) / 2
    k[:, 0] = cx
    k[:, 2] = 1
    k[0, 1] = y0 + 0.08 * h                   # Nase
    k[5, 1] = k[6, 1] = y0 + 0.2 * h          # Schultern
    k[9, 1] = k[10, 1] = (y0 + 0.02 * h) if wrists_up else (y0 + 0.5 * h)
    k[11, 1] = k[12, 1] = y0 + 0.5 * h        # Hüfte
    k[13, 1] = k[14, 1] = y0 + 0.72 * h       # Knie
    k[15, 1], k[16, 1] = y0 + feet[0] * h, y0 + feet[1] * h
    return [x0, y0, x1, y1], k


def _out(*persons):
    return {"boxes": np.array([p[0] for p in persons], float), "scores": np.ones(len(persons)),
            "keypoints": np.array([p[1] for p in persons], float)}


def test_pose_features():
    calm = pose_features(_out(_person(400, 300, 520, 800)), 1000, 1000)
    kick = pose_features(_out(_person(400, 300, 560, 800, feet=(0.95, 0.55))), 1000, 1000)
    duel = pose_features(_out(_person(350, 300, 500, 800), _person(460, 320, 610, 800)), 1000, 1000)
    cheer = pose_features(_out(_person(400, 300, 520, 800, wrists_up=True)), 1000, 1000)
    assert calm.score < 0.3
    assert kick.kick > 0.8 and kick.moment == "schuss"
    assert duel.duel > 0.6 and duel.moment == "zweikampf"
    assert cheer.arms_up == 1.0 and cheer.moment == "jubel"
    empty = pose_features({"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                           "keypoints": np.zeros((0, 17, 3))}, 1000, 1000)
    assert empty.score == 0 and empty.moment is None


def test_combine_sources():
    c = combine({}, (0.9, "schuss"), None)
    assert c["action"] == 0.9 and c["moment"] == "schuss" and c["action_source"] == "clip"
    fallback = combine({"face_count": 2, "max_face_height": 0.1, "subject_fraction": 0.3}, None, None)
    assert fallback["action_source"] == "classical" and 0 < fallback["action"] <= 1


def _items(n=40, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    t = 0.0
    for i in range(n):
        t += 0.3 if i % 4 else 6.0            # Bursts von 4 Bildern, 6 s Pause dazwischen
        a = {"sharpness": 100.0, "subject_sharpness": 50 + rng.normal(0, 3), "exposure_score": 0.8, "subject_fraction": 0.25,
             "median": 0.4, "action": float(rng.uniform(0, 1)),
             "scene": {"sport_floodlight": 0.8, "portrait": 0.2}}
        emb = rng.normal(size=16).astype(np.float32)
        emb /= np.linalg.norm(emb)
        items.append(CullItem(i, t, a, emb, None))
    return items


def test_highlights_mode_prefers_action_and_one_per_scene():
    cs = CullingSettings(keep_ratio=0.5)
    normal = cull(_items(), cs)
    hl_cs = CullingSettings(keep_ratio=0.5, highlights=True, highlights_ratio=0.25)
    hl = cull(_items(), hl_cs)
    kept_n = [it for it in normal if it.keep]
    kept_h = [it for it in hl if it.keep]
    assert 0 < len(kept_h) <= 10 < len(kept_n)
    # höchstens ein Bild pro Spielszene
    assert len({it.series for it in kept_h}) == len(kept_h)
    # nur Bilder mit viel Action
    assert all(it.feats["action"] >= 0.55 for it in kept_h)
    assert np.mean([it.a["action"] for it in kept_h]) > np.mean([it.a["action"] for it in kept_n])
    assert any("kein_moment" in it.reasons or "szene" in it.reasons for it in hl if not it.keep)


def test_max_keep_and_no_action_for_portraits():
    cs = CullingSettings(keep_ratio=0.5, max_keep=5)
    assert sum(it.keep for it in cull(_items(), cs)) == 5
    items = _items()
    for it in items:
        it.a["scene"] = {"portrait": 0.9, "event": 0.1}
    from imagomat.culling.engine import action_weight, build_features

    build_features(items)
    assert action_weight(items, CullingSettings()) == 0.0


def test_moment_only_when_sure_and_sporty():
    assert combine({"scene": {"event": 0.8, "sport_day": 0.2}}, (0.9, "zweikampf"), None)["moment"] is None
    assert combine({"scene": {"sport_day": 0.9}}, (0.9, "zweikampf"), None)["moment"] == "zweikampf"
    from imagomat.vision.action import PoseResult

    duel_only = PoseResult(2, 0.9, 0.0, 0.0, 0.0)
    assert combine({"scene": {"sport_day": 0.9}}, None, duel_only)["moment"] is None


def test_near_duplicates_not_kept_twice():
    from imagomat.culling.engine import CullItem

    items = _items(12)
    for i, it in enumerate(items):
        it.t = i * 60.0                      # jede Minute ein Bild -> eigene Serien
        it.phash = "f" * 16 if i < 6 else format(i * 12345678901, "016x")[:16]
    out = cull(items, CullingSettings(keep_ratio=0.5, burst_keep=0))
    kept_same = [it for it in out if it.keep and it.phash == "f" * 16]
    assert len(kept_same) == 1


def test_clean_mode_only_drops_bad_and_bursts():
    """Standard: keine Quote. Schlechte raus, aus 10 fast gleichen Bildern bleiben 2."""
    items = _items(20)
    for i, it in enumerate(items):
        if i < 10:
            it.t = i * 0.1                               # Serie: 10 Bilder in einer Sekunde, gleiches Motiv
            it.phash = "f" * 16
        else:
            it.t = 100.0 + i * 30.0                      # verschiedene Szenen
            it.phash = format(i * 12345678901, "016x")[:16]
    out = cull(items, CullingSettings(burst_keep=2))
    burst = [it for it in out[:10] if it.keep]
    others = [it for it in out[10:] if it.keep or it.hard]
    assert len(burst) <= 2 and len(others) == 10         # alles andere bleibt (ausser echte Fehler)


def test_content_tags_from_scores():
    from imagomat.vision.tags import labels, tags_from_scores

    keys = ["fans", "fans", "team", "trainer", "_neg"]
    assert tags_from_scores(keys, np.array([30.0, 29.0, 20.0, 18.0, 22.0])) == ["fans"]
    assert tags_from_scores(keys, np.array([20.0, 20.0, 20.0, 20.0, 26.0])) == []      # nichts Eindeutiges
    assert labels(["fans", "torhueter", "unbekannt"]) == ["Fans", "Torhüter"]

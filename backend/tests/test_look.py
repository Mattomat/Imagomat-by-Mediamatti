"""Saubere Vorschau (Kamera-JPEG + RAW-Licht) und Look aus fertigen Bildern."""

from pathlib import Path

import cv2
import numpy as np

from tagmatti.io.color import XYZ_TO_SRGB
from tagmatti.render.pipeline import _srgb_encode, render, render_hybrid


def _scene(h: int, w: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w] / h
    lin = np.zeros((h, w, 3), np.float32)
    lin[:] = np.array([0.02, 0.02, 0.03])                     # Nachthimmel
    lin[(yy > 0.3) & (yy < 0.55)] = np.array([0.05, 0.045, 0.04])   # Zuschauer
    lin[yy >= 0.55] = np.array([0.04, 0.1, 0.03])             # Rasen
    subj = ((xx - 0.7) ** 2 / 0.02 + (yy - 0.6) ** 2 / 0.08) < 1
    lin[subj] = np.array([0.3, 0.04, 0.04])                   # rotes Trikot
    lin[subj & (yy > 0.7)] = np.array([0.5, 0.5, 0.48])       # weisse Hose
    return lin, subj.astype(np.float32)


def test_hybrid_preview_is_clean_and_keeps_raw_tones():
    rng = np.random.default_rng(1)
    clean, _ = _scene(600, 900)
    noisy = np.clip(clean + rng.normal(0, 1, clean.shape).astype(np.float32) * np.sqrt(clean * 0.002 + 4e-5), 0, None)
    camera_jpeg = (_srgb_encode(np.clip(clean * 2.2, 0, 1)) * 255).astype(np.uint8)   # anderer Kamera-Look
    crs = {"Exposure2012": 0.5, "Contrast2012": 20, "Whites2012": 10,
           "MaskGroupBasedCorrections": [{"CorrectionAmount": 1, "LocalExposure2012": -0.3, "LocalSharpness": -0.6,
                                          "CorrectionMasks": [{"What": "Mask/Gradient", "ZeroX": 0.5, "ZeroY": 0.7,
                                                               "FullX": 0.5, "FullY": 1.0}]}]}
    plain = render(noisy, XYZ_TO_SRGB, np.ones(3), crs, 1, {}, 900).astype(np.float32)
    hyb = render_hybrid(noisy, XYZ_TO_SRGB, np.ones(3), crs, camera_jpeg, 1, {}, 900).astype(np.float32)
    ideal = render(clean, XYZ_TO_SRGB, np.ones(3), crs, 1, {}, 900).astype(np.float32)
    assert hyb.shape == plain.shape
    crowd = (slice(200, 300), slice(50, 300))                  # flache Fläche: nur Rauschen
    assert hyb[crowd].std() < 0.5 * plain[crowd].std()
    # Helligkeit und Farbe wie die RAW-Entwicklung, nicht wie das Kamera-JPEG
    for region in (crowd, (slice(450, 520), slice(50, 300)), (slice(330, 380), slice(600, 660))):
        assert np.abs(hyb[region].mean((0, 1)) - ideal[region].mean((0, 1))).max() < 6


def test_hybrid_falls_back_without_matching_preview():
    lin, _ = _scene(200, 300)
    a = render_hybrid(lin, XYZ_TO_SRGB, np.ones(3), {}, None, 1, {}, 300)
    b = render_hybrid(lin, XYZ_TO_SRGB, np.ones(3), {}, np.zeros((300, 300, 3), np.uint8), 1, {}, 300)
    assert a.shape == b.shape == (200, 300, 3)


def test_fit_look_matches_reference_statistics():
    from tagmatti.lightroom.dialect import Dialect
    from tagmatti.style.look import measure, fit_look

    lin, subj = _scene(160, 224)
    target = {"Exposure2012": 0.4, "Contrast2012": 30, "Whites2012": 20, "Blacks2012": -10,
              "MaskGroupBasedCorrections": [{"CorrectionName": "x", "CorrectionAmount": 1, "LocalExposure2012": -0.35,
                                             "CorrectionMasks": [{"What": "Mask/Gradient", "ZeroX": 0.5, "ZeroY": 0.42,
                                                                  "FullX": 0.5, "FullY": 0.0}]}]}
    ref = measure(render(lin, XYZ_TO_SRGB, np.ones(3), target, 1, {"subject": subj}, None), subj)
    warm = lin * np.array([1.12, 1.0, 0.82], np.float32)        # Bild mit Farbstich
    crs = {"Exposure2012": 0.0, "WhiteBalance": "As Shot"}
    notes = fit_look(crs, warm, {"stats": ref}, 1, subj, 5000, 0, Dialect.load())
    assert notes
    assert crs["WhiteBalance"] == "Custom" and crs["Temperature"] < 5000      # Stich ausgeglichen (kühler)
    names = [c["CorrectionName"] for c in crs.get("MaskGroupBasedCorrections", [])]
    assert "Look: oben" in names
    from tagmatti.style.look import _WB

    wb = _WB({}, 5000, 0)
    wb.dm = 1e6 / crs["Temperature"] - 200
    wb.dt = crs["Tint"]
    got = measure(render(warm, XYZ_TO_SRGB, np.ones(3), {**crs, **wb.preview_keys()}, 1, {"subject": subj}, None), subj)
    assert abs(got["subj_L50"] - ref["subj_L50"]) < 5
    assert abs(got["top_L50"] - ref["top_L50"]) < 3
    assert abs(got["n_b"] - ref["n_b"]) < 4


def test_learn_look_and_develop_with_it(tmp_path: Path):
    from tagmatti.analysis import import_folder
    from tagmatti.db import Database
    from tagmatti.jobs import JobManager
    from tagmatti.style import compare
    import tagmatti.pipeline  # noqa: F401 - registriert alle Aufträge
    from tagmatti.culling import engine as _cull  # noqa: F401
    from tagmatti.style.look import list_looks

    from .synth import write_shoot

    refs = tmp_path / "fertig"
    refs.mkdir()
    lin, _ = _scene(400, 600)
    for i in range(3):
        img = (_srgb_encode(np.clip(lin * (2.0 + 0.3 * i), 0, 1)) * 255).astype(np.uint8)
        cv2.imwrite(str(refs / f"r{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    write_shoot(tmp_path / "s", n=3)
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    jm = JobManager(db)
    assert jm.run_sync(db.create_job("learn_look", None, {"name": "Test", "folder": str(refs)}))["status"] == "done"
    assert [lk["name"] for lk in list_looks()] == ["Test"]
    assert "look:Test" in [s["key"] for s in compare.styles()]
    for k in ("analyze", "cull"):
        assert jm.run_sync(db.create_job(k, sid, {}))["status"] == "done"
    assert jm.run_sync(db.create_job("develop", sid, {"profile": "look:Test", "only_keep": False}))["status"] == "done"
    rows = db.query("SELECT profile FROM edits")
    assert rows and all(r["profile"] == "look:Test" for r in rows)
    iid = db.images(sid)[0]["id"]
    crs = compare.develop_preview(db, sid, iid, "look:Test")
    assert "Exposure2012" in crs


def test_straighten_uses_ad_boards_not_legs():
    from tagmatti.vision.geometry import estimate_tilt

    rng = np.random.default_rng(3)
    img = np.full((683, 1024, 3), 40, np.uint8)
    # Werbebande: lange Kanten, um 2° verdreht
    M = cv2.getRotationMatrix2D((512, 341), 2.0, 1.0)
    board = np.full_like(img, 40)
    cv2.rectangle(board, (-200, 420), (1224, 470), (230, 230, 230), -1)
    img = cv2.warpAffine(board, M, (1024, 683), borderValue=(40, 40, 40))
    # Beine: kurze, schräge Striche (laufende Spieler) dürfen nicht zählen
    for _ in range(25):
        x, y = int(rng.integers(100, 900)), int(rng.integers(480, 640))
        a = np.radians(90 + rng.uniform(-7, 7))
        cv2.line(img, (x, y), (int(x + 45 * np.cos(a)), int(y + 45 * np.sin(a))), (200, 30, 30), 5)
    t = estimate_tilt(img)
    assert t.source in ("horizontal", "both")
    assert abs(abs(t.angle) - 2.0) < 0.5 and t.confidence >= 0.4


def test_hybrid_handles_camera_tone_curve():
    """Sony-JPEG mit steiler Kurve und anderer Farbe: Struktur im Publikum wie in der RAW-Entwicklung
    (vorher Flecken/Lichthöfe, weil das Verhältnis JPEG/RAW von der Helligkeit abhing)."""
    rng = np.random.default_rng(2)
    clean, _ = _scene(600, 900)
    tex = cv2.GaussianBlur(rng.random((600, 900)).astype(np.float32), (0, 0), 2.5)
    tex = (tex - tex.mean()) / tex.std()
    band = ((np.arange(600) / 600 > 0.3) & (np.arange(600) / 600 < 0.55))[:, None]
    clean = clean * np.where(band, np.exp(0.9 * tex), 1)[..., None].astype(np.float32)   # Zuschauer
    g = _srgb_encode(np.clip(clean * np.array([2.6, 2.2, 1.6], np.float32), 0, 1))
    camera_jpeg = (np.clip(0.5 + (g - 0.5) * 1.5, 0, 1) * 255).astype(np.uint8)   # steil, Tiefen abgeschnitten
    crs = {"Exposure2012": 0.8, "Shadows2012": 30, "Highlights2012": -60}
    hyb = render_hybrid(clean, XYZ_TO_SRGB, np.ones(3), crs, camera_jpeg, 1, {}, 900).astype(np.float32)
    ideal = render(clean, XYZ_TO_SRGB, np.ones(3), crs, 1, {}, 900).astype(np.float32)
    crowd = (slice(200, 320), slice(50, 850))
    assert np.abs(hyb[crowd] - ideal[crowd]).mean() < 5.5             # alte Übertragung: ~7.2
    assert np.abs(hyb[crowd].mean((0, 1)) - ideal[crowd].mean((0, 1))).max() < 4


def test_white_balance_on_already_balanced_data():
    """Apple RAW-Engine / Vorschau liefern schon weissabgeglichene Daten: dein Weissabgleich (z. B. 3600 K, wie
    aufgenommen) darf das Bild nicht verfärben (früher wurde von D65 aus gerechnet: starker Blaustich)."""
    from tagmatti.render.pipeline import AS_SHOT_TEMP, AS_SHOT_TINT

    lin, _ = _scene(200, 300)
    base = render(lin, XYZ_TO_SRGB, np.ones(3), {}, 1, {}, 300).astype(float)
    crs = {"WhiteBalance": "Custom", "Temperature": 3600, "Tint": 40}
    same = render(lin, XYZ_TO_SRGB, np.ones(3), {**crs, AS_SHOT_TEMP: 3600, AS_SHOT_TINT: 40}, 1, {}, 300)
    assert np.abs(same.astype(float) - base).mean() < 1.0
    warmer = render(lin, XYZ_TO_SRGB, np.ones(3), {**crs, "Temperature": 4600, AS_SHOT_TEMP: 3600,
                                                  AS_SHOT_TINT: 40}, 1, {}, 300).astype(float)
    assert warmer[..., 0].mean() > base[..., 0].mean() + 2                   # höher = wärmer, wie in Lightroom

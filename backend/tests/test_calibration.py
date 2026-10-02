"""Kalibrierung an echten Lightroom-Bearbeitungen: Weiss an den Spielern neutral, Nacht-Belichtung,
Begradigen-Richtung, Spieler-Weiss in der Vorschau, Waveform und Änderungsliste."""

from pathlib import Path

import numpy as np

from tagmatti.config import load_settings
from tagmatti.io.color import XYZ_TO_SRGB
from tagmatti.lightroom.dialect import Dialect
from tagmatti.render.pipeline import _srgb_encode, render
from tagmatti.style.develop import ImageDevelop, build_settings, predict_all
from tagmatti.style.model import Record


def _jersey_scene(cast=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    h, w = 300, 200
    yy, xx = np.mgrid[0:h, 0:w] / h
    lin = np.zeros((h, w, 3), np.float32)
    lin[:] = (0.004, 0.006, 0.006)                       # Nachthimmel
    lin[yy > 0.55] = (0.05, 0.16, 0.03)                  # heller Rasen unter Flutlicht
    subj = ((xx - 0.33) ** 2 / 0.01 + (yy - 0.55) ** 2 / 0.05) < 1
    lin[subj] = (0.55, 0.55, 0.55)                       # weisses Trikot
    lin[subj & (yy > 0.62)] = (0.4, 0.04, 0.04)          # rote Hose
    img = (_srgb_encode(np.clip(lin * np.asarray(cast, np.float32), 0, 1)) * 255).astype(np.uint8)
    return img, subj.astype(np.float32)


def test_white_balance_on_player_white():
    from tagmatti.style.whitebalance import white_gains, white_patch_shift

    img, subj = _jersey_scene(cast=(0.85, 1.0, 0.8))    # Grünstich wie unter LED-Flutlicht
    g = white_gains(img, subj)
    assert g is not None and abs(g[0] - 1 / 0.85) < 0.06 and abs(g[2] - 1 / 0.8) < 0.06
    dm, dt = white_patch_shift(img, subj)
    assert dt > 10                                      # Richtung Magenta (gegen Grün), wie du in Lightroom
    neutral, _ = _jersey_scene()
    g0 = white_gains(neutral, subj)
    assert np.abs(g0 - 1).max() < 0.03                   # schon neutral: nichts ändern
    assert white_gains(img, None) is None                # ohne Personen-Maske lieber gar nicht


def test_mediamatti_preset_follows_your_lightroom_values():
    a = {"lin_log_median": -6.8, "lin_log_p99": -1.6, "lin_log_p05": -9.0, "lin_log_p01": -9.6, "median": 0.2,
         "as_shot_temp": 4000, "as_shot_tint": 5, "subject_fraction": 0.1, "subject_bbox": [0.3, 0.3, 0.45, 0.8],
         "subject_luma": 0.45, "noise_sigma_mid": 0.006, "preview_w": 1365, "preview_h": 2048,
         "scene": {"sport_floodlight": 0.9}, "bottom_luma": 0.35, "mid_luma": 0.3}
    it = ImageDevelop(1, Record(a, {"iso": 4000}, None), 8, 4672, 7008)
    s = load_settings()
    predict_all([it], None, s, "fb_mediamatti")
    build_settings(it, s, Dialect())
    crs = it.crs
    assert 0.2 < crs["Exposure2012"] < 1.6                    # Nacht bleibt Nacht (du: +0.5 bis +1.1)
    assert crs["Highlights2012"] <= -30 and crs["Shadows2012"] >= 30
    assert crs["SaturationAdjustmentGreen"] == -33 and crs["LuminanceAdjustmentGreen"] == -19
    corr = {c["CorrectionName"]: c for c in crs["MaskGroupBasedCorrections"]}
    assert abs(corr["Spieler"]["LocalWhites2012"] - 0.84) < 1e-6
    fade = corr["Verlauf unten"]
    assert fade["LocalSharpness"] == -1 and fade["LocalClarity2012"] == -1 and fade["LocalTexture"] == -1
    assert fade["LocalExposure2012"] * 4 <= -1.0


def test_night_target_keeps_floodlight_images_dark():
    from tagmatti.style.presets import PRESETS, apply

    night = {"lin_log_median": -6.0, "lin_log_p99": -2.5, "lin_log_p05": -9, "median": 0.2}
    day = {"lin_log_median": -3.0, "lin_log_p99": -0.5, "lin_log_p05": -6, "median": 0.4}
    for key in ("fb_signature", "sport_floodlight", "fb_mediamatti"):
        assert apply(PRESETS[key], night)["Exposure2012"] < 1.6, key   # früher +3.0 (zerdrückte Lichter)
    assert apply(PRESETS["sport_day"], day)["Exposure2012"] > 0


def test_crop_angle_direction_matches_lightroom():
    """Werbebande intern um -2.5° gekippt -> Lightroom braucht CropAngle +2.5 (an echten XMPs bestätigt)."""
    import cv2

    from tagmatti.render.pipeline import _geometry
    from tagmatti.vision.geometry import estimate_tilt, plan_crop, to_lightroom_crop

    img = np.full((683, 1024, 3), 40, np.uint8)
    cv2.rectangle(img, (-200, 420), (1224, 470), (230, 230, 230), -1)
    M = cv2.getRotationMatrix2D((512, 341), -2.5, 1.0)      # Inhalt im Uhrzeigersinn verdreht
    img = cv2.warpAffine(img, M, (1024, 683), borderValue=(40, 40, 40))
    t = estimate_tilt(img)
    crs = to_lightroom_crop(plan_crop(1024, 683, t.angle), 1)
    assert crs["CropAngle"] > 2.0
    fixed = (_geometry(img.astype(np.float32) / 255, crs, 1) * 255).astype(np.uint8)
    assert abs(estimate_tilt(fixed).angle) < 0.6


def test_local_whites_brighten_players_in_preview():
    lin, subj = np.full((120, 80, 3), 0.02, np.float32), np.zeros((120, 80), np.float32)
    lin[40:90, 30:50] = 0.3
    subj[40:90, 30:50] = 1
    corr = {"CorrectionAmount": 1, "LocalWhites2012": 0.84,
            "CorrectionMasks": [{"What": "Mask/Image", "MaskSubType": "1", "MaskInverted": "false"}]}
    base = render(lin, XYZ_TO_SRGB, np.ones(3), {}, 1, {"subject": subj}, None)
    up = render(lin, XYZ_TO_SRGB, np.ones(3), {"MaskGroupBasedCorrections": [corr]}, 1, {"subject": subj}, None)
    assert up[60, 40].mean() > base[60, 40].mean() + 5
    assert abs(int(up[5, 5].mean()) - int(base[5, 5].mean())) <= 1


def test_waveform_and_changes_api(tmp_path: Path):
    from fastapi.testclient import TestClient

    import tagmatti.pipeline  # noqa: F401 - registriert alle Aufträge
    from tagmatti.analysis import import_folder
    from tagmatti.db import Database
    from tagmatti.jobs import JobManager
    from tagmatti.server.app import create_app

    from .synth import write_shoot

    write_shoot(tmp_path / "s", n=2)
    db = Database(tmp_path / "a.db")
    sid = import_folder(db, tmp_path / "s")
    jm = JobManager(db)
    for k in ("analyze", "cull"):
        assert jm.run_sync(db.create_job(k, sid, {}))["status"] == "done"
    assert jm.run_sync(db.create_job("develop", sid, {"profile": "preset:fb_mediamatti", "only_keep": False}))[
        "status"] == "done"
    iid = db.images(sid)[0]["id"]
    with TestClient(create_app(str(tmp_path / "a.db"))) as c:
        for src in ("before", "after", "style:preset:fb_bw"):
            r = c.get(f"/api/images/{iid}/waveform", params={"src": src})
            assert r.status_code == 200 and r.headers["content-type"] == "image/png", src
        ch = c.get(f"/api/images/{iid}/changes").json()
        labels = [s["label"] for s in ch["sliders"]]
        assert "Belichtung" in labels and "Weissabgleich" in labels
        assert any(m["name"] == "Verlauf unten" for m in ch["masks"])
        assert c.get(f"/api/images/{iid}/changes", params={"style": "preset:fb_bw"}).json()["sliders"]


def test_waveform_marks_clipping():
    from tagmatti.style.scopes import waveform

    img = np.zeros((100, 100, 3), np.uint8)
    img[:, 50:] = 255
    wf = waveform(img, 100, 50)
    assert wf.shape == (50, 100, 3)
    assert (wf[0, 60] == (255, 80, 80)).all() and (wf[49, 20] == (255, 80, 80)).all()

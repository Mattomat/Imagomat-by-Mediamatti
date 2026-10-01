import datetime as dt
import math
from pathlib import Path

import cv2
import numpy as np

from imagomat.analysis import import_folder
from imagomat.config import load_settings
from imagomat.db import Database
from imagomat.io.dng import mosaic_rggb, write_dng
from imagomat.jobs import JobManager
from imagomat.lightroom.dialect import Dialect
from imagomat.lightroom.xmp import XmpDoc, write_sidecar
from imagomat.style import jobs as style_jobs  # noqa: F401 (registriert Jobs)
from imagomat.style.develop import ImageDevelop, build_settings
from imagomat.style.model import Record, StyleModel
from imagomat.style.presets import PRESETS, apply
from imagomat.style.targets import decode, encode

from .synth import scene

USER_MASK = {"What": "Correction", "CorrectionAmount": 1, "CorrectionActive": True, "CorrectionName": "Motiv",
             "LocalExposure2012": 0.1, "LocalClarity2012": 0.15,
             "CorrectionMasks": [{"What": "Mask/Image", "MaskSubType": "1", "MaskVersion": "2",
                                  "MaskName": "Motiv 1", "MaskActive": True, "MaskDigest": "D1"}]}


def _write(folder: Path, i: int, brightness: float, t0: dt.datetime) -> Path:
    img = scene(i % 5) * brightness
    rng = np.random.default_rng(i)
    cam = img * np.array([0.5, 1.0, 0.65])
    raw = np.clip(cam * 30000 + 800 + rng.normal(0, 200, cam.shape), 0, 65535).astype(np.uint16)
    p = folder / f"IMG{i:04d}.dng"
    write_dng(p, mosaic_rggb(raw), cfa=True, black=800, white=60000, as_shot_neutral=(0.5, 1.0, 0.65), iso=3200,
              exposure_time=1 / 800, fnumber=2.8, focal_length=135, capture_time=t0 + dt.timedelta(seconds=i * 3))
    return p


def _user_edit(brightness: float) -> dict:
    return {"ProcessVersion": "15.4", "Version": "18.0", "WhiteBalance": "As Shot",
            "Exposure2012": round(-math.log2(brightness) + 0.4, 2), "Contrast2012": 25, "Vibrance": 12,
            "SaturationAdjustmentGreen": -20, "CameraProfile": "Adobe Standard", "LensProfileEnable": 1,
            "EnhanceDenoiseLumaAmount": 60, "EnhanceDenoiseVersion": "3",
            "MaskGroupBasedCorrections": [USER_MASK]}


def test_targets_roundtrip():
    crs = {"WhiteBalance": "Custom", "Temperature": 4300, "Tint": 12, "Exposure2012": 0.5,
           "SplitToningShadowHue": 350, "SplitToningShadowSaturation": 20,
           "ToneCurvePV2012": ["0, 0", "64, 50", "192, 205", "255, 255"]}
    t = encode(crs, 5000, 5)
    back = decode(t, 5000, 5)
    assert back["WhiteBalance"] == "Custom" and abs(back["Temperature"] - 4300) < 2 and back["Tint"] == 12
    assert back["SplitToningShadowHue"] == 350 and back["SplitToningShadowSaturation"] == 20
    assert back["ToneCurveName2012"] == "Custom"


def test_preset_adapts_exposure_and_denoise():
    dark = {"lin_log_median": -7.0, "lin_log_p99": -3, "lin_log_p05": -10, "noise_sigma_mid": 0.01,
            "median": 0.15}
    bright = {"lin_log_median": -2.0, "lin_log_p99": -0.1, "lin_log_p05": -5, "raw_clip": 0.15,
              "noise_sigma_mid": 0.002, "median": 0.6}
    p = PRESETS["sport_floodlight"]
    td, tb = apply(p, dark), apply(p, bright)
    assert td["Exposure2012"] > 2 and tb["Exposure2012"] < 0
    assert tb["Highlights2012"] < td["Highlights2012"]
    assert td["denoise"] > tb["denoise"]


def test_train_from_xmp_folder_and_develop(tmp_path: Path):
    train = tmp_path / "train"
    train.mkdir()
    rng = np.random.default_rng(42)
    t0 = dt.datetime(2026, 4, 1, 20, 0)
    for i in range(48):
        b = float(np.exp(rng.uniform(np.log(0.25), np.log(2.5))))
        p = _write(train, i, b, t0)
        write_sidecar(p, XmpDoc(rating=3, label="Grün", crs=_user_edit(b)))
    db = Database(tmp_path / "s.db")
    jm = JobManager(db)
    j = jm.run_sync(db.create_job("train_profile", None, {"name": "Test", "folders": [str(train)],
                                                         "base_preset": "sport_floodlight"}))
    assert j["status"] == "done", j["error"]
    model = StyleModel.load("Test")
    assert model.n == 48
    # "Jetzt fertigstellen": nur bereits berechnete Bilder, ohne neu zu rechnen
    j = jm.run_sync(db.create_job("train_profile", None, {"name": "Schnell", "folders": [str(train)],
                                                         "cached_only": True, "learn_people": False}))
    assert j["status"] == "done", j["error"]
    assert StyleModel.load("Schnell").n == 48
    assert model.templates and model.templates[0].sig.startswith("image:1")
    d = Dialect.load()
    assert d.process_version == "15.4" and d.denoise_amount_key == "EnhanceDenoiseLumaAmount"
    assert d.labels["green"] == "Grün"
    # Neuer Shoot
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    truth = {}
    for i in range(10):
        b = float([0.3, 0.5, 0.8, 1.2, 2.0][i % 5])
        p = _write(shoot, 100 + i, b, t0 + dt.timedelta(hours=1))
        truth[p.name] = -math.log2(b) + 0.4
    sid = import_folder(db, shoot, profile="Test")
    assert jm.run_sync(db.create_job("analyze", sid, {}))["status"] == "done"
    settings = load_settings()
    settings.develop.shoot_consistency = 0.0
    settings.develop.punch = 0.0              # hier nur prüfen, ob dein Stil exakt übernommen wird
    from imagomat.config import save_settings

    save_settings(settings)
    j = jm.run_sync(db.create_job("develop", sid, {"only_keep": False}))
    assert j["status"] == "done", j["error"]
    import json

    errs = []
    for r in db.query("SELECT i.filename, e.params, e.masks, e.denoise FROM edits e JOIN images i ON i.id=e.image_id"):
        params = json.loads(r["params"])
        errs.append(abs(params["Exposure2012"] - truth[r["filename"]]))
        assert abs(params["Contrast2012"] - 25) < 4
        assert abs(params["SaturationAdjustmentGreen"] + 20) < 4
        assert params["CameraProfile"] == "Adobe Standard"
        assert params["EnhanceDenoiseLumaAmount"] == "60" or abs(int(params["EnhanceDenoiseLumaAmount"]) - 60) <= 5
        masks = json.loads(r["masks"])
        assert masks and masks[0]["CorrectionMasks"][0]["What"] == "Mask/Image"
        assert "MaskDigest" not in masks[0]["CorrectionMasks"][0]
    assert float(np.mean(errs)) < 0.35, errs
    # Stil-Vergleich: gleiche Bilder mit verschiedenen Stilen, ohne zu speichern
    from imagomat.style import compare

    iid = db.images(sid)[0]["id"]
    before = db.one("SELECT params FROM edits WHERE image_id=?", (iid,))[0]
    mine, std = (compare.develop_preview(db, sid, iid, k) for k in ("Test", compare.AUTO))
    assert mine["Exposure2012"] != std["Exposure2012"] or mine.get("Contrast2012") != std.get("Contrast2012")
    assert db.one("SELECT params FROM edits WHERE image_id=?", (iid,))[0] == before
    assert {s["key"] for s in compare.styles()} >= {"Test", compare.AUTO}
    assert len(compare.sample_images(db, sid, 3)) >= 1


def test_build_settings_preset_only():
    a = {"lin_log_median": -6, "median": 0.2, "as_shot_temp": 4200, "as_shot_tint": 10, "subject_fraction": 0.2,
         "noise_sigma_mid": 0.008, "preview_w": 1200, "preview_h": 800, "tilt_angle": 2.0, "tilt_conf": 0.9,
         "scene": {"concert_stage": 0.9, "sport_day": 0.1}}
    it = ImageDevelop(1, Record(a, {"iso": 12800}, None), 1, 6000, 4000)
    from imagomat.style.develop import predict_all

    s = load_settings()
    predict_all([it], None, s)
    build_settings(it, s, Dialect())
    assert it.preset == "concert_stage"
    assert it.crs["HasCrop"] is True and abs(it.crs["CropAngle"] - 2.0) < 1e-6
    assert it.crs["MaskGroupBasedCorrections"][0]["CorrectionMasks"][0]["What"] == "Mask/Image"
    assert it.denoise and it.crs["EnhanceDenoiseLumaAmount"] == str(it.denoise)
    assert it.crs["Exposure2012"] > 1


def test_paint_fallback_and_radial():
    from imagomat.style.masks import mask_to_dabs, paint_components, radial_component

    m = np.zeros((200, 300), np.float32)
    cv2.circle(m, (150, 100), 60, 1.0, -1)
    dabs = mask_to_dabs(m)
    cov = np.zeros_like(m)
    for x, y, r in dabs:
        cv2.circle(cov, (int(x * 300), int(y * 200)), int(r * 300), 1.0, -1)
    assert (cov * m).sum() / m.sum() > 0.9 and (cov * (1 - m)).sum() / m.sum() < 0.25
    comps = paint_components(m, orientation=6)
    assert comps and all(c["What"] == "Mask/Paint" and c["Dabs"][0].startswith("d ") for c in comps)
    rc = radial_component((0.2, 0.1, 0.4, 0.5), orientation=1)
    assert rc["Left"] == 0.2 and rc["Bottom"] == 0.5


def test_feedback_uses_last_export(tmp_path: Path):
    import json

    from imagomat.export import exporter  # noqa: F401
    from imagomat.lightroom.xmp import read_xmp

    train = tmp_path / "train"
    train.mkdir()
    t0 = dt.datetime(2026, 4, 1, 20, 0)
    for i in range(6):
        b = [0.4, 0.8, 1.6][i % 3]
        write_sidecar(_write(train, i, b, t0), XmpDoc(crs=_user_edit(b)))
    db = Database(tmp_path / "f.db")
    jm = JobManager(db)
    assert jm.run_sync(db.create_job("train_profile", None, {"name": "FB", "folders": [str(train)]}))["status"] == "done"
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    for i in range(4):
        _write(shoot, 50 + i, 1.0, t0)
    sid = import_folder(db, shoot, profile="FB")
    for kind, params in (("analyze", {}), ("develop", {"only_keep": False}),
                         ("export", {"target": str(tmp_path / "exp")})):
        assert jm.run_sync(db.create_job(kind, sid, params))["status"] == "done"
    # "Korrektur in Lightroom": Belichtung auf +1.5 setzen
    x = tmp_path / "exp" / "IMG0050.xmp"
    doc = read_xmp(x)
    doc.crs["Exposure2012"] = "+1.50"
    x.write_bytes(__import__("imagomat.lightroom.xmp", fromlist=["serialize"]).serialize(doc))
    j = jm.run_sync(db.create_job("feedback", sid, {"profile": "FB"}))
    assert j["status"] == "done", j["error"]
    assert StyleModel.load("FB").n == 7   # nur das korrigierte Bild kommt dazu
    up = db.one("SELECT user_params FROM edits e JOIN images i ON i.id=e.image_id WHERE i.filename='IMG0050.dng'")
    assert json.loads(up[0])["Exposure2012"] == "+1.50"


def test_gbdt_with_torch_loaded():
    """macOS: PyTorch + Gradient Boosting im selben Prozess darf nicht abstürzen."""
    import pytest

    pytest.importorskip("torch")
    import torch  # noqa: F401

    from imagomat.style.model import gbdt_factory

    make = gbdt_factory()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=200)
    m = make().fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.9


def test_bottom_fade_adapts_per_image():
    from imagomat.style.develop import predict_all

    s = load_settings()

    def fade(bottom, mid):
        a = {"lin_log_median": -3, "median": 0.4, "scene": {"sport_day": 0.9}, "preview_w": 1200, "preview_h": 800,
             "subject_bbox": [0.4, 0.2, 0.6, 0.8], "bottom_luma": bottom, "mid_luma": mid}
        it = ImageDevelop(1, Record(a, {"iso": 800}, None), 1, 6000, 4000)
        predict_all([it], None, s)
        build_settings(it, s, Dialect())
        corr = [c for c in it.crs["MaskGroupBasedCorrections"] if c["CorrectionName"] == "Verlauf unten"]
        assert len(corr) == 1 and corr[0]["CorrectionMasks"][0]["What"] == "Mask/Gradient"
        assert corr[0]["LocalSharpness"] < -0.3 and corr[0]["LocalTexture"] < 0     # weich
        return corr[0]["LocalExposure2012"] * 4

    bright, dark = fade(0.55, 0.4), fade(0.08, 0.3)
    assert bright < -1.2 and -0.6 < dark < 0 and bright < dark


def test_punch_sets_white_point_and_background_contrast():
    from imagomat.style.develop import punch, punch_background

    flat = {"Exposure2012": 0.5, "Whites2012": 0, "Blacks2012": 0, "Highlights2012": -40, "Contrast2012": 10}
    a = {"lin_log_p99": -2.0, "lin_log_p01": -6.0}
    crs = dict(flat)
    punch(crs, a, 1.0)
    assert crs["Whites2012"] >= 30 and crs["Blacks2012"] < -3 and crs["Contrast2012"] == 14
    bright = dict(flat, Whites2012=40)
    punch(bright, {"lin_log_p99": 0.0, "lin_log_p01": -9.0}, 1.0)
    assert bright["Whites2012"] == 40 and bright["Blacks2012"] == 0      # schon hell/dunkel genug
    bg = {"LocalExposure2012": -0.1, "CorrectionMasks": [{"What": "Mask/Image", "MaskInverted": True}]}
    punch_background(bg, 1.0)
    assert bg["LocalContrast2012"] > 0 and bg["LocalDehaze"] > 0
    mine = {"LocalContrast2012": -0.2, "CorrectionMasks": [{"What": "Mask/Image", "MaskInverted": True}]}
    punch_background(mine, 1.0)
    assert mine["LocalContrast2012"] == -0.2                              # bewusst so gewählt: bleibt


def test_scopes_touch_white_and_black(tmp_path: Path):
    """Wie am Lumetri-Waveform: oben und unten schlägt ein kleiner Teil leicht an."""
    from imagomat.io.color import XYZ_TO_SRGB
    from imagomat.render.pipeline import render
    from imagomat.style.scopes import HI_TARGET, LO_TARGET, fit_scopes, preview_linear, scopes

    rng = np.random.default_rng(1)
    base = np.linspace(0, 1, 900)[None, :] * np.linspace(0.6, 1, 600)[:, None]
    img = np.clip(5 + 250 * (base + rng.normal(0, 0.03, base.shape)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "p.jpg"), np.repeat(img[..., None], 3, 2))
    lin = preview_linear(str(tmp_path / "p.jpg"))
    crs = {"Exposure2012": 0.9, "Contrast2012": 15, "Whites2012": 0, "Blacks2012": 0, "Highlights2012": -30}
    fit_scopes(crs, lin)
    sc = scopes(render(lin, XYZ_TO_SRGB, np.ones(3), {**crs, "WhiteBalance": "As Shot"}, 1, {}, None))
    assert abs(sc["p_hi"] - HI_TARGET) < 0.03 and sc["p_lo"] <= LO_TARGET + 0.02     # Schwarz nie angehoben
    assert sc["white_clip"] < 0.02 and sc["black_clip"] < 0.06            # leicht, nicht ausgefressen


def test_football_presets_build_with_masks():
    from imagomat.style.develop import predict_all
    from imagomat.style.presets import FOOTBALL

    s = load_settings()
    a = {"lin_log_median": -4, "lin_log_p99": -1.5, "lin_log_p01": -9, "median": 0.3, "as_shot_temp": 4200,
         "as_shot_tint": 12, "subject_fraction": 0.2, "subject_bbox": [0.35, 0.2, 0.6, 0.85], "noise_sigma_mid": 0.006,
         "preview_w": 1200, "preview_h": 800, "scene": {"sport_floodlight": 0.9}, "bottom_luma": 0.4, "mid_luma": 0.35}
    for p in FOOTBALL:
        it = ImageDevelop(1, Record(a, {"iso": 6400}, None), 1, 6000, 4000)
        predict_all([it], None, s, p.key)
        build_settings(it, s, Dialect())
        assert it.preset == p.key
        names = [c["CorrectionName"] for c in it.crs["MaskGroupBasedCorrections"]]
        assert "Spieler" in names and "Verlauf unten" in names, (p.key, names)
        kinds = {m["What"] for c in it.crs["MaskGroupBasedCorrections"] for m in c["CorrectionMasks"]}
        if any(r.kind == "spotlight" for r in p.masks):
            assert "Mask/CircularGradient" in kinds
    bw = ImageDevelop(1, Record(a, {"iso": 6400}, None), 1, 6000, 4000)
    predict_all([bw], None, s, "fb_bw")
    build_settings(bw, s, Dialect())
    assert float(bw.crs["Saturation"]) == -100

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

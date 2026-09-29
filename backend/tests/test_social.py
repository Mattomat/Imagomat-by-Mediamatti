"""Social-Media-Export: Story/Post-Format, Zuschnitt um den Fokuspunkt."""

from pathlib import Path

import cv2
import numpy as np

from imagomat.analysis import import_folder
from imagomat.db import Database
from imagomat.export import social
from imagomat.jobs import JobManager
from imagomat.style import jobs  # noqa: F401

from .synth import write_shoot


def test_smart_crop_keeps_focus():
    img = np.zeros((1000, 1500, 3), np.uint8)
    out = social.smart_crop(img, 9 / 16, (0.9, 0.5))
    assert out.shape[0] == 1000 and abs(out.shape[1] - 562) <= 1
    left = social.smart_crop(img, 9 / 16, (0.05, 0.5))
    assert left.shape == out.shape


def test_social_job(tmp_path: Path):
    folder = tmp_path / "Match"
    write_shoot(folder, n=6)
    db = Database(tmp_path / "s.db")
    sid = import_folder(db, folder)
    jm = JobManager(db)
    for kind, params in (("analyze", {}), ("cull", {"keep_ratio": 0.5}), ("develop", {})):
        assert jm.run_sync(db.create_job(kind, sid, params))["status"] == "done"
    out = tmp_path / "story"
    j = jm.run_sync(db.create_job("social", sid, {"format": "story", "selection": "keep", "target": str(out)}))
    assert j["status"] == "done", j["error"]
    files = sorted(out.glob("*_story.jpg"))
    kept = db.query("SELECT COUNT(*) n FROM culling c JOIN images i ON i.id=c.image_id "
                    "WHERE i.shoot_id=? AND c.decision='keep'", (sid,))[0]["n"]
    assert len(files) == kept > 0
    assert cv2.imread(str(files[0])).shape[:2] == (1920, 1080)

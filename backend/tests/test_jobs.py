"""Job-Spuren: Lernen blockiert das Verarbeiten von Shoots nicht."""

import threading
import time
from pathlib import Path

from imagomat.db import Database
from imagomat.jobs import JobManager, job, lane_of

_release = threading.Event()


@job("train_profile_dummy")
def _slow(ctx, **_):
    _release.wait(10)


@job("cull_dummy")
def _fast(ctx, **_):
    pass


def test_lanes(tmp_path: Path, monkeypatch):
    import imagomat.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "LEARN_KINDS", jobs_mod.LEARN_KINDS | {"train_profile_dummy"})
    assert lane_of("train_profile") == "learn" and lane_of("pipeline") == "shoot"
    db = Database(tmp_path / "t.db")
    jm = JobManager(db)
    jm.start()
    slow = jm.submit("train_profile_dummy")
    fast = jm.submit("cull_dummy")
    for _ in range(100):
        if db.job(fast)["status"] == "done":
            break
        time.sleep(0.05)
    assert db.job(fast)["status"] == "done"      # lief, obwohl das Training noch läuft
    assert db.job(slow)["status"] == "running"
    _release.set()
    jm.stop()

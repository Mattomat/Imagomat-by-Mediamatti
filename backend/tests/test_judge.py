"""Presets beurteilen jedes Bild einzeln (Probe-Entwicklung messen und nachregeln)."""

import numpy as np

from imagomat.lightroom.params import to_number


def _night(gain: float = 1.0, jersey: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    h, w = 160, 240
    yy, xx = np.mgrid[0:h, 0:w] / h
    lin = np.full((h, w, 3), 0.004, np.float32)
    lin[yy > 0.35] = (0.03, 0.06, 0.02)
    lin[(yy > 0.35) & (yy < 0.5)] = (0.04, 0.035, 0.03)
    subj = (((xx - 0.75) ** 2 / 0.02 + (yy - 0.6) ** 2 / 0.06) < 1)
    lin[subj] = (jersey, jersey, jersey * 0.97)
    return lin * gain, subj.astype(np.float32)


def test_darker_scene_gets_more_exposure():
    from imagomat.style.judge import judge

    a = {"lin_log_median": -6.5}                       # Nacht
    out = []
    for gain in (1.0, 0.4):
        lin, subj = _night(gain)
        crs = {"Exposure2012": 0.5, "Highlights2012": -20}
        judge(crs, lin, a, 1, subj)
        out.append(float(to_number(crs["Exposure2012"])))
    assert out[1] > out[0] + 0.6                        # 1.3 EV dunkler aufgenommen -> deutlich mehr Belichtung


def test_white_jersey_not_blown():
    from imagomat.style.judge import judge

    lin, subj = _night(1.0, jersey=2.5)                 # Trikot ausgefressen
    crs = {"Exposure2012": 0.5, "Highlights2012": -20}
    notes = judge(crs, lin, {"lin_log_median": -6.5}, 1, subj)
    assert float(to_number(crs["Highlights2012"])) < -20 and any("Lichter" in n for n in notes)

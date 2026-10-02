from pathlib import Path

import cv2
import numpy as np
import rawpy

from tagmatti.denoise.local import denoise_to_dng
from tagmatti.io.dng import mosaic_rggb, write_dng
from tagmatti.lightroom.xmp import XmpDoc, serialize
from tagmatti.style.sources import embedded_xmp


def _hf(x):
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(float)
    return (g - cv2.GaussianBlur(g, (0, 0), 4)).std()


def test_local_denoise_to_linear_dng(tmp_path: Path):
    h, w = 400, 600
    _, xx = np.mgrid[0:h, 0:w]
    img = np.stack([0.1 + 0.2 * xx / w] * 3, -1) * np.array([0.5, 1, 0.65])
    rng = np.random.default_rng(1)
    raw = np.clip(img * 50000 + 800 + rng.normal(0, 900, img.shape), 0, 65535).astype(np.uint16)
    src = tmp_path / "n.dng"
    write_dng(src, mosaic_rggb(raw), cfa=True, black=800, white=60000, as_shot_neutral=(0.5, 1, 0.65),
              iso=12800, orientation=6)
    before = src.read_bytes()
    out = tmp_path / "n-dn.dng"
    res = denoise_to_dng(src, out, 60, xmp=serialize(XmpDoc(rating=4, crs={"Exposure2012": 0.3})))
    assert src.read_bytes() == before                     # Original unverändert
    a = rawpy.imread(str(src)).postprocess()
    b = rawpy.imread(str(out)).postprocess()
    assert a.shape == b.shape                              # Orientierung erhalten
    assert _hf(b) < _hf(a) * 0.5
    assert abs(float(b.mean()) - float(a.mean())) / float(a.mean()) < 0.05
    doc = embedded_xmp(out)
    assert doc.rating == 4 and doc.crs["Exposure2012"] == "+0.30"
    assert res["method"] in ("classical", "nafnet")

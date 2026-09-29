import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "imagomat_home"
    home.mkdir()
    monkeypatch.setenv("IMAGOMAT_HOME", str(home))
    monkeypatch.setenv("IMAGOMAT_OFFLINE", "1")
    # Tests laufen immer mit den schnellen Ersatzverfahren (keine Modell-Downloads, deterministisch)
    import json

    (home / "settings.json").write_text(json.dumps({
        "embedding_backend": "classical", "face_backend": "haar", "segmentation_backend": "classical",
        "ocr_backend": "none", "device": "cpu"}), "utf-8")
    # Caches von Backends zurücksetzen, damit jeder Test die Offline-Fallbacks nutzt
    from imagomat.vision import embeddings, faces, models, segmentation

    models.torch_device.cache_clear()
    faces.get_backend.cache_clear()
    embeddings.get_embedder.cache_clear()
    segmentation._birefnet.cache_clear()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    yield home

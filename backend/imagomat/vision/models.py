"""Modellverwaltung: Gerätewahl (MPS/CUDA/CPU), Download beim ersten Start, Lizenz-Manifest.

Jedes Modell hat einen Lizenzhinweis. Nicht-kommerzielle Modelle sind markiert, damit sie
vor einem späteren Verkauf der App ersetzt werden können.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import load_settings, models_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    filename: str
    license: str
    commercial_ok: bool
    sha256: str | None = None
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        "YuNet Gesichtserkennung",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "face_detection_yunet_2023mar.onnx", "MIT", True),
    "sface": ModelSpec(
        "SFace Gesichts-Embedding",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx",
        "face_recognition_sface_2021dec.onnx", "Apache-2.0", True),
    "mediapipe_face_landmarker": ModelSpec(
        "MediaPipe Face Landmarker (Augen offen/zu)",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
        "face_landmarker.task",
        "face_landmarker.task", "Apache-2.0", True),
    "laion_aesthetic": ModelSpec(
        "LAION Improved Aesthetic Predictor (auf CLIP ViT-L/14)",
        "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/"
        "sac+logos+ava1-l14-linearMSE.pth",
        "laion_aesthetic_l14.pth", "Apache-2.0", True),
    "nafnet_sidd": ModelSpec(
        "NAFNet SIDD width64 (Denoise, RGB)",
        "https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet-SIDD-width64.pth",
        "NAFNet-SIDD-width64.pth", "MIT", True,
        note="Falls der Download scheitert: Datei manuell aus dem NAFNet-Repository laden."),
}

# Modelle, die über ihre eigenen Bibliotheken geladen werden (nur Lizenz-Info)
LIBRARY_MODELS = {
    "insightface_buffalo_l": ("InsightFace buffalo_l", "nur nicht-kommerzielle Forschung", False),
    "clip_vit_l14": ("OpenAI CLIP ViT-L/14 (open_clip)", "MIT", True),
    "dinov2": ("DINOv2", "Apache-2.0", True),
    "birefnet": ("BiRefNet", "MIT", True),
    "easyocr": ("EasyOCR", "Apache-2.0", True),
    "apple_vision": ("Apple Vision (macOS)", "Systembestandteil", True),
}


def model_path(key: str, download: bool = True) -> Path | None:
    spec = MODELS[key]
    p = models_dir() / spec.filename
    if p.exists() and p.stat().st_size > 0:
        return p
    if not download or os.environ.get("IMAGOMAT_OFFLINE"):
        return None
    try:
        log.info("Lade Modell %s ...", spec.name)
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(spec.url, timeout=60) as resp, open(tmp, "wb") as f:
            h = hashlib.sha256()
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
        if spec.sha256 and h.hexdigest() != spec.sha256:
            tmp.unlink(missing_ok=True)
            log.warning("Prüfsumme für %s stimmt nicht", spec.name)
            return None
        # Git-LFS-Zeiger statt Datei erkennen
        if tmp.stat().st_size < 2048 and b"git-lfs" in tmp.read_bytes()[:200]:
            tmp.unlink(missing_ok=True)
            return None
        tmp.rename(p)
        return p
    except Exception as e:  # noqa: BLE001 - Netzwerkfehler aller Art -> Fallback
        log.warning("Modell %s nicht verfügbar: %s", spec.name, e)
        return None


@lru_cache(maxsize=1)
def torch_device() -> str:
    pref = load_settings().device
    try:
        import torch
    except ImportError:
        return "none"
    if pref in ("mps", "cuda", "cpu"):
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
    except ImportError:
        return []
    order = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail]


def license_report() -> list[dict[str, object]]:
    rows = [{"key": k, "name": s.name, "license": s.license, "commercial_ok": s.commercial_ok,
             "installed": (models_dir() / s.filename).exists()} for k, s in MODELS.items()]
    rows += [{"key": k, "name": n, "license": lic, "commercial_ok": ok, "installed": None}
             for k, (n, lic, ok) in LIBRARY_MODELS.items()]
    return rows

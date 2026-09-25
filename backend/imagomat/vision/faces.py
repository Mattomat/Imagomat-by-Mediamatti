"""Gesichtserkennung, Embeddings, Augen offen/zu, Kopfdrehung, Gesichts- und Augenschärfe.

Backends (in dieser Reihenfolge bei "auto"):
1. InsightFace buffalo_l: beste Qualität, **nur nicht-kommerzielle Nutzung**.
2. OpenCV YuNet + SFace: kommerziell nutzbar (MIT/Apache).
3. Haar-Kaskaden: immer verfügbar, schwach, nur als Notlösung (und für Tests).

Augen offen/zu: MediaPipe-Blendshapes, falls installiert, sonst Eye-Aspect-Ratio aus
Landmarken, sonst Haar-Augendetektor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from ..config import load_settings, models_dir
from . import quality
from .models import has_module, model_path, onnx_providers

log = logging.getLogger(__name__)


@dataclass
class Face:
    bbox: tuple[float, float, float, float]            # normiert, x0 y0 x1 y1
    score: float
    kps: list[tuple[float, float]] | None = None       # 5 Punkte normiert: Auge L/R (Bild), Nase, Mund L/R
    embedding: np.ndarray | None = None
    landmarks: np.ndarray | None = None                # (N, 2) normiert
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    eyes_open: float | None = None                     # 0..1
    sharpness: float = 0.0
    eye_sharpness: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "score": self.score, "kps": self.kps, "yaw": self.yaw,
                "pitch": self.pitch, "roll": self.roll, "eyes_open": self.eyes_open,
                "sharpness": self.sharpness, "eye_sharpness": self.eye_sharpness}


class FaceBackend(Protocol):
    name: str
    commercial_ok: bool

    def detect(self, img: np.ndarray) -> list[Face]: ...


def _yaw_from_kps(kps: list[tuple[float, float]]) -> float | None:
    if not kps or len(kps) < 3:
        return None
    (lx, ly), (rx, ry), (nx, ny) = kps[0], kps[1], kps[2]
    ied = max(abs(rx - lx), 1e-6)
    mid = (lx + rx) / 2
    return float(np.clip((nx - mid) / ied * 120.0, -90, 90))


class InsightFaceBackend:
    name = "insightface"
    commercial_ok = False

    def __init__(self, det_size: int = 1024):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", root=str(models_dir() / "insightface"),
                                providers=onnx_providers() or None)
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        out = []
        for f in self.app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            x0, y0, x1, y1 = [float(v) for v in f.bbox]
            kps = [(float(x) / w, float(y) / h) for x, y in f.kps] if f.kps is not None else None
            lm = getattr(f, "landmark_2d_106", None)
            pose = getattr(f, "pose", None)
            face = Face(
                bbox=(max(0, x0 / w), max(0, y0 / h), min(1, x1 / w), min(1, y1 / h)), score=float(f.det_score),
                kps=kps, embedding=np.asarray(f.normed_embedding, np.float32) if f.embedding is not None else None,
                landmarks=(np.asarray(lm, np.float32) / np.array([w, h], np.float32)) if lm is not None else None,
            )
            if pose is not None:
                face.pitch, face.yaw, face.roll = (float(v) for v in pose)
            else:
                face.yaw = _yaw_from_kps(kps or [])
            out.append(face)
        return out


class YuNetBackend:
    name = "yunet"
    commercial_ok = True

    def __init__(self):
        det = model_path("yunet")
        rec = model_path("sface")
        if det is None:
            raise RuntimeError("YuNet-Modell nicht verfügbar")
        self.det = cv2.FaceDetectorYN.create(str(det), "", (320, 320), score_threshold=0.6, nms_threshold=0.3)
        self.rec = cv2.FaceRecognizerSF.create(str(rec), "") if rec else None

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.det.setInputSize((w, h))
        _, rows = self.det.detect(bgr)
        out = []
        for row in (rows if rows is not None else []):
            x, y, bw, bh = row[:4]
            pts = row[4:14].reshape(5, 2)
            # YuNet: rechtes Auge der Person (links im Bild) zuerst
            kps = [(float(px) / w, float(py) / h) for px, py in pts]
            emb = None
            if self.rec is not None:
                aligned = self.rec.alignCrop(bgr, row)
                e = self.rec.feature(aligned).ravel().astype(np.float32)
                emb = e / (np.linalg.norm(e) + 1e-9)
            out.append(Face(bbox=(max(0, x / w), max(0, y / h), min(1, (x + bw) / w), min(1, (y + bh) / h)),
                            score=float(row[14]), kps=kps, embedding=emb, yaw=_yaw_from_kps(kps)))
        return out


class HaarBackend:
    """Notlösung ohne Modelldownload. Embedding = normierter Gesichtsausschnitt (schwach)."""

    name = "haar"
    commercial_ok = True

    def __init__(self):
        # OpenCV 5 hat die Haar-Kaskaden in opencv-contrib verschoben
        cc = getattr(cv2, "CascadeClassifier", None)
        base = getattr(getattr(cv2, "data", None), "haarcascades", "")
        self.face = cc(base + "haarcascade_frontalface_default.xml") if cc and base else None

    def detect(self, img: np.ndarray) -> list[Face]:
        if self.face is None:
            return []
        h, w = img.shape[:2]
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        min_size = max(20, int(min(h, w) * 0.03))
        boxes = list(self.face.detectMultiScale(g, 1.1, 5, minSize=(min_size, min_size)))
        out = []
        for (x, y, bw, bh) in boxes:
            crop = g[y:y + bh, x:x + bw]
            emb = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            emb = (emb - emb.mean()) / (emb.std() + 1e-6)
            emb /= np.linalg.norm(emb) + 1e-9
            out.append(Face(bbox=(x / w, y / h, (x + bw) / w, (y + bh) / h), score=0.5, embedding=emb, yaw=0.0))
        return out


@lru_cache(maxsize=4)
def get_backend(name: str | None = None) -> FaceBackend:
    name = name or load_settings().face_backend
    order = ["insightface", "yunet", "haar"] if name == "auto" else [name, "haar"]
    for n in order:
        try:
            if n == "insightface" and has_module("insightface"):
                return InsightFaceBackend()
            if n == "yunet":
                return YuNetBackend()
            if n == "haar":
                return HaarBackend()
        except Exception as e:  # noqa: BLE001 - Backend nicht verfügbar -> nächstes
            log.warning("Gesichts-Backend %s nicht verfügbar: %s", n, e)
    return HaarBackend()


# ---------------------------------------------------------------------------
# Augen offen/zu
# ---------------------------------------------------------------------------

class _MediaPipeEyes:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        p = model_path("mediapipe_face_landmarker")
        if p is None:
            raise RuntimeError("MediaPipe-Modell fehlt")
        self.mp = mp
        self.lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(p)), output_face_blendshapes=True, num_faces=1))

    def openness(self, crop: np.ndarray) -> float | None:
        if min(crop.shape[:2]) < 32:
            return None
        crop = cv2.resize(crop, (256, int(256 * crop.shape[0] / crop.shape[1])))
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
        if not res.face_blendshapes:
            return None
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
        blink = max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))
        return float(1.0 - blink)


@lru_cache(maxsize=1)
def _mediapipe() -> _MediaPipeEyes | None:
    if not has_module("mediapipe"):
        return None
    try:
        return _MediaPipeEyes()
    except Exception as e:  # noqa: BLE001
        log.info("MediaPipe nicht verfügbar: %s", e)
        return None


@lru_cache(maxsize=1)
def _eye_cascade():
    cc = getattr(cv2, "CascadeClassifier", None)
    base = getattr(getattr(cv2, "data", None), "haarcascades", "")
    return cc(base + "haarcascade_eye_tree_eyeglasses.xml") if cc and base else None


def eye_aspect_from_landmarks(face: Face, aspect: float) -> float | None:
    """Eye-Aspect-Ratio aus dichten Landmarken, die nahe der 5-Punkt-Augenzentren liegen.

    Unabhängig von der Punktnummerierung des Modells: wir nehmen alle Landmarken im Umkreis
    von 22 % des Augenabstands um jedes Augenzentrum.
    """
    if face.landmarks is None or not face.kps:
        return None
    lm = face.landmarks * np.array([aspect, 1.0])
    ears = []
    eyes = [np.array(face.kps[0]) * [aspect, 1.0], np.array(face.kps[1]) * [aspect, 1.0]]
    ied = float(np.linalg.norm(eyes[0] - eyes[1]))
    if ied <= 0:
        return None
    roll = np.arctan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])
    rot = np.array([[np.cos(-roll), -np.sin(-roll)], [np.sin(-roll), np.cos(-roll)]])
    for c in eyes:
        pts = lm[np.linalg.norm(lm - c, axis=1) < 0.22 * ied]
        if len(pts) < 4:
            continue
        p = (pts - c) @ rot.T
        width = p[:, 0].max() - p[:, 0].min()
        height = p[:, 1].max() - p[:, 1].min()
        if width > 0:
            ears.append(height / width)
    return float(np.mean(ears)) if ears else None


def eyes_open_score(img: np.ndarray, face: Face, ear_threshold: float = 0.22) -> float | None:
    h, w = img.shape[:2]
    if face.height * h < 28:
        return None  # zu klein für eine verlässliche Aussage
    crop = quality.crop_region(img, face.bbox, pad=0.25)
    mp = _mediapipe()
    if mp is not None:
        v = mp.openness(crop)
        if v is not None:
            return v
    ear = eye_aspect_from_landmarks(face, w / h)
    if ear is not None:
        # weicher Übergang um die Schwelle
        return float(1 / (1 + np.exp(-(ear - ear_threshold) * 40)))
    cascade = _eye_cascade()
    if cascade is None:
        return None
    upper = crop[: max(1, crop.shape[0] * 3 // 5)]
    g = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    ms = max(8, g.shape[1] // 8)
    eyes = cascade.detectMultiScale(g, 1.1, 3, minSize=(ms, ms))
    return {0: 0.35, 1: 0.6}.get(len(eyes), 0.85)


def face_quality(img: np.ndarray, face: Face, ear_threshold: float = 0.22) -> Face:
    """Ergänzt Schärfe, Augenschärfe und Augen-offen-Wert."""
    g = quality.gray(img)
    face.sharpness = quality.sharpness(quality.crop_region(g, face.bbox), reference_side=256)
    if face.kps:
        h, w = g.shape[:2]
        fw = face.bbox[2] - face.bbox[0]
        vals = []
        for (ex, ey) in face.kps[:2]:
            r = fw * 0.18
            box = (ex - r, ey - r * w / h, ex + r, ey + r * w / h)
            vals.append(quality.sharpness(quality.crop_region(g, box), reference_side=128))
        face.eye_sharpness = float(max(vals)) if vals else face.sharpness
    else:
        face.eye_sharpness = face.sharpness
    face.eyes_open = eyes_open_score(img, face, ear_threshold)
    return face


def detect_faces(img: np.ndarray, backend: str | None = None, ear_threshold: float = 0.22,
                 min_score: float = 0.5) -> list[Face]:
    faces = [f for f in get_backend(backend).detect(img) if f.score >= min_score]
    return [face_quality(img, f, ear_threshold) for f in faces]

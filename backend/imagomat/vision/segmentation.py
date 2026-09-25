"""Motiv- und Himmelsmasken.

Wofür wir eigene Masken brauchen, obwohl Lightroom KI-Masken selbst berechnet:
- Merkmale fürs Stilmodell (Motivhelligkeit vs. Hintergrund, Himmelsanteil),
- Platzierung von Radial-/Verlaufsfiltern und Zuschnitt,
- Vorschau in der App,
- Pinsel-Fallback (Mask/Paint), falls KI-Masken-Deklarationen nicht funktionieren.

Backends: BiRefNet (MIT, über transformers) oder klassisch (Gesichter -> Körper, Saliency, GrabCut).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..config import load_settings
from .models import has_module, torch_device

log = logging.getLogger(__name__)
WORK = 512  # Arbeitsauflösung der Masken (lange Seite)


@dataclass
class Segmentation:
    subject: np.ndarray            # float32 0..1, (h, w) in Arbeitsauflösung
    sky: np.ndarray
    subject_bbox: tuple[float, float, float, float] | None

    def stats(self, img: np.ndarray) -> dict[str, float]:
        small = cv2.resize(img, (self.subject.shape[1], self.subject.shape[0]), interpolation=cv2.INTER_AREA)
        y = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        s = self.subject
        bg = 1 - np.maximum(s, self.sky)
        def wmean(v, w):
            return float((v * w).sum() / max(w.sum(), 1e-6))
        return {
            "subject_fraction": float(s.mean()), "sky_fraction": float(self.sky.mean()),
            "subject_luma": wmean(y, s), "background_luma": wmean(y, bg), "sky_luma": wmean(y, self.sky),
            "subject_cx": float(((self.subject_bbox or (0, 0, 1, 1))[0] + (self.subject_bbox or (0, 0, 1, 1))[2]) / 2),
            "subject_cy": float(((self.subject_bbox or (0, 0, 1, 1))[1] + (self.subject_bbox or (0, 0, 1, 1))[3]) / 2),
        }


def _resize_work(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = WORK / max(h, w)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def mask_bbox(m: np.ndarray, thr: float = 0.5) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(m > thr)
    if len(xs) < 10:
        return None
    h, w = m.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def body_boxes(face_boxes: list[tuple[float, float, float, float]], aspect: float) -> list[tuple[float, ...]]:
    """Grobe Körperbox aus Gesichtsbox (Sport: ganze Person ~ 7 Kopfhöhen)."""
    out = []
    for x0, y0, x1, y1 in face_boxes:
        fw, fh = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        out.append((max(0, cx - fw * 1.8), max(0, y0 - fh * 0.3), min(1, cx + fw * 1.8), min(1, y1 + fh * 6.5)))
    return out


def sky_mask_classical(img: np.ndarray) -> np.ndarray:
    small = _resize_work(img)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    tex = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    tex = cv2.blur(tex, (9, 9))
    h, w = g.shape
    bright = hsv[..., 2] > 110
    blueish = ((hsv[..., 0] > 85) & (hsv[..., 0] < 135)) | (hsv[..., 1] < 40)
    cand = (bright & blueish & (tex < 0.03)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        comp = lab == i
        if comp[0].any() and comp.sum() > 0.01 * h * w:  # muss den oberen Rand berühren
            keep |= comp.astype(np.uint8)
    return cv2.GaussianBlur(keep.astype(np.float32), (7, 7), 0)


def _saliency(small: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (64, 64))
    f = np.fft.fft2(g)
    log_amp = np.log(np.abs(f) + 1e-6)
    spectral = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(spectral + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)
    sal = cv2.resize(sal, (small.shape[1], small.shape[0]))
    return sal / (sal.max() + 1e-9)


def subject_mask_classical(img: np.ndarray, face_boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    small = _resize_work(img)
    h, w = small.shape[:2]
    prior = np.zeros((h, w), np.float32)
    if face_boxes:
        for x0, y0, x1, y1 in body_boxes(face_boxes, w / h):
            prior[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
    else:
        sal = _saliency(small)
        yy, xx = np.mgrid[0:h, 0:w]
        center = np.exp(-(((xx / w - 0.5) ** 2) + ((yy / h - 0.5) ** 2)) / 0.08)
        s = sal * 0.6 + center * 0.4
        prior = (s > np.percentile(s, 80)).astype(np.float32)
    if prior.sum() < 20:
        return prior
    # GrabCut verfeinert die Box-/Saliency-Initialisierung
    gc = np.where(prior > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    ring = cv2.dilate(prior, np.ones((15, 15), np.uint8)) > 0
    gc[ring & (prior == 0)] = cv2.GC_PR_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    except cv2.error:
        m = prior
    return cv2.GaussianBlur(m, (5, 5), 0)


class BiRefNetSegmenter:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.torch = torch
        self.device = torch_device()
        self.model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        if self.device in ("mps", "cuda"):
            self.model = self.model.half()

    def subject(self, img: np.ndarray) -> np.ndarray:
        x = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(self.device)
        if self.device in ("mps", "cuda"):
            t = t.half()
        with self.torch.no_grad():
            pred = self.model(t)[-1].sigmoid().float().cpu().numpy()[0, 0]
        small = _resize_work(img)
        return cv2.resize(pred, (small.shape[1], small.shape[0]))


@lru_cache(maxsize=1)
def _birefnet() -> BiRefNetSegmenter | None:
    if load_settings().segmentation_backend not in ("auto", "birefnet") or not has_module("transformers"):
        return None
    try:
        return BiRefNetSegmenter()
    except Exception as e:  # noqa: BLE001
        log.warning("BiRefNet nicht verfügbar: %s", e)
        return None


def segment(img: np.ndarray, face_boxes: list[tuple[float, float, float, float]] | None = None) -> Segmentation:
    face_boxes = face_boxes or []
    net = _birefnet()
    subj = net.subject(img) if net is not None else subject_mask_classical(img, face_boxes)
    sky = sky_mask_classical(img) * (1 - subj)
    return Segmentation(subject=subj.astype(np.float32), sky=sky.astype(np.float32), subject_bbox=mask_bbox(subj))

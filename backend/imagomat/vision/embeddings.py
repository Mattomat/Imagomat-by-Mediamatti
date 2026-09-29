"""Bild-Embeddings, Szenen-Erkennung (Zero-Shot) und Ästhetik-Score.

Backends:
- "clip": OpenAI CLIP ViT-L/14 über open_clip (MIT). Liefert zusätzlich Zero-Shot-Szenen und,
  mit dem LAION-Kopf (Apache-2.0), einen Ästhetik-Score.
- "classical": Farb-/Gradienten-Histogramme. Immer verfügbar, gut genug für Serien-
  und Duplikaterkennung, nicht für Ästhetik.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import cv2
import numpy as np

from ..config import load_settings
from .models import has_module, model_path, torch_device

log = logging.getLogger(__name__)

# Situationen für die mitgelieferten Presets (siehe style/presets.py)
SCENE_PROMPTS: dict[str, list[str]] = {
    "sport_day": ["a photo of a football match on a sunny day", "outdoor soccer game in daylight",
                  "athletes playing sports outdoors during the day"],
    "sport_floodlight": ["a football match at night under stadium floodlights",
                         "soccer players in a stadium at night"],
    "sport_indoor": ["indoor sports in a gymnasium", "basketball or handball in a sports hall",
                     "ice hockey in an arena"],
    "concert_stage": ["a concert on a big stage with colorful stage lights", "a band performing live on stage",
                      "a singer on stage with spotlights"],
    "concert_club": ["a musician performing in a small dark club", "a dj in a nightclub with laser lights"],
    "portrait": ["a portrait photo of a person", "a headshot of a person"],
    "event": ["people at a party or corporate event", "guests at an indoor event"],
    "wedding": ["a wedding ceremony", "a bride and groom"],
    "landscape": ["a landscape photo", "a city skyline", "mountains and nature"],
}


class ClassicalEmbedder:
    name = "classical"
    dim = 128 + 192 + 128

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self._one(im) for im in images]).astype(np.float32)

    @staticmethod
    def _one(img: np.ndarray) -> np.ndarray:
        small = cv2.resize(img, (256, int(256 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hist = np.sqrt(hist / max(hist.sum(), 1))
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        thumb = cv2.resize(g, (16, 12), interpolation=cv2.INTER_AREA).ravel()
        thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6) / np.sqrt(thumb.size)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = (ang / (2 * np.pi) * 8).astype(int) % 8
        h, w = g.shape
        hog = []
        for i in range(4):
            for j in range(4):
                sl = (slice(i * h // 4, (i + 1) * h // 4), slice(j * w // 4, (j + 1) * w // 4))
                hog.append(np.bincount(bins[sl].ravel(), weights=mag[sl].ravel(), minlength=8))
        hog = np.concatenate(hog)
        hog = np.sqrt(hog / max(hog.sum(), 1e-6))
        v = np.concatenate([hist * 0.8, thumb * 0.6, hog * 0.8])
        return v / (np.linalg.norm(v) + 1e-9)

    def scenes(self, images: list[np.ndarray]) -> list[dict[str, float]] | None:
        return None

    def aesthetic(self, embs: np.ndarray) -> np.ndarray | None:
        return None


class ClipEmbedder:
    name = "clip"

    def __init__(self, model: str = "ViT-L-14", pretrained: str = "openai"):
        import open_clip
        import torch

        self.torch = torch
        self.device = torch_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        try:
            self.model = self.model.to(self.device).eval()
        except RuntimeError as e:  # z. B. MPS in virtuellen Maschinen nicht nutzbar
            log.warning("CLIP auf %s nicht möglich (%s), nutze CPU", self.device, e)
            self.device = "cpu"
            self.model = self.model.to("cpu").eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.dim = int(self.model.visual.output_dim)
        self._scene_keys: list[str] = []
        self._scene_emb = None
        self._aesthetic = self._load_aesthetic()

    def _load_aesthetic(self):
        if self.dim != 768:
            return None
        p = model_path("laion_aesthetic")
        if p is None:
            return None
        nn = self.torch.nn
        mlp = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
        sd = self.torch.load(p, map_location="cpu")
        sd = {k.replace("layers.", ""): v for k, v in sd.items()}
        mlp.load_state_dict(sd)
        return mlp.to(self.device).eval()

    def embed(self, images: list[np.ndarray], batch: int = 16) -> np.ndarray:
        from PIL import Image

        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                x = self.torch.stack([self.preprocess(Image.fromarray(im)) for im in images[i:i + batch]])
                f = self.model.encode_image(x.to(self.device)).float()
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def _scenes_text(self):
        if self._scene_emb is None:
            keys, prompts = [], []
            for k, ps in SCENE_PROMPTS.items():
                for p in ps:
                    keys.append(k)
                    prompts.append(p)
            with self.torch.no_grad():
                t = self.model.encode_text(self.tokenizer(prompts).to(self.device)).float()
                t = t / t.norm(dim=-1, keepdim=True)
            self._scene_keys, self._scene_emb = keys, t.cpu().numpy()
        return self._scene_keys, self._scene_emb

    def scenes_from_embeddings(self, embs: np.ndarray) -> list[dict[str, float]]:
        keys, temb = self._scenes_text()
        logits = 100.0 * embs @ temb.T
        out = []
        for row in logits:
            per: dict[str, float] = {}
            for k, v in zip(keys, row):
                per[k] = max(per.get(k, -1e9), float(v))
            vals = np.array(list(per.values()))
            p = np.exp(vals - vals.max())
            p /= p.sum()
            out.append(dict(zip(per.keys(), p.tolist())))
        return out

    def aesthetic(self, embs: np.ndarray) -> np.ndarray | None:
        if self._aesthetic is None:
            return None
        with self.torch.no_grad():
            s = self._aesthetic(self.torch.from_numpy(embs).to(self.device)).cpu().numpy().ravel()
        return np.clip((s - 3.0) / 5.0, 0, 1)


_LOCK = threading.RLock()   # zwei Job-Spuren teilen sich ein Modell


class _Locked:
    """Serialisiert Aufrufe auf ein geteiltes Modell (Torch/MPS ist nicht threadsicher)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if callable(attr) and name in ("embed", "scenes_from_embeddings", "aesthetic", "scenes"):
            def call(*a, **k):
                with _LOCK:
                    return attr(*a, **k)
            return call
        return attr


def get_embedder(name: str | None = None):
    with _LOCK:
        return _get_embedder(name)


get_embedder.cache_info = lambda: _get_embedder.cache_info()  # type: ignore[attr-defined]
get_embedder.cache_clear = lambda: _get_embedder.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=2)
def _get_embedder(name: str | None = None):
    name = name or load_settings().embedding_backend
    if name in ("auto", "clip") and has_module("open_clip"):
        try:
            return _Locked(ClipEmbedder())
        except Exception as e:  # noqa: BLE001
            log.warning("CLIP nicht verfügbar (%s), nutze klassische Embeddings", e)
    return ClassicalEmbedder()


def heuristic_scene(meta: dict) -> dict[str, float]:
    """Szenen-Schätzung ohne KI aus Belichtung, Farbe und EXIF (Fallback)."""
    iso = meta.get("iso") or 100
    median = meta.get("median", 0.4)
    neon = meta.get("neon_fraction", 0.0)
    temp = meta.get("as_shot_temp") or 5500
    faces = meta.get("face_count", 0)
    face_h = meta.get("max_face_height", 0.0)
    s = {k: 0.02 for k in SCENE_PROMPTS}
    if neon > 0.03 or (iso >= 3200 and median < 0.3 and meta.get("sat_p95", 0) > 0.8):
        s["concert_stage"] += 0.6
        s["concert_club"] += 0.3 if median < 0.2 else 0.1
    elif iso >= 2500 and temp < 4800:
        s["sport_indoor"] += 0.4
        s["sport_floodlight"] += 0.35
    elif iso >= 2500:
        s["sport_floodlight"] += 0.5
    else:
        s["sport_day"] += 0.5
        s["landscape"] += 0.1 if faces == 0 else 0.0
    if face_h > 0.2:
        s["portrait"] += 0.5
    tot = sum(s.values())
    return {k: v / tot for k, v in s.items()}

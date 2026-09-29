"""Action-Momente erkennen: Zweikampf, Schuss, Kopfball, Parade, Jubel, Bühnenmoment.

Drei Quellen, je nachdem, was verfügbar ist:

- CLIP (Zero-Shot): vergleicht das Bild-Embedding mit Beschreibungen typischer Momente
  ("zwei Spieler kämpfen um den Ball") und langweiliger Bilder ("Spieler stehen herum").
- Pose (torchvision Keypoint R-CNN, BSD-3): Körperhaltungen der Spieler. Ein Fuss hoch =
  Schuss, zwei Spieler überlappen = Zweikampf, Arme über dem Kopf = Jubel/Kopfball,
  gestreckter Körper = Parade/Grätsche. Läuft nur mit GPU (MPS/CUDA), sonst zu langsam.
- Klassisch: Gesichter und Motivgrösse (grober Ersatz ohne KI).

Ergebnis pro Bild: ``action`` (0..1) und ``moment`` (z. B. "zweikampf").
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..config import load_settings
from .models import has_module, torch_device

log = logging.getLogger(__name__)

MOMENTS_DE = {
    "zweikampf": "Zweikampf",
    "schuss": "Schuss",
    "kopfball": "Kopfball",
    "parade": "Parade",
    "jubel": "Jubel",
    "dribbling": "Dribbling",
    "emotion": "Emotion",
    "performance": "Bühnenmoment",
    "publikum": "Publikum",
    "dynamik": "Dynamik",
}

# (Moment, positiv?) -> Beschreibungen für CLIP
ACTION_PROMPTS: dict[tuple[str, bool], list[str]] = {
    ("zweikampf", True): ["two football players fighting for the ball", "a hard tackle in a soccer match",
                          "two players challenging each other for the ball in a duel",
                          "players battling shoulder to shoulder for the ball"],
    ("schuss", True): ["a football player shooting the ball at goal", "a soccer player kicking the ball hard",
                       "a striker taking a powerful shot", "a player volleying the ball"],
    ("kopfball", True): ["a football player heading the ball", "players jumping high for a header"],
    ("parade", True): ["a goalkeeper diving to save the ball", "a goalkeeper jumping to catch the ball",
                       "a sliding tackle on the grass"],
    ("jubel", True): ["football players celebrating a goal", "a player screaming with joy and arms raised",
                      "teammates hugging and celebrating", "fans and players cheering"],
    ("dribbling", True): ["a soccer player dribbling the ball at full speed", "a player sprinting with the ball",
                          "a winger running past a defender"],
    ("emotion", True): ["a frustrated football player with hands on his head", "an emotional close-up of an athlete",
                        "a referee showing a card to a player"],
    ("performance", True): ["a singer passionately singing into a microphone", "a guitarist playing an intense solo",
                            "a musician jumping high on stage", "a drummer hitting the drums with energy"],
    ("publikum", True): ["an excited crowd at a concert with hands in the air"],
    ("ruhig", False): ["football players standing around on the pitch", "a soccer player walking slowly",
                       "players waiting before a free kick", "a player standing still, seen from behind",
                       "the substitutes bench", "a wide shot of the football pitch with small players far away",
                       "spectators sitting in the stands", "players jogging without the ball",
                       "a musician standing still on stage", "an empty stage with lights", "a blurry photo"],
}

POSITIVE = [k for (k, pos) in ACTION_PROMPTS if pos]


@dataclass
class PoseResult:
    persons: int
    duel: float
    kick: float
    arms_up: float
    stretch: float

    @property
    def score(self) -> float:
        return float(max(self.duel, self.kick, 0.8 * self.arms_up, 0.85 * self.stretch))

    @property
    def moment(self) -> str | None:
        vals = {"zweikampf": self.duel, "schuss": self.kick, "jubel": 0.8 * self.arms_up,
                "dynamik": 0.85 * self.stretch}
        k = max(vals, key=vals.get)
        return k if vals[k] >= 0.45 else None

    def to_dict(self) -> dict[str, float]:
        return {"persons": self.persons, "duel": self.duel, "kick": self.kick, "arms_up": self.arms_up,
                "stretch": self.stretch}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

def clip_action(embedder: Any, embs: np.ndarray) -> list[tuple[float, str | None]]:
    """Wahrscheinlichkeit für einen Action-Moment aus CLIP-Bild-Embeddings."""
    temb, keys = _clip_text(embedder)
    logits = 100.0 * embs @ temb.T
    out = []
    for row in logits:
        per: dict[str, float] = {}
        for k, v in zip(keys, row):
            per[k] = max(per.get(k, -1e9), float(v))
        names = list(per)
        vals = np.array([per[n] for n in names])
        p = np.exp(vals - vals.max())
        p /= p.sum()
        prob = dict(zip(names, p.tolist()))
        pos = sum(prob.get(k, 0.0) for k in POSITIVE)
        best = max(POSITIVE, key=lambda k: prob.get(k, 0.0))
        out.append((float(pos), best if prob.get(best, 0.0) >= 0.2 else None))
    return out


def _clip_text(embedder: Any):
    cached = getattr(embedder, "_action_text", None)
    if cached is not None:
        return cached
    torch = embedder.torch
    keys, prompts = [], []
    for (k, _pos), ps in ACTION_PROMPTS.items():
        for p in ps:
            keys.append(k)
            prompts.append(p)
    with torch.no_grad():
        t = embedder.model.encode_text(embedder.tokenizer(prompts).to(embedder.device)).float()
        t = t / t.norm(dim=-1, keepdim=True)
    embedder._action_text = (t.cpu().numpy(), keys)
    return embedder._action_text


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

class PoseDetector:
    """Personen + 17 Körperpunkte (COCO) mit torchvision Keypoint R-CNN."""

    def __init__(self) -> None:
        import torch
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

        self.torch = torch
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        self.device = torch_device()
        try:
            self.model.to(self.device)
            self._run([np.zeros((64, 96, 3), np.uint8)])
        except Exception as e:  # noqa: BLE001 - manche Operatoren fehlen auf MPS
            log.info("Pose auf %s nicht möglich (%s), nutze CPU", self.device, e)
            self.device = "cpu"
            self.model.to("cpu")

    def _run(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        torch = self.torch
        with torch.no_grad():
            xs = [torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(self.device) for im in images]
            return [{k: v.cpu().numpy() for k, v in o.items()} for o in self.model(xs)]

    def analyze(self, images: list[np.ndarray], side: int = 640) -> list[PoseResult]:
        import cv2

        small = []
        for im in images:
            s = side / max(im.shape[:2])
            small.append(cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
                         if s < 1 else im)
        outs = self._run(small)
        return [pose_features(o, im.shape[1], im.shape[0]) for o, im in zip(outs, small)]


def pose_features(out: dict[str, Any], w: int, h: int, min_score: float = 0.75) -> PoseResult:
    """Action-Merkmale aus erkannten Personen (Boxen + Keypoints, Pixelkoordinaten)."""
    boxes = np.asarray(out.get("boxes", np.zeros((0, 4))), dtype=float)
    scores = np.asarray(out.get("scores", np.zeros(0)), dtype=float)
    kps = np.asarray(out.get("keypoints", np.zeros((0, 17, 3))), dtype=float).copy()
    if "keypoints_scores" in out and len(kps):
        # Sichtbarkeit aus der Keypoint-Konfidenz (torchvision liefert sonst immer 1)
        kps[:, :, 2] = (np.asarray(out["keypoints_scores"], dtype=float) > 1.0).astype(float)
    keep = [i for i in range(len(boxes)) if scores[i] >= min_score and (boxes[i, 3] - boxes[i, 1]) / h > 0.12]
    boxes, kps = boxes[keep], kps[keep]
    if len(boxes) == 0:
        return PoseResult(0, 0.0, 0.0, 0.0, 0.0)
    heights = (boxes[:, 3] - boxes[:, 1]) / h
    # Grosse Personen zählen mehr (Spieler im Vordergrund statt Zuschauer)
    size_w = np.clip((heights - 0.12) / 0.35, 0.2, 1.0)

    kick = arms = stretch = 0.0
    for b, k, sw in zip(boxes, kps, size_w):
        bh = max(b[3] - b[1], 1.0)
        bw = b[2] - b[0]
        la, ra, lk, rk = k[15], k[16], k[13], k[14]
        # Ein Fuss deutlich höher als der andere (Schuss, Sprint, Sprung)
        if la[2] > 0 and ra[2] > 0:
            lift = abs(la[1] - ra[1]) / bh
            kick = max(kick, sw * _clip01((lift - 0.12) / 0.25))
            # Fuss über dem Knie des anderen Beins = echter Schuss/Volley
            if min(la[1], ra[1]) < max(lk[1], rk[1]):
                kick = max(kick, sw * 0.9)
        nose, lw, rw = k[0], k[9], k[10]
        ls, rs = k[5], k[6]
        top = nose[1] if nose[2] > 0 else min(ls[1], rs[1]) - 0.1 * bh
        up = sum(1 for wr in (lw, rw) if wr[2] > 0 and wr[1] < top)
        arms = max(arms, sw * (0.6 * up if up < 2 else 1.0))
        stretch = max(stretch, sw * _clip01((bw / bh - 0.7) / 0.6))

    duel = 0.0
    order = np.argsort(-heights)
    for a_i in range(min(3, len(order))):
        for b_i in range(a_i + 1, min(4, len(order))):
            i, j = order[a_i], order[b_i]
            bi, bj = boxes[i], boxes[j]
            if min(heights[i], heights[j]) < 0.2:
                continue
            ix = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
            gap = max(bi[0], bj[0]) - min(bi[2], bj[2])
            mean_w = 0.5 * ((bi[2] - bi[0]) + (bj[2] - bj[0]))
            similar = min(heights[i], heights[j]) / max(heights[i], heights[j])  # ähnliche Tiefe
            close = 1.0 if ix > 0 else _clip01(1.0 - gap / max(mean_w * 0.6, 1.0))
            duel = max(duel, close * _clip01((similar - 0.55) / 0.3) * min(size_w[i], size_w[j]) ** 0.5)
    return PoseResult(int(len(boxes)), round(duel, 3), round(kick, 3), round(arms, 3), round(stretch, 3))


def _pose_wanted() -> bool:
    mode = getattr(load_settings(), "action_backend", "auto")
    if mode in ("none", "clip") or os.environ.get("IMAGOMAT_OFFLINE"):
        return False
    if not (has_module("torch") and has_module("torchvision")):
        return False
    return mode == "pose" or torch_device() in ("mps", "cuda")


@lru_cache(maxsize=1)
def get_pose_detector() -> PoseDetector | None:
    if not _pose_wanted():
        return None
    try:
        return PoseDetector()
    except Exception as e:  # noqa: BLE001
        log.warning("Pose-Erkennung nicht verfügbar: %s", e)
        return None


# ---------------------------------------------------------------------------
# Klassischer Ersatz und Kombination
# ---------------------------------------------------------------------------

def classical_action(a: dict[str, Any]) -> float:
    """Grobe Schätzung ohne KI: mehrere grosse Gesichter nah beieinander, grosses Motiv."""
    faces = int(a.get("face_count") or 0)
    fh = float(a.get("max_face_height") or 0.0)
    subj = float(a.get("subject_fraction") or 0.0)
    s = 0.3 * _clip01(subj * 3) + 0.3 * _clip01(fh / 0.12)
    if faces >= 2:
        s += 0.25
    return _clip01(s + 0.1 * _clip01(float(a.get("motion_anisotropy") or 0.0)))


def combine(a: dict[str, Any], clip: tuple[float, str | None] | None, pose: PoseResult | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    parts: list[tuple[float, float]] = []
    moment = None
    if clip is not None:
        out["action_clip"] = round(clip[0], 4)
        parts.append((0.55, clip[0]))
        moment = clip[1]
    if pose is not None:
        out["pose"] = pose.to_dict()
        out["action_pose"] = round(pose.score, 4)
        parts.append((0.45, pose.score))
        if moment is None or (pose.moment == "zweikampf" and pose.duel > 0.7):
            moment = pose.moment or moment
    if parts:
        out["action"] = round(sum(w * v for w, v in parts) / sum(w for w, _ in parts), 4)
        out["action_source"] = "+".join(n for n, x in (("clip", clip), ("pose", pose)) if x is not None)
    else:
        out["action"] = round(classical_action(a), 4)
        out["action_source"] = "classical"
    out["moment"] = moment if moment in MOMENTS_DE else None
    return out

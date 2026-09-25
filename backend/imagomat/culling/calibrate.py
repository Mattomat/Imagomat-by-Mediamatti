"""Culling an die eigene Auswahl anpassen und messen.

Grundwahrheit: Bilder, die du in Lightroom behalten hast (Pick-Flag, Sterne >= Schwelle oder
Farblabel), gegenüber allen Bildern desselben Ordners. Das Modell ist eine logistische
Regression auf den (shoot-normierten) Culling-Merkmalen: robust mit wenigen hundert
Beispielen und erklärbar (Gewicht pro Merkmal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import profiles_dir
from .engine import FEATURE_NAMES, CullItem


@dataclass
class CullingModel:
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]

    def predict(self, feats: dict[str, float]) -> float:
        x = (np.array([feats.get(k, 0.0) for k in FEATURE_NAMES]) - self.mean) / self.std
        return float(1 / (1 + np.exp(-(x @ np.array(self.weights) + self.bias))))

    def save(self, profile: str) -> Path:
        p = profiles_dir() / profile / "culling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, indent=2), "utf-8")
        return p

    @classmethod
    def load(cls, profile: str) -> "CullingModel | None":
        p = profiles_dir() / profile / "culling.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text("utf-8")))


def fit(items: list[CullItem], labels: list[int]) -> CullingModel:
    from sklearn.linear_model import LogisticRegression

    X = np.array([[it.feats.get(k, 0.0) for k in FEATURE_NAMES] for it in items], dtype=float)
    y = np.asarray(labels)
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500)
    clf.fit((X - mean) / std, y)
    return CullingModel(clf.coef_[0].tolist(), float(clf.intercept_[0]), mean.tolist(), std.tolist())


def agreement(items: list[CullItem], labels: dict[int, int]) -> dict[str, float]:
    """Übereinstimmung der App-Auswahl mit deiner eigenen Auswahl."""
    tp = sum(1 for it in items if it.keep and labels.get(it.image_id))
    fp = sum(1 for it in items if it.keep and not labels.get(it.image_id))
    fn = sum(1 for it in items if not it.keep and labels.get(it.image_id))
    tn = sum(1 for it in items if not it.keep and not labels.get(it.image_id))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "accuracy": (tp + tn) / max(len(items), 1), "kept_by_you": tp + fn, "kept_by_app": tp + fp}

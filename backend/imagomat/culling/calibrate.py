"""Culling an die eigene Auswahl anpassen und messen.

Grundwahrheit: Bilder, die du in Lightroom behalten hast (Pick-Flag, Sterne >= Schwelle oder
Farblabel), gegenüber allen Bildern desselben Ordners. Das Modell ist eine logistische
Regression auf den (shoot-normierten) Culling-Merkmalen: robust mit wenigen hundert
Beispielen und erklärbar (Gewicht pro Merkmal).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import profiles_dir
from .engine import FEATURE_NAMES, LEGACY_FEATURES, CullItem


@dataclass
class CullingModel:
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    features: list[str] = field(default_factory=lambda: list(LEGACY_FEATURES))  # ältere Modelle ohne "action"

    def predict(self, feats: dict[str, float]) -> float:
        x = (np.array([feats.get(k, 0.0) for k in self.features]) - self.mean) / self.std
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
    return CullingModel(clf.coef_[0].tolist(), float(clf.intercept_[0]), mean.tolist(), std.tolist(),
                        list(FEATURE_NAMES))


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


def user_labels_from_folder(folder: Path, min_rating: int) -> dict[str, int]:
    """Deine Auswahl aus XMP-Sidecars: behalten = Sterne >= min_rating."""
    from ..lightroom.xmp import read_xmp

    out = {}
    for x in Path(folder).rglob("*.xmp"):
        try:
            d = read_xmp(x)
        except Exception:  # noqa: BLE001
            continue
        for ext in (".ARW", ".arw", ".CR3", ".cr3", ".NEF", ".nef", ".DNG", ".dng", ".RAF", ".raf"):
            p = x.with_suffix(ext)
            if p.exists():
                out[str(p.resolve())] = int((d.rating or 0) >= min_rating)
    return out


def user_labels_from_catalog(catalog: Path, min_rating: int) -> dict[str, int]:
    from ..lightroom.catalog import CatalogReader

    with CatalogReader(catalog) as r:
        return {str(im.path.resolve()): int(im.pick > 0 or (im.rating or 0) >= min_rating) for im in r.images()}


def _register_job():
    from ..analysis import analyze_shoot, import_folder
    from ..jobs import JobContext, job
    from .engine import build_features, cull, load_items

    @job("calibrate_culling")
    def calibrate_job(ctx: JobContext, profile: str, folders: list[str], catalog: str | None = None,
                      min_rating: int = 2) -> None:
        from ..config import load_settings

        db = ctx.db
        labels: dict[str, int] = {}
        if catalog:
            labels.update(user_labels_from_catalog(Path(catalog), min_rating))
        items: list[CullItem] = []
        y: list[int] = []
        paths: dict[int, str] = {}
        for f in folders:
            labels.update({k: v for k, v in user_labels_from_folder(Path(f), min_rating).items() if k not in labels})
            sid = import_folder(db, Path(f), name=f"Kalibrierung: {Path(f).name}")
            analyze_shoot(ctx, sid)
            its = load_items(db, sid)
            for r in db.images(sid):
                paths[r["id"]] = str(Path(r["path"]).resolve())
            its = [it for it in its if paths.get(it.image_id) in labels]
            if not its:
                continue
            build_features(its)
            items += its
            y += [labels[paths[it.image_id]] for it in its]
        if len(items) < 20 or sum(y) < 5 or sum(y) > len(y) - 5:
            raise ValueError(f"Zu wenige gelabelte Bilder für die Kalibrierung ({len(items)}, davon {sum(y)} behalten)")
        cs = load_settings().culling
        cs.keep_ratio = sum(y) / len(y)
        before = agreement(cull([CullItem(i.image_id, i.t, i.a, i.emb, i.phash) for i in items], cs),
                           {i.image_id: v for i, v in zip(items, y)})
        model = fit(items, y)
        after = agreement(cull([CullItem(i.image_id, i.t, i.a, i.emb, i.phash) for i in items], cs, model),
                          {i.image_id: v for i, v in zip(items, y)})
        model.save(profile)
        report = {"n": len(items), "kept_by_you": int(sum(y)), "keep_ratio": cs.keep_ratio,
                  "heuristik": before, "kalibriert": after,
                  "gewichte": dict(zip(model.features, model.weights))}
        (profiles_dir() / profile / "culling_report.json").write_text(json.dumps(report, indent=2), "utf-8")
        ctx.progress(message=f"Culling kalibriert: F1 {before['f1']:.2f} -> {after['f1']:.2f} (in-sample)")


_register_job()

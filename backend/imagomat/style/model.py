"""Stilmodell: sagt pro Bild individuelle Lightroom-Einstellungen voraus.

Architektur (Begründung im Machbarkeitsbericht, Abschnitt d):
1. Merkmalsraum: standardisierte Bildmerkmale + PCA des Bild-Embeddings.
2. kNN-Retrieval: "Was hast du bei den ähnlichsten Bildern gemacht?" (gewichteter Mittelwert).
3. LightGBM pro Zielgrösse auf Merkmalen + kNN-Schätzung (Leave-one-out beim Training).
4. Prior: das adaptive Basis-Preset. Gewicht des Modells = n / (n + 30); bei wenigen
   eigenen Beispielen dominiert das Preset.
5. Kategorische/übrige Einstellungen (Profil, Look, Objektivkorrektur ...) vom ähnlichsten
   Trainingsbild übernommen.
6. Masken: Vorlagen aus den Trainingsdaten, Nutzung per kNN-Abstimmung + logistischer
   Regression, Werte per kNN.
7. Unsicherheit: Streuung der Nachbarn relativ zur Gesamtstreuung.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DenoiseSettings, profiles_dir
from ..lightroom.develop import DIGEST_KEYS
from .features import NUMERIC_KEYS, numeric_features, vectorize
from .masks import MaskTemplate, learn_templates, local_values, signature
from .presets import PRESETS, apply as apply_preset, choose_preset
from .targets import DIRECT, TARGETS, encode

log = logging.getLogger(__name__)

PRIOR_N0 = 12.0
EXP_J = TARGETS.index("Exposure2012")


def exposure_reference(a: dict[str, Any]) -> float:
    """Gemessene Szenenhelligkeit (log2, linear). Belichtung wird als Ausgabehelligkeit
    (= Referenz + Belichtung) gelernt: die ist bei den meisten Fotografen fast konstant."""
    from .presets import measured_log

    m = measured_log(a, 0.5)
    if m is not None:
        return float(m)
    med = max(float(a.get("median") or 0.4), 0.02)
    return 2.2 * float(np.log2(med)) - 3.0
PCA_DIMS = 24
# Diese Schlüssel werden vom Modell erzeugt und nie vom Nachbarn kopiert
LEARNED_KEYS = set(DIRECT) | {
    "WhiteBalance", "Temperature", "Tint", "ToneCurvePV2012", "ToneCurveName2012", "SplitToningShadowHue",
    "SplitToningShadowSaturation", "SplitToningHighlightHue", "SplitToningHighlightSaturation",
    "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeGlobalHue", "ColorGradeGlobalSat",
    "MaskGroupBasedCorrections", "HasCrop", "CropTop", "CropLeft", "CropBottom", "CropRight", "CropAngle",
    "CropConstrainToWarp", "RawFileName", "AlreadyApplied", "HasSettings",
    # Vorsicht bei Ebenen/Retusche: nie auf andere Bilder übertragen
    "RetouchInfo", "RetouchAreas", "CircularGradientBasedCorrections", "GradientBasedCorrections",
    "PaintBasedCorrections", "DepthBasedCorrections", "LensBlur", "UprightTransform_0", "UprightTransform_1",
    "UprightTransform_2", "UprightTransform_3", "UprightTransform_4", "UprightTransform_5",
} | DIGEST_KEYS
FEATURE_WEIGHTS = {"lin_log_median": 3.0, "lin_log_p95": 2.0, "lin_log_p05": 2.0, "as_shot_mired": 2.0,
                   "as_shot_tint": 1.5, "log_iso": 1.5, "subject_luma": 1.5, "face_luma": 1.5, "median": 2.0}


@dataclass
class Record:
    """Ein Trainings- oder Vorhersagebeispiel."""
    analysis: dict[str, Any]
    exif: dict[str, Any]
    embedding: np.ndarray | None
    crs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    path: str = ""

    @property
    def feats(self) -> dict[str, float]:
        return numeric_features(self.analysis, self.exif)


@dataclass
class Prediction:
    targets: dict[str, float]
    extras: dict[str, Any]
    masks: list[tuple[MaskTemplate, dict[str, float], float]]
    confidence: float
    neighbors: list[tuple[str, float]]
    preset: str
    has_crop_prob: float
    crop_area: float


class StyleModel:
    def __init__(self, name: str, base_preset: str | None = None):
        self.name = name
        self.base_preset = base_preset          # None = automatisch pro Bild
        self.created = _dt.datetime.now().isoformat(timespec="seconds")
        self.n = 0
        self.denoise_key: str | None = None
        self.look_override: dict[str, Any] = {}  # aus Lightroom-Presets
        self.metrics: dict[str, Any] = {}
        # interne Daten
        self._mu = self._sd = None
        self._pca_mean = self._pca_comp = None
        self._Z: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._ystd: np.ndarray | None = None
        self._const: dict[str, float] = {}
        self._gbdt: dict[str, Any] = {}
        self._extras: list[dict[str, Any]] = []
        self._paths: list[str] = []
        self.templates: list[MaskTemplate] = []
        self._tmpl_use: np.ndarray | None = None          # (n, T)
        self._tmpl_vals: list[list[dict[str, float] | None]] = []
        self._tmpl_clf: list[Any] = []
        self._crop: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _space(self, X: np.ndarray, E: np.ndarray | None) -> np.ndarray:
        Xs = (X - self._mu) / self._sd
        w = np.array([FEATURE_WEIGHTS.get(k, 1.0) for k in NUMERIC_KEYS])
        Xs = Xs * w
        parts = [Xs / np.sqrt(len(NUMERIC_KEYS))]
        if self._pca_comp is not None and E is not None:
            P = (E - self._pca_mean) @ self._pca_comp.T
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
            parts.append(P * 1.2)
        Z = np.concatenate(parts, axis=1)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    def _knn(self, z: np.ndarray, k: int, exclude: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        sims = self._Z @ z
        if exclude is not None:
            sims[exclude] = -np.inf
        k = min(k, len(sims) - (1 if exclude is not None else 0))
        idx = np.argpartition(-sims, k - 1)[:k] if k > 0 else np.array([], int)
        idx = idx[np.argsort(-sims[idx])]
        w = np.exp((sims[idx] - sims[idx].max()) * 12.0) * self._w[idx]
        return idx, w / max(w.sum(), 1e-12)

    # ------------------------------------------------------------------
    def fit(self, records: list[Record], denoise_key: str | None = None) -> "StyleModel":
        from ..vision.geometry import from_lightroom_crop

        if len(records) < 3:
            raise ValueError("Mindestens 3 bearbeitete Bilder nötig")
        self.denoise_key = denoise_key
        self.n = len(records)
        feats = [r.feats for r in records]
        X = vectorize(feats)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-6
        embs = [r.embedding for r in records]
        dims = {len(e) for e in embs if e is not None}
        E = None
        if len(dims) == 1 and all(e is not None for e in embs):
            E = np.stack(embs).astype(np.float64)
            self._pca_mean = E.mean(0)
            _, _, vt = np.linalg.svd(E - self._pca_mean, full_matrices=False)
            self._pca_comp = vt[: min(PCA_DIMS, len(records) - 1, vt.shape[0])]
        self._Z = self._space(X, E)
        self._w = np.array([r.weight for r in records], dtype=float)
        Y = np.array([[encode(r.crs, r.analysis.get("as_shot_temp"), r.analysis.get("as_shot_tint"),
                              denoise_key)[t] for t in TARGETS] for r in records], dtype=float)
        self._ref = np.array([exposure_reference(r.analysis) for r in records])
        Y[:, EXP_J] += self._ref
        self._Y = Y
        self._ystd = Y.std(0) + 1e-6
        self._const = {t: float(np.median(Y[:, j])) for j, t in enumerate(TARGETS) if Y[:, j].std() < 1e-6}
        self._extras = [{k: v for k, v in r.crs.items() if k not in LEARNED_KEYS and not k.startswith("Enhance")}
                        for r in records]
        self._paths = [r.path for r in records]
        self._crop = np.array([[
            (from_lightroom_crop(r.crs) or {}).get("crop_aspect_rel", 1.0)] for r in records])
        # kNN (Leave-one-out) als Zusatzmerkmal für GBDT
        k = min(10, self.n - 1)
        knn_Y = np.zeros_like(Y)
        for i in range(self.n):
            idx, w = self._knn(self._Z[i], k, exclude=i)
            knn_Y[i] = w @ Y[idx]
        mae_knn = np.abs(knn_Y - Y).mean(0)
        self._gbdt = {}
        mae_model = mae_knn.copy()
        if self.n >= 40:
            mae_model = self._fit_gbdt(X, Y, knn_Y)
        self.metrics = {
            "n": self.n,
            "mae_knn": {t: float(v) for t, v in zip(TARGETS, mae_knn)},
            "mae_model": {t: float(v) for t, v in zip(TARGETS, mae_model)},
        }
        self._fit_masks(records)
        return self

    def _fit_gbdt(self, X: np.ndarray, Y: np.ndarray, knn_Y: np.ndarray) -> np.ndarray:
        try:
            import lightgbm as lgb

            def make():
                return lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=8,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                         verbose=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            def make():
                return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15)
        from sklearn.model_selection import KFold

        Xs = (X - self._mu) / self._sd
        mae = np.zeros(Y.shape[1])
        kf = KFold(n_splits=min(5, max(2, self.n // 20)), shuffle=True, random_state=0)
        for j, t in enumerate(TARGETS):
            if t in self._const:
                continue
            A = np.column_stack([Xs, knn_Y[:, j]])
            y = Y[:, j]
            errs = []
            for tr, te in kf.split(A):
                m = make().fit(A[tr], y[tr], sample_weight=self._w[tr])
                errs.append(np.abs(m.predict(A[te]) - y[te]).mean())
            mae[j] = float(np.mean(errs))
            # GBDT nur behalten, wenn es den reinen kNN schlägt
            knn_err = float(np.abs(knn_Y[:, j] - y).mean())
            if mae[j] < knn_err * 0.98:
                self._gbdt[t] = make().fit(A, y, sample_weight=self._w)
            else:
                mae[j] = knn_err
        return mae

    def _fit_masks(self, records: list[Record]) -> None:
        self.templates = learn_templates([(r.crs, r.analysis) for r in records])
        T = len(self.templates)
        use = np.zeros((self.n, T))
        vals: list[list[dict[str, float] | None]] = [[None] * T for _ in range(self.n)]
        for i, r in enumerate(records):
            for corr in r.crs.get("MaskGroupBasedCorrections", []) or []:
                if not isinstance(corr, dict):
                    continue
                sig = signature(corr)
                for j, t in enumerate(self.templates):
                    if t.sig == sig and vals[i][j] is None:
                        use[i, j] = 1
                        vals[i][j] = local_values(corr)
        self._tmpl_use, self._tmpl_vals = use, vals
        self._tmpl_clf = []
        from sklearn.linear_model import LogisticRegression

        for j in range(T):
            y = use[:, j]
            if 5 <= y.sum() <= self.n - 5:
                self._tmpl_clf.append(LogisticRegression(C=0.3, max_iter=300).fit(self._Z, y))
            else:
                self._tmpl_clf.append(None)

    # ------------------------------------------------------------------
    def predict(self, rec: Record, ds: DenoiseSettings | None = None, preset_key: str | None = None) -> Prediction:
        feats = rec.feats
        X = vectorize([feats])
        E = None if rec.embedding is None or self._pca_mean is None or len(rec.embedding) != len(self._pca_mean) \
            else np.asarray(rec.embedding, dtype=float)[None]
        z = self._space(X, E)[0]
        idx, w = self._knn(z, min(10, self.n))
        knn = w @ self._Y[idx]
        Xs = (X - self._mu) / self._sd
        out: dict[str, float] = {}
        for j, t in enumerate(TARGETS):
            if t in self._const:
                out[t] = self._const[t]
            elif t in self._gbdt:
                out[t] = float(self._gbdt[t].predict(np.column_stack([Xs, [[knn[j]]]]))[0])
            else:
                out[t] = float(knn[j])
        # Preset-Prior
        key = preset_key or self.base_preset or choose_preset(rec.analysis.get("scene"))
        prior = apply_preset(PRESETS[key], {**rec.analysis, "iso": rec.exif.get("iso")}, ds)
        ref = exposure_reference(rec.analysis)
        prior = {**prior, "Exposure2012": prior.get("Exposure2012", 0.0) + ref}
        wm = self.n / (self.n + PRIOR_N0)
        for t in TARGETS:
            if t not in self._const:
                out[t] = wm * out[t] + (1 - wm) * prior.get(t, out[t])
        out["Exposure2012"] = float(np.clip(out["Exposure2012"] - ref, -5, 5))
        # Unsicherheit
        spread = np.sqrt(w @ (self._Y[idx] - knn) ** 2) / self._ystd
        key_t = [TARGETS.index(t) for t in ("Exposure2012", "wb_dmired", "Contrast2012", "Highlights2012")]
        conf = float(np.exp(-np.mean(spread[key_t])) * min(1.0, float(self._Z[idx[0]] @ z) + 0.2))
        # Übrige Einstellungen vom ähnlichsten Bild
        extras = {k: v for k, v in self._extras[idx[0]].items()}
        extras.update(self.look_override)
        # Masken
        masks = []
        for j, t in enumerate(self.templates):
            p_knn = float(w @ self._tmpl_use[idx, j])
            clf = self._tmpl_clf[j]
            p = p_knn if clf is None else 0.5 * p_knn + 0.5 * float(clf.predict_proba(z[None])[0, 1])
            if p < 0.5:
                continue
            vals_w = [(wi, self._tmpl_vals[i][j]) for i, wi in zip(idx, w) if self._tmpl_vals[i][j]]
            if vals_w:
                tot = sum(wi for wi, _ in vals_w)
                v = {k: sum(wi * d.get(k, t.medians.get(k, 0.0)) for wi, d in vals_w) / tot for k in t.params}
            else:
                v = dict(t.medians)
            masks.append((t, v, p))
        has_crop = float(w @ self._Y[idx, TARGETS.index("has_crop")])
        area = float(w @ self._Y[idx, TARGETS.index("crop_area")])
        return Prediction(out, extras, masks, conf, [(self._paths[i], float(wi)) for i, wi in zip(idx, w)], key,
                          has_crop, area)

    # ------------------------------------------------------------------
    def save(self) -> Path:
        d = profiles_dir() / self.name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "model.pkl", "wb") as f:
            pickle.dump(self, f)
        meta = {"name": self.name, "base_preset": self.base_preset, "created": self.created, "n": self.n,
                "templates": [{"sig": t.sig, "name": t.name, "count": t.count} for t in self.templates],
                "metrics": self.metrics}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        return d

    @classmethod
    def load(cls, name: str) -> "StyleModel | None":
        p = profiles_dir() / name / "model.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)


def list_profiles() -> list[dict[str, Any]]:
    out = []
    for d in sorted(profiles_dir().glob("*/meta.json")):
        try:
            out.append(json.loads(d.read_text("utf-8")))
        except json.JSONDecodeError:
            continue
    return out

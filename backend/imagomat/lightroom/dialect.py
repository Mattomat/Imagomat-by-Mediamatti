"""Der "Lightroom-Dialekt" einer Installation, gelernt aus echten XMPs bzw. dem Katalog.

Einige Details schreibt jede Lightroom-Version etwas anders (Prozessversion,
Camera-Raw-Version, Namen der Farblabels, Felder für Denoise, Werte für KI-Masken-Typen).
Statt sie zu raten, lesen wir sie aus Dateien, die Lightroom selbst geschrieben hat.
Nur wenn nichts vorliegt, greifen die Fallback-Werte unten.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import data_dir
from .develop import strip_digests
from .params import to_number

# Fallbacks (nicht verifiziert, siehe docs/phase0 Hypothesen H1/H2)
FALLBACK_PROCESS_VERSION = "11.0"
FALLBACK_CRS_VERSION = "17.5"
FALLBACK_DENOISE = {"EnhanceDenoiseAlreadyApplied": "True", "EnhanceDenoiseVersion": "1"}
FALLBACK_DENOISE_AMOUNT_KEY = "EnhanceDenoiseLumaAmount"
FALLBACK_LABELS = {"red": "Rot", "yellow": "Gelb", "green": "Grün", "blue": "Blau", "purple": "Lila"}
# Werte für crs:MaskSubType bei What="Mask/Image" (Fallback, echte Werte werden gelernt)
FALLBACK_AI_MASK = {
    "subject": {"What": "Mask/Image", "MaskSubType": "1", "MaskVersion": "1"},
    "sky": {"What": "Mask/Image", "MaskSubType": "2", "MaskVersion": "1"},
    "person": {"What": "Mask/Image", "MaskSubType": "3", "MaskVersion": "1"},
}

_DENOISE_RE = re.compile(r"(Denoise|EnhanceDenoise)", re.I)


@dataclass
class Dialect:
    process_version: str = FALLBACK_PROCESS_VERSION
    crs_version: str = FALLBACK_CRS_VERSION
    labels: dict[str, str] = field(default_factory=lambda: dict(FALLBACK_LABELS))
    denoise_fields: dict[str, str] = field(default_factory=lambda: dict(FALLBACK_DENOISE))
    denoise_amount_key: str = FALLBACK_DENOISE_AMOUNT_KEY
    ai_masks: dict[str, dict[str, Any]] = field(default_factory=lambda: {k: dict(v) for k, v in FALLBACK_AI_MASK.items()})
    learned_from: int = 0
    verified: dict[str, bool] = field(default_factory=dict)

    def label(self, color_or_name: str | None) -> str | None:
        if not color_or_name:
            return None
        return self.labels.get(color_or_name.lower(), color_or_name)

    def denoise(self, amount: int) -> dict[str, Any]:
        d = dict(self.denoise_fields)
        d[self.denoise_amount_key] = str(int(amount))
        return d

    def save(self, path: Path | None = None) -> None:
        (path or dialect_path()).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "Dialect":
        p = path or dialect_path()
        if p.exists():
            try:
                d = json.loads(p.read_text("utf-8"))
                return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()


def dialect_path() -> Path:
    return data_dir() / "lightroom_dialect.json"


def _mask_kind(mask: dict[str, Any]) -> str | None:
    """Ordnet eine gelernte KI-Maske anhand ihres Namens einer Kategorie zu."""
    if mask.get("What") != "Mask/Image":
        return None
    name = str(mask.get("MaskName", "")).lower()
    for kind, words in {
        "sky": ("himmel", "sky", "ciel", "cielo"),
        "subject": ("motiv", "subject", "sujet", "soggetto"),
        "person": ("person", "people", "personen", "haut", "skin", "gesicht", "face"),
        "background": ("hintergrund", "background"),
    }.items():
        if any(w in name for w in words):
            return kind
    return None


def learn_dialect(develops: Iterable[dict[str, Any]], labels: Iterable[str] = (),
                  base: Dialect | None = None) -> Dialect:
    """Lernt den Dialekt aus crs-Dicts (XMP oder Katalog) und genutzten Label-Texten."""
    d = base or Dialect()
    pv, ver, n = Counter(), Counter(), 0
    denoise_keys: Counter[str] = Counter()
    denoise_examples: dict[str, Counter[str]] = {}
    for crs in develops:
        n += 1
        if crs.get("ProcessVersion"):
            pv[str(crs["ProcessVersion"])] += 1
        if crs.get("Version"):
            ver[str(crs["Version"])] += 1
        for k, v in crs.items():
            if _DENOISE_RE.search(k) and not isinstance(v, (dict, list)):
                denoise_keys[k] += 1
                denoise_examples.setdefault(k, Counter())[str(v)] += 1
        for corr in crs.get("MaskGroupBasedCorrections", []) or []:
            if not isinstance(corr, dict):
                continue
            for m in corr.get("CorrectionMasks", []) or []:
                if isinstance(m, dict):
                    kind = _mask_kind(m)
                    if kind and kind not in d.verified:
                        clean = strip_digests({k: v for k, v in m.items() if k not in ("MaskName", "MaskSyncID")})
                        d.ai_masks[kind] = clean
                        d.verified[kind] = True
    if pv:
        d.process_version = pv.most_common(1)[0][0]
        d.verified["process_version"] = True
    if ver:
        d.crs_version = max(ver, key=_version_key)
        d.verified["crs_version"] = True
    if denoise_keys:
        # Der Mengen-Schlüssel ist der, dessen Werte am stärksten variieren.
        amount_key = max(denoise_examples, key=lambda k: (len(denoise_examples[k]), "amount" in k.lower()))
        fields = {}
        for k, c in denoise_examples.items():
            if k != amount_key:
                fields[k] = c.most_common(1)[0][0]
        d.denoise_fields = fields
        d.denoise_amount_key = amount_key
        d.verified["denoise"] = True
    used = Counter(l for l in labels if l)
    if used:
        d.labels = _map_labels(used, d.labels)
        d.verified["labels"] = True
    d.learned_from += n
    return d


def _version_key(v: str) -> tuple[float, ...]:
    return tuple(to_number(x) or 0 for x in re.split(r"[.\s]", v) if x)


_LABEL_WORDS = {
    "red": ("rot", "red", "rouge", "rosso"),
    "yellow": ("gelb", "yellow", "jaune", "giallo"),
    "green": ("grün", "gruen", "green", "vert", "verde"),
    "blue": ("blau", "blue", "bleu", "blu"),
    "purple": ("lila", "violett", "purple", "violet", "viola"),
}


def _map_labels(used: Counter[str], current: dict[str, str]) -> dict[str, str]:
    out = dict(current)
    for text in used:
        low = text.lower()
        for color, words in _LABEL_WORDS.items():
            if low in words:
                out[color] = text
    return out

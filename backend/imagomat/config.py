"""Pfade und Einstellungen.

Alle Daten liegen lokal im App-Datenordner (Standard: ~/Library/Application Support/Imagomat
auf macOS, ~/.imagomat sonst). Mit IMAGOMAT_HOME lässt sich der Ordner überschreiben
(z. B. für Tests).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw", ".3fr", ".iiq",
}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".tif", ".tiff", ".heic"}


def data_dir() -> Path:
    env = os.environ.get("IMAGOMAT_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "Imagomat"
    elif os.name == "nt":
        p = Path(os.environ.get("APPDATA", Path.home())) / "Imagomat"
    else:
        p = Path.home() / ".imagomat"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = data_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_key(image_id: int, path: str | Path) -> str:
    """Name für Zwischenspeicher eines Bildes: Nummer + Pfad. Nach Löschen/Neuimport kann eine Nummer an ein
    anderes Bild gehen; mit dem Pfad im Namen wird nie die Vorschau eines anderen Bildes gezeigt."""
    import zlib

    return f"{image_id}_{zlib.crc32(str(path).encode('utf-8')):08x}"


def profiles_dir() -> Path:
    p = data_dir() / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class CullingSettings:
    keep_ratio: float = 0.20            # "behalte ca. 20 %"
    min_rating_keep: int = 2            # Sterne für behaltene Bilder: 2..5
    reject_rating: int = 1              # Sterne für aussortierte Bilder (0 = keine)
    series_gap_seconds: float = 2.0     # max. Zeitabstand innerhalb einer Serie
    series_similarity: float = 0.90     # Embedding-Ähnlichkeit für Serien
    duplicate_similarity: float = 0.975
    label_keep_best: str | None = "Grün"     # bestes Bild einer Serie
    label_rejected: str | None = None
    label_denoise: str | None = "Lila"
    label_review: str | None = "Gelb"         # unsichere Vorhersagen
    eyes_closed_threshold: float = 0.22       # Eye-Aspect-Ratio
    # Action-Momente (Zweikampf, Schuss, Jubel ...): Anteil am Score bei Sport/Konzert, 0 = nur Technik
    action_weight: float = 0.35
    # "Nur Highlights": sehr streng, ein Bild pro Spielszene, nur echte Action-Momente
    highlights: bool = False
    highlights_ratio: float = 0.05
    highlights_action_weight: float = 0.65
    moment_gap_seconds: float = 4.0     # Bilder innerhalb dieses Abstands = dieselbe Spielszene
    max_keep: int | None = None         # höchstens so viele Bilder behalten (None = nur Prozent)
    # "Nur Schlechte raus": keine Prozent-Quote; es fliegen nur technisch schlechte Bilder raus und aus einer
    # Serie mit (fast) gleichem Motiv bleiben höchstens so viele. 0 = alte Prozent-Auswahl (keep_ratio).
    burst_keep: int = 2
    weights: dict[str, float] = field(default_factory=lambda: {
        "sharpness": 0.30, "face_quality": 0.25, "exposure": 0.15, "aesthetic": 0.20, "composition": 0.10,
    })


@dataclass
class KeywordSettings:
    people_root: str = "Personen"
    culling_root: str = "Imagomat|Culling"
    denoise_keyword: str = "Imagomat|Denoise"
    review_keyword: str = "Imagomat|Prüfen"
    write_face_regions: bool = True
    write_parent_keywords: bool = False
    # "name": nur der Name (z. B. "Luca Zuffi"); "team": Personen|Team|Name
    person_keyword_style: str = "name"
    # Imagomat-Arbeitsstichwörter (Behalten, Denoise, Prüfen, Moment) mit ausgeben
    workflow_keywords: bool = False
    # Inhalts-Stichwörter (Fans, Team, Jubel, Trainer ...): "no_person" = nur wenn keine Person erkannt,
    # "always" = immer zusätzlich, "off" = nie
    content_keywords: str = "no_person"


@dataclass
class DenoiseSettings:
    # "lightroom": Denoise-Wert ins XMP schreiben, Lightroom rechnet ("KI-Einstellungen aktualisieren").
    # "local": eigene Entrauschung, Ergebnis als lineare DNG.
    # "mark": nur Stichwort + Farblabel.
    mode: str = "lightroom"
    always: bool = True                 # Sport/Konzert: immer entrauschen, Stärke variiert
    default_amount: int = 50
    min_amount: int = 25
    max_amount: int = 80
    local_model: str = "auto"           # auto | nafnet | scunet | classical


@dataclass
class DevelopSettings:
    auto_straighten: bool = True
    max_straighten_deg: float = 8.0
    auto_crop: bool = True
    keep_aspect: str = "learned"        # learned | original | "3:2" | "4:5" ...
    write_masks: bool = True
    ai_masks: bool = True               # KI-Masken als Deklaration (Lightroom rechnet neu)
    paint_mask_fallback: bool = True
    # Dunkler, weicher Verlauf von unten (Rasen zurücknehmen); Stärke wird pro Bild angepasst
    bottom_fade: bool = True
    bottom_fade_strength: float = 1.0   # 0.5 = halb so stark, 1.5 = kräftiger
    # Knackigkeit: Weiss-/Schwarzpunkt pro Bild setzen, etwas mehr Kontrast, Hintergrund mit Kontrast
    # statt nur dunkler (0 = aus, 1 = normal, 1.5 = kräftig)
    punch: float = 1.0
    shoot_consistency: float = 0.6      # 0 = aus, 1 = voll angleichen
    process_version: str | None = None  # None = aus Katalog/XMP gelernt, Fallback "11.0"
    camera_raw_version: str | None = None


@dataclass
class Settings:
    culling: CullingSettings = field(default_factory=CullingSettings)
    keywords: KeywordSettings = field(default_factory=KeywordSettings)
    denoise: DenoiseSettings = field(default_factory=DenoiseSettings)
    develop: DevelopSettings = field(default_factory=DevelopSettings)
    device: str = "auto"                # auto | mps | cuda | cpu
    face_backend: str = "auto"          # auto | insightface | yunet | haar
    embedding_backend: str = "auto"     # auto | clip | dinov2 | classical
    segmentation_backend: str = "auto"  # auto | birefnet | classical
    ocr_backend: str = "auto"           # auto | vision | easyocr | none
    action_backend: str = "auto"        # auto (Pose nur mit GPU) | pose | clip | none
    # Wörter auf Trikots, die nie ein Spielername sind (Sponsoren); weitere erkennt Imagomat selbst
    ignored_shirt_words: list[str] = field(default_factory=lambda: ["KELLER", "INIT"])
    workers: int = max(2, (os.cpu_count() or 4) - 2)
    default_profile: str | None = None
    # Ablageort: hierhin kopiert Imagomat die RAWs beim Import (Ordner je Shoot); dort verlinkt sie auch
    # Lightroom ("Hinzufügen"). None = beim ersten Import fragen.
    library_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        s = cls()
        for key, sub in (("culling", CullingSettings), ("keywords", KeywordSettings),
                         ("denoise", DenoiseSettings), ("develop", DevelopSettings)):
            if key in d and isinstance(d[key], dict):
                known = {k: v for k, v in d[key].items() if k in sub.__dataclass_fields__}
                setattr(s, key, sub(**{**asdict(getattr(s, key)), **known}))
        for k, v in d.items():
            if k in cls.__dataclass_fields__ and not isinstance(v, dict):
                setattr(s, k, v)
        return s


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    p = settings_path()
    if p.exists():
        try:
            return Settings.from_dict(json.loads(p.read_text("utf-8")))
        except (json.JSONDecodeError, TypeError):
            pass
    return Settings()


def save_settings(s: Settings) -> None:
    settings_path().write_text(json.dumps(s.to_dict(), indent=2, ensure_ascii=False), "utf-8")

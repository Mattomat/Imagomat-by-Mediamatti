"""Eigene Vorgaben (Lightroom-Presets) und die neutrale Ausgangslage.

- Neutral: wie ein frisch importiertes RAW in Lightroom (alle Regler auf 0, Weissabgleich wie Aufnahme). Das ist
  der Standard nach dem Import; eine Vorgabe kommt nur dazu, wenn man sie wählt.
- Eigene Vorgaben: Lightroom-Presets (.xmp aus "Vorgabe exportieren" bzw. dem CameraRaw/Settings-Ordner, oder
  ältere .lrtemplate) importieren. Anwenden wie in Lightroom: nur die Regler, die das Preset enthält, werden
  gesetzt, Masken kommen dazu; alles andere bleibt.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from ..config import data_dir

NEUTRAL = "neutral"
PREFIX = "up:"
# Gehören nicht zur Bearbeitung (Verwaltung des Presets) oder zum einzelnen Bild
META_KEYS = {"Name", "ShortName", "SortName", "Group", "Copyright", "Description", "UUID", "Cluster", "PresetType",
             "SupportsAmount", "SupportsAmount2", "SupportsColor", "SupportsMonochrome", "SupportsHighDynamicRange",
             "SupportsNormalDynamicRange", "SupportsSceneReferred", "SupportsOutputReferred", "CameraModelRestriction",
             "ContactInfo", "Version", "HasSettings", "AlreadyApplied", "RawFileName", "Amount"}
NEUTRAL_CRS: dict[str, Any] = {
    "CameraProfile": "Camera Standard",      # bei 0 wie das Kamera-Original (Lightroom: Kamera-Profil)
    "WhiteBalance": "As Shot", "Exposure2012": 0.0, "Contrast2012": 0, "Highlights2012": 0, "Shadows2012": 0,
    "Whites2012": 0, "Blacks2012": 0, "Texture": 0, "Clarity2012": 0, "Dehaze": 0, "Vibrance": 0, "Saturation": 0,
    "ToneCurveName2012": "Linear", "ToneCurvePV2012": ["0, 0", "255, 255"], "HasCrop": False, "CropAngle": 0.0,
}


def presets_dir() -> Path:
    p = data_dir() / "presets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _file(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in " -_äöüÄÖÜ.()").strip() or "Vorgabe"
    return presets_dir() / f"{safe}.json"


def neutral_crs() -> dict[str, Any]:
    """Wie ein frisch importiertes RAW in Lightroom (mit den Standard-Kennzeichen, die Lightroom erwartet)."""
    from ..config import load_settings
    from ..lightroom.dialect import Dialect
    from ..lightroom.params import BASE_FLAGS

    s, d = load_settings(), Dialect.load()
    return {"Version": s.develop.camera_raw_version or d.crs_version,
            "ProcessVersion": s.develop.process_version or d.process_version, **BASE_FLAGS,
            "LensProfileEnable": 1, "AutoLateralCA": 1, **copy.deepcopy(NEUTRAL_CRS)}


def suggest_denoise(iso: float | None, settings: Any) -> int | None:
    """Vorschlag für KI-Entrauschen nach ISO (aus ab ISO 1600; 1600 -> 25, 6400 -> ~60, 12800 -> ~80)."""
    if not iso or iso < 1600:
        return None
    d = settings.denoise
    amt = 25 + math.log2(iso / 1600) * 18
    return int(round(min(max(amt, d.min_amount), d.max_amount)))


def _text(v: Any) -> str:
    if isinstance(v, dict):               # rdf:Alt -> {"x-default": "..."}
        return str(next(iter(v.values()), "") or "")
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def _clean(crs: dict[str, Any]) -> dict[str, Any]:
    from .template import DROP_KEYS

    out = {}
    for k, v in crs.items():
        if k in META_KEYS or k in DROP_KEYS or k.startswith("Table_") or re.search(r"Digest$", k):
            continue
        out[k] = copy.deepcopy(v)
    return out


def parse_preset_file(path: Path) -> tuple[str, dict[str, Any]]:
    """Lightroom-Preset lesen -> (Name, Einstellungen)."""
    if path.suffix.lower() == ".lrtemplate":
        from ..lightroom.lua import parse_lua

        src = path.read_text("utf-8", errors="replace")
        data = parse_lua(src[src.index("{"):] if "{" in src else src)
        value = data.get("value") if isinstance(data, dict) else None
        settings = (value or {}).get("settings") if isinstance(value, dict) else None
        if not isinstance(settings, dict):
            raise ValueError(f"{path.name}: keine Entwicklungseinstellungen gefunden")
        title = data.get("title") if isinstance(data, dict) else None
        name = _text(title) if title else path.stem
        return re.sub(r"^\$\$\$/\S+=", "", name) or path.stem, _clean(settings)
    from ..lightroom.xmp import read_xmp

    doc = read_xmp(path)
    if not doc.crs:
        raise ValueError(f"{path.name}: keine Lightroom-Einstellungen gefunden")
    name = _text(doc.crs.get("Name")) or path.stem
    return name, _clean(doc.crs)


def import_presets(paths: list[Path]) -> tuple[list[str], list[str]]:
    """Mehrere Dateien (oder Ordner) importieren -> (importierte Namen, Fehler)."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in (".xmp", ".lrtemplate")]
        else:
            files.append(p)
    done, errors = [], []
    for f in files:
        try:
            name, crs = parse_preset_file(f)
            if not crs:
                raise ValueError(f"{f.name}: leer")
            _file(name).write_text(json.dumps({"name": name, "crs": crs, "file": f.name}, indent=1,
                                              ensure_ascii=False, default=str), "utf-8")
            done.append(name)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    return done, errors


def list_presets() -> list[dict[str, Any]]:
    out = []
    for f in sorted(presets_dir().glob("*.json")):
        try:
            d = json.loads(f.read_text("utf-8"))
            out.append({"name": d["name"], "n_keys": len(d.get("crs") or {})})
        except Exception:  # noqa: BLE001
            continue
    return out


def load_preset(name: str) -> dict[str, Any] | None:
    f = _file(name)
    if not f.exists():
        return None
    return json.loads(f.read_text("utf-8"))


def delete_preset(name: str) -> bool:
    f = _file(name)
    if f.exists():
        f.unlink()
        return True
    return False


def preset_version(name: str) -> float:
    f = _file(name)
    return f.stat().st_mtime if f.exists() else 0.0


def apply_preset(base: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    """Wie in Lightroom: Regler aus dem Preset überschreiben, Masken des Presets kommen dazu."""
    out = copy.deepcopy(base)
    for k, v in preset.items():
        if k == "MaskGroupBasedCorrections":
            continue
        out[k] = copy.deepcopy(v)
    if "Temperature" in preset and "WhiteBalance" not in preset:
        out["WhiteBalance"] = "Custom"
    masks = preset.get("MaskGroupBasedCorrections")
    if isinstance(masks, list) and masks:
        have = {str(m.get("CorrectionName")) for m in out.get("MaskGroupBasedCorrections") or [] if isinstance(m, dict)}
        add = [copy.deepcopy(m) for m in masks if isinstance(m, dict) and str(m.get("CorrectionName")) not in have]
        out["MaskGroupBasedCorrections"] = list(out.get("MaskGroupBasedCorrections") or []) + add
    return out

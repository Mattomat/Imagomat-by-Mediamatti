"""Umwandlung von Entwicklungseinstellungen zwischen Katalog (Lua) und XMP (crs)."""

from __future__ import annotations

import uuid
from typing import Any

from .params import TONE_CURVE_KEYS, format_curve, parse_curve, to_number

# Schlüssel, die nur Zwischenergebnisse beschreiben und nie kopiert werden sollen.
DIGEST_KEYS = {"MaskDigest", "InputDigest", "UprightDependentDigest", "UprightGuidedDependentDigest",
               "AutoToneDigest", "AutoToneDigestNoSat", "ToggleStyleDigest", "CameraProfileDigest",
               "LensProfileDigest", "RawFileName"}


def lua_to_crs(d: dict[str, Any]) -> dict[str, Any]:
    """Lua-Tabelle aus dem Katalog -> crs-Dict im XMP-Stil."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        if k in TONE_CURVE_KEYS or k.startswith("ToneCurvePV2012"):
            out[k] = format_curve(parse_curve(v))
        elif isinstance(v, bool):
            out[k] = "True" if v else "False"
        elif isinstance(v, dict):
            out[k] = _struct(v)
        elif isinstance(v, list):
            out[k] = [_struct(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _struct(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, dict):
            out[k] = _struct(v)
        elif isinstance(v, list):
            out[k] = [_struct(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def crs_to_lua(d: dict[str, Any]) -> dict[str, Any]:
    """crs-Dict -> Lua-Tabelle für den Katalog (Zahlen/Booleans typisiert)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.startswith("ToneCurvePV2012"):
            flat: list[float] = []
            for x, y in parse_curve(v):
                flat += [int(round(x)), int(round(y))]
            out[k] = flat
        else:
            out[k] = _lua_value(v)
    return out


def _lua_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _lua_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_lua_value(x) for x in v]
    if isinstance(v, str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        n = to_number(v)
        if n is not None and not any(c in v for c in (" ", ",")):
            return int(n) if n.is_integer() and "." not in v else n
        return v
    return v


def strip_digests(v: Any) -> Any:
    """Entfernt Digests rekursiv, damit Lightroom KI-Masken neu berechnet."""
    if isinstance(v, dict):
        return {k: strip_digests(x) for k, x in v.items() if k not in DIGEST_KEYS}
    if isinstance(v, list):
        return [strip_digests(x) for x in v]
    return v


def new_sync_id() -> str:
    return uuid.uuid4().hex.upper()


def refresh_sync_ids(v: Any) -> Any:
    """Neue Sync-IDs vergeben, wenn Masken aus einem anderen Bild übernommen werden."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in ("CorrectionSyncID", "MaskSyncID"):
                out[k] = new_sync_id()
            else:
                out[k] = refresh_sync_ids(x)
        return out
    if isinstance(v, list):
        return [refresh_sync_ids(x) for x in v]
    return v


def numeric_settings(crs: dict[str, Any]) -> dict[str, float]:
    """Alle skalaren Zahlenwerte eines crs-Dicts (für Training)."""
    out = {}
    for k, v in crs.items():
        if isinstance(v, (dict, list)):
            continue
        n = to_number(v)
        if n is not None:
            out[k] = n
    return out

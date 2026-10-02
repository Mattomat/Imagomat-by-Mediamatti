"""Retusche wie in Lightroom: Entfernen (KI) und Reparieren.

- Entfernen: über das Störende malen -> ein Inpainting-Netz (LaMa, ONNX) füllt die Stelle aus der Umgebung auf.
  Ohne Modell (offline) wird repariert (siehe unten).
- Reparieren: Tagmatti sucht eine passende Stelle in der Nähe (Umgebung stimmt am besten überein), übernimmt
  deren Struktur und gleicht Helligkeit/Farbe nahtlos an (Poisson-Überblendung, wie der Reparaturpinsel).
  "Andere Quelle" nimmt die nächstbeste Stelle.

Gerechnet wird einmal in voller Auflösung auf den linearen RAW-Daten (Anzeigeorientierung). Das Ergebnis (Ausschnitt
+ weiche Maske) wird zwischengespeichert und für jede Vorschau, den Editor und den Export nur noch eingeblendet.
Lightroom kann solche Retuschen nicht aus XMP lesen: das retuschierte Bild geht als DNG an Lightroom.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import cache_dir

log = logging.getLogger(__name__)
GAMMA = 1 / 2.2
LAMA_SIDE = 512


def store_dir(key: str) -> Path:
    d = cache_dir() / "retouch" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def mask_from_dabs(dabs: list[Any], W: int, H: int, feather: float = 40) -> np.ndarray:
    """Pinsel-Tupfer (x, y, Radius relativ zur langen Seite; Anzeige-Koordinaten) -> weiche Maske 0..1."""
    m = np.zeros((H, W), np.float32)
    long = max(W, H)
    rmax = 1.0
    for d in dabs:
        x, y, r = float(d[0]), float(d[1]), float(d[2])
        R = max(1.0, r * long)
        rmax = max(rmax, R)
        cv2.circle(m, (int(round(x * W)), int(round(y * H))), int(round(R)), 1.0, -1, lineType=cv2.LINE_AA)
    f = max(0.0, min(100.0, float(feather))) / 100
    if f > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(0.6, rmax * 0.35 * f))
        m = np.clip(m * (1 + f), 0, 1)
    return m


def _to_disp(lin: np.ndarray, wb: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(lin * wb[None, None, :3] * gain, 0, 1) ** GAMMA


def _from_disp(disp: np.ndarray, wb: np.ndarray, gain: float) -> np.ndarray:
    return (np.clip(disp, 0, 1) ** (1 / GAMMA) / (wb[None, None, :3] * gain)).astype(np.float32)


# ---------------------------------------------------------------------------------------------- Reparieren
def heal_candidates(img: np.ndarray, mask: np.ndarray, n: int = 6) -> list[tuple[int, int]]:
    """Verschiebungen (dx, dy) zu Quellstellen, deren Umgebung am besten zur Umgebung der Maske passt."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask > 0.05)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    size = max(bw, bh)
    # Vergleich in kleinem Massstab (schnell), Ring um die Maske
    s = min(1.0, 160 / max(size * 3, 1))
    small = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA).astype(np.float32)
    ms = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    k = max(3, int(size * s * 0.35) | 1)
    hole = (ms > 0.05).astype(np.uint8)
    ring = (cv2.dilate(hole, np.ones((k, k), np.uint8)) > 0) & (hole == 0)
    ry, rx = np.nonzero(ring)
    hy, hx = np.nonzero(hole)
    if len(rx) == 0:
        return []
    target = small[ry, rx]
    out = []
    for scale in (1.1, 1.4, 1.9, 2.6, 3.5):
        for i in range(24):
            a = i / 24 * 2 * math.pi
            # kleinste Verschiebung, die aus der Maske herausführt (lange Striche: quer dazu verschieben)
            ext = abs(math.cos(a)) * bw + abs(math.sin(a)) * bh
            dx, dy = int(round(math.cos(a) * ext * scale)), int(round(math.sin(a) * ext * scale))
            if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > W or y1 + dy > H:
                continue
            sdx, sdy = int(round(dx * s)), int(round(dy * s))
            sy, sx = ry + sdy, rx + sdx
            ok = (sy >= 0) & (sy < small.shape[0]) & (sx >= 0) & (sx < small.shape[1])
            if ok.mean() < 0.9:
                continue
            src = small[np.clip(sy, 0, small.shape[0] - 1), np.clip(sx, 0, small.shape[1] - 1)]
            ring_err = float(np.mean((src - target) ** 2))
            # Quelle soll nicht selbst in der Maske liegen und ähnlich strukturiert sein wie die Umgebung
            iy, ix = np.clip(hy + sdy, 0, small.shape[0] - 1), np.clip(hx + sdx, 0, small.shape[1] - 1)
            overlap = float((ms[iy, ix] > 0.05).mean())
            tex = abs(float(small[iy, ix].std()) - float(target.std()))
            out.append((ring_err + tex * tex * 0.5 + overlap * 1e4 + math.hypot(dx, dy) / max(size, 1) * 3.0, dx, dy))
    out.sort()
    picked: list[tuple[int, int]] = []
    for _, dx, dy in out:
        if all(math.hypot(dx - px, dy - py) > size * 0.6 for px, py in picked):
            picked.append((dx, dy))
        if len(picked) >= n:
            break
    return picked


def heal(disp: np.ndarray, mask: np.ndarray, variant: int = 0) -> np.ndarray:
    """Reparaturpinsel: Struktur einer passenden Stelle, Farbe/Helligkeit nahtlos angeglichen."""
    img8 = (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8)
    cands = heal_candidates(img8, mask)
    hole = (mask > 0.02).astype(np.uint8) * 255
    if not cands:
        return cv2.inpaint(img8, hole, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
    dx, dy = cands[min(variant, len(cands) - 1)]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    H, W = mask.shape
    src = cv2.warpAffine(disp.astype(np.float32), M, (W, H), borderMode=cv2.BORDER_REFLECT)
    ys, xs = np.nonzero(hole)
    size = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    # Struktur (Feindetails) aus der Quelle, Farbe/Helligkeit glatt aus dem Rand der Stelle (wie ein
    # Reparaturpinsel): stabil auch an Farbkanten, keine Farbverschiebung
    sig = max(2.0, size * 0.12)
    sd = float(np.clip(size * 0.04, 1.2, 5.0))         # nur feine Struktur (Korn, Rasen), keine Kanten
    detail = src - cv2.GaussianBlur(src, (0, 0), sd)
    s = min(1.0, 256 / max(H, W))
    small = cv2.resize(img8, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((3, 3), np.uint8))
    low = cv2.inpaint(small, hs, 3, cv2.INPAINT_TELEA)
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    low = cv2.GaussianBlur(low, (0, 0), sig)
    base = cv2.GaussianBlur(disp.astype(np.float32), (0, 0), sig)
    m = cv2.GaussianBlur((hole > 0).astype(np.float32), (0, 0), sig)[..., None]
    smooth = base * (1 - m) + low * m                  # ausserhalb unverändert, innen aus dem Rand
    return np.clip(smooth + detail, 0, 1)


# ---------------------------------------------------------------------------------------------- Entfernen (KI)
_LAMA: Any = None
_LAMA_LOCK = threading.Lock()


def _lama() -> Any:
    global _LAMA
    with _LAMA_LOCK:
        if _LAMA is None:
            from .vision.models import has_module, model_path

            if not has_module("onnxruntime"):
                _LAMA = False
                return None
            p = model_path("lama")
            if p is None:
                _LAMA = False
                return None
            import onnxruntime as ort

            import os

            so = ort.SessionOptions()
            so.intra_op_num_threads = max(2, (os.cpu_count() or 4))
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # CPU: LaMa nutzt FFT-Schichten, die CoreML nicht kann (sonst langsames Hin und Her)
            _LAMA = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        return _LAMA or None


def inpaint_ai(disp: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """LaMa auf einem Ausschnitt um die Maske (512 x 512). None, wenn das Modell nicht verfügbar ist."""
    sess = _lama()
    if sess is None:
        return None
    H, W = mask.shape
    ys, xs = np.nonzero(mask > 0.02)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    # kleine Stellen: Ausschnitt auf 512 vergrössern (volle Schärfe); grosse: so wenig Umgebung wie nötig
    side = int(min(max(size * 2.0 + 48, 160), max(H, W)))
    x0 = int(min(max(cx - side / 2, 0), max(W - side, 0)))
    y0 = int(min(max(cy - side / 2, 0), max(H - side, 0)))
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    crop, mcrop = disp[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    a = cv2.resize(crop, (LAMA_SIDE, LAMA_SIDE),
                   interpolation=cv2.INTER_AREA if side > LAMA_SIDE else cv2.INTER_CUBIC).astype(np.float32)
    m = (cv2.resize(mcrop, (LAMA_SIDE, LAMA_SIDE), interpolation=cv2.INTER_LINEAR) > 0.02).astype(np.float32)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    names = [i.name for i in sess.get_inputs()]
    feed = {names[0]: a.transpose(2, 0, 1)[None], names[1]: m[None, None]}
    res = sess.run(None, feed)[0][0].transpose(1, 2, 0).astype(np.float32)
    if res.max() > 2.0:
        res = res / 255.0
    res = cv2.resize(np.clip(res, 0, 1), (x1 - x0, y1 - y0),
                     interpolation=cv2.INTER_AREA if side < LAMA_SIDE else cv2.INTER_CUBIC)
    if side > LAMA_SIDE * 1.1:
        # das Netz rechnet in 512 px: feine Struktur (Korn, Rasen, Stoff) aus einer passenden Stelle zurückholen
        img8 = (np.clip(crop, 0, 1) * 255 + 0.5).astype(np.uint8)
        cands = heal_candidates(img8, mcrop, n=1)
        if cands:
            dx, dy = cands[0]
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            src = cv2.warpAffine(crop.astype(np.float32), M, (crop.shape[1], crop.shape[0]), borderMode=cv2.BORDER_REFLECT)
            sd = side / LAMA_SIDE * 0.9
            res = np.clip(res + (src - cv2.GaussianBlur(src, (0, 0), sd)) * 0.9, 0, 1)
    out = disp.copy()
    out[y0:y1, x0:x1] = res
    return out


# ---------------------------------------------------------------------------------------------- Ausführen
def warmup() -> bool:
    """KI-Modell schon laden (beim Wählen des Werkzeugs), damit der erste Strich nicht wartet."""
    try:
        return _lama() is not None
    except Exception:  # noqa: BLE001
        return False


def compute_patch(lin: np.ndarray, wb: np.ndarray, op: dict[str, Any]) -> dict[str, Any]:
    """Retusche (Arbeitsauflösung ~3500 px) -> Ausschnitt (x, y), lineares Ergebnis und Deckkraft."""
    H, W = lin.shape[:2]
    mask = mask_from_dabs(op.get("dabs") or [], W, H, float(op.get("feather", 40)))
    ys, xs = np.nonzero(mask > 0.003)
    if len(xs) == 0:
        return {"x": 0, "y": 0, "W": W, "H": H, "lin": np.zeros((0, 0, 3), np.float16), "alpha": np.zeros((0, 0), np.uint8)}
    size = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    pad = int(size * 3.2) + 16                      # Umgebung für Quellsuche / Netz
    X0, Y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    X1, Y1 = min(W, xs.max() + 1 + pad), min(H, ys.max() + 1 + pad)
    region = lin[Y0:Y1, X0:X1].astype(np.float32)
    mreg = mask[Y0:Y1, X0:X1]
    wb3 = np.asarray(wb, np.float32)[:3]
    gain = min(8.0, 1.0 / max(float(np.percentile(region * wb3[None, None], 99.5)), 1e-3))
    disp = _to_disp(region, wb3, gain)
    method = "heal"
    out = None
    if op.get("mode") == "remove":
        try:
            out = inpaint_ai(disp, mreg)
            if out is not None:
                method = "ai"
        except Exception as e:  # noqa: BLE001
            log.info("KI-Entfernen nicht möglich (%s), repariere", e)
    if out is None:
        out = heal(disp, mreg, int(op.get("variant", 0)))
    res = _from_disp(out, wb3, gain)
    # nur der Bereich der Maske wird gespeichert
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max() + 1, ys.max() + 1
    lx0, ly0 = x0 - X0, y0 - Y0
    return {"x": int(x0), "y": int(y0), "W": W, "H": H, "method": method,
            "lin": res[ly0:ly0 + (y1 - y0), lx0:lx0 + (x1 - x0)].astype(np.float16),
            "alpha": (np.clip(mask[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8)}


def save_patch(key: str, op_id: str, p: dict[str, Any]) -> None:
    np.savez(store_dir(key) / f"{op_id}.npz", **{k: v for k, v in p.items()})


def load_patch(key: str, op_id: str) -> dict[str, Any] | None:
    f = store_dir(key) / f"{op_id}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def ensure_patches(key: str, ops: list[dict[str, Any]], full: Any) -> list[dict[str, Any]]:
    """Ausschnitte laden, fehlende (z. B. Zwischenspeicher geleert) neu rechnen. ``full``: Funktion, die
    (lin_voll, wb) liefert (nur bei Bedarf aufgerufen)."""
    out, cache = [], None
    for op in ops:
        p = load_patch(key, str(op.get("id")))
        if p is None:
            if cache is None:
                cache = full()
            p = compute_patch(cache[0], cache[1], op)
            save_patch(key, str(op.get("id")), p)
        out.append(p)
    return out


def apply_patches(lin: np.ndarray, patches: list[dict[str, Any]]) -> np.ndarray:
    """Retusche in ein (beliebig verkleinertes) lineares Bild einblenden."""
    if not patches:
        return lin
    out = lin.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    for p in patches:
        pl = p["lin"]
        if pl.size == 0:
            continue
        sx, sy = w / float(p["W"]), h / float(p["H"])
        x0, y0 = int(math.floor(p["x"] * sx)), int(math.floor(p["y"] * sy))
        x1 = min(w, int(math.ceil((p["x"] + pl.shape[1]) * sx)))
        y1 = min(h, int(math.ceil((p["y"] + pl.shape[0]) * sy)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        tw, th = x1 - x0, y1 - y0
        interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
        patch = cv2.resize(pl.astype(np.float32), (tw, th), interpolation=interp)
        a = cv2.resize(p["alpha"].astype(np.float32) / 255, (tw, th), interpolation=interp)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + patch * a
    return out

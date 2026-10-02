"""Lokales Denoise -> lineare DNG (Alternative zu Lightroom-Denoise).

Ehrliche Einordnung: Adobe Denoise arbeitet auf den Bayer-Rohdaten mit einem auf echtes
Kamerarauschen trainierten Netz und ist bei ISO 6400+ klar besser. Dieser Weg ist gedacht,
wenn Lightroom-Denoise nicht genutzt werden soll oder die XMP-Deklaration (Hypothese H1)
nicht funktioniert.

Ablauf:
1. RAW demosaicen (linear, Kamerafarben, ohne Weissabgleich).
2. Weissabgleich + Gamma: das Netz (NAFNet, SIDD) ist auf sRGB-ähnlichen Daten trainiert.
3. Entrauschen: NAFNet in Kacheln (MPS/CUDA/CPU) oder klassisch
   (Anscombe-Transformation + Non-Local-Means auf Luminanz, stärkere Glättung der Chroma).
4. Zurück in lineare Kamerafarben, als LinearRaw-DNG mit Farbmatrix, AsShotNeutral und
   eingebettetem XMP (Lightroom liest bei DNG nur das eingebettete XMP).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..io import raw as raw_io
from ..io.dng import write_dng
from ..vision.models import has_module, model_path, torch_device

log = logging.getLogger(__name__)
GAMMA = 1 / 2.2


def _to_display(lin: np.ndarray, wb: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(lin * wb[None, None, :3] * gain, 0, 1) ** GAMMA


def _from_display(disp: np.ndarray, wb: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(disp, 0, 1) ** (1 / GAMMA) / (wb[None, None, :3] * gain)


def classical(img: np.ndarray, strength: float) -> np.ndarray:
    """img: gamma-kodiert 0..1, float32. strength ~ Rauschsigma (0..0.1)."""
    ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y16 = np.clip(ycc[..., 0] * 65535, 0, 65535).astype(np.uint16)
    h = float(np.clip(strength * 65535 * 1.1, 200, 12000))
    y = cv2.fastNlMeansDenoising(y16, h=[h], templateWindowSize=5, searchWindowSize=15,
                                 normType=cv2.NORM_L1).astype(np.float32) / 65535
    chroma = [cv2.bilateralFilter(ycc[..., c], 9, strength * 4 + 0.02, 5) for c in (1, 2)]
    chroma = [cv2.GaussianBlur(c, (0, 0), 1.2) for c in chroma]
    out = np.stack([y, *chroma], -1)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)


class _Nafnet:
    def __init__(self):
        import torch

        from .nafnet import load_nafnet

        p = model_path("nafnet_sidd")
        if p is None:
            raise RuntimeError("NAFNet-Gewichte fehlen")
        self.torch = torch
        self.device = torch_device()
        self.net = load_nafnet(str(p), self.device)

    def __call__(self, img: np.ndarray, tile: int = 512, overlap: int = 32) -> np.ndarray:
        torch = self.torch
        h, w = img.shape[:2]
        out = np.zeros_like(img, dtype=np.float32)
        weight = np.zeros((h, w, 1), np.float32)
        ramp = np.ones((tile, tile, 1), np.float32)
        r = np.linspace(0, 1, overlap, dtype=np.float32)
        ramp[:overlap] *= r[:, None, None]
        ramp[-overlap:] *= r[::-1, None, None]
        ramp[:, :overlap] *= r[None, :, None]
        ramp[:, -overlap:] *= r[None, ::-1, None]
        step = tile - overlap
        with torch.no_grad():
            for y0 in range(0, max(h - overlap, 1), step):
                for x0 in range(0, max(w - overlap, 1), step):
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    ys, xs = max(0, y1 - tile), max(0, x1 - tile)
                    patch = img[ys:y1, xs:x1]
                    t = torch.from_numpy(patch.transpose(2, 0, 1)[None].copy()).to(self.device)
                    res = self.net(t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
                    wgt = ramp[: y1 - ys, : x1 - xs]
                    out[ys:y1, xs:x1] += res * wgt
                    weight[ys:y1, xs:x1] += wgt
        return out / np.maximum(weight, 1e-6)


_NET: _Nafnet | None = None


def denoise_image(disp: np.ndarray, strength: float, method: str = "auto") -> tuple[np.ndarray, str]:
    global _NET
    if method in ("auto", "nafnet") and has_module("torch"):
        try:
            if _NET is None:
                _NET = _Nafnet()
            return _NET(disp.astype(np.float32)), "nafnet"
        except Exception as e:  # noqa: BLE001
            log.info("NAFNet nicht verfügbar (%s), nutze klassisches Verfahren", e)
    return classical(disp, strength), "classical"


def denoise_linear(lin: np.ndarray, wb: np.ndarray, sigma_mid: float, amount: int = 50,
                   method: str = "auto") -> tuple[np.ndarray, str]:
    """KI-Entrauschen linearer Kamerafarben (ohne Weissabgleich): Ergebnis wieder linear, gleiche Skala."""
    wb = np.asarray(wb, dtype=np.float32)[:3]
    # Headroom: Lichter nicht abschneiden
    gain = 1.0 / max(float(np.percentile(lin * wb[None, None, :3], 99.9)), 1e-3)
    gain = min(gain, 8.0)
    disp = _to_display(lin, wb, gain).astype(np.float32)
    # Rauschsigma im Anzeigeraum grob an der Mitteltonstelle (Gamma-Ableitung)
    sigma_disp = float(sigma_mid) * gain * GAMMA * (0.18 * gain) ** (GAMMA - 1)
    strength = sigma_disp * (0.5 + amount / 100.0)
    den, used = denoise_image(disp, strength, method)
    # Stärke steuern: Mischung mit dem Original (amount 100 = volle Wirkung)
    mix = float(np.clip(amount / 70.0, 0.2, 1.0))
    den = mix * den + (1 - mix) * disp
    return _from_display(den, wb, gain).astype(np.float32), used


def denoise_to_dng(src: Path, dst: Path, amount: int = 50, method: str = "auto", xmp: bytes | None = None,
                   exif: dict | None = None) -> dict:
    """Entrauscht eine RAW-Datei und schreibt eine lineare DNG. Das Original bleibt unverändert."""
    lin, info = raw_io.decode(src, half_size=False, oriented=False)
    noise = raw_io.estimate_noise(src)
    wb = np.asarray(info.camera_wb, dtype=np.float32)
    out_lin, used = denoise_linear(lin, wb, noise["sigma_mid"], amount, method)
    data16 = np.clip(out_lin * 65535, 0, 65535).astype(np.uint16)
    e = exif or {}
    write_dng(dst, data16, cfa=False, color_matrix=info.xyz_to_cam,
              as_shot_neutral=tuple((1.0 / wb[:3]).tolist()), make=str(e.get("make") or "Tagmatti"),
              model=str(e.get("camera") or "Denoised"), iso=int(e["iso"]) if e.get("iso") else None,
              exposure_time=e.get("exposure_time"), fnumber=e.get("aperture"), focal_length=e.get("focal_length"),
              orientation=info.orientation, xmp=xmp)
    return {"method": used, "sigma": noise["sigma_mid"], "amount": amount, "path": str(dst)}

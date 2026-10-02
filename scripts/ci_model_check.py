"""Prüft in der Python-Umgebung der App, ob die KI-Modelle laden (für CI und Fehlersuche).

Aufruf: <venv>/bin/python scripts/ci_model_check.py
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from tagmatti.vision.models import has_module, torch_device

ok = True
print("Python", sys.version.split()[0], "| Gerät:", torch_device())
for mod in ("torch", "torchvision", "open_clip", "transformers", "onnxruntime", "mediapipe", "insightface",
            "ocrmac"):
    print(f"  {mod:13s}", "ja" if has_module(mod) else "FEHLT")

img = np.random.default_rng(0).integers(0, 255, (1365, 2048, 3), dtype=np.uint8)

try:
    from tagmatti.vision import action
    from tagmatti.vision.embeddings import ClipEmbedder

    t = time.time()
    emb = ClipEmbedder()
    print(f"CLIP geladen in {time.time() - t:.0f} s (dim {emb.dim}, Ästhetik: {emb._aesthetic is not None})")
    t = time.time()
    vec = emb.embed([img] * 8)
    print(f"CLIP 8 Bilder: {time.time() - t:.2f} s")
    print("Action (CLIP):", action.clip_action(emb, vec[:1]))
except Exception:  # noqa: BLE001
    ok = False
    print("CLIP FEHLER:")
    traceback.print_exc()

try:
    from tagmatti.vision.action import PoseDetector

    t = time.time()
    pose = PoseDetector()
    print(f"Pose geladen in {time.time() - t:.0f} s auf {pose.device}")
    t = time.time()
    res = pose.analyze([img] * 4)
    print(f"Pose 4 Bilder: {time.time() - t:.2f} s -> {res[0]}")
except Exception:  # noqa: BLE001
    ok = False
    print("POSE FEHLER:")
    traceback.print_exc()

sys.exit(0 if ok else 1)

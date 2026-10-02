"""Protokoll: alles Wichtige landet in <Datenordner>/logs/tagmatti.log (rotierend).

In der App unter Einstellungen › Protokoll sichtbar und kopierbar.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import data_dir


def logs_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_file() -> Path:
    return logs_dir() / "tagmatti.log"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_tagmatti", False) for h in root.handlers):
        return
    fh = logging.handlers.RotatingFileHandler(log_file(), maxBytes=5 << 20, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    fh.setLevel(level)
    # Eigene Meldungen ab INFO, fremde Bibliotheken (open_clip, huggingface ...) erst ab WARNING
    fh.addFilter(lambda r: r.name.startswith("tagmatti") or r.levelno >= logging.WARNING)
    fh._tagmatti = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    root.setLevel(min(root.level or logging.WARNING, level))
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    for noisy in ("httpx", "urllib3", "PIL", "matplotlib", "uvicorn.access", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def tail(path: Path, lines: int = 300) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400_000))
        data = f.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-lines:])

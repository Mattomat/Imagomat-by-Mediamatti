#!/usr/bin/env bash
# Imagomat – Einrichtung auf macOS (Apple Silicon)
#
# Installiert: Homebrew-Pakete (ExifTool, Node, uv), Python-Umgebung mit den KI-Paketen
# (PyTorch mit Apple-MPS), Frontend. KI-Pakete, die sich nicht installieren lassen, werden
# übersprungen – die App nutzt dann die eingebauten Ersatzverfahren.
# Die KI-Modelle selbst werden beim ersten Start automatisch geladen.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "==> Homebrew-Pakete"
# Homebrew ist installiert, aber noch nicht im PATH (typisch direkt nach der Installation)
if ! command -v brew >/dev/null && [ -x /opt/homebrew/bin/brew ]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! command -v brew >/dev/null; then
  echo "Homebrew fehlt. Zuerst installieren:" >&2
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"' >&2
  exit 1
fi
for pkg in exiftool node uv; do
  brew list "$pkg" >/dev/null 2>&1 || brew install "$pkg" || echo "  Warnung: $pkg konnte nicht installiert werden"
done
xcode-select -p >/dev/null 2>&1 || { echo "Xcode-Kommandozeilentools fehlen: xcode-select --install"; exit 1; }

echo "==> Python-Umgebung (backend/.venv)"
cd "$ROOT/backend"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev,scrape]"

echo "==> KI-Pakete (einzeln, Fehler werden übersprungen)"
for pkg in "torch>=2.3" "torchvision>=0.18" "open_clip_torch>=2.24" "transformers>=4.44" "timm>=1.0" \
           "einops>=0.8" "kornia>=0.7" "onnxruntime>=1.18" "mediapipe>=0.10.14" "insightface>=0.7.3" "ocrmac>=1.0"; do
  uv pip install "$pkg" >/dev/null 2>&1 && echo "  ok   $pkg" || echo "  FEHLT $pkg (Ersatzverfahren wird genutzt)"
done

echo "==> Tests"
IMAGOMAT_OFFLINE=1 python -m pytest -q

echo "==> Frontend"
cd "$ROOT/frontend"
npm install --no-audit --no-fund
npm run build

cat <<EOF

Fertig. Starten:
  cd "$ROOT/backend" && source .venv/bin/activate && imagomat serve
  dann im Browser öffnen:  http://127.0.0.1:8765

Beim ersten Import lädt Imagomat die KI-Modelle (einmalig, einige hundert MB).
EOF

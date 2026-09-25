#!/usr/bin/env bash
# Imagomat – Einrichtung auf macOS (Apple Silicon)
#
# Installiert: Homebrew-Pakete (ExifTool, Node, Rust), Python-Umgebung mit allen KI-Paketen
# (PyTorch mit Apple-MPS), Frontend-Abhängigkeiten. Die KI-Modelle werden beim ersten Start
# automatisch heruntergeladen (danach läuft alles offline).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "==> Homebrew-Pakete"
if ! command -v brew >/dev/null; then
  echo "Homebrew fehlt: https://brew.sh" >&2
  exit 1
fi
brew install exiftool node python@3.12 uv rustup-init || true
command -v cargo >/dev/null || rustup-init -y

echo "==> Python-Umgebung (backend/.venv)"
cd "$ROOT/backend"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[ml,mac,scrape,dev]"

echo "==> Tests"
IMAGOMAT_OFFLINE=1 python -m pytest -q

echo "==> Frontend"
cd "$ROOT/frontend"
npm install
npm run build

cat <<EOF

Fertig. Starten:
  Browser-Modus:   cd backend && source .venv/bin/activate && imagomat serve   ->  http://127.0.0.1:8765
  Desktop-App:     cd backend && source .venv/bin/activate && imagomat serve &
                   cd frontend && IMAGOMAT_DEV_BACKEND=1 npm run tauri dev
  App bauen:       scripts/build_app.sh

Erster Schritt: Stil-Profil aus deinem Katalog trainieren (Tab "Stil-Profile") und die
Lightroom-Roundtrip-Checkliste in docs/lightroom-roundtrip.md durchgehen.
EOF

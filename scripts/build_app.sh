#!/usr/bin/env bash
# Baut die Desktop-App: Python-Backend als eigenständige Binärdatei (PyInstaller) + Tauri-Bundle (.app/.dmg).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
TARGET="$(rustc -vV | sed -n 's/host: //p')"

cd "$ROOT/backend"
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install pyinstaller
pyinstaller --noconfirm --clean --name tagmatti-server --onefile \
  --collect-all tagmatti --collect-all rawpy --collect-all lightgbm --collect-all open_clip \
  --collect-all insightface --collect-all mediapipe --collect-data cv2 \
  --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on \
  -p . tagmatti/cli.py
mkdir -p "$ROOT/frontend/src-tauri/binaries"
cp dist/tagmatti-server "$ROOT/frontend/src-tauri/binaries/tagmatti-server-$TARGET"

cd "$ROOT/frontend"
npm run tauri build
echo "Fertig: frontend/src-tauri/target/release/bundle/"

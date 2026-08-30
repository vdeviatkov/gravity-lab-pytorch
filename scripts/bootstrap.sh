#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GAME_REPO=${GRAVITY_LAB_REPO:-/Users/vdeviatkov/Documents/game}
PYTHON=${PYTHON:-}
if [ -z "$PYTHON" ]; then
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON=$candidate; break; fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "Python 3.10+ was not found. Install it and rerun, or set PYTHON." >&2
  exit 1
fi
if [ ! -d "$GAME_REPO/python/gravity_lab" ]; then
  echo "Gravity Lab Python package missing at $GAME_REPO/python/gravity_lab" >&2
  echo "Set GRAVITY_LAB_REPO to the game repository." >&2
  exit 1
fi
if [ ! -d "$ROOT/.venv" ]; then "$PYTHON" -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$GAME_REPO" -e "$ROOT[test]"

LIBRARY=${GRAVITY_LAB_CLASSIC_LIBRARY:-$GAME_REPO/build-classic-rl/libgravity_lab_classic.dylib}
VIEWER=$GAME_REPO/build-classic-rl/gravity_lab_classic_viewer
if [ ! -f "$LIBRARY" ]; then
  echo "Native classic library missing: $LIBRARY" >&2
  echo "Build it per $GAME_REPO/docs/classic-rl.md or set GRAVITY_LAB_CLASSIC_LIBRARY." >&2
  exit 1
fi
if [ ! -x "$VIEWER" ]; then
  echo "Graphical viewer missing or not executable: $VIEWER" >&2
  exit 1
fi
echo "Setup complete."
echo "Run: ./scripts/train_one_minute.sh"
echo "Then: ./scripts/play_latest.sh"


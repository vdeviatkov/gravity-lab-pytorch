#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GAME_REPO=${GRAVITY_LAB_REPO:-$ROOT/gravity-lab}
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
if [ ! -d "$GAME_REPO/python/gravity_lab" ] && [ "$GAME_REPO" = "$ROOT/gravity-lab" ]; then
  echo "Initializing Gravity Lab submodule..."
  git -C "$ROOT" submodule update --init --recursive
fi
if [ ! -d "$GAME_REPO/python/gravity_lab" ]; then
  echo "Gravity Lab Python package missing at $GAME_REPO/python/gravity_lab" >&2
  echo "Run 'git submodule update --init --recursive' or set GRAVITY_LAB_REPO." >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) LIBRARY_NAME=libgravity_lab_classic.dylib ;;
  MINGW*|MSYS*|CYGWIN*) LIBRARY_NAME=gravity_lab_classic.dll ;;
  *) LIBRARY_NAME=libgravity_lab_classic.so ;;
esac
BUILD_DIR=$GAME_REPO/build-classic-rl
LIBRARY=${GRAVITY_LAB_CLASSIC_LIBRARY:-$BUILD_DIR/$LIBRARY_NAME}
VIEWER=$BUILD_DIR/gravity_lab_classic_viewer
if [ ! -f "$LIBRARY" ] || [ ! -x "$VIEWER" ]; then
  if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake is required to build Gravity Lab. Install the prerequisites listed in README.md." >&2
    exit 1
  fi
  echo "Building the Gravity Lab classic native library and viewer..."
  cmake -S "$GAME_REPO" -B "$BUILD_DIR" \
    -DGRAVITY_LAB_BUILD_CLASSIC=ON \
    -DGRAVITY_LAB_BUILD_DESKTOP=OFF
  cmake --build "$BUILD_DIR" --config Release
fi

if [ ! -d "$ROOT/.venv" ]; then "$PYTHON" -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$GAME_REPO" -e "$ROOT[test]"

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

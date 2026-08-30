#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
"$ROOT/.venv/bin/gravity-lab-rl" train "$@"
exec "$ROOT/.venv/bin/gravity-lab-rl" play --latest


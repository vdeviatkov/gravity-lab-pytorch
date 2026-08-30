#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -lt 1 ]; then
  echo "usage: $0 status|pause|resume|stop [--run-id ID|--latest]" >&2
  exit 2
fi
ACTION=$1
shift
exec "$ROOT/.venv/bin/gravity-lab-rl" control "$ACTION" "$@"


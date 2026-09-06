# Complete game-map plates

30 reusable PNG backgrounds: `lg{group}_t{track}.png`, groups 0–2, tracks 0–9.
Each has a JSON sidecar with world-to-image coordinates. `index.jpg` previews all 30.

Generated directly from the original game renderer, with full terrain and flags,
one fixed perspective, and no bike, shadow, or HUD. No trained policy is required.

Regenerate with `.venv/bin/python scripts/generate_map_plates.py` from the repo root.
See [video documentation](../../docs/map-videos.md) for checkpoint replay generation.

Derived from the vendored GPL-2.0-only Gravity Defied game and its original assets.
See [upstream credits](../../gravity-lab/classic/README.md) and
[license](../../gravity-lab/classic/LICENSE.md).

#!/usr/bin/env python3
"""Generate all 30 complete empty maps using the game's renderer, without policies.

Each PNG has original track lines and flag sprites, with one fixed perspective.
The JSON sidecar maps world pixels to image pixels: (x - min_ox, -y - min_oy).
Usage: .venv/bin/python scripts/generate_map_plates.py [--tracks 1:2,2:9]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PLATES_DIR = ROOT / 'assets' / 'map_plates'
import sys
sys.path.insert(0, str(ROOT / 'src'))
from gravity_lab_rl.playback import integration_paths
VIEWER = integration_paths()[2]
HEADLESS_ENV = {**os.environ, 'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'}


def parse_tracks(spec: str | None) -> list[tuple[int, int]]:
    if spec is None or spec == 'all':
        return [(group, track) for group in range(3) for track in range(10)]
    tracks = list(dict.fromkeys(tuple(map(int, item.split(':'))) for item in spec.split(',')))
    if any(len(pair) != 2 or not 0 <= pair[0] < 3 or not 0 <= pair[1] < 10 for pair in tracks):
        raise ValueError('tracks must be group:track pairs, group 0–2 and track 0–9')
    return tracks


def render_plate(level_group: int, track: int, output_dir: Path = PLATES_DIR,
                 level_pack: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'lg{level_group}_t{track}.png'
    command = [str(VIEWER), '--group', str(level_group), '--track', str(track),
               '--map-plate', str(path)]
    if level_pack:
        command += ['--level-pack', str(level_pack)]
    subprocess.run(command, check=True, env=HEADLESS_ENV, capture_output=True)
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    metadata.update(level_group=level_group, track=track, level_pack=level_pack,
                    renderer='classic fixed perspective; no bike, shadow, or HUD')
    sidecar.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'  {path.name}: {metadata["width"]} × {metadata["height"]}', flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', default=None, help='group:track pairs, default all 30')
    parser.add_argument('--output-dir', type=Path, default=PLATES_DIR)
    parser.add_argument('--level-pack')
    args = parser.parse_args()
    for group, track in parse_tracks(args.tracks):
        render_plate(group, track, args.output_dir, args.level_pack)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

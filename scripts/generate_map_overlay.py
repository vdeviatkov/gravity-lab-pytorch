#!/usr/bin/env python3
"""Replay all saved training policies together over complete, reusable game maps.

This visualizes deterministic checkpoint evaluations, not historical exploratory
training episodes. Defaults to all environments in the run's evaluation protocol.
Usage: .venv/bin/python scripts/generate_map_overlay.py --run-id ID --tracks 1:2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / '.matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
from gravity_lab_rl.config import curriculum_environments
from gravity_lab_rl.control import resolve_run
from generate_map_plates import HEADLESS_ENV, PLATES_DIR, VIEWER, parse_tracks, render_plate

FRAME_SIZE = (640, 480)
# With look-ahead disabled the renderer puts the bike reference at this point.
BIKE_CENTER = (320, 240)


def checkpoint_files(run_dir: Path) -> list[tuple[int, Path]]:
    snapshots = sorted((int(p.stem.split('_')[1]), p)
                       for p in (run_dir / 'timelapse').glob('t_*.gdp'))
    final = run_dir / 'final.gdp'
    if final.is_file():
        summary_path = run_dir / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        elapsed = int(summary.get('active_training_duration_seconds',
                                  snapshots[-1][0] + 1 if snapshots else 0))
        snapshots.append((elapsed, final))
    if not snapshots:
        raise ValueError(f'No timelapse snapshots or final.gdp found in {run_dir}')
    return snapshots


def record_checkpoint(policy: Path, level_group: int, track: int, league: int, max_steps: int,
                      seed: int, out_dir: Path, frame_skip: int = 2,
                      level_pack: str | None = None) -> list[tuple[int, int, int]]:
    """Capture an isolated bike layer and viewport origins, including reset and terminal frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(VIEWER), '--policy', str(policy), '--group', str(level_group),
               '--track', str(track), '--league', str(league), '--episodes', '1',
               '--fps', '0', '--hold-ms', '0', '--frame-skip', str(frame_skip),
               '--max-steps', str(max_steps), '--seed', str(seed),
               '--record-dir', str(out_dir), '--bike-only']
    if level_pack:
        command += ['--level-pack', str(level_pack)]
    result = subprocess.run(command, check=True, env=HEADLESS_ENV, capture_output=True, text=True)
    (out_dir / 'outcome.txt').write_text(result.stdout)
    with (out_dir / 'positions.csv').open(newline='') as stream:
        return [(int(row['frame']), int(row['bike_x']), int(row['bike_y']))
                for row in csv.DictReader(stream)]


def bike_canvas_pos(x: int, y: int, min_ox: int, min_oy: int) -> tuple[int, int]:
    # CSV values are camera origin, despite the legacy bike_x/bike_y column names.
    return x + BIKE_CENTER[0] - min_ox, -y + BIKE_CENTER[1] - min_oy


def load_plate(level_group: int, track: int, plates_dir: Path = PLATES_DIR,
               level_pack: str | None = None):
    from PIL import Image
    path = plates_dir / f'lg{level_group}_t{track}.png'
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if (not path.exists() or metadata.get('format') != 'gravity-lab-map-plate-v2'
            or metadata.get('level_pack') != level_pack):
        render_plate(level_group, track, plates_dir, level_pack)
        metadata = json.loads(sidecar.read_text())
    with Image.open(path) as source:
        return source.convert('RGB'), metadata['min_ox'], metadata['min_oy']


def sprite_alpha_mask(frame):
    """The dedicated bike layer contains no terrain; preserve every non-white bike pixel."""
    from PIL import Image, ImageChops
    difference = ImageChops.difference(frame.convert('RGB'), Image.new('RGB', frame.size, 'white'))
    r, g, b = difference.split()
    return ImageChops.lighter(r, ImageChops.lighter(g, b)).point(lambda p: 255 if p else 0)


def fit_plate_to_paths(plate, min_ox, min_oy, paths, margin=100):
    """Extend the fixed world view so even off-track crashes and high jumps stay visible."""
    from PIL import Image
    points = [bike_canvas_pos(x, y, min_ox, min_oy) for path in paths for _, x, y in path]
    left = min([0] + [x - margin for x, _ in points])
    top = min([0] + [y - margin for _, y in points])
    right = max([plate.width] + [x + margin for x, _ in points])
    bottom = max([plate.height] + [y + margin for _, y in points])
    width, height = right - left, bottom - top
    result = Image.new('RGB', (width + width % 2, height + height % 2), 'white')
    result.paste(plate, (-left, -top))
    return result, min_ox + left, min_oy + top


def selected_environments(config, tracks=None, league=None):
    environments = curriculum_environments(config)
    if tracks is not None:
        selected = []
        for group, track in tracks:
            matches = [e for e in environments if (e['level_group'], e['track']) == (group, track)]
            selected.extend(matches or [{**config['environment'], 'level_group': group,
                                         'track': track, 'league': group}])
        environments = selected
    unique = {}
    for env in environments:
        env = dict(env)
        if league is not None:
            env['league'] = league
        unique[(env['level_group'], env['track'], env['league'])] = env
    return list(unique.values())


def generate_video(run_dir, env, checkpoints, args):
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    group, track, league = env['level_group'], env['track'], env['league']
    label = f'lg{group}_t{track}'
    if league != group:
        label += f'_league{league}'
    level_pack = env.get('level_pack')
    plates_dir = run_dir / 'map_plates' if level_pack else PLATES_DIR
    plate, min_ox, min_oy = load_plate(group, track, plates_dir, level_pack)
    cmap = matplotlib.colormaps['plasma']
    colors = [tuple(int(v * 255) for v in cmap(i / max(1, len(checkpoints) - 1))[:3])
              for i in range(len(checkpoints))]
    work = Path(tempfile.mkdtemp(prefix=f'{label}_', dir=run_dir))
    try:
        paths, frame_dirs, attempts = [], [], []
        for i, (elapsed, policy) in enumerate(checkpoints):
            folder = work / f'checkpoint_{i}'
            positions = record_checkpoint(policy, group, track, league, env['max_episode_steps'],
                                          args.seed, folder, env['frame_skip'], level_pack)
            if not positions:
                raise RuntimeError(f'{policy}: empty recording')
            paths.append(positions)
            frame_dirs.append(folder)
            outcome = (folder / 'outcome.txt').read_text().strip().splitlines()[-1]
            attempts.append(dict(policy=str(policy.relative_to(run_dir)), elapsed_seconds=elapsed,
                                 frames=len(positions), outcome=outcome, color=colors[i]))
            print(f'  [{label}] {i + 1}/{len(checkpoints)} t={elapsed}s: {outcome}', flush=True)
        plate, min_ox, min_oy = fit_plate_to_paths(plate, min_ox, min_oy, paths)
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=14)
        columns = max(1, plate.width // 280)
        header = 66 + math.ceil(len(checkpoints) / columns) * 26
        header += header % 2
        canvas = Image.new('RGB', (plate.width, plate.height + header), 'white')
        canvas.paste(plate, (0, header))
        d = ImageDraw.Draw(canvas)
        d.text((18, 10), f'{label} | {len(checkpoints)} saved policies | seed {args.seed}', font=font, fill='black')
        d.text((18, 36), 'Checkpoint replays | full map and paths | episode time shown below', font=small, fill='#555555')
        for i, ((elapsed, policy), color) in enumerate(zip(checkpoints, colors)):
            x, y = 18 + (i % columns) * 280, 64 + (i // columns) * 26
            d.rectangle((x, y + 3, x + 14, y + 17), fill=color)
            suffix = ' (final)' if policy.name == 'final.gdp' else ''
            d.text((x + 22, y), f'{i + 1}: train {elapsed}s{suffix}', font=small, fill='black')
        def point(row):
            x, y = bike_canvas_pos(row[1], row[2], min_ox, min_oy)
            return x, y + header
        longest = max(map(len, paths))
        steps = list(range(0, longest, args.step_stride))
        fps = 50 / (env['frame_skip'] * args.step_stride) * args.speedup
        # Padding keeps playback timing correct when the terminal step is off the sampling grid.
        if steps[-1] != longest - 1:
            steps.append(longest - 1)
        output = run_dir / f'map_overlay_{label}.mp4'
        temporary_output = work / 'video.mp4'
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{canvas.width}x{canvas.height}', '-framerate', str(fps),
                   '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_output)]
        encoder_log = work / 'ffmpeg.log'
        with encoder_log.open('wb') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame_no, step in enumerate(steps):
                    frame = canvas.copy()
                    draw = ImageDraw.Draw(frame, 'RGBA')
                    for i, (path, folder, color) in enumerate(zip(paths, frame_dirs, colors)):
                        current = min(step, len(path) - 1)
                        start = max(0, current - args.trail_length) if args.trail_length else 0
                        trail = [point(row) for row in path[start:current + 1]]
                        if len(trail) > 1:
                            draw.line(trail, fill=color + (125,), width=2)
                        row = path[current]
                        px, py = point(row)
                        with Image.open(folder / f'frame_{row[0]:06d}.png') as image:
                            source = image.convert('RGB')
                        mask = sprite_alpha_mask(source)
                        bbox = mask.getbbox()
                        if bbox:
                            sprite = source.crop(bbox)
                            tint = Image.new('RGB', sprite.size, color)
                            frame.paste(Image.blend(sprite, tint, 0.25),
                                        (px + bbox[0] - BIKE_CENTER[0], py + bbox[1] - BIKE_CENTER[1]),
                                        mask.crop(bbox))
                        draw.text((px + 25, py - 40), str(i + 1), font=small, fill=color + (255,))
                    draw.rectangle((canvas.width - 165, 8, canvas.width - 8, 31), fill='white')
                    draw.text((canvas.width - 160, 10), f'{step * env["frame_skip"] * .02:.2f}s', font=font, fill='black')
                    if args.keep_frames:
                        frame.save(work / f'combined_{frame_no:06d}.png')
                    process.stdin.write(frame.tobytes())
                # Brief final hold so all terminal positions can be inspected.
                for _ in range(max(1, round(fps))):
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(encoder_log.read_text())
            except BaseException:
                process.stdin.close()
                process.wait()
                raise
        temporary_output.replace(output)
        metadata = dict(format='gravity-lab-map-overlay-v2', source='deterministic checkpoint replays',
                        environment=env, seed=args.seed, fps=fps, step_stride=args.step_stride,
                        speedup=args.speedup, min_ox=min_ox, min_oy=min_oy, header_height=header,
                        width=canvas.width, height=canvas.height, attempts=attempts)
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'  -> {output}', flush=True)
        return output
    finally:
        if not args.keep_frames:
            shutil.rmtree(work)
        else:
            print(f'  capture frames: {work}', flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--run-id')
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--tracks', help='group:track pairs or all; default run evaluation tracks')
    parser.add_argument('--league', type=int, choices=range(4))
    parser.add_argument('--seed', type=int, default=2000007)
    parser.add_argument('--step-stride', type=int, default=1)
    parser.add_argument('--speedup', type=float, default=1.0)
    parser.add_argument('--trail-length', type=int, default=0, help='0 keeps complete paths (default)')
    parser.add_argument('--keep-frames', action='store_true')
    args = parser.parse_args(argv)
    if args.step_stride < 1 or not math.isfinite(args.speedup) or args.speedup <= 0 or args.trail_length < 0:
        parser.error('step-stride/speedup must be positive and trail-length nonnegative')
    if not VIEWER.exists() or not shutil.which('ffmpeg'):
        parser.error('build the classic viewer and install ffmpeg first')
    run_dir = args.run_dir.resolve() if args.run_dir else resolve_run(args.run_id, args.latest or args.run_id is None)
    config = json.loads((run_dir / 'config.json').read_text())
    checkpoints = checkpoint_files(run_dir)
    envs = selected_environments(config, parse_tracks(args.tracks) if args.tracks else None, args.league)
    print(f'{len(checkpoints)} policies, {len(envs)} maps', flush=True)
    for env in envs:
        generate_video(run_dir, env, checkpoints, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

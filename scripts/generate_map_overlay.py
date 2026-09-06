#!/usr/bin/env python3
"""Replay every recorded training attempt in batches over complete game maps.

Legacy runs without action recordings use explicitly labeled checkpoint replays.
Defaults to the first map in each level group (three videos).
Usage: .venv/bin/python scripts/generate_map_overlay.py --run-id ID --tracks 1:2
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
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
from gravity_lab_rl.recording import training_episodes
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
    best = run_dir / 'best.gdp'
    if best.is_file() and (not final.is_file() or best.read_bytes() != final.read_bytes()):
        summary_path = run_dir / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        snapshots.append((int(summary.get('best_policy_active_training_seconds', 0)), best))
    if not snapshots:
        raise ValueError(f'No timelapse snapshots or final.gdp found in {run_dir}')
    return snapshots


def record_checkpoint(policy: Path, level_group: int, track: int, league: int, max_steps: int,
                      seed: int, out_dir: Path, frame_skip: int = 2,
                      level_pack: str | None = None, *, actions=False) -> list[tuple[int, int, int]]:
    """Capture an isolated bike layer and viewport origins, including reset and terminal frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(VIEWER), '--actions' if actions else '--policy', str(policy), '--group', str(level_group),
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


def generate_video(run_dir, env, checkpoints, args, *, recordings=None, output_path=None,
                   output_size=None, offset=0):
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont, ImageOps
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
            record = recordings[i] if recordings is not None else None
            positions = record_checkpoint(policy, group, track, league, env['max_episode_steps'],
                                          record['seed'] if record else args.seed, folder,
                                          env['frame_skip'], level_pack, actions=record is not None)
            if record and len(positions) != record['action_count'] + 1:
                raise RuntimeError(f"Incomplete replay: {record['id']}")
            if not positions:
                raise RuntimeError(f'{policy}: empty recording')
            paths.append(positions)
            frame_dirs.append(folder)
            outcome = (folder / 'outcome.txt').read_text().strip().splitlines()[-1]
            if record and 'progress' in record:
                fields = dict(item.split('=') for item in outcome.split())
                same_progress = math.isclose(float(fields['progress']), record['progress'], rel_tol=1e-5, abs_tol=1e-6)
                same_outcome = all(bool(int(fields[key])) == record[key] for key in ('finished', 'crashed', 'truncated'))
                if not same_progress or not same_outcome:
                    raise RuntimeError(f"Replay outcome differs from recorded attempt: {record['id']}")
            attempt = dict(elapsed_seconds=elapsed, frames=len(positions), outcome=outcome, color=colors[i])
            if record:
                attempt.update(episode_id=record['id'], number=offset + i + 1,
                               actions=str(policy.relative_to(run_dir)), seed=record['seed'],
                               status=record['status'], practice_prefix_steps=record['practice_prefix_steps'])
            else:
                attempt['policy'] = str(policy.relative_to(run_dir))
            attempts.append(attempt)
            print(f'  [{label}] {i + 1}/{len(checkpoints)} t={elapsed}s: {outcome}', flush=True)
        plate, min_ox, min_oy = fit_plate_to_paths(plate, min_ox, min_oy, paths)
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=14)
        columns = max(1, plate.width // 280)
        legend_count = args.batch_size if recordings is not None else len(checkpoints)
        header = 66 + math.ceil(legend_count / columns) * 26
        header += header % 2
        canvas = Image.new('RGB', (plate.width, plate.height + header), 'white')
        canvas.paste(plate, (0, header))
        d = ImageDraw.Draw(canvas)
        title = (f'{label} | actual attempts {offset + 1}-{offset + len(checkpoints)}' if recordings is not None
                 else f'{label} | {len(checkpoints)} saved policies | seed {args.seed}')
        subtitle = ('Recorded training actions | full map and paths | episode time' if recordings is not None
                    else 'Checkpoint replays | full map and paths | episode time shown below')
        d.text((18, 10), title, font=font, fill='black')
        d.text((18, 36), subtitle, font=small, fill='#555555')
        for i, ((elapsed, policy), color) in enumerate(zip(checkpoints, colors)):
            x, y = 18 + (i % columns) * 280, 64 + (i // columns) * 26
            d.rectangle((x, y + 3, x + 14, y + 17), fill=color)
            suffix = {'final.gdp': ' (final)', 'best.gdp': ' (best)'}.get(policy.name, '')
            if recordings is not None:
                record = recordings[i]
                suffix = ' partial' if record['status'] != 'terminal' else ''
                if record['practice_prefix_steps']:
                    suffix += f" prefix={record['practice_prefix_steps']}"
            d.text((x + 22, y), f'{offset + i + 1}: train {elapsed}s{suffix}', font=small, fill='black')
        def point(row):
            x, y = bike_canvas_pos(row[1], row[2], min_ox, min_oy)
            return x, y + header
        longest = max(map(len, paths))
        steps = list(range(0, longest, args.step_stride))
        fps = 50 / (env['frame_skip'] * args.step_stride) * args.speedup
        # Padding keeps playback timing correct when the terminal step is off the sampling grid.
        if steps[-1] != longest - 1:
            steps.append(longest - 1)
        output = output_path or run_dir / f'map_overlay_{label}.mp4'
        temporary_output = work / 'video.mp4'
        encoded_size = output_size or canvas.size
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                   '-video_size', f'{encoded_size[0]}x{encoded_size[1]}', '-framerate', str(fps),
                   '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                   '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_output)]
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
                        draw.text((px + 25, py - 40), str(offset + i + 1), font=small, fill=color + (255,))
                    draw.rectangle((canvas.width - 165, 8, canvas.width - 8, 31), fill='white')
                    draw.text((canvas.width - 160, 10), f'{step * env["frame_skip"] * .02:.2f}s', font=font, fill='black')
                    if args.keep_frames:
                        frame.save(work / f'combined_{frame_no:06d}.png')
                    encoded_frame = ImageOps.pad(frame, encoded_size, color="white") if output_size else frame
                    process.stdin.write(encoded_frame.tobytes())
                # Brief final hold so all terminal positions can be inspected.
                for _ in range(max(1, round(fps))):
                    process.stdin.write(encoded_frame.tobytes())
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(encoder_log.read_text())
            except BaseException:
                process.stdin.close()
                process.wait()
                raise
        temporary_output.replace(output)
        metadata = dict(format='gravity-lab-map-overlay-v2', source='actual training actions' if recordings is not None else 'deterministic checkpoint replays',
                        environment=env, seed=args.seed, fps=fps, step_stride=args.step_stride,
                        speedup=args.speedup, min_ox=min_ox, min_oy=min_oy, header_height=header,
                        width=encoded_size[0], height=encoded_size[1], attempts=attempts)
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'  -> {output}', flush=True)
        return output
    finally:
        if not args.keep_frames:
            shutil.rmtree(work)
        else:
            print(f'  capture frames: {work}', flush=True)



def generate_recorded_video(run_dir, env, records, args):
    """Bound raw captures to one batch; concatenate all batches into one map video."""
    group, track, league = env['level_group'], env['track'], env['league']
    label = f'lg{group}_t{track}' + (f'_league{league}' if league != group else '')
    selected = [r for r in records if r['environment'] == env]
    if not selected:
        print(f'  [{label}] no recorded attempts; skipped', flush=True)
        return None
    plate, _, _ = load_plate(group, track, run_dir / 'map_plates' if env.get('level_pack') else PLATES_DIR,
                            env.get('level_pack'))
    # One stable encoding size per map; fit each complete batch without cropping.
    columns = max(1, plate.width // 280)
    height = plate.height + 266 + math.ceil(args.batch_size / columns) * 26
    size = (plate.width + 200 + plate.width % 2, height + height % 2)
    work = Path(tempfile.mkdtemp(prefix=f'{label}_batches_', dir=run_dir))
    output = run_dir / f'map_overlay_{label}.mp4'
    try:
        batches = []
        attempts = []
        for offset in range(0, len(selected), args.batch_size):
            batch = selected[offset:offset + args.batch_size]
            path = work / f'batch_{len(batches):06d}.mp4'
            generate_video(run_dir, env, [(int(r['started_seconds']), r['actions_path']) for r in batch],
                           args, recordings=batch, output_path=path, output_size=size, offset=offset)
            metadata = json.loads(path.with_suffix('.json').read_text())
            attempts.extend(metadata['attempts'])
            batches.append(path)
        playlist = work / 'concat.txt'
        # Generated basenames contain no quoting characters; resolve relative to playlist.
        playlist.write_text(''.join(f"file '{p.name}'\n" for p in batches))
        temporary = work / 'complete.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '1',
                        '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart', str(temporary)], check=True)
        temporary.replace(output)
        output.with_suffix('.json').write_text(json.dumps(dict(
            format='gravity-lab-map-overlay-v3', source='actual training actions', environment=env,
            batch_size=args.batch_size, batch_count=len(batches), attempt_count=len(attempts),
            width=size[0], height=size[1], fps=metadata['fps'], attempts=attempts), indent=2) + '\n')
        return output
    finally:
        if not args.keep_frames:
            shutil.rmtree(work)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--run-id')
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--tracks', default='0:0,1:0,2:0', help='group:track pairs or all; default first map in each group')
    parser.add_argument('--source', choices=['auto', 'training', 'checkpoints'], default='auto')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--league', type=int, choices=range(4))
    parser.add_argument('--seed', type=int, default=2000007)
    parser.add_argument('--step-stride', type=int, default=1)
    parser.add_argument('--speedup', type=float, default=1.0)
    parser.add_argument('--trail-length', type=int, default=0, help='0 keeps complete paths (default)')
    parser.add_argument('--keep-frames', action='store_true')
    parser.add_argument('--jobs', type=int, default=2, help='independent map render jobs (default 2)')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.jobs < 1 or args.step_stride < 1 or not math.isfinite(args.speedup) or args.speedup <= 0 or args.trail_length < 0:
        parser.error('step-stride/speedup must be positive and trail-length nonnegative')
    if not VIEWER.exists() or not shutil.which('ffmpeg'):
        parser.error('build the classic viewer and install ffmpeg first')
    run_dir = args.run_dir.resolve() if args.run_dir else resolve_run(args.run_id, args.latest or args.run_id is None)
    config = json.loads((run_dir / 'config.json').read_text())
    sessions_path = run_dir / 'recording_sessions.jsonl'
    sessions = [json.loads(line) for line in sessions_path.read_text().splitlines()] if sessions_path.exists() else []
    history_complete = bool(sessions and sessions[0]['transitions'] == 0 and all(s['enabled'] for s in sessions))
    records = training_episodes(run_dir)
    recorded = args.source == 'training' or (args.source == 'auto' and (run_dir / 'training_episodes').exists())
    tracks = parse_tracks(args.tracks) if args.tracks else None
    if recorded:
        if args.step_stride != 1:
            parser.error('actual training videos require step-stride 1 to retain every movement')
        if not records:
            parser.error('no recorded training attempts; historical actions cannot be recovered from policies')
        unique = {}
        for record in records:
            env = record['environment']
            if tracks is not None and (env['level_group'], env['track']) not in tracks:
                continue
            if args.league is not None and env['league'] != args.league:
                continue
            unique[json.dumps(env, sort_keys=True)] = env
        envs = list(unique.values())
        # Multiple simulation settings for one map cannot share a video silently.
        keys = [(e['level_group'], e['track'], e['league']) for e in envs]
        if len(keys) != len(set(keys)):
            parser.error('recordings contain changed simulation settings for the same map; split this run before rendering')
        if not envs:
            parser.error('no recorded attempts for the selected maps')
        if not history_complete:
            print('Recording does not cover the entire run; rendering all available recorded attempts.', flush=True)
        checkpoints = []
        print(f'{len(records)} recorded attempts, {len(envs)} selected maps', flush=True)
    else:
        checkpoints = checkpoint_files(run_dir)
        envs = selected_environments(config, tracks, args.league)
        print(f'Legacy checkpoint replays: {len(checkpoints)} policies, {len(envs)} maps (not actual training history)', flush=True)
    # Prepare shared plates and matplotlib before threads; the actual native
    # environments always run in separate viewer subprocesses.
    import matplotlib
    for env in envs:
        pack = env.get('level_pack')
        directory = run_dir / 'map_plates' if pack else PLATES_DIR
        load_plate(env['level_group'], env['track'], directory, pack)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outputs = list(pool.map(lambda env: generate_recorded_video(run_dir, env, records, args)
                                if recorded else generate_video(run_dir, env, checkpoints, args), envs))
    (run_dir / 'map_overlay_manifest.json').write_text(json.dumps({
        'format': 'gravity-lab-map-overlay-manifest-v1',
        'videos': [str(path.relative_to(run_dir)) for path in outputs],
        'map_count': len(outputs), 'policy_count': len(checkpoints),
        'source': 'actual training actions' if recorded else 'deterministic checkpoint replays',
        'recorded_attempt_count': len(records) if recorded else 0,
        'history_recorded_from_start': history_complete if recorded else False,
    }, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render a timelapse video per track showing a policy's improvement across training checkpoints.

Reads the periodic policy snapshots a trainer writes to <run_dir>/timelapse/t_<seconds>.gdp when
the training config sets experiment.timelapse_interval_seconds, plays each one headlessly through
gravity_lab_classic_viewer (SDL_VIDEODRIVER=dummy, so no window/display is needed), captions each
frame with its checkpoint's elapsed training time, and concatenates all checkpoints in order into
one video per (level_group, track) pair -- watch the same track get played worse-to-better as
training progressed.

Two output modes:
  - Default: one video per track (as above).
  - `--grid`: one combined video with every requested track tiled side by side in the same frame,
    all playing the same checkpoint simultaneously -- the standard "compare attempts at a glance"
    layout, advancing to the next checkpoint once every tile's episode has ended (shorter episodes
    hold their last frame so tiles stay in sync).

`--speedup N` multiplies the output framerate (default 1x = real training-time pacing), for
watching a run's improvement faster than it happened without re-simulating anything.

Usage:
    scripts/generate_timelapse.py --run-id <run_id> [--tracks 0:0,1:0,2:0] [--fps 25]
    scripts/generate_timelapse.py --run-id <run_id> --grid --speedup 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravity_lab_rl.control import resolve_run  # noqa: E402

VIEWER = ROOT / "gravity-lab" / "build-classic-rl" / "gravity_lab_classic_viewer"
HEADLESS_ENV = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}


def parse_tracks(spec: str) -> list[tuple[int, int]]:
    pairs = []
    for item in spec.split(","):
        level_group, track = item.split(":")
        pairs.append((int(level_group), int(track)))
    return pairs


def checkpoint_files(run_dir: Path) -> list[tuple[int, Path]]:
    timelapse_dir = run_dir / "timelapse"
    if not timelapse_dir.is_dir():
        raise SystemExit(
            f"no timelapse/ directory in {run_dir} -- set experiment.timelapse_interval_seconds "
            "in the training config before training (or resuming) to produce snapshots"
        )
    result = []
    for path in sorted(timelapse_dir.glob("t_*.gdp")):
        elapsed = int(path.stem.split("_")[1])
        result.append((elapsed, path))
    return result


def render_checkpoint_frames(policy: Path, level_group: int, track: int, max_steps: int,
                             seed: int, out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    subprocess.run(
        [str(VIEWER), "--policy", str(policy), "--group", str(level_group), "--track", str(track),
         "--league", str(level_group), "--episodes", "1", "--fps", "0", "--hold-ms", "0",
         "--max-steps", str(max_steps), "--seed", str(seed), "--record-dir", str(out_dir)],
        check=True, env=HEADLESS_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    return len(list(out_dir.glob("frame_*.png")))


def caption_and_renumber(frames_dir: Path, caption: str, start_index: int, combined_dir: Path) -> int:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    index = start_index
    for frame in sorted(frames_dir.glob("frame_*.png")):
        image = Image.open(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), (image.width, 26)], fill=(0, 0, 0))
        draw.text((6, 4), caption, fill=(255, 255, 255), font=font)
        image.save(combined_dir / f"frame_{index:07d}.png")
        index += 1
    return index


def render_track_frames(policy: Path, level_group: int, track: int, max_steps: int, seed: int,
                        out_dir: Path) -> list[Path]:
    """Like render_checkpoint_frames, but returns the ordered frame paths instead of a count."""
    render_checkpoint_frames(policy, level_group, track, max_steps, seed, out_dir)
    return sorted(out_dir.glob("frame_*.png"))


def compose_grid_frame(tile_frames: list[Path | None], tile_labels: list[str], caption: str) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 18)
        tile_font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 14)
    except OSError:
        font = tile_font = ImageFont.load_default()

    tiles = [Image.open(path).convert("RGB") if path else None for path in tile_frames]
    tile_width, tile_height = next(t.size for t in tiles if t is not None)
    blank = Image.new("RGB", (tile_width, tile_height), (20, 20, 20))
    tiles = [t if t is not None else blank for t in tiles]

    columns = len(tiles) if len(tiles) <= 3 else math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    caption_height = 26
    canvas = Image.new("RGB", (tile_width * columns, caption_height + tile_height * rows), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), caption, fill=(255, 255, 255), font=font)
    for index, (tile, label) in enumerate(zip(tiles, tile_labels)):
        row, column = divmod(index, columns)
        x, y = column * tile_width, caption_height + row * tile_height
        canvas.paste(tile, (x, y))
        tile_draw = ImageDraw.Draw(canvas)
        tile_draw.rectangle([(x, y), (x + tile_width, y + 18)], fill=(0, 0, 0))
        tile_draw.text((x + 4, y + 1), label, fill=(255, 255, 0), font=tile_font)
    return canvas


def render_grid_checkpoints(checkpoints: list[tuple[int, Path]], tracks: list[tuple[int, int]],
                            max_steps: int, seed: int, raw_root: Path,
                            combined_dir: Path) -> int:
    index = 0
    tile_labels = [f"level {lg} track {t}" for lg, t in tracks]
    for i, (elapsed, policy) in enumerate(checkpoints):
        per_track_frames = []
        ok = True
        for lg, t in tracks:
            track_dir = raw_root / f"lg{lg}_t{t}"
            try:
                per_track_frames.append(render_track_frames(policy, lg, t, max_steps, seed, track_dir))
            except subprocess.CalledProcessError as error:
                stderr = error.stderr.decode(errors="replace")[-300:] if error.stderr else ""
                print(f"  [grid] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s) track "
                      f"{lg}:{t} failed to render, skipping checkpoint: {stderr}")
                ok = False
                break
        if not ok:
            continue
        frame_count = max(len(frames) for frames in per_track_frames)
        minutes, seconds = divmod(elapsed, 60)
        caption = f"t={minutes}:{seconds:02d}  checkpoint {i + 1}/{len(checkpoints)}"
        for frame_index in range(frame_count):
            tile_frames = [frames[min(frame_index, len(frames) - 1)] if frames else None
                           for frames in per_track_frames]
            grid_frame = compose_grid_frame(tile_frames, tile_labels, caption)
            grid_frame.save(combined_dir / f"frame_{index:07d}.png")
            index += 1
        print(f"  [grid] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s): "
              f"{frame_count} frames x {len(tracks)} tracks")
    return index


def encode_video(frames_dir: Path, fps: float, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%07d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--tracks", default=None,
                        help="comma-separated level_group:track pairs, default 0:0,1:0,2:0 "
                             "(one per curriculum stage)")
    parser.add_argument("--fps", type=float, default=None,
                        help="output video framerate; default derived from the run's frame_skip "
                             "so playback matches real training-time pacing")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="multiply the output framerate by this factor for faster playback "
                             "(default 1x = real training-time pacing)")
    parser.add_argument("--grid", action="store_true",
                        help="render one combined video with all --tracks tiled side by side, "
                             "all playing the same checkpoint at once, instead of one video per "
                             "track")
    parser.add_argument("--seed", type=int, default=2000007)
    parser.add_argument("--keep-frames", action="store_true",
                        help="do not delete the intermediate PNG sequence after encoding")
    args = parser.parse_args()
    if args.speedup <= 0.0:
        raise SystemExit("--speedup must be positive")

    if not VIEWER.is_file():
        raise SystemExit(f"viewer not found: {VIEWER} -- build gravity-lab/build-classic-rl first")

    run_dir = resolve_run(args.run_id, args.latest or args.run_id is None)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    frame_skip = int(config["environment"]["frame_skip"])
    max_steps = int(config["environment"]["max_episode_steps"])
    fps = (args.fps or (1000.0 / (20.0 * frame_skip))) * args.speedup

    tracks = parse_tracks(args.tracks) if args.tracks else [(0, 0), (1, 0), (2, 0)]
    checkpoints = checkpoint_files(run_dir)
    if not checkpoints:
        raise SystemExit(f"no timelapse snapshots found under {run_dir / 'timelapse'}")
    print(f"{len(checkpoints)} checkpoints spanning t={checkpoints[0][0]}s to t={checkpoints[-1][0]}s")

    work_root = run_dir / "timelapse_work"
    raw_dir = work_root / "_raw"

    if args.grid:
        combined_dir = work_root / "grid"
        if combined_dir.exists():
            shutil.rmtree(combined_dir)
        combined_dir.mkdir(parents=True)
        index = render_grid_checkpoints(checkpoints, tracks, max_steps, args.seed, raw_dir,
                                        combined_dir)
        if index == 0:
            print("  [grid] no frames rendered, skipping video")
        else:
            output = run_dir / "timelapse_grid.mp4"
            encode_video(combined_dir, fps, output)
            print(f"  -> {output} ({index} frames @ {fps:.1f}fps = {index / fps:.1f}s)")
            if not args.keep_frames:
                shutil.rmtree(combined_dir)
        if not args.keep_frames and raw_dir.exists():
            shutil.rmtree(raw_dir)
        if not args.keep_frames and work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()
        return 0

    for level_group, track in tracks:
        label = f"lg{level_group}_t{track}"
        combined_dir = work_root / label
        if combined_dir.exists():
            shutil.rmtree(combined_dir)
        combined_dir.mkdir(parents=True)
        index = 0
        for i, (elapsed, policy) in enumerate(checkpoints):
            try:
                frame_count = render_checkpoint_frames(policy, level_group, track, max_steps,
                                                       args.seed, raw_dir)
            except subprocess.CalledProcessError as error:
                stderr = error.stderr.decode(errors="replace")[-300:] if error.stderr else ""
                print(f"  [{label}] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s) "
                      f"failed to render, skipping: {stderr}")
                continue
            minutes, seconds = divmod(elapsed, 60)
            caption = f"level {level_group} track {track}  t={minutes}:{seconds:02d}  checkpoint {i + 1}/{len(checkpoints)}"
            index = caption_and_renumber(raw_dir, caption, index, combined_dir)
            print(f"  [{label}] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s): {frame_count} frames")
        if index == 0:
            print(f"  [{label}] no frames rendered, skipping video")
            continue
        output = run_dir / f"timelapse_{label}.mp4"
        encode_video(combined_dir, fps, output)
        print(f"  -> {output} ({index} frames @ {fps:.1f}fps = {index / fps:.1f}s)")
        if not args.keep_frames:
            shutil.rmtree(combined_dir)
    if not args.keep_frames:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

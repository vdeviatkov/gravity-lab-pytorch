#!/usr/bin/env python3
"""Train one PPO specialist per (level_group, track) in the 30-track roster.

Motivation (see docs/training-runs.md, "Session synthesis"): every attempt to train one shared
network across all 30 tracks plateaued in the same 7-9/30 range regardless of algorithm, reward
design, replay balancing, or curriculum strategy, while single-track training has reliably reached
~100% every time it's been tried. This trains 30 independent small networks, one per track, each
solving the easy single-task problem instead of the hard 30-task one.

Each specialist is a separate OS process (the native engine allows only one active environment per
*process*, not system-wide), so specialists run in parallel shards for wall-clock efficiency:
tracks are assigned to shards round-robin (`track_index % shards == shard`), and each shard trains
its assigned tracks sequentially within one process.

Each track's training itself also runs as its own subprocess (`gravity-lab-rl train`), supervised
with a hard wall-clock timeout and killed if exceeded -- the vendored physics engine can hang
inside a native call on certain tracks (the same bug `train_watchdog.py` exists for; observed here
on "Hole"), which blocks forever and doesn't respond to any in-process Python-level timeout. One
retry is attempted before giving up on a track and moving on, so a single stuck track doesn't block
the rest of the shard.

Usage:
    scripts/train_specialists.py --shard 0 --shards 6 --duration-seconds 300
(run once per shard, e.g. via separate background invocations for shards 0..5)
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravity_lab_rl.config import configured, load_config  # noqa: E402

LEVEL_GROUPS = range(3)
TRACKS_PER_GROUP = 10
GRAVITY_LAB_RL = str(ROOT / ".venv" / "bin" / "gravity-lab-rl")


def specialist_run_id(level_group: int, track: int) -> str:
    return f"specialist_lg{level_group}_t{track}"


def run_supervised(command: list[str], run_id: str, budget_seconds: float) -> bool:
    """Run a `gravity-lab-rl` subcommand under a hard wall-clock timeout. Returns True on success."""
    timeout = budget_seconds + 120.0  # generous margin over the active-training target
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    start = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - start > timeout:
            print(f"[{run_id}] TIMED OUT after {timeout:.0f}s -- killing (likely the native "
                  "physics-engine hang bug)", flush=True)
            process.kill()
            process.wait()
            return False
        time.sleep(1.0)
    return process.returncode == 0


def train_one_track(config_path: Path, run_id: str, duration_seconds: float) -> bool:
    return run_supervised(
        [GRAVITY_LAB_RL, "train", "--config", str(config_path),
         "--duration-seconds", str(duration_seconds), "--run-id", run_id],
        run_id, duration_seconds)


def resume_one_track(run_id: str, cumulative_duration_seconds: float, remaining_budget: float) -> bool:
    return run_supervised(
        [GRAVITY_LAB_RL, "resume", "--run-id", run_id,
         "--duration-seconds", str(cumulative_duration_seconds)],
        run_id, remaining_budget)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(ROOT / "configs" / "classic_intro_ppo.json"))
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--policies-dir", default=str(ROOT / "policies" / "specialists"))
    parser.add_argument("--only", help="comma-separated lg:track pairs to (re)train, e.g. 0:7,1:2 "
                                       "-- overrides the shard/shards roster split")
    parser.add_argument("--resume-to-seconds", type=float,
                        help="resume each track's existing run toward this cumulative active-"
                             "training target instead of training from scratch; skips tracks "
                             "whose checkpoint already reports finish_rate 1.0")
    args = parser.parse_args()

    base = load_config(args.base_config)
    policies_dir = Path(args.policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = ROOT / "artifacts" / "specialist_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        assigned = [tuple(int(x) for x in pair.split(":")) for pair in args.only.split(",")]
    else:
        roster = [(lg, t) for lg in LEVEL_GROUPS for t in range(TRACKS_PER_GROUP)]
        assigned = [(lg, t) for i, (lg, t) in enumerate(roster) if i % args.shards == args.shard]
    print(f"shard {args.shard}/{args.shards}: {len(assigned)} tracks: {assigned}", flush=True)

    failed: list[tuple[int, int]] = []
    for level_group, track in assigned:
        run_id = specialist_run_id(level_group, track)
        run_dir = ROOT / "artifacts" / run_id

        if args.resume_to_seconds is not None:
            checkpoint_path = run_dir / "latest.pt"
            if not checkpoint_path.is_file():
                print(f"[{run_id}] no existing checkpoint, skipping resume", flush=True)
                failed.append((level_group, track))
                continue
            import torch  # local import: only needed for the resume path

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_score = checkpoint.get("best_score")
            current_active = float(checkpoint["active_training_duration_seconds"])
            if best_score is not None and best_score[0] >= 1.0:
                print(f"[{run_id}] already at finish_rate=1.0, skipping", flush=True)
                continue
            if current_active >= args.resume_to_seconds:
                print(f"[{run_id}] already trained {current_active:.0f}s >= target "
                      f"{args.resume_to_seconds:.0f}s, skipping", flush=True)
                continue
            remaining = args.resume_to_seconds - current_active
            print(f"[{run_id}] resuming from {current_active:.0f}s toward "
                  f"{args.resume_to_seconds:.0f}s ({remaining:.0f}s more)...", flush=True)
            ok = resume_one_track(run_id, args.resume_to_seconds, remaining)
            if not ok:
                print(f"[{run_id}] retrying resume once...", flush=True)
                ok = resume_one_track(run_id, args.resume_to_seconds, remaining)
            if not ok:
                print(f"[{run_id}] FAILED twice, skipping", flush=True)
                failed.append((level_group, track))
                continue
        else:
            cfg = copy.deepcopy(base)
            cfg["environment"]["level_group"] = level_group
            cfg["environment"]["track"] = track
            cfg["environment"]["league"] = level_group
            cfg = configured(cfg, duration_seconds=args.duration_seconds)
            config_path = configs_dir / f"{run_id}.json"
            config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

            print(f"[{run_id}] training...", flush=True)
            ok = train_one_track(config_path, run_id, args.duration_seconds)
            if not ok:
                print(f"[{run_id}] retrying once...", flush=True)
                shutil.rmtree(run_dir, ignore_errors=True)
                ok = train_one_track(config_path, run_id, args.duration_seconds)
            if not ok:
                print(f"[{run_id}] FAILED twice, skipping", flush=True)
                failed.append((level_group, track))
                continue

        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            print(f"[{run_id}] no summary.json produced, skipping", flush=True)
            failed.append((level_group, track))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best_evaluation") or summary["final_evaluation"]
        print(f"[{run_id}] done: finish_rate={best['finish_rate']:.2f} "
              f"mean_progress={best['mean_progress']:.3f}", flush=True)
        source = run_dir / "best.gdp"
        if not source.is_file():
            source = run_dir / "final.gdp"
        destination = policies_dir / f"lg{level_group}_t{track}.gdp"
        shutil.copy(source, destination)
        sidecar = source.with_suffix(source.suffix + ".json")
        if sidecar.is_file():
            shutil.copy(sidecar, destination.with_suffix(destination.suffix + ".json"))
        print(f"[{run_id}] deployed -> {destination}", flush=True)

    if failed:
        print(f"shard {args.shard}/{args.shards} complete with failures: {failed}", flush=True)
    else:
        print(f"shard {args.shard}/{args.shards} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

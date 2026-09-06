#!/usr/bin/env python3
"""Plot every training-checkpoint attempt at one track overlaid on a single map.

Unlike generate_timelapse.py (one video per checkpoint, played in sequence or tiled side by side),
this draws every checkpoint's full path as a translucent line on one shared 2D plot -- see every
attempt "at the same time," the way trajectory-swarm plots are usually shown for RL: a bundle of
paths converging (or not) toward the goal as training progresses.

Positions come from Environment::bike_position() (gravity-lab's render camera's tracked bike
position -- see the submodule's classic_c_api/classic_env.py "Add bike_position diagnostic
accessor" change), read directly through the headless ClassicGravityEnv while replaying each
timelapse/*.gdp snapshot with greedy argmax action selection -- no image rendering or ffmpeg
needed, so this is much cheaper than generate_timelapse.py and works for any number of checkpoints.

Usage:
    scripts/generate_trajectory_swarm.py --run-id <run_id> [--tracks 0:0,1:0,2:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravity_lab_rl.control import resolve_run  # noqa: E402
from gravity_lab_rl.playback import require_integration  # noqa: E402


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


def rollout_path(policy, level_group: int, track: int, league: int, frame_skip: int,
                 max_episode_steps: int, obstacle_ray_count: int,
                 seed: int) -> tuple[list[int], list[int], str]:
    from gravity_lab import ClassicAction, ClassicConfig, ClassicGravityEnv

    config = ClassicConfig(level_group=level_group, track=track, league=league,
                           frame_skip=frame_skip, max_episode_steps=max_episode_steps,
                           obstacle_ray_count=obstacle_ray_count, seed=seed)
    xs: list[int] = []
    ys: list[int] = []
    outcome = "truncated"
    with ClassicGravityEnv(config) as env:
        observation = env.reset(seed)
        x, y = env.bike_position()
        xs.append(x)
        ys.append(y)
        while True:
            q_values = policy.evaluate(observation)
            action = ClassicAction(max(range(len(q_values)), key=q_values.__getitem__))
            result = env.step(action)
            observation = result.observation
            x, y = env.bike_position()
            xs.append(x)
            ys.append(y)
            if result.finished:
                outcome = "finished"
                break
            if result.crashed:
                outcome = "crashed"
                break
            if result.truncated:
                outcome = "truncated"
                break
    return xs, ys, outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--tracks", default=None,
                        help="comma-separated level_group:track pairs, default 0:0,1:0,2:0 "
                             "(one plot per curriculum stage)")
    parser.add_argument("--seed", type=int, default=2000007)
    args = parser.parse_args()

    require_integration(require_viewer=False)
    from gravity_lab import DenseQPolicy

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = resolve_run(args.run_id, args.latest or args.run_id is None)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    frame_skip = int(config["environment"]["frame_skip"])
    max_steps = int(config["environment"]["max_episode_steps"])
    obstacle_ray_count = int(config["environment"].get("obstacle_ray_count", 8))

    tracks = parse_tracks(args.tracks) if args.tracks else [(0, 0), (1, 0), (2, 0)]
    checkpoints = checkpoint_files(run_dir)
    if not checkpoints:
        raise SystemExit(f"no timelapse snapshots found under {run_dir / 'timelapse'}")
    print(f"{len(checkpoints)} checkpoints spanning t={checkpoints[0][0]}s to t={checkpoints[-1][0]}s")

    cmap = matplotlib.colormaps["plasma"]
    outcome_markers = {"finished": ("*", "#2a9d2a", 220, "black"), "crashed": ("x", "#d1495b", 90, None),
                       "truncated": ("o", "#888888", 40, "black")}

    for level_group, track in tracks:
        label = f"lg{level_group}_t{track}"
        figure, axis = plt.subplots(figsize=(11, 6))
        for i, (elapsed, policy_path) in enumerate(checkpoints):
            policy = DenseQPolicy.load(policy_path)
            try:
                xs, ys, outcome = rollout_path(policy, level_group, track, level_group, frame_skip,
                                               max_steps, obstacle_ray_count, args.seed)
            except Exception as error:  # noqa: BLE001
                print(f"  [{label}] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s) "
                      f"failed to roll out, skipping: {error}")
                continue
            color = cmap(i / max(1, len(checkpoints) - 1))
            minutes, seconds = divmod(elapsed, 60)
            axis.plot(xs, [-y for y in ys], color=color, alpha=0.65, linewidth=1.6,
                     label=f"t={minutes}:{seconds:02d}")
            marker, marker_color, size, edge = outcome_markers[outcome]
            scatter_kwargs = {"edgecolors": edge, "linewidths": 0.5} if edge else {}
            axis.scatter([xs[-1]], [-ys[-1]], marker=marker, color=marker_color, s=size,
                        zorder=5, **scatter_kwargs)
            print(f"  [{label}] checkpoint {i + 1}/{len(checkpoints)} (t={elapsed}s): "
                  f"{len(xs)} steps, {outcome}")
        axis.scatter([], [], marker="*", color="#2a9d2a", s=120, label="finished", edgecolors="black")
        axis.scatter([], [], marker="x", color="#d1495b", s=60, label="crashed")
        axis.scatter([], [], marker="o", color="#888888", s=40, label="truncated")
        axis.set_title(f"{run_dir.name}: level {level_group} track {track} -- "
                       f"{len(checkpoints)} attempts across training")
        axis.set_xlabel("bike x position")
        axis.set_ylabel("bike y position")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncols=1)
        figure.tight_layout()
        output = run_dir / f"trajectory_swarm_{label}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"  -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

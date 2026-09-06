#!/usr/bin/env python3
"""Plot a training run's progress: cumulative distinct maps passed and rolling finish rate.

Reads <run_dir>/metrics.jsonl (written once per completed training episode by every trainer) and
produces one PNG with two panels:
  1. Cumulative distinct (level_group, track) pairs that finished at least once during training
     rollout, vs active training time -- "how many maps has the noisy, exploring policy ever
     stumbled onto a finish for." This is NOT the strict deterministic best-eval score reported in
     docs/training-runs.md: it counts one-off lucky finishes under exploration noise, so it is
     always >= the formal eval score for the same run. Pass --best-score to overlay the formal
     number as a dashed reference line so the two are never confused for each other.
  2. Rolling finish rate over a sliding window of recent episodes, vs the same time axis -- shows
     training dynamics (plateaus, regressions) that the monotonic top panel alone hides.

The denominator on panel 1 is always the full TRACK_ID_SIZE curriculum roster (30), not just the
tracks this run happened to attempt -- a run that never unlocked the last curriculum stage still
reports out of the full roster, so "13/20" (20 = tracks attempted) can't be misread as "13/20 of
the whole game" when it was actually 13/30.

Usage:
    scripts/plot_progress.py --run-id <run_id> [--window 150] [--best-score 9] [--output progress.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gravity_lab_rl import TRACK_ID_SIZE  # noqa: E402
from gravity_lab_rl.control import resolve_run  # noqa: E402


def load_episodes(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"no metrics.jsonl in {run_dir}")
    with metrics_path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def cumulative_maps_passed(episodes: list[dict]) -> tuple[list[float], list[int]]:
    seen: set[tuple[int, int]] = set()
    times, counts = [], []
    for row in episodes:
        if row["finished"]:
            seen.add((row["level_group"], row["track"]))
        times.append(row["active_training_seconds"])
        counts.append(len(seen))
    return times, counts


def rolling_finish_rate(episodes: list[dict], window: int) -> tuple[list[float], list[float]]:
    times, rates = [], []
    finished = [1.0 if row["finished"] else 0.0 for row in episodes]
    running_sum = 0.0
    for i, row in enumerate(episodes):
        running_sum += finished[i]
        if i >= window:
            running_sum -= finished[i - window]
        denominator = min(i + 1, window)
        times.append(row["active_training_seconds"])
        rates.append(running_sum / denominator)
    return times, rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--window", type=int, default=150,
                        help="episode window for the rolling finish-rate panel")
    parser.add_argument("--best-score", type=int, default=None,
                        help="formal deterministic best-eval score (out of the full "
                             f"{TRACK_ID_SIZE}-track roster) to overlay as a reference line, e.g. "
                             "the number quoted in docs/training-runs.md for this run")
    parser.add_argument("--output", type=Path, default=None,
                        help="output PNG path, default <run_dir>/progress.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = resolve_run(args.run_id, args.latest or args.run_id is None)
    episodes = load_episodes(run_dir)
    if not episodes:
        raise SystemExit(f"{run_dir}/metrics.jsonl has no completed episodes yet")

    map_times, map_counts = cumulative_maps_passed(episodes)
    tracks_attempted = len({(row["level_group"], row["track"]) for row in episodes})
    rate_times, rates = rolling_finish_rate(episodes, args.window)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    top.step([t / 60.0 for t in map_times], map_counts, where="post", color="#2a6f97",
             label="ever finished during training (noisy)")
    if args.best_score is not None:
        top.axhline(args.best_score, color="#2a6f97", linestyle="--", linewidth=1.2, alpha=0.7,
                    label=f"formal best-eval score ({args.best_score}/{TRACK_ID_SIZE})")
        top.legend(loc="lower right", fontsize=8)
    top.set_ylabel(f"distinct maps (of {TRACK_ID_SIZE})")
    top.set_title(
        f"{run_dir.name}: maps ever finished during training rollout "
        f"({map_counts[-1]}/{TRACK_ID_SIZE}, {tracks_attempted} attempted) -- noisy, "
        "not the formal eval score",
        fontsize=10,
    )
    top.set_ylim(0, TRACK_ID_SIZE)
    top.grid(True, alpha=0.3)

    bottom.plot([t / 60.0 for t in rate_times], rates, color="#d1495b", linewidth=1.2)
    bottom.set_ylabel(f"finish rate (last {args.window} episodes)")
    bottom.set_xlabel("active training time (minutes)")
    bottom.set_ylim(0.0, 1.0)
    bottom.grid(True, alpha=0.3)

    figure.tight_layout()
    output = args.output or (run_dir / "progress.png")
    figure.savefig(output, dpi=150)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

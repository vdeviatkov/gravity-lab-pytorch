"""TensorBoard logging for the SAC trainer.

Everything is written under ``<run_dir>/tensorboard`` with the transition count as the step,
so a resumed run continues the same curves. When the ``tensorboard`` package is missing, or
``experiment.tensorboard`` is false, every call is a no-op and training is unaffected.

View with ``tensorboard --logdir artifacts --bind_all``.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

UPDATE_LOG_INTERVAL = 250
CURRICULUM_LOG_INTERVAL_SECONDS = 60.0
EPISODE_WINDOW = 50


def episode_kind(prefix_steps: int, greedy: bool) -> str:
    if not prefix_steps:
        return "full_start"
    return "demo_greedy" if greedy else "demo_explore"


class TrainingLogger:
    def __init__(self, run_dir: Path, config: dict[str, Any], enabled: bool = True):
        self.writer = None
        self.finished: dict[str, deque[bool]] = {}
        self._rate_mark: tuple[float, int, int] | None = None
        self._last_curriculum_log = float("-inf")
        if not enabled or not config["experiment"].get("tensorboard", True):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("warning: tensorboard is not installed; pip install tensorboard to enable logging",
                  file=sys.stderr)
            return
        self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), max_queue=1000, flush_secs=20)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def episode(self, step: int, metrics: dict[str, Any]) -> None:
        if self.writer is None:
            return
        kind = episode_kind(metrics["practice_prefix_steps"], metrics["greedy"])
        window = self.finished.setdefault(kind, deque(maxlen=EPISODE_WINDOW))
        window.append(bool(metrics["finished"]))
        write = self.writer.add_scalar
        write(f"episode_{kind}/finish_rate_last{EPISODE_WINDOW}", sum(window) / len(window), step)
        write(f"episode_{kind}/reward", metrics["reward"], step)
        write(f"episode_{kind}/length", metrics["length"], step)
        write(f"episode_{kind}/peak_progress", metrics["peak_progress"], step)
        write(f"episode_{kind}/crashed", float(metrics["crashed"]), step)

    def updates(self, step: int, update_count: int, stats: dict[str, torch.Tensor]) -> None:
        if self.writer is None or update_count % UPDATE_LOG_INTERVAL:
            return
        for name, value in stats.items():
            self.writer.add_scalar(f"train/{name}", float(value), step)

    def throughput(self, step: int, active_seconds: float, update_count: int) -> None:
        if self.writer is None:
            return
        now = time.monotonic()
        if self._rate_mark is not None:
            then, transitions, updates = self._rate_mark
            elapsed = now - then
            if elapsed > 0:
                self.writer.add_scalar("speed/env_steps_per_second", (step - transitions) / elapsed, step)
                self.writer.add_scalar("speed/updates_per_second", (update_count - updates) / elapsed, step)
        self._rate_mark = (now, step, update_count)
        self.writer.add_scalar("speed/active_training_hours", active_seconds / 3600.0, step)

    def curriculum(self, step: int, active_seconds: float, update_count: int,
                   summary: dict[str, Any] | None, force: bool = False) -> None:
        """Throughput and demo-curriculum progress, at most once a minute unless forced."""
        if self.writer is None:
            return
        if not force and active_seconds - self._last_curriculum_log < CURRICULUM_LOG_INTERVAL_SECONDS:
            return
        self._last_curriculum_log = active_seconds
        self.throughput(step, active_seconds, update_count)
        if summary is None:
            return
        lengths = {int(k): v for k, v in summary["demo_length"].items()}
        prefixes = {int(k): v for k, v in summary["prefix"].items()}
        total = sum(lengths.values())
        if total:
            self.writer.add_scalar("curriculum/walk_back_fraction",
                                   1.0 - sum(prefixes.values()) / total, step)
        self.writer.add_scalar("curriculum/graduated_maps", len(summary["graduated"]), step)
        self.writer.add_scalar("curriculum/advances", summary["advances"], step)
        self.writer.add_scalar("curriculum/retreats", summary["retreats"], step)
        for track, length in sorted(lengths.items()):
            self.writer.add_scalar(f"curriculum_walked_back/track_{track:02d}",
                                   1.0 - prefixes.get(track, 0) / length, step)

    def evaluation(self, step: int, evaluation: dict[str, Any], best_score: tuple | None) -> None:
        if self.writer is None:
            return
        write = self.writer.add_scalar
        episodes = evaluation["episodes"]
        write("eval/finished_maps", sum(1 for row in episodes if row["finished"]), step)
        write("eval/finish_rate", evaluation["finish_rate"], step)
        write("eval/mean_progress", evaluation["mean_progress"], step)
        write("eval/crash_rate", evaluation["crash_rate"], step)
        if best_score is not None:
            write("eval/best_finished_maps", round(best_score[0] * len(episodes)), step)
        for row in episodes:
            name = f"lg{row['level_group']}_t{row['track']}_{row['track_name']}"
            write(f"eval_progress/{name}", row["progress"], step)
        finished = [f"`lg{row['level_group']}_t{row['track']}` {row['track_name']}"
                    for row in episodes if row["finished"]]
        self.writer.add_text("eval/finished_list", ", ".join(finished) or "none", step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

"""Go-Explore style search for one finishing action sequence per map.

The classic environment is fully deterministic (three evaluation seeds always produce identical
episodes), so a map is solved once *one* action sequence reaches the finish. Policy noise almost
never discovers a multi-second setup maneuver (brake, lean back, hold throttle) because it must
repeat the same choice for dozens of consecutive 0.04 s decisions, so this module finds those
sequences with search instead:

- an archive of *cells* (coarse progress / speed / pitch buckets), each remembering the shortest
  action prefix that reaches it;
- a loop that picks a cell, reconstructs its state by replaying the prefix (the same exact
  reconstruction `practice.py` relies on; no teleporting), then explores with *sticky* random
  actions held for several steps, adding every new or shorter-reached cell to the archive;
- the first prefix that finishes is the map's demonstration.

Demonstrations are plain JSON (`Demo`) and are consumed by the trainer's backward start
curriculum and behavior-cloning loss (see `sac_trainer.py`). The search has no neural network,
runs at raw simulator speed, and is one process per map because the engine allows one active
environment per process.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import DEFAULT_OBSTACLE_RAY_COUNT, TRACKS_PER_LEVEL_GROUP

# Observation indices (see gravity-lab/docs/classic-rl.md): progress, center velocity, front and
# rear wheel offsets relative to the center point.
_PROGRESS, _CENTER_VX = 0, 6
_FRONT_DX, _FRONT_DY, _REAR_DX, _REAR_DY = 8, 9, 12, 13


@dataclass
class Demo:
    """One finishing action sequence plus everything needed to replay it exactly."""

    level_group: int
    track: int
    league: int
    frame_skip: int
    max_episode_steps: int
    obstacle_ray_count: int
    seed: int
    actions: list[int]
    track_name: str = ""
    final_progress: float = 0.0
    search_seconds: float = 0.0
    iterations: int = 0

    @property
    def track_id(self) -> int:
        return self.level_group * TRACKS_PER_LEVEL_GROUP + self.track

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps({"format": "gravity-lab-demo-v1", **asdict(self)}) + "\n")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> Demo:
        data = json.loads(Path(path).read_text())
        if data.pop("format", None) != "gravity-lab-demo-v1":
            raise ValueError(f"unsupported demo format: {path}")
        return cls(**data)


def demo_filename(level_group: int, track: int) -> str:
    return f"lg{level_group}_t{track}.json"


def load_demos(directory: str | Path, environments: list[dict[str, Any]]) -> dict[int, Demo]:
    """Demos keyed by track id for every curriculum environment that has one, verified to match
    the environment settings they will be replayed under."""
    result: dict[int, Demo] = {}
    for env_cfg in environments:
        path = Path(directory) / demo_filename(env_cfg["level_group"], env_cfg["track"])
        if not path.exists():
            continue
        demo = Demo.load(path)
        for key in ("league", "frame_skip", "max_episode_steps"):
            if int(getattr(demo, key)) != int(env_cfg[key]):
                raise ValueError(f"{path}: {key} {getattr(demo, key)} does not match config {env_cfg[key]}")
        ray_count = int(env_cfg.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT))
        if demo.obstacle_ray_count != ray_count:
            raise ValueError(f"{path}: obstacle_ray_count {demo.obstacle_ray_count} does not match config {ray_count}")
        result[demo.track_id] = demo
    return result


def pitch_of(observation: tuple[float, ...]) -> float:
    """Bike pitch in radians from the rear-to-front wheel vector (0 = level, positive = nose up)."""
    return math.atan2(observation[_FRONT_DY] - observation[_REAR_DY],
                      observation[_FRONT_DX] - observation[_REAR_DX])


@dataclass(frozen=True)
class CellSpec:
    progress_bin: float = 0.02
    speed_bin: float = 0.4
    pitch_bin: float = math.pi / 6

    def key(self, observation: tuple[float, ...]) -> tuple[int, int, int]:
        speed = max(-4, min(4, int(round(observation[_CENTER_VX] / self.speed_bin))))
        pitch = int(round(pitch_of(observation) / self.pitch_bin))
        return (int(math.floor(observation[_PROGRESS] / self.progress_bin)), speed, pitch)


@dataclass
class Cell:
    actions: list[int]
    progress: float
    chosen: int = 0
    seen: int = 1


@dataclass
class SearchConfig:
    explore_steps: int = 150
    hold_mean: float = 8.0          # mean sticky-action hold length (geometric)
    hold_choices: tuple[float, ...] = ()  # if set, each iteration draws its hold mean from these
    frontier_fraction: float = 0.5  # share of picks from the highest-progress cells
    frontier_window: float = 0.06   # progress span below the best cell that counts as frontier
    end_margin: int = 40            # keep this many steps of episode budget after any prefix
    cells: CellSpec = field(default_factory=CellSpec)


class MapSearch:
    """Go-Explore archive + loop for one map. `env` must be the only environment in the process."""

    def __init__(self, env: Any, env_cfg: dict[str, Any], seed: int, search: SearchConfig | None = None,
                 rng_seed: int = 0) -> None:
        self.env, self.env_cfg, self.seed = env, env_cfg, int(seed)
        self.config = search or SearchConfig()
        self.rng = random.Random(rng_seed)
        self.max_prefix = int(env_cfg["max_episode_steps"]) - self.config.end_margin
        self.archive: dict[tuple[int, int, int], Cell] = {}
        self.best_progress = -math.inf
        self.iterations = 0
        self.steps = 0
        self.demo: Demo | None = None
        observation = tuple(env.reset(self.seed))
        self._offer(observation, [])

    # -- archive -------------------------------------------------------------------------------
    def _offer(self, observation: tuple[float, ...], actions: list[int]) -> bool:
        if len(actions) > self.max_prefix:
            return False
        key = self.config.cells.key(observation)
        progress = float(observation[_PROGRESS])
        self.best_progress = max(self.best_progress, progress)
        cell = self.archive.get(key)
        if cell is None:
            self.archive[key] = Cell(list(actions), progress)
            return True
        cell.seen += 1
        if len(actions) < len(cell.actions):
            cell.actions, cell.progress = list(actions), progress
            return True
        return False

    def _select(self) -> Cell:
        cells = list(self.archive.values())
        if self.rng.random() < self.config.frontier_fraction:
            frontier = [c for c in cells if c.progress >= self.best_progress - self.config.frontier_window]
            return self.rng.choice(frontier)
        weights = [1.0 / math.sqrt(1.0 + c.chosen) for c in cells]
        return self.rng.choices(cells, weights=weights, k=1)[0]

    # -- one iteration -------------------------------------------------------------------------
    def _restore(self, actions: list[int]) -> tuple[float, ...] | None:
        observation = tuple(self.env.reset(self.seed))
        for action in actions:
            step = self.env.step(action)
            self.steps += 1
            if step.terminated or step.truncated:
                return None
            observation = tuple(step.observation)
        return observation

    def iterate(self) -> bool:
        """Run one explore iteration. Returns True once a finishing sequence has been found."""
        cell = self._select()
        cell.chosen += 1
        self.iterations += 1
        actions = list(cell.actions)
        observation = self._restore(actions)
        if observation is None:
            return False  # archive prefix no longer valid; should not happen in a deterministic engine
        hold, action = 0, 0
        hold_mean = self.rng.choice(self.config.hold_choices) if self.config.hold_choices else self.config.hold_mean
        for _ in range(self.config.explore_steps):
            if hold <= 0:
                action = self.rng.randrange(9)
                # Geometric hold length with the configured mean: P(stop) = 1 / mean each step.
                hold = 1
                while self.rng.random() > 1.0 / hold_mean:
                    hold += 1
            hold -= 1
            step = self.env.step(action)
            self.steps += 1
            actions.append(action)
            if step.finished:
                self.demo = Demo(int(self.env_cfg["level_group"]), int(self.env_cfg["track"]),
                                 int(self.env_cfg["league"]), int(self.env_cfg["frame_skip"]),
                                 int(self.env_cfg["max_episode_steps"]),
                                 int(self.env_cfg.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT)),
                                 self.seed, actions, getattr(self.env, "track_name", ""),
                                 float(step.observation[_PROGRESS]))
                return True
            if step.terminated or step.truncated:
                return False
            self._offer(tuple(step.observation), actions)
            if len(actions) >= self.max_prefix:
                return False
        return False

    def run(self, time_budget: float, iteration_budget: int | None = None,
            progress_callback: Callable[[MapSearch], None] | None = None,
            report_every: float = 30.0, archive_path: str | Path | None = None,
            archive_every: float = 15.0) -> Demo | None:
        """`archive_path`, when given, is rewritten every `archive_every` seconds so a search killed
        mid-way (the native engine can hang inside a step) can be resumed with `load_archive`; its
        modification time doubles as a liveness heartbeat for the supervising process."""
        started = last_report = last_archive = time.monotonic()
        while time.monotonic() - started < time_budget:
            if iteration_budget is not None and self.iterations >= iteration_budget:
                break
            if self.iterate():
                assert self.demo is not None
                self.demo.search_seconds = time.monotonic() - started
                self.demo.iterations = self.iterations
                return self.demo
            now = time.monotonic()
            if progress_callback and now - last_report >= report_every:
                last_report = now
                progress_callback(self)
            if archive_path is not None and now - last_archive >= archive_every:
                last_archive = now
                self.save_archive(archive_path)
        if archive_path is not None:
            self.save_archive(archive_path)
        return None

    def save_archive(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "gravity-lab-search-archive-v1", "iterations": self.iterations, "steps": self.steps,
                   "cells": [[list(key), cell.actions, cell.progress, cell.chosen, cell.seen]
                             for key, cell in self.archive.items()]}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(destination)

    def load_archive(self, path: str | Path) -> int:
        """Merge a saved archive into this one (shorter prefix wins per cell). Returns cells loaded."""
        data = json.loads(Path(path).read_text())
        if data.get("format") != "gravity-lab-search-archive-v1":
            raise ValueError(f"unsupported search archive: {path}")
        for key, actions, progress, chosen, seen in data["cells"]:
            if len(actions) > self.max_prefix:
                continue
            cell = self.archive.get(tuple(key))
            if cell is None or len(actions) < len(cell.actions):
                self.archive[tuple(key)] = Cell(list(actions), float(progress), int(chosen), int(seen))
            self.best_progress = max(self.best_progress, float(progress))
        self.iterations += int(data.get("iterations", 0))
        self.steps += int(data.get("steps", 0))
        return len(data["cells"])


def verify_demo(env: Any, demo: Demo) -> bool:
    """Replay the demo from its seed and confirm it finishes in the same number of steps."""
    env.reset(demo.seed)
    for index, action in enumerate(demo.actions):
        step = env.step(action)
        if step.finished:
            return index == len(demo.actions) - 1
        if step.terminated or step.truncated:
            return False
    return False

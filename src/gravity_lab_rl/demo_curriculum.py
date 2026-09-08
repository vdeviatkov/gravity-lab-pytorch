"""Backward start curriculum and demonstration replay built from searched demos.

One finishing action sequence per map (`explore.py`) turns the two problems that stalled every
shared-network run -- discovering a hard maneuver, and keeping 30 solutions in one network --
into supervised-flavored ones:

- **Backward start curriculum** (Salimans & Chen 2018): an episode on a map with a demo starts by
  replaying the demo's first `prefix` actions (exact state reconstruction, as in `practice.py`)
  and hands control to the policy from there. The takeover point begins a few seconds before the
  finish and moves toward the start each time the policy succeeds from the current point, so the
  policy only ever has to learn a short extension of what it already does. A share of episodes
  always starts from the real start, and evaluation is full-start only.
- **Demonstration replay**: every demo transition (with the same reward and n-step treatment as
  online data) lives permanently in its own replay buffer; the trainer mixes it into critic
  batches and adds a behavior-cloning cross-entropy term on the actor (DQfD-style), so each map
  has an anchor that other maps' gradient updates cannot erase.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import DEFAULT_OBSTACLE_RAY_COUNT, TRACKS_PER_LEVEL_GROUP
from .explore import Demo, load_demos
from .reward import EpisodeReward, RewardConfig

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "directory": "demos",
    "initial_remaining": 50,       # first takeover point: this many demo steps before the finish
    "step_back": 50,               # how far the takeover point moves toward the start per advance
    "advance_window": 4,           # advance when >= advance_successes of the last advance_window succeed
    "advance_successes": 3,
    "retreat_window": 6,           # retreat when the last retreat_window episodes all failed
    "full_start_probability": 0.2, # share of episodes that ignore the demo and start at the real start
    "bc_weight": 1.0,              # behavior-cloning cross-entropy weight on the actor
    "bc_batch_size": 64,           # demo transitions mixed into every optimizer step
    "sticky_action_probability": 0.5,  # rollout: repeat the previous action instead of resampling
    # Share of demo-start episodes run greedily (argmax, no sticky repeats). Only those decide
    # advances/retreats, so the curriculum is gated on the same deterministic policy that formal
    # evaluation measures; the rest keep exploring. 0 means every demo-start episode counts.
    "greedy_probability": 0.0,
}


def track_id_of(env_cfg: dict[str, Any]) -> int:
    return int(env_cfg["level_group"]) * TRACKS_PER_LEVEL_GROUP + int(env_cfg["track"])


@dataclass
class EpisodeStart:
    observation: tuple[float, ...]
    seed: int
    actions: list[int]
    peak_progress: float
    greedy: bool = False


class DemoCurriculum:
    def __init__(self, config: dict[str, Any], environments: list[dict[str, Any]], seed: int) -> None:
        self.config = {**DEFAULTS, **config}
        self.enabled = bool(self.config["enabled"])
        self.rng = random.Random(seed)
        self.demos: dict[int, Demo] = load_demos(self.config["directory"], environments) if self.enabled else {}
        self.prefix: dict[int, int] = {}
        self.history: dict[int, list[bool]] = {}
        self.advances = 0
        self.retreats = 0
        for track_id, demo in self.demos.items():
            self.prefix[track_id] = self.initial_prefix(demo)
            self.history[track_id] = []

    def initial_prefix(self, demo: Demo) -> int:
        return max(0, len(demo.actions) - int(self.config["initial_remaining"]))

    def graduated(self, track_id: int) -> bool:
        return track_id in self.demos and self.prefix.get(track_id, 0) == 0

    def start(self, env: Any, env_cfg: dict[str, Any], track_id: int, seed: int) -> EpisodeStart:
        demo = self.demos.get(track_id)
        prefix = self.prefix.get(track_id, 0)
        if demo is None or prefix == 0 or self.rng.random() < float(self.config["full_start_probability"]):
            observation = tuple(env.reset(seed))
            return EpisodeStart(observation, seed, [], observation[0])
        observation = tuple(env.reset(demo.seed))
        peak = observation[0]
        for action in demo.actions[:prefix]:
            step = env.step(action)
            if step.terminated or step.truncated:
                raise RuntimeError(f"demo prefix for track {track_id} ended early; environment is not reproducible")
            observation = tuple(step.observation)
            peak = max(peak, observation[0])
        if hasattr(env, "mark_practice_prefix"):
            env.mark_practice_prefix(prefix)
        greedy = self.rng.random() < float(self.config["greedy_probability"])
        return EpisodeStart(observation, demo.seed, list(demo.actions[:prefix]), peak, greedy)

    def record(self, track_id: int, prefix_steps: int, finished: bool, greedy: bool = True) -> None:
        """Outcome of an episode that started `prefix_steps` into the demo (0 = full start)."""
        if prefix_steps == 0 or track_id not in self.demos or prefix_steps != self.prefix.get(track_id):
            return  # full-start episodes and stale pointers do not move the takeover point
        if float(self.config["greedy_probability"]) > 0.0 and not greedy:
            return  # exploratory episodes do not decide the curriculum
        history = self.history[track_id]
        history.append(bool(finished))
        advance_window, retreat_window = int(self.config["advance_window"]), int(self.config["retreat_window"])
        recent = history[-advance_window:]
        if sum(recent) >= int(self.config["advance_successes"]):
            self.prefix[track_id] = max(0, prefix_steps - int(self.config["step_back"]))
            self.history[track_id] = []
            self.advances += 1
            return
        recent = history[-retreat_window:]
        initial = self.initial_prefix(self.demos[track_id])
        if len(recent) >= retreat_window and not any(recent) and prefix_steps < initial:
            self.prefix[track_id] = min(initial, prefix_steps + int(self.config["step_back"]))
            self.history[track_id] = []
            self.retreats += 1

    def summary(self) -> dict[str, Any]:
        return {
            "prefix": {str(track): steps for track, steps in sorted(self.prefix.items())},
            "demo_length": {str(track): len(demo.actions) for track, demo in sorted(self.demos.items())},
            "graduated": sorted(track for track in self.demos if self.graduated(track)),
            "advances": self.advances, "retreats": self.retreats,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"prefix": dict(self.prefix), "history": {k: list(v) for k, v in self.history.items()},
                "rng": self.rng.getstate(), "advances": self.advances, "retreats": self.retreats}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for track, steps in state["prefix"].items():
            if int(track) in self.demos:
                self.prefix[int(track)] = int(steps)
        for track, history in state["history"].items():
            if int(track) in self.demos:
                self.history[int(track)] = list(history)
        self.rng.setstate(state["rng"])
        self.advances, self.retreats = int(state.get("advances", 0)), int(state.get("retreats", 0))


def demo_transitions(demos: dict[int, Demo], environments: list[dict[str, Any]], reward_config: RewardConfig,
                     n_step: int, gamma: float, observation_size: int, open_environment: Any,
                     ) -> list[tuple[np.ndarray, int, float, np.ndarray, bool, bool, int, int]]:
    """Replay every demo once and return its n-step transitions plus raw observations.

    `open_environment(env_cfg)` must return a context-managed environment; only one may be open
    per process, so demos are replayed sequentially.
    """
    from .trainer import NStepAccumulator

    result = []
    for env_cfg in environments:
        track_id = track_id_of(env_cfg)
        demo = demos.get(track_id)
        if demo is None:
            continue
        accumulator = NStepAccumulator(n_step, gamma)
        with open_environment(env_cfg) as env:
            observation = tuple(env.reset(demo.seed))
            tracker = EpisodeReward(reward_config, observation[0])
            for index, action in enumerate(demo.actions):
                step = env.step(action)
                reward, _ = tracker.step(step.observation[0], step.finished, step.crashed)
                for ready in accumulator.push(np.asarray(observation[:observation_size], dtype=np.float32), action,
                                              reward, np.asarray(step.observation[:observation_size], dtype=np.float32),
                                              step.terminated, step.truncated):
                    result.append((*ready, track_id))
                observation = tuple(step.observation)
                if step.terminated or step.truncated:
                    if not step.finished or index != len(demo.actions) - 1:
                        raise RuntimeError(f"demo for track {track_id} did not finish on replay")
                    break
    return result


def compute_normalization(observations: np.ndarray, track_start: int, track_end: int,
                          max_scale: float = 10.0) -> tuple[list[float], list[float]]:
    """Per-feature standardization from sampled observations, as fixed `input_scale`/`input_bias`
    vectors: constant features and the track one-hot region are left untouched (scale 1, bias 0)."""
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    scale = np.ones(observations.shape[1])
    bias = np.zeros(observations.shape[1])
    for index in range(observations.shape[1]):
        if track_start <= index < track_end or std[index] < 1e-6:
            continue
        scale[index] = min(max_scale, 1.0 / float(std[index]))
        bias[index] = -float(mean[index]) * scale[index]
    return [float(v) for v in scale], [float(v) for v in bias]


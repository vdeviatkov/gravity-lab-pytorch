"""Reward shaping for gravity-lab-classic-v1.

Reward design is a training concern, not game logic, so it lives here rather than in the
`gravity-lab` engine submodule: the environment exposes only raw game state (progress at
observation index 0, and the `finished`/`crashed`/`truncated` flags on each step), and this module
turns that into the scalar signal every trainer (DQN, PPO, SAC+REDQ) and `evaluation.py` optimizes
against. All shaping constants are config-driven (the `reward` block of a training config) so
tuning them is a config edit, not a C++ change + two-tree rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Two explicit bonuses plus one penalty, deliberately simple and literal rather than a
    derived shaping formula (see docs/training-runs.md for the earlier potential-based-shaping
    design this replaced): a big bonus for finishing the track, a bonus for each percent of new
    forward progress made this step (past the episode's peak so far -- see `peak_progress` in
    `step_reward`), a bonus for how *fast* that progress arrived, and a small penalty on any step
    that makes no new forward progress (standing still, moving backward, or re-covering
    already-visited ground). Without that last penalty, freezing in place costs nothing while
    attempting an obstacle risks a crash penalty, which made "never move" look like the safe choice
    -- the exact failure mode this session repeatedly had to fix (see "reward tuning experiments"
    and "Redesign: v2" in docs/training-runs.md).
    """

    finish_bonus: float = 10.0  # the big, one-time bonus for reaching the finish line
    crash_penalty: float = 5.0
    # For idling-forever to discount worse than one crash: idle_penalty / (1 - gamma) >
    # crash_penalty, i.e. idle_penalty > crash_penalty * (1 - gamma) = 5.0 * 0.01 = 0.05
    # (gamma=0.99 in every shipped config). Default keeps a 2x margin.
    idle_penalty: float = 0.1
    progress_percent_bonus: float = 0.1  # per 1% of new peak progress advanced this step
    # The per-percent bonus scales up with how far along the track progress already is (1x at the
    # very start, 2x right at the finish), so pushing further into a track -- past whatever hard
    # obstacle sits later on -- earns more than repeating the same easy early percent over and
    # over. Without this, a policy that dies early and restarts has no reason to prefer reaching
    # further: the first 10% of a hard track and the last 10% are worth exactly the same, so cheap
    # early progress can dominate the training signal even though the late obstacle is what's
    # actually unsolved.
    progress_ramp_factor: float = 1.0
    # Rewards covering the same distance in fewer steps, on top of the linear per-percent bonus
    # above. Each environment step spans a fixed slice of game time (`frame_skip` physics ticks),
    # so percent_moved this step *is* an instantaneous speed measurement, not just a distance one
    # -- covering a given span of track in fewer, bigger-percent-moved steps earns strictly more
    # total reward than spreading the same total distance over more, smaller-percent-moved steps,
    # since this term is quadratic (not linear) in percent_moved. Calibrated against a measured
    # full-throttle run: typical positive percent_moved is ~0.36-0.46 per step, at which this term
    # is roughly equal to the linear per-percent bonus (0.1 * 0.4 = 0.04 vs 0.25 * 0.4^2 = 0.04) --
    # doubling reward for genuinely fast movement while barely touching slow/cautious steps (at
    # percent_moved=0.05, this term is 1/8th of the linear one). Applied only to positive steps,
    # so it doesn't interact with the idle-vs-crash discounted-value math above.
    speed_bonus_scale: float = 0.25

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RewardConfig:
        overrides = config.get("reward", {})
        return cls(**{field: overrides[field] for field in cls.__dataclass_fields__ if field in overrides})


def validate_reward_config(config: dict[str, Any]) -> None:
    reward = config.get("reward", {})
    if not isinstance(reward, dict):
        raise ValueError("reward must be an object")
    known = set(RewardConfig.__dataclass_fields__)
    unknown = set(reward) - known
    if unknown:
        raise ValueError(f"unknown reward fields: {sorted(unknown)}")
    for field, value in reward.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"reward.{field} must be a number")
    if float(reward.get("idle_penalty", RewardConfig.idle_penalty)) <= 0.0:
        raise ValueError("reward.idle_penalty must be positive")
    if float(reward.get("crash_penalty", RewardConfig.crash_penalty)) <= 0.0:
        raise ValueError("reward.crash_penalty must be positive")


def step_reward(reward_config: RewardConfig, peak_progress: float, current_progress: float,
                finished: bool, crashed: bool) -> tuple[float, float]:
    """Reward for one step, plus the episode's updated peak progress.

    Only progress past `peak_progress` (the furthest point reached so far this episode, not merely
    past the previous step) earns the bonuses below -- retreating and re-covering already-visited
    ground earns nothing beyond the flat idle penalty. This closes a retreat-then-surge
    reward-hacking path a plain previous-step-relative version would allow: since the speed bonus
    is quadratic in the distance covered, a policy could otherwise retreat cheaply (a flat penalty
    regardless of how far it backs up) and then surge forward over the same ground to farm a
    disproportionate one-step bonus for covering ground it had already covered before.
    """
    new_peak = max(peak_progress, current_progress)
    percent_moved = (new_peak - peak_progress) * 100.0
    if percent_moved > 0.0:
        multiplier = 1.0 + reward_config.progress_ramp_factor * current_progress
        reward = (reward_config.progress_percent_bonus * percent_moved * multiplier
                 + reward_config.speed_bonus_scale * percent_moved * percent_moved)
    else:
        reward = -reward_config.idle_penalty
    if finished:
        reward += reward_config.finish_bonus
    if crashed:
        reward -= reward_config.crash_penalty
    return reward, new_peak

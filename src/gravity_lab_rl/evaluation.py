from __future__ import annotations

import statistics
from typing import Any

import torch

from .config import curriculum_environments
from .model import DenseQNetwork


def evaluate_model(model: DenseQNetwork, config: dict[str, Any], episodes: int | None = None,
                   seed: int | None = None, device: torch.device | str = "cpu") -> dict[str, Any]:
    from gravity_lab import ClassicConfig, ClassicGravityEnv

    count = int(episodes or config["experiment"]["evaluation_episodes"])
    first_seed = int(seed if seed is not None else config["seeds"]["final_evaluation"])
    rows: list[dict[str, Any]] = []
    model.eval()
    # The environment always returns OBSERVATION_SIZE values; truncate to this model's actual
    # input width via the shared compatible prefix (see docs/policy-comparison.md).
    observation_size = len(model.input_scale)
    environments = curriculum_environments(config)
    for environment_index, env_cfg in enumerate(environments):
        actual_seed = first_seed + environment_index * count
        classic_config = ClassicConfig(
            level_group=env_cfg["level_group"], track=env_cfg["track"], league=env_cfg["league"],
            frame_skip=env_cfg["frame_skip"], max_episode_steps=env_cfg["max_episode_steps"],
            seed=actual_seed,
        )
        # The engine allows one active environment per process, so evaluation opens these
        # sequentially and closes each before advancing to the next track.
        with ClassicGravityEnv(classic_config, env_cfg.get("level_pack")) as env:
            track_name = env.track_name
            for episode in range(count):
                episode_seed = actual_seed + episode
                observation = env.reset(episode_seed)[:observation_size]
                reward_total = 0.0
                last = None
                for length in range(1, env_cfg["max_episode_steps"] + 1):
                    with torch.inference_mode():
                        q_values = model(torch.tensor(observation, dtype=torch.float32,
                                                      device=device))
                        action = int(torch.argmax(q_values).item())
                    last = env.step(action)
                    reward_total += last.reward
                    observation = last.observation[:observation_size]
                    if last.terminated or last.truncated:
                        break
                assert last is not None
                rows.append({
                    "episode": episode, "seed": episode_seed, "reward": reward_total,
                    "length": length, "progress": float(last.observation[0]),
                    "finished": last.finished, "crashed": last.crashed,
                    "truncated": last.truncated, "level_group": env_cfg["level_group"],
                    "track": env_cfg["track"], "league": env_cfg["league"],
                    "track_name": track_name,
                })
    rewards = [row["reward"] for row in rows]
    return {
        "epsilon": 0.0, "episodes": rows, "episode_count": len(rows),
        "environment_count": len(environments), "episodes_per_environment": count,
        "mean_reward": statistics.fmean(rewards), "median_reward": statistics.median(rewards),
        "mean_progress": statistics.fmean(row["progress"] for row in rows),
        "finish_rate": statistics.fmean(float(row["finished"]) for row in rows),
        "crash_rate": statistics.fmean(float(row["crashed"]) for row in rows),
        "truncation_rate": statistics.fmean(float(row["truncated"]) for row in rows),
        "mean_episode_length": statistics.fmean(row["length"] for row in rows),
    }


def double_dqn_targets(rewards: torch.Tensor, next_observations: torch.Tensor,
                       terminated: torch.Tensor, online_network: DenseQNetwork,
                       target_network: DenseQNetwork, gamma: float) -> torch.Tensor:
    """Double-DQN target; truncation deliberately does not disable bootstrapping."""
    with torch.no_grad():
        next_actions = online_network(next_observations).argmax(dim=1, keepdim=True)
        next_values = target_network(next_observations).gather(1, next_actions).squeeze(1)
        return rewards + float(gamma) * (~terminated).to(rewards.dtype) * next_values

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import (
    ACTION_COUNT,
    BASE_OBSERVATION_SIZE,
    DEFAULT_OBSTACLE_RAY_COUNT,
    ENVIRONMENT_ID,
    MAX_OBSTACLE_RAY_COUNT,
    OBSERVATION_SIZE,
    OBSTACLE_REGION_END,
)


def valid_observation_size(size: int) -> bool:
    """True if `size` is a real region boundary: base-only, base+some ray count, or full width."""
    if size == BASE_OBSERVATION_SIZE or size == OBSERVATION_SIZE:
        return True
    return BASE_OBSERVATION_SIZE < size <= OBSTACLE_REGION_END


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "classic_intro.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else default_config_path()
    config = json.loads(source.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("format") != "gravity-lab-rl-config-v1":
        raise ValueError("unsupported configuration format")
    if config.get("environment_id") != ENVIRONMENT_ID:
        raise ValueError(f"environment_id must be {ENVIRONMENT_ID}")
    env = config["environment"]
    if not 0 <= int(env["level_group"]) <= 2 or not 0 <= int(env["league"]) <= 3:
        raise ValueError("invalid group or league")
    if int(env["track"]) < 0 or not 1 <= int(env["frame_skip"]) <= 100:
        raise ValueError("invalid track or frame_skip")
    if int(env["max_episode_steps"]) <= 0:
        raise ValueError("max_episode_steps must be positive")
    ray_count = int(env.get("obstacle_ray_count", DEFAULT_OBSTACLE_RAY_COUNT))
    if not 1 <= ray_count <= MAX_OBSTACLE_RAY_COUNT:
        raise ValueError(f"obstacle_ray_count must be in [1, {MAX_OBSTACLE_RAY_COUNT}]")
    algo = config["algorithm"]
    if list(algo["hidden_sizes"]) != [128, 128]:
        raise ValueError("v1 portable architecture requires hidden_sizes [128, 128]")
    if int(algo["batch_size"]) <= 0 or int(algo["replay_capacity"]) < int(algo["batch_size"]):
        raise ValueError("invalid replay or batch size")
    norm = config["normalization"]
    if len(norm["input_scale"]) != len(norm["input_bias"]) or not valid_observation_size(
        len(norm["input_scale"])
    ):
        raise ValueError(
            f"normalization must contain {BASE_OBSERVATION_SIZE} matching scale and bias values "
            f"(optionally +1..{MAX_OBSTACLE_RAY_COUNT} for obstacle rays), or {OBSERVATION_SIZE} "
            "for the full sensor+acceleration vector"
        )
    for name, seed in config["seeds"].items():
        if not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError(f"seed {name} must be a nonnegative integer below 2^63")
    curriculum = config.get("curriculum")
    if curriculum and curriculum.get("enabled", False):
        if int(curriculum.get("episodes_per_track", 0)) <= 0:
            raise ValueError("curriculum episodes_per_track must be positive")
        stages = curriculum.get("stages", [])
        if not stages:
            raise ValueError("enabled curriculum requires stages")
        for stage in stages:
            group, league = int(stage["level_group"]), int(stage["league"])
            tracks = stage.get("tracks", [])
            if not 0 <= group <= 2 or not 0 <= league <= 3 or not tracks:
                raise ValueError("invalid curriculum stage")
            if any(int(track) < 0 for track in tracks):
                raise ValueError("curriculum tracks must be nonnegative")
    threads = int(config["experiment"].get("torch_num_threads", 1))
    if threads <= 0:
        raise ValueError("torch_num_threads must be positive")


def model_input_size(config: dict[str, Any]) -> int:
    """Observation width this config's model consumes (BASE_OBSERVATION_SIZE or OBSERVATION_SIZE)."""
    return len(config["normalization"]["input_scale"])


def curriculum_environments(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the configured level/league curriculum into concrete environments."""
    base = config["environment"]
    curriculum = config.get("curriculum")
    if not curriculum or not curriculum.get("enabled", False):
        return [copy.deepcopy(base)]
    result: list[dict[str, Any]] = []
    for stage in curriculum["stages"]:
        for track in stage["tracks"]:
            environment = copy.deepcopy(base)
            environment.update({"level_group": int(stage["level_group"]),
                                "track": int(track), "league": int(stage["league"])})
            result.append(environment)
    return result


def curriculum_environment_index(config: dict[str, Any], completed_episodes: int) -> int:
    """Return the environment to use next, with complete cycles repeating indefinitely."""
    environments = curriculum_environments(config)
    curriculum = config.get("curriculum")
    episodes_per_track = (int(curriculum["episodes_per_track"])
                          if curriculum and curriculum.get("enabled", False) else 1)
    return (int(completed_episodes) // episodes_per_track) % len(environments)


def configured(config: dict[str, Any], *, duration_seconds: float | None = None,
               device: str | None = None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    if duration_seconds is not None:
        if duration_seconds <= 0:
            raise ValueError("duration must be positive")
        result["experiment"]["duration_seconds"] = float(duration_seconds)
    if device is not None:
        result["experiment"]["device"] = device
    validate_config(result)
    return result

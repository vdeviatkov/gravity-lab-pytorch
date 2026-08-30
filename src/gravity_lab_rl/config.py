from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import ACTION_COUNT, ENVIRONMENT_ID, OBSERVATION_SIZE


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
    algo = config["algorithm"]
    if list(algo["hidden_sizes"]) != [128, 128]:
        raise ValueError("v1 portable architecture requires hidden_sizes [128, 128]")
    if int(algo["batch_size"]) <= 0 or int(algo["replay_capacity"]) < int(algo["batch_size"]):
        raise ValueError("invalid replay or batch size")
    norm = config["normalization"]
    if len(norm["input_scale"]) != OBSERVATION_SIZE or len(norm["input_bias"]) != OBSERVATION_SIZE:
        raise ValueError("normalization must contain 28 scale and bias values")
    for name, seed in config["seeds"].items():
        if not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError(f"seed {name} must be a nonnegative integer below 2^63")


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

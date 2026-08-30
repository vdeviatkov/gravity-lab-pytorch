from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT = "gravity-lab-pytorch-checkpoint-v1"


def rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_global"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"format": FORMAT, **payload}, temporary)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("unsupported checkpoint format")
    return checkpoint


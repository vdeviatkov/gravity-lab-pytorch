from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import OBSERVATION_SIZE


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, observation_size: int = OBSERVATION_SIZE) -> None:
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, observation: object, action: int, reward: float, next_observation: object,
            terminated: bool, truncated: bool) -> None:
        i = self.position
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size > self.size:
            raise ValueError("not enough replay entries")
        return self.rng.choice(self.size, size=batch_size, replace=False)

    def sample(self, batch_size: int, device: torch.device | str) -> ReplayBatch:
        indices = self.sample_indices(batch_size)
        tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=device)
        return ReplayBatch(
            tensor(self.observations[indices]), tensor(self.actions[indices]),
            tensor(self.rewards[indices]), tensor(self.next_observations[indices]),
            tensor(self.terminated[indices], torch.bool), tensor(self.truncated[indices], torch.bool),
        )

    def state_dict(self) -> dict[str, Any]:
        n = self.size
        return {
            "capacity": self.capacity, "position": self.position, "size": n,
            "observations": self.observations[:n].copy(), "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(), "next_observations": self.next_observations[:n].copy(),
            "terminated": self.terminated[:n].copy(), "truncated": self.truncated[:n].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        n = int(state["size"])
        for name in ("observations", "actions", "rewards", "next_observations", "terminated", "truncated"):
            getattr(self, name)[:n] = state[name]
        self.size, self.position = n, int(state["position"])
        self.rng.bit_generator.state = state["rng_state"]


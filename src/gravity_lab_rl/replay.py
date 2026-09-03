from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import OBSERVATION_SIZE, TRACK_ID_SIZE


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    steps: torch.Tensor


class ReplayBuffer:
    """Ring-buffer replay with O(1) per-track-balanced sampling.

    A multi-task curriculum (many tracks sharing one buffer) naturally biases plain uniform
    sampling toward whatever tracks currently produce the longest episodes -- a track the policy
    already solves finishes in hundreds of steps, while one it can't solve crashes in a fraction of
    that, so the buffer (and therefore every training batch) ends up structurally dominated by
    transitions from already-easy tracks. `sample()` instead draws close to `batch_size / (number
    of tracks currently represented)` from each track's own bucket, so a struggling track gets as
    much training signal per batch as a mastered one regardless of episode-length imbalance. With
    a single track (`track_id` always 0, the default), this degenerates to plain uniform sampling.

    Membership in each track's bucket is maintained incrementally in O(1) per `add()` via
    swap-remove, not recomputed per sample -- an O(buffer size) scan per optimizer step would be
    far too slow at the transition rates this trains at.
    """

    def __init__(self, capacity: int, seed: int, observation_size: int = OBSERVATION_SIZE,
                 max_tracks: int = TRACK_ID_SIZE) -> None:
        self.capacity = int(capacity)
        self.max_tracks = int(max_tracks)
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        # Number of environment steps this transition's reward and next_observation actually span
        # (see NStepAccumulator): 1 for a plain transition, up to the configured n for a full
        # n-step return, and fewer than n only for the last few steps before an episode ends.
        # Needed because the Bellman target's bootstrap discount is gamma ** steps, not a fixed
        # gamma ** n, at those episode-ending edges.
        self.steps = np.empty(capacity, dtype=np.int64)
        # Per-slot track id (level_group * TRACKS_PER_LEVEL_GROUP + track), for balanced sampling.
        self.track_id = np.zeros(capacity, dtype=np.int64)
        # track_buckets[t, :track_bucket_len[t]] holds the buffer positions currently belonging to
        # track t; position_in_bucket[i] is where position i sits within its own track's bucket, so
        # both directions of the mapping support O(1) swap-remove.
        self.track_buckets = np.zeros((self.max_tracks, capacity), dtype=np.int64)
        self.track_bucket_len = np.zeros(self.max_tracks, dtype=np.int64)
        self.position_in_bucket = np.zeros(capacity, dtype=np.int64)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _bucket_remove(self, position: int) -> None:
        track = int(self.track_id[position])
        slot = int(self.position_in_bucket[position])
        last = int(self.track_bucket_len[track]) - 1
        moved = int(self.track_buckets[track, last])
        self.track_buckets[track, slot] = moved
        self.position_in_bucket[moved] = slot
        self.track_bucket_len[track] -= 1

    def _bucket_insert(self, position: int, track: int) -> None:
        slot = int(self.track_bucket_len[track])
        self.track_buckets[track, slot] = position
        self.position_in_bucket[position] = slot
        self.track_bucket_len[track] += 1
        self.track_id[position] = track

    def add(self, observation: object, action: int, reward: float, next_observation: object,
            terminated: bool, truncated: bool, steps: int = 1, track_id: int = 0) -> None:
        i = self.position
        if self.size == self.capacity:
            self._bucket_remove(i)
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_observation
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.steps[i] = steps
        self._bucket_insert(i, int(track_id))
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size > self.size:
            raise ValueError("not enough replay entries")
        active = np.nonzero(self.track_bucket_len > 0)[0]
        count = len(active)
        base, remainder = divmod(batch_size, count)
        quotas = np.full(count, base)
        if remainder:
            quotas[self.rng.choice(count, size=remainder, replace=False)] += 1
        chosen = []
        for track, quota in zip(active, quotas):
            if quota == 0:
                continue
            bucket_length = int(self.track_bucket_len[track])
            local = self.rng.choice(bucket_length, size=int(quota), replace=quota > bucket_length)
            chosen.append(self.track_buckets[track, local])
        return np.concatenate(chosen)

    def sample(self, batch_size: int, device: torch.device | str) -> ReplayBatch:
        indices = self.sample_indices(batch_size)
        tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=device)
        return ReplayBatch(
            tensor(self.observations[indices]), tensor(self.actions[indices]),
            tensor(self.rewards[indices]), tensor(self.next_observations[indices]),
            tensor(self.terminated[indices], torch.bool), tensor(self.truncated[indices], torch.bool),
            tensor(self.steps[indices], torch.int64),
        )

    def state_dict(self) -> dict[str, Any]:
        n = self.size
        return {
            "capacity": self.capacity, "position": self.position, "size": n,
            "observations": self.observations[:n].copy(), "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(), "next_observations": self.next_observations[:n].copy(),
            "terminated": self.terminated[:n].copy(), "truncated": self.truncated[:n].copy(),
            "steps": self.steps[:n].copy(), "track_id": self.track_id[:n].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        n = int(state["size"])
        for name in ("observations", "actions", "rewards", "next_observations", "terminated", "truncated"):
            getattr(self, name)[:n] = state[name]
        # Older checkpoints (pre n-step returns / pre track-balanced sampling) lack these fields;
        # those transitions were all single-step and untracked, so default to 1 and track 0.
        self.steps[:n] = state["steps"] if "steps" in state else 1
        track_id = state["track_id"] if "track_id" in state else np.zeros(n, dtype=np.int64)
        self.track_bucket_len[:] = 0
        for i in range(n):
            self._bucket_insert(i, int(track_id[i]))
        self.size, self.position = n, int(state["position"])
        self.rng.bit_generator.state = state["rng_state"]


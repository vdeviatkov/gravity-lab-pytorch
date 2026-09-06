"""Exact-state obstacle practice through deterministic action-prefix reconstruction.

No teleporting or approximate physics snapshots: reset with the original seed and
execute the original actions to reconstruct all integration and collision state.
Only suffix transitions are learned; evaluation always starts from the real start.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass
class PracticeStart:
    observation: tuple[float, ...]
    seed: int
    actions: list[int]
    peak_progress: float


class PracticeBank:
    def __init__(self, config: dict[str, Any], seed: int):
        self.config = dict(config)
        self.rng = random.Random(seed)
        self.entries: dict[str, dict[int, dict]] = {}
        self.restored_episodes = 0
        self.reconstructed_steps = 0

    @staticmethod
    def key(env: dict) -> str:
        return f'{env["level_group"]}:{env["track"]}:{env["league"]}'

    def observe(self, env_config: dict, seed: int, actions: list[int], step) -> None:
        if not self.config.get('enabled', False) or step.terminated or step.truncated:
            return
        stride = int(self.config.get('checkpoint_stride', 25))
        if len(actions) % stride or len(actions) > env_config['max_episode_steps'] - 100:
            return
        progress = float(step.observation[0])
        if not .03 <= progress <= .9:
            return
        bucket = int(progress * 20)
        entries = self.entries.setdefault(self.key(env_config), {})
        previous = entries.get(bucket)
        # Prefer shorter valid routes to each region, leaving more episode budget
        # for learning the obstacle beyond it. Keep at most 19 progress bins/map.
        if previous is None or len(actions) < len(previous['actions']):
            entries[bucket] = {'seed': seed, 'actions': list(actions),
                               'observation': tuple(step.observation)}

    def start(self, env, env_config: dict, seed: int) -> PracticeStart:
        entries = self.entries.get(self.key(env_config), {})
        if (not self.config.get('enabled', False) or not entries
                or self.rng.random() >= self.config.get('probability', .5)):
            observation = tuple(env.reset(seed))
            return PracticeStart(observation, seed, [], observation[0])
        # Half of practice targets the furthest reachable region; half rehearses
        # earlier regions so the policy can connect the learned pieces.
        keys = sorted(entries)
        bucket = keys[-1] if self.rng.random() < .5 else self.rng.choice(keys)
        saved = entries[bucket]
        observation = tuple(env.reset(saved['seed']))
        peak = observation[0]
        for action in saved['actions']:
            step = env.step(action)
            if step.terminated or step.truncated:
                raise RuntimeError('practice prefix ended early; environment is not reproducible')
            observation = tuple(step.observation)
            peak = max(peak, observation[0])
        if observation != tuple(saved['observation']):
            raise RuntimeError('practice state reconstruction mismatch')
        self.restored_episodes += 1
        self.reconstructed_steps += len(saved['actions'])
        return PracticeStart(observation, saved['seed'], list(saved['actions']), peak)

    def state_dict(self) -> dict:
        return {'entries': self.entries, 'rng': self.rng.getstate(),
                'restored_episodes': self.restored_episodes,
                'reconstructed_steps': self.reconstructed_steps}

    def load_state_dict(self, state: dict) -> None:
        self.entries = state['entries']
        self.rng.setstate(state['rng'])
        self.restored_episodes = state.get('restored_episodes', 0)
        self.reconstructed_steps = state.get('reconstructed_steps', 0)

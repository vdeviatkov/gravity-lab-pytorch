"""Compact, append-only records of actual training actions, including partial attempts."""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from .control import atomic_write_json


class RecordedEnvironment:
    """Proxy the training environment; never record separate evaluation rollouts.

    Each successful step writes one byte without userspace buffering. A killed
    process leaves a replayable prefix and an explicitly incomplete metadata file.
    Unique IDs prevent overwriting attempts when a checkpoint is resumed.
    """
    def __init__(self, env, run_dir: Path, configuration: dict, elapsed):
        self.env = env
        self.directory = run_dir / 'training_episodes'
        self.directory.mkdir(exist_ok=True)
        self.configuration = dict(configuration)
        self.elapsed = elapsed
        self.stream = None
        self.metadata = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed):
        self.finish('reset-interrupted')
        observation = self.env.reset(seed)
        identifier = f'{time.time_ns():020d}_{uuid.uuid4().hex[:8]}'
        self.path = self.directory / f'{identifier}.json'
        self.metadata = dict(format='gravity-lab-training-episode-v1', id=identifier,
                             environment=self.configuration, seed=int(seed),
                             started_seconds=self.elapsed(), status='incomplete',
                             practice_prefix_steps=0, actions_file=f'{identifier}.actions')
        atomic_write_json(self.path, self.metadata)
        self.stream = self.path.with_suffix('.actions').open('wb', buffering=0)
        return observation

    def mark_practice_prefix(self, steps):
        self.metadata['practice_prefix_steps'] = steps
        atomic_write_json(self.path, self.metadata)

    def step(self, action):
        if self.stream is None:
            raise RuntimeError('training recording requires reset before step')
        result = self.env.step(action)
        self.stream.write(bytes([int(action)]))
        self.last_result = result
        if result.terminated or result.truncated:
            self.finish('terminal')
        return result

    def finish(self, reason='interrupted'):
        if self.stream is None:
            return
        count = self.stream.tell()
        self.stream.close()
        self.stream = None
        self.metadata.update(status=reason, action_count=count, ended_seconds=self.elapsed())
        if count:
            step = self.last_result
            self.metadata.update(progress=float(step.observation[0]), finished=bool(step.finished),
                                 crashed=bool(step.crashed), truncated=bool(step.truncated),
                                 final_observation=list(step.observation))
        atomic_write_json(self.path, self.metadata)

    def close(self):
        self.finish()
        self.env.close()


def record_environment(env, run_dir, configuration, config, elapsed):
    if config['experiment'].get('record_training_episodes', True):
        return RecordedEnvironment(env, run_dir, configuration, elapsed)
    return env


def training_episodes(run_dir: Path):
    """Load all nonempty attempts, including prefixes surviving an unclean exit."""
    records = []
    for path in sorted((run_dir / 'training_episodes').glob('*.json')):
        record = json.loads(path.read_text())
        actions = path.parent / record['actions_file']
        if not actions.exists() and record.get('action_count', 0):
            raise ValueError(f'missing recorded actions: {path}')
        if actions.exists() and actions.stat().st_size:
            record['actions_path'] = actions
            count = actions.stat().st_size
            if record.get('action_count', count) != count:
                raise ValueError(f'action count mismatch: {path}')
            record['action_count'] = count
            records.append(record)
    return records


def begin_recording_session(run_dir, config, transitions, elapsed):
    """Expose gaps when recording is enabled only after an older checkpoint."""
    with (run_dir / 'recording_sessions.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(started_ns=time.time_ns(), transitions=transitions,
                                     active_seconds=elapsed,
                                     enabled=config['experiment'].get('record_training_episodes', True))) + '\n')

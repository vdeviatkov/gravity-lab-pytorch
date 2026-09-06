"""Actual-action fidelity, interrupted attempts, and complete 20-bike batching."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gravity_lab import ClassicConfig, ClassicGravityEnv
from gravity_lab_rl.recording import RecordedEnvironment, training_episodes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_map_overlay as overlay

ENV = dict(level_group=0, track=0, league=0, frame_skip=2,
           max_episode_steps=5, obstacle_ray_count=32, level_pack=None)


def open_env():
    return ClassicGravityEnv(ClassicConfig(**{k: v for k, v in ENV.items() if k != 'level_pack'}))


def test_actions_reproduce_exact_observations_and_preserve_partial_resets(tmp_path):
    env = RecordedEnvironment(open_env(), tmp_path, ENV, lambda: 12.0)
    env.reset(13)
    first = [env.step(a).observation for a in (1, 3, 2)]
    env.reset(14)
    second = [env.step(a).observation for a in (2, 0, 4, 1, 3)]
    env.close()
    records = training_episodes(tmp_path)
    assert [r['status'] for r in records] == ['reset-interrupted', 'terminal']
    for record, expected in zip(records, (first, second)):
        with open_env() as replay:
            replay.reset(record['seed'])
            actual = [replay.step(a).observation for a in record['actions_path'].read_bytes()]
        assert actual == expected
        assert list(actual[-1]) == record['final_observation']
    # Resuming creates new attempts even with repeated seeds and training time.
    env = RecordedEnvironment(open_env(), tmp_path, ENV, lambda: 12.0)
    env.reset(13)
    env.step(1)
    # Metadata intentionally still says incomplete: simulates process exit before close.
    record = training_episodes(tmp_path)[-1]
    assert record['status'] == 'incomplete'
    assert record['action_count'] == 1
    env.close()


@pytest.mark.skipif(not overlay.VIEWER.exists() or not shutil.which('ffmpeg'), reason='requires viewer and ffmpeg')
def test_twenty_bike_batches_include_every_attempt_and_native_terminal_frame(tmp_path):
    env = RecordedEnvironment(open_env(), tmp_path, ENV, lambda: 10.0)
    for attempt in range(21):
        env.reset(100 + attempt)
        for action in (1, 2, 3, 0, 1)[:2 + attempt % 4]:
            env.step(action)
    env.close()
    records = training_episodes(tmp_path)
    args = SimpleNamespace(seed=999, keep_frames=False, step_stride=1, speedup=1.0,
                           trail_length=0, batch_size=20)
    output = overlay.generate_recorded_video(tmp_path, ENV, records, args)
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['batch_count'] == 2
    assert metadata['attempt_count'] == 21
    assert [r['episode_id'] for r in metadata['attempts']] == [r['id'] for r in records]
    for source, rendered in zip(records, metadata['attempts']):
        assert rendered['frames'] == source['action_count'] + 1
        fields = dict(item.split('=') for item in rendered['outcome'].split())
        assert int(fields['steps']) == source['action_count']
        assert float(fields['progress']) == pytest.approx(source['progress'], abs=1e-6)
        assert int(fields['truncated']) == source['truncated']
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                            '-of', 'json', str(output)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)['streams'][0]
    # First batch: six frames + 25-frame hold; second: three frames + hold.
    assert int(stream['nb_read_frames']) == 59


@pytest.mark.parametrize('filename', ['classic_intro.json', 'classic_intro_ppo.json', 'classic_intro_sac.json'])
def test_each_trainer_records_all_transitions_and_calls_finalizer(tmp_path, filename):
    from unittest.mock import patch
    from gravity_lab_rl.cli import _trainer_class
    from gravity_lab_rl.config import load_config
    config = load_config(ROOT / 'configs' / filename)
    config['environment'].update(max_episode_steps=5)
    config['curriculum'] = {'enabled': False}
    config['experiment'].update(duration_seconds=.25, evaluation_episodes=1,
                                best_checkpoint_eval_interval_seconds=.05,
                                training_plot_after_training=False, map_overlay_after_training=False)
    config['algorithm']['hidden_sizes'] = [16, 16]
    if config['algorithm'].get('kind') == 'ppo':
        config['algorithm'].update(rollout_length=8, minibatch_size=8, ppo_epochs=1)
    else:
        config['algorithm'].update(replay_capacity=1000, replay_warmup=500, batch_size=8)
    with patch('gravity_lab_rl.video.generate_training_videos') as finalize:
        summary = _trainer_class(config)(config, tmp_path / filename).run()
    records = training_episodes(tmp_path / filename)
    assert records
    assert sum(r['action_count'] for r in records) == summary['transition_count']
    assert sum(r['status'] == 'terminal' for r in records) == summary['completed_episode_count']
    finalize.assert_called_once()

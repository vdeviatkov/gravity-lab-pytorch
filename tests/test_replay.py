import numpy as np

from gravity_lab_rl.replay import ReplayBuffer


def add_rows(buffer, count=8):
    for i in range(count):
        observation = np.full(72, i, dtype=np.float32)
        buffer.add(observation, i % 9, float(i), observation + 1, i % 3 == 0, i % 4 == 0)


def test_replay_insertion_wrap_and_sampling():
    replay = ReplayBuffer(5, 17)
    add_rows(replay, 7)
    assert len(replay) == 5
    assert replay.position == 2
    batch = replay.sample(3, "cpu")
    assert batch.observations.shape == (3, 72)
    assert batch.actions.shape == (3,)


def test_replay_sampling_uses_independent_named_seed():
    a, b, c = ReplayBuffer(20, 9), ReplayBuffer(20, 9), ReplayBuffer(20, 10)
    for replay in (a, b, c):
        add_rows(replay, 20)
    assert np.array_equal(a.sample_indices(8), b.sample_indices(8))
    assert not np.array_equal(a.sample_indices(8), c.sample_indices(8))


def test_replay_state_restores_position_contents_and_rng():
    source = ReplayBuffer(20, 4)
    add_rows(source, 12)
    source.sample_indices(3)
    state = source.state_dict()
    expected = source.sample_indices(5)
    restored = ReplayBuffer(20, 999)
    restored.load_state_dict(state)
    assert restored.position == source.position and len(restored) == len(source)
    assert np.array_equal(restored.observations[:12], source.observations[:12])
    assert np.array_equal(restored.sample_indices(5), expected)


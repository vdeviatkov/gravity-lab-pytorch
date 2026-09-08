import numpy as np
from gravity_lab import ClassicConfig, ClassicGravityEnv

from gravity_lab_rl import ACCELERATION_REGION_END, TRACK_ID_REGION_END
from gravity_lab_rl.demo_curriculum import DemoCurriculum, compute_normalization, demo_transitions
from gravity_lab_rl.explore import Demo, MapSearch, SearchConfig, demo_filename, verify_demo
from gravity_lab_rl.reward import RewardConfig

ENV = dict(level_group=0, track=0, league=0, frame_skip=2, max_episode_steps=2000, obstacle_ray_count=32)


def _search_intro(tmp_path):
    with ClassicGravityEnv(ClassicConfig(**ENV, seed=7)) as env:
        demo = MapSearch(env, ENV, 7, SearchConfig(), rng_seed=0).run(120.0)
        assert demo is not None and verify_demo(env, demo)
    demo.save(tmp_path / demo_filename(0, 0))
    return demo


def test_search_finds_and_verifies_intro_and_demo_round_trips(tmp_path):
    demo = _search_intro(tmp_path)
    loaded = Demo.load(tmp_path / demo_filename(0, 0))
    assert loaded == demo and loaded.track_id == 0 and loaded.actions[-1] == demo.actions[-1]


def test_backward_curriculum_reconstructs_state_and_moves_pointer(tmp_path):
    demo = _search_intro(tmp_path)
    cfg = {"enabled": True, "directory": str(tmp_path), "initial_remaining": 30, "step_back": 30,
           "full_start_probability": 0.0}
    curriculum = DemoCurriculum(cfg, [ENV], 1)
    assert curriculum.prefix[0] == len(demo.actions) - 30
    with ClassicGravityEnv(ClassicConfig(**ENV, seed=7)) as env:
        start = curriculum.start(env, ENV, 0, 99)
        assert len(start.actions) == len(demo.actions) - 30
        # The remaining demo suffix must finish from the reconstructed state.
        finished = False
        for action in demo.actions[len(start.actions):]:
            step = env.step(action)
            finished = step.finished
            if step.terminated or step.truncated:
                break
        assert finished
    for _ in range(3):
        curriculum.record(0, curriculum.prefix[0], True)
    assert curriculum.prefix[0] == len(demo.actions) - 60 and curriculum.advances == 1
    pointer = curriculum.prefix[0]
    for _ in range(6):
        curriculum.record(0, pointer, False)
    assert curriculum.prefix[0] == len(demo.actions) - 30 and curriculum.retreats == 1
    curriculum.record(0, 12345, True)  # stale pointer is ignored
    restored = DemoCurriculum(cfg, [ENV], 2)
    restored.load_state_dict(curriculum.state_dict())
    assert restored.prefix == curriculum.prefix and restored.summary() == curriculum.summary()


def test_demo_transitions_and_normalization(tmp_path):
    demo = _search_intro(tmp_path)
    curriculum = DemoCurriculum({"enabled": True, "directory": str(tmp_path)}, [ENV], 1)

    def open_env(env_cfg):
        return ClassicGravityEnv(ClassicConfig(**{k: env_cfg[k] for k in ENV}, seed=7))

    transitions = demo_transitions(curriculum.demos, [ENV], RewardConfig(), 3, 0.99, TRACK_ID_REGION_END, open_env)
    assert len(transitions) == len(demo.actions)
    assert transitions[-1][4] and transitions[-1][6] == 1 and transitions[-1][7] == 0
    observations = np.stack([t[0] for t in transitions])
    scale, bias = compute_normalization(observations, ACCELERATION_REGION_END, TRACK_ID_REGION_END)
    assert scale[ACCELERATION_REGION_END] == 1.0 and bias[ACCELERATION_REGION_END] == 0.0
    normalized = observations * np.asarray(scale) + np.asarray(bias)
    assert abs(float(normalized[:, 0].mean())) < 1e-4 and abs(float(normalized[:, 0].std()) - 1.0) < 1e-3


def test_only_greedy_episodes_decide_the_curriculum(tmp_path):
    demo = _search_intro(tmp_path)
    cfg = {"enabled": True, "directory": str(tmp_path), "initial_remaining": 30, "step_back": 30,
           "full_start_probability": 0.0, "greedy_probability": 0.5}
    curriculum = DemoCurriculum(cfg, [ENV], 1)
    pointer = curriculum.prefix[0]
    for _ in range(5):
        curriculum.record(0, pointer, True, greedy=False)
    assert curriculum.prefix[0] == pointer and curriculum.advances == 0
    for _ in range(3):
        curriculum.record(0, pointer, True, greedy=True)
    assert curriculum.prefix[0] == pointer - 30 and curriculum.advances == 1
    with ClassicGravityEnv(ClassicConfig(**ENV, seed=7)) as env:
        flags = {curriculum.start(env, ENV, 0, 5).greedy for _ in range(30)}
    assert flags == {True, False}

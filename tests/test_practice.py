from collections import Counter
import random

from gravity_lab import ClassicConfig, ClassicGravityEnv
from gravity_lab_rl.practice import PracticeBank
from gravity_lab_rl.reward import EpisodeReward, RewardConfig
from gravity_lab_rl.sac_trainer import SACREDQTrainer


def test_reconstructed_state_and_future_physics_are_exact_after_other_map():
    cfg = dict(level_group=0, track=0, league=0, frame_skip=2, max_episode_steps=500)
    bank = PracticeBank({'enabled': True, 'probability': 1, 'checkpoint_stride': 25}, 23)
    with ClassicGravityEnv(ClassicConfig(**cfg)) as env:
        env.reset(7)
        actions = []
        for _ in range(125):
            actions.append(1)
            step = env.step(1)
            if step.terminated or step.truncated:
                break
            bank.observe(cfg, 7, actions, step)
        assert bank.entries
        start = bank.start(env, cfg, 99)
        assert start.actions
        continuation = [0, 5, 1, 6, 2]
        expected = [env.step(a) for a in continuation]
    # Exercise the engine's process-global loader/physics state on a different map.
    with ClassicGravityEnv(ClassicConfig(level_group=2, track=4)) as other:
        other.reset(13)
        for _ in range(20):
            other.step(1)
    with ClassicGravityEnv(ClassicConfig(**cfg)) as env:
        env.reset(start.seed)
        for action in start.actions:
            step = env.step(action)
        assert tuple(step.observation) == start.observation
        assert [env.step(a) for a in continuation] == expected
    restored = PracticeBank(bank.config, 100)
    restored.load_state_dict(bank.state_dict())
    assert restored.state_dict() == bank.state_dict()


def test_setup_grace_expires_and_resets_only_on_new_peak():
    config = RewardConfig(idle_grace_steps=2, speed_bonus_scale=0)
    tracker = EpisodeReward(config, .5)
    assert tracker.step(.49, False, False)[0] == 0
    assert tracker.step(.5, False, False)[0] == 0
    assert tracker.step(.5, False, False)[0] == -config.idle_penalty
    assert tracker.step(.51, False, False)[0] > 0
    assert tracker.step(.50, False, True)[0] == -config.crash_penalty


def test_coverage_turns_reach_every_map_despite_failed_track_bias():
    trainer = SACREDQTrainer.__new__(SACREDQTrainer)
    trainer.config = {'curriculum': {'guaranteed_coverage': True}}
    trainer.track_full_start_counts = {}
    trainer.track_success_ema = {i: .99 if i else 0 for i in range(30)}
    trainer.curriculum_rng = random.Random(17)
    envs = [dict(level_group=g, track=t, league=g) for g in range(3) for t in range(10)]
    visits = Counter()
    for episode in range(60):
        trainer.completed_episode_count = episode
        env = trainer._select_next_environment(envs)
        track = trainer._track_id(env)
        visits[track] += 1
        trainer.track_full_start_counts[track] = visits[track]
    assert len(visits) == 30

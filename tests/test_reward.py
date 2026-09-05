import pytest

from gravity_lab_rl.config import validate_reward_config
from gravity_lab_rl.reward import RewardConfig, step_reward


def test_idle_step_earns_flat_penalty():
    reward, peak = step_reward(RewardConfig(), peak_progress=0.5, current_progress=0.5,
                               finished=False, crashed=False)
    assert reward == -RewardConfig().idle_penalty
    assert peak == 0.5


def test_retreat_earns_flat_penalty_not_proportional_to_distance():
    small_retreat, _ = step_reward(RewardConfig(), peak_progress=0.5, current_progress=0.49,
                                   finished=False, crashed=False)
    large_retreat, _ = step_reward(RewardConfig(), peak_progress=0.5, current_progress=0.1,
                                   finished=False, crashed=False)
    assert small_retreat == large_retreat == -RewardConfig().idle_penalty


def test_new_peak_progress_earns_positive_bonus_and_updates_peak():
    config = RewardConfig()
    reward, peak = step_reward(config, peak_progress=0.5, current_progress=0.51,
                               finished=False, crashed=False)
    percent_moved = 1.0
    expected_linear = config.progress_percent_bonus * percent_moved * (1.0 + config.progress_ramp_factor * 0.51)
    expected_speed = config.speed_bonus_scale * percent_moved * percent_moved
    assert reward == pytest.approx(expected_linear + expected_speed)
    assert peak == pytest.approx(0.51)


def test_recovering_to_below_peak_earns_nothing_beyond_idle_penalty():
    # Retreat from peak 0.5 down to 0.3, then partially recover to 0.4 (still short of the old
    # peak) -- this must earn the same flat idle penalty as a pure retreat, not a positive bonus,
    # since 0.4 is not new territory. Closes the retreat-then-surge reward-hacking path.
    reward, peak = step_reward(RewardConfig(), peak_progress=0.5, current_progress=0.4,
                               finished=False, crashed=False)
    assert reward == -RewardConfig().idle_penalty
    assert peak == 0.5


def test_surging_past_old_peak_only_rewards_the_new_slice():
    # From peak 0.5, jumping to 0.6 in one step should be rewarded as if only 0.5->0.6 (1 percentage
    # point of *new* progress) happened, not as if the whole 0->0.6 span were covered this step.
    config = RewardConfig()
    reward, peak = step_reward(config, peak_progress=0.5, current_progress=0.6,
                               finished=False, crashed=False)
    percent_moved = 10.0
    expected_linear = config.progress_percent_bonus * percent_moved * (1.0 + config.progress_ramp_factor * 0.6)
    expected_speed = config.speed_bonus_scale * percent_moved * percent_moved
    assert reward == pytest.approx(expected_linear + expected_speed)
    assert peak == pytest.approx(0.6)


def test_finish_bonus_and_crash_penalty_are_additive():
    config = RewardConfig()
    finish_reward, _ = step_reward(config, peak_progress=0.99, current_progress=1.0,
                                   finished=True, crashed=False)
    crash_reward, _ = step_reward(config, peak_progress=0.5, current_progress=0.5,
                                  finished=False, crashed=True)
    assert finish_reward > config.finish_bonus  # includes the small positive progress term too
    assert crash_reward == -config.idle_penalty - config.crash_penalty


def test_validate_reward_config_accepts_known_fields():
    validate_reward_config({"reward": {"finish_bonus": 20.0, "idle_penalty": 0.2}})


def test_validate_reward_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        validate_reward_config({"reward": {"bogus_field": 1.0}})


def test_validate_reward_config_rejects_non_positive_idle_penalty():
    with pytest.raises(ValueError):
        validate_reward_config({"reward": {"idle_penalty": 0.0}})


def test_reward_config_from_config_applies_overrides_and_defaults():
    config = RewardConfig.from_config({"reward": {"finish_bonus": 25.0}})
    assert config.finish_bonus == 25.0
    assert config.crash_penalty == RewardConfig().crash_penalty


def test_reward_config_from_config_defaults_when_absent():
    assert RewardConfig.from_config({}) == RewardConfig()

from pathlib import Path

from gravity_lab_rl.config import (curriculum_environment_index,
                                   curriculum_environments, load_config)


def test_all_tracks_curriculum_advances_group_and_bike_league_together():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "classic_all_tracks.json")
    environments = curriculum_environments(config)
    assert len(environments) == 30
    assert [(environment["level_group"], environment["league"])
            for environment in environments] == [(0, 0)] * 10 + [(1, 1)] * 10 + [(2, 2)] * 10
    assert [environment["track"] for environment in environments] == list(range(10)) * 3
    assert curriculum_environment_index(config, 0) == 0
    assert curriculum_environment_index(config, 4) == 0
    assert curriculum_environment_index(config, 5) == 1
    assert curriculum_environment_index(config, 50) == 10
    assert curriculum_environment_index(config, 100) == 20
    assert curriculum_environment_index(config, 150) == 0

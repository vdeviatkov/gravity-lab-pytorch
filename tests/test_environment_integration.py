import os
from pathlib import Path

import pytest


def test_classic_environment_smoke():
    game = Path(os.environ.get("GRAVITY_LAB_REPO", "/Users/vdeviatkov/Documents/game"))
    library = game / "build-classic-rl" / "libgravity_lab_classic.dylib"
    if not library.is_file():
        pytest.skip(f"classic native library unavailable: {library}")
    os.environ.setdefault("GRAVITY_LAB_CLASSIC_LIBRARY", str(library))
    try:
        from gravity_lab import ClassicConfig, ClassicGravityEnv
    except ImportError as error:
        pytest.skip(f"Gravity Lab Python bindings unavailable: {error}")
    with ClassicGravityEnv(ClassicConfig(level_group=0, track=0, league=0, frame_skip=2,
                                         max_episode_steps=10, seed=7)) as env:
        observation = env.reset(7)
        result = env.step(1)
        assert len(observation) == 28 and len(result.observation) == 28
        assert isinstance(result.reward, float)


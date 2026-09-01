"""Minimal Double DQN training for Gravity Lab Classic."""

__version__ = "0.1.0"
ENVIRONMENT_ID = "gravity-lab-classic-v1"
# The environment returns up to OBSERVATION_SIZE values in fixed regions: [0, BASE_OBSERVATION_SIZE)
# is the original bike/track state; [BASE_OBSERVATION_SIZE, BASE_OBSERVATION_SIZE + ray_count) is
# the obstacle-ray sensor (ray_count is per-environment, up to MAX_OBSTACLE_RAY_COUNT);
# [OBSTACLE_REGION_END, OBSERVATION_SIZE) is per-component acceleration, always present. A model
# may be built for any width in this layout (28, 28+ray_count, or the full OBSERVATION_SIZE) since
# every shorter width is an unchanged, compatible prefix of every longer one. See
# docs/policy-comparison.md for how to run each.
BASE_OBSERVATION_SIZE = 28
MAX_OBSTACLE_RAY_COUNT = 32
DEFAULT_OBSTACLE_RAY_COUNT = 8
OBSTACLE_REGION_END = BASE_OBSERVATION_SIZE + MAX_OBSTACLE_RAY_COUNT
ACCELERATION_SIZE = 12
OBSERVATION_SIZE = OBSTACLE_REGION_END + ACCELERATION_SIZE
ACTION_COUNT = 9


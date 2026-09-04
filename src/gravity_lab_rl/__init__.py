"""Minimal Double DQN training for Gravity Lab Classic."""

__version__ = "0.1.0"
ENVIRONMENT_ID = "gravity-lab-classic-v1"
# The environment returns up to OBSERVATION_SIZE values in fixed regions: [0, BASE_OBSERVATION_SIZE)
# is the original bike/track state; [BASE_OBSERVATION_SIZE, BASE_OBSERVATION_SIZE + ray_count) is
# the obstacle-ray sensor (ray_count is per-environment, up to MAX_OBSTACLE_RAY_COUNT);
# [OBSTACLE_REGION_END, ACCELERATION_REGION_END) is per-component acceleration, always present;
# [ACCELERATION_REGION_END, TRACK_ID_REGION_END) is a one-hot over (level_group, track), always
# present; [TRACK_ID_REGION_END, TRACK_ID_REGION_END + ray_count) is the head-clearance sensor --
# the same ray_count/angles as the obstacle sensor, cast from the rider's head instead of the bike
# center, reporting remaining clearance (distance minus the head's own collision radius) rather
# than raw distance. A model may be built for any width in this layout (28, 28+ray_count,
# ACCELERATION_REGION_END, TRACK_ID_REGION_END, TRACK_ID_REGION_END+ray_count, or the full
# OBSERVATION_SIZE) since every shorter width is an unchanged, compatible prefix of every longer
# one. See docs/training-runs.md ("SAC + REDQ: a new algorithm family") for how the head-clearance
# sensor was designed, and docs/policy-comparison.md for how to run each width.
BASE_OBSERVATION_SIZE = 28
MAX_OBSTACLE_RAY_COUNT = 32
DEFAULT_OBSTACLE_RAY_COUNT = 8
OBSTACLE_REGION_END = BASE_OBSERVATION_SIZE + MAX_OBSTACLE_RAY_COUNT
ACCELERATION_SIZE = 12
ACCELERATION_REGION_END = OBSTACLE_REGION_END + ACCELERATION_SIZE
LEVEL_GROUP_COUNT = 3
TRACKS_PER_LEVEL_GROUP = 10
TRACK_ID_SIZE = LEVEL_GROUP_COUNT * TRACKS_PER_LEVEL_GROUP
TRACK_ID_REGION_END = ACCELERATION_REGION_END + TRACK_ID_SIZE
# The head-clearance sensor always reuses obstacle_ray_count (see the layout comment above), so
# its reserved region is the same width (MAX_OBSTACLE_RAY_COUNT) as the obstacle sensor's.
MAX_HEAD_CLEARANCE_RAY_COUNT = MAX_OBSTACLE_RAY_COUNT
OBSERVATION_SIZE = TRACK_ID_REGION_END + MAX_HEAD_CLEARANCE_RAY_COUNT
ACTION_COUNT = 9


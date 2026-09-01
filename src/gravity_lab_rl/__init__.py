"""Minimal Double DQN training for Gravity Lab Classic."""

__version__ = "0.1.0"
ENVIRONMENT_ID = "gravity-lab-classic-v1"
# The environment always returns OBSERVATION_SIZE values; a model may instead be built for only
# the leading BASE_OBSERVATION_SIZE of them (the pre-obstacle-sensor layout), since indices
# [0, BASE_OBSERVATION_SIZE) are an unchanged, compatible prefix of the full vector. See
# docs/policy-comparison.md for how to run each.
BASE_OBSERVATION_SIZE = 28
OBSERVATION_SIZE = 36
ACTION_COUNT = 9


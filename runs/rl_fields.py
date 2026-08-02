"""Combat RL profile with four static refractive fields per environment.

This is a smoke/future-training profile, not an obstacle-avoidance objective.
Fields receive no universal proximity or interface shaping; their value emerges
from combat, navigation, handling, speed, and health outcomes.
"""

from dataclasses import replace

from boost_and_broadside.config import FieldMapConfig, ScaleConfig
from runs.rl import RL_TRAIN_CONFIG
from runs.shared import FIELD_REWARDS

_NUM_FIELDS = 4
_NUM_SHIPS = RL_TRAIN_CONFIG.scales[0].env_config.num_ships
_NUM_MINIBATCHES = RL_TRAIN_CONFIG.num_minibatches
_BASE_ENVS = RL_TRAIN_CONFIG.scales[0].num_envs
_NUM_ENVS = (
    (_BASE_ENVS * _NUM_SHIPS // (_NUM_SHIPS + _NUM_FIELDS)) // _NUM_MINIBATCHES * _NUM_MINIBATCHES
)

RL_FIELDS_TRAIN_CONFIG = replace(
    RL_TRAIN_CONFIG,
    rewards=FIELD_REWARDS,
    scales=(
        ScaleConfig(
            env_config=replace(
                RL_TRAIN_CONFIG.scales[0].env_config,
                num_fields=_NUM_FIELDS,
            ),
            num_envs=_NUM_ENVS,
        ),
    ),
    field_map=FieldMapConfig(
        cache_size=512,
        max_generation_attempts=256,
        nesting_probability=0.35,
    ),
)

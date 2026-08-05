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
    # Its own ladder: fields compress the skill scale badly (random sits at +170
    # here against -351 without them), so borrowing the zero-field ratings would
    # misdirect every proximity draw and every early rating update. Fitted by
    # `--mode semi_random --profile rl_fields`
    # (checkpoints/rl_fields/semi_random_tournament.json).
    reference_ladder=(
        (0.2, 257.7),
        (0.3, 347.0),
        (0.4, 475.9),
        (0.5, 554.2),
        (0.6, 647.7),
        (0.7, 737.3),
        (0.8, 841.3),
        (0.9, 897.9),
        (0.95, 960.5),
    ),
    random_elo=170.2,
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

"""The launch arithmetic's token count must match the one the environment builds.

Everything derived from the logical batch -- environment width, shard count, the
VRAM preset ceilings, the micro-batch bound -- is computed from a *prediction* of
how wide the observation's token axis will be. That prediction used to be written
out twice, and the two copies disagreed: the validator omitted Frontline's zone
and boundary tokens and so capped the micro-batch at two thirds of the minibatch
that actually existed.

These tests pin the prediction against an observation a real environment builds,
so a token kind added to the environment without a matching term in
``entity_token_count`` fails here rather than silently resizing the batch.
"""

import pytest
import torch

from boost_and_broadside.config import EnvConfig, FrontlineConfig, ShipConfig
from boost_and_broadside.config.core import entity_token_count
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.env.observation import observation_from_state


def _frontline() -> FrontlineConfig:
    return FrontlineConfig(
        zone_radius=330.0,
        zone_ring_radius=1200.0,
        playable_radius=2600.0,
        capture_seconds=20.0,
        defense_damage_per_second=2.0,
        respawn_health=25.0,
        spawn_heal_per_second=12.0,
        enemy_spawn_damage_per_second=8.0,
        boundary_damage_per_second=5.0,
        boundary_damage_per_pixel_second=0.05,
        front_win_threshold=5,
    )


def _built_token_width(env_config: EnvConfig, ship_config: ShipConfig) -> int:
    env = TensorEnv(2, ship_config, env_config, torch.device("cpu"))
    env.reset()
    observation = observation_from_state(env.state, ship_config)
    return observation.pos.shape[1]


@pytest.mark.parametrize("num_ships", [2, 8])
@pytest.mark.parametrize("num_fields", [0, 10])
def test_prediction_matches_a_frontline_observation(num_ships: int, num_fields: int) -> None:
    ship_config = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)
    env_config = EnvConfig(
        num_ships=num_ships,
        max_bullets=0,
        max_episode_steps=600,
        num_fields=num_fields,
        frontline=_frontline(),
    )
    assert env_config.entity_tokens == _built_token_width(env_config, ship_config)


@pytest.mark.parametrize("num_fields", [0, 4])
def test_prediction_matches_an_elimination_observation(num_fields: int) -> None:
    ship_config = ShipConfig()
    env_config = EnvConfig(
        num_ships=4,
        max_bullets=0,
        max_episode_steps=600,
        num_fields=num_fields,
    )
    assert env_config.entity_tokens == _built_token_width(env_config, ship_config)


def test_the_property_and_the_function_are_one_derivation() -> None:
    env_config = EnvConfig(
        num_ships=8,
        max_bullets=0,
        max_episode_steps=600,
        num_fields=10,
        frontline=_frontline(),
    )
    assert env_config.entity_tokens == entity_token_count(
        env_config.num_ships, env_config.num_fields, env_config.frontline
    )


def test_frontline_adds_the_zone_and_boundary_tokens() -> None:
    """Stated as a difference so the six is never silently absorbed."""
    shared = {"num_ships": 8, "max_bullets": 0, "max_episode_steps": 600, "num_fields": 10}
    plain = EnvConfig(**shared)
    fronted = EnvConfig(**shared, frontline=_frontline())
    assert fronted.entity_tokens - plain.entity_tokens == 6

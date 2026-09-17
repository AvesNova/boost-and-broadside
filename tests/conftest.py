"""Shared pytest fixtures for all test modules."""

import os

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig
from boost_and_broadside.env.state import TensorState

# Torch sizes its intra-op pool from the whole machine, which is right for one
# process and wrong for sixteen: under ``-n auto`` every worker claims every
# core and they spend the run descheduling each other. Measured on a 16-core
# laptop, the full suite is 443s sequential, 391s at ``-n 8`` with this pin, and
# 445s at ``-n 8`` without it -- parallelism that costs more than it returns.
# One thread per worker and one worker per core is 195s.
#
# Only under xdist. A sequential run is a single process and should have the
# machine.
if os.environ.get("PYTEST_XDIST_WORKER"):
    torch.set_num_threads(1)


@pytest.fixture
def ship_config() -> ShipConfig:
    """Default ship physics config (reference game values)."""
    return ShipConfig()


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=500)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(d_model=64, n_heads=4, n_yemong_blocks=2)


@pytest.fixture
def device() -> str:
    return "cpu"


def make_state(
    num_envs: int = 2,
    max_ships: int = 4,
    max_bullets: int = 5,
    device: str = "cpu",
    ship_config: ShipConfig | None = None,
    num_fields: int = 0,
) -> TensorState:
    """Build a TensorState with sane initial values for unit testing."""
    if ship_config is None:
        ship_config = ShipConfig()
    dev = torch.device(device)
    return TensorState(
        step_count=torch.zeros((num_envs,), dtype=torch.int32, device=dev),
        ship_pos=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_vel=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_attitude=torch.ones((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_ang_vel=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=dev),
        ship_shield_delay=torch.zeros((num_envs, max_ships), device=dev),
        ship_shield_recharge=torch.zeros((num_envs, max_ships), device=dev),
        ship_health=torch.full((num_envs, max_ships), ship_config.max_health, device=dev),
        ship_power=torch.full((num_envs, max_ships), ship_config.max_power, device=dev),
        ship_cooldown=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=dev),
        ship_team_id=torch.zeros((num_envs, max_ships), dtype=torch.int32, device=dev),
        ship_alive=torch.ones((num_envs, max_ships), dtype=torch.bool, device=dev),
        ship_is_shooting=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=dev),
        map_center=torch.zeros((num_envs,), dtype=torch.complex64, device=dev),
        playable_boundary_radius=torch.zeros((num_envs,), dtype=torch.float32, device=dev),
        front_position=torch.zeros((num_envs,), dtype=torch.long, device=dev),
        front_delta=torch.zeros((num_envs,), dtype=torch.int8, device=dev),
        front_win_threshold=torch.zeros((num_envs,), dtype=torch.long, device=dev),
        match_max_steps=torch.zeros((num_envs,), dtype=torch.long, device=dev),
        match_result=torch.full((num_envs,), -1, dtype=torch.int8, device=dev),
        zone_pos=torch.zeros((num_envs, 0), dtype=torch.complex64, device=dev),
        zone_radius=torch.zeros((num_envs, 0), dtype=torch.float32, device=dev),
        zone_roles=torch.zeros((num_envs, 0), dtype=torch.int8, device=dev),
        zone_capture_progress=torch.zeros((num_envs, 0), dtype=torch.float32, device=dev),
        zone_capture_direction=torch.zeros((num_envs, 0), dtype=torch.int8, device=dev),
        team0_captured=torch.zeros((num_envs,), dtype=torch.bool, device=dev),
        team1_captured=torch.zeros((num_envs,), dtype=torch.bool, device=dev),
        simultaneous_capture=torch.zeros((num_envs,), dtype=torch.bool, device=dev),
        prev_action=torch.zeros((num_envs, max_ships, 3), dtype=torch.float32, device=dev),
        bullet_pos=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.complex64, device=dev
        ),
        bullet_vel=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.complex64, device=dev
        ),
        bullet_time=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.float32, device=dev
        ),
        bullet_active=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.bool, device=dev),
        bullet_local_index=torch.ones(
            (num_envs, max_ships, max_bullets), dtype=torch.float32, device=dev
        ),
        bullet_field_gradient=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.complex64, device=dev
        ),
        bullet_cursor=torch.zeros((num_envs, max_ships), dtype=torch.long, device=dev),
        damage_matrix=torch.zeros(
            (num_envs, max_ships, max_ships), dtype=torch.float32, device=dev
        ),
        cumulative_damage_matrix=torch.zeros(
            (num_envs, max_ships, max_ships), dtype=torch.float32, device=dev
        ),
        field_pos=torch.zeros((num_envs, num_fields), dtype=torch.complex64, device=dev),
        field_radius=torch.zeros((num_envs, num_fields), dtype=torch.float32, device=dev),
        field_transition_width=torch.zeros((num_envs, num_fields), dtype=torch.float32, device=dev),
        field_index_level=torch.zeros((num_envs, num_fields), dtype=torch.int8, device=dev),
        field_index=torch.ones((num_envs, num_fields), dtype=torch.float32, device=dev),
        ship_local_index=torch.ones((num_envs, max_ships), dtype=torch.float32, device=dev),
        ship_field_gradient=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_combat_damage=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=dev),
        ship_combat_death=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=dev),
        ship_boundary_damage=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=dev),
        ship_boundary_death=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=dev),
        ship_respawned=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=dev),
    )


def activate_bullet(
    state: TensorState,
    config: ShipConfig,
    *,
    env: int = 0,
    owner: int = 0,
    slot: int = 0,
    position: complex | torch.Tensor = 0.0j,
    velocity: complex | torch.Tensor = 0.0j,
    lifetime: float | None = None,
) -> None:
    """Activate one internally consistent test bullet.

    Field-enabled tests should refresh the bullet's field cache after placement.
    """
    key = (env, owner, slot)
    state.bullet_pos[key] = position
    state.bullet_vel[key] = velocity
    state.bullet_time[key] = config.bullet_lifetime if lifetime is None else lifetime
    state.bullet_active[key] = True

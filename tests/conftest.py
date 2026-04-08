"""Shared pytest fixtures for all test modules."""

import torch
import pytest

from boost_and_broadside.config import ShipConfig, EnvConfig, ModelConfig, PhaseConfig
from boost_and_broadside.env.state import TensorState


@pytest.fixture
def ship_config() -> ShipConfig:
    """Default ship physics config (reference game values)."""
    return ShipConfig()


@pytest.fixture
def env_config() -> EnvConfig:
    return EnvConfig(num_ships=8, max_bullets=20, max_episode_steps=500)


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(d_model=64, n_heads=4, n_fourier_freqs=8, n_transformer_blocks=2)


@pytest.fixture
def base_phase() -> PhaseConfig:
    return PhaseConfig(
        step=0,
        learning_rate=3e-4,
        pg_coef=1.0,
        ent_coef=0.01,
        bc_coef=0.0,
        vf_coef=0.5,
        scripted_frac=0.0,
        avg_model_frac=0.0,
        league_frac=0.0,
        allow_avg_model_updates=False,
        allow_scripted_in_roster=False,
        elo_eval_games=16,
        elo_eval_interval=0,
        checkpoint_interval=0,
        true_reward_scale=1.0,
        important_scale=1.0,
        aux_scale=1.0,
        victory_weight=1.0,
        death_weight=0.5,
        damage_weight=0.01,
        facing_weight=0.01,
        exposure_weight=0.01,
        turn_rate_weight=0.01,
        closing_speed_weight=0.01,
        proximity_weight=0.01,
        positioning_weight=0.05,
        power_range_weight=0.01,
        speed_range_weight=0.01,
        shoot_quality_weight=0.01,
        positioning_radius=300.0,
        proximity_radius=300.0,
        power_range_lo=0.2,
        power_range_hi=0.8,
        speed_range_lo=40.0,
        speed_range_hi=120.0,
        shoot_quality_radius=200.0,
        enemy_neg_lambda_components=frozenset({"damage", "death", "victory", "exposure"}),
        disabled_rewards=frozenset(),
    )


@pytest.fixture
def device() -> str:
    return "cpu"


def make_state(
    num_envs: int = 2,
    max_ships: int = 4,
    max_bullets: int = 5,
    device: str = "cpu",
    ship_config: ShipConfig | None = None,
) -> TensorState:
    """Build a TensorState with sane initial values for unit testing."""
    if ship_config is None:
        ship_config = ShipConfig()
    dev = torch.device(device)
    return TensorState(
        step_count=torch.zeros((num_envs,), dtype=torch.int32, device=dev),
        ship_pos=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_vel=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=dev),
        ship_attitude=torch.ones(
            (num_envs, max_ships), dtype=torch.complex64, device=dev
        ),
        ship_ang_vel=torch.zeros(
            (num_envs, max_ships), dtype=torch.float32, device=dev
        ),
        ship_health=torch.full(
            (num_envs, max_ships), ship_config.max_health, device=dev
        ),
        ship_power=torch.full((num_envs, max_ships), ship_config.max_power, device=dev),
        ship_cooldown=torch.zeros(
            (num_envs, max_ships), dtype=torch.float32, device=dev
        ),
        ship_team_id=torch.zeros((num_envs, max_ships), dtype=torch.int32, device=dev),
        ship_alive=torch.ones((num_envs, max_ships), dtype=torch.bool, device=dev),
        ship_is_shooting=torch.zeros(
            (num_envs, max_ships), dtype=torch.bool, device=dev
        ),
        prev_action=torch.zeros(
            (num_envs, max_ships, 3), dtype=torch.float32, device=dev
        ),
        bullet_pos=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.complex64, device=dev
        ),
        bullet_vel=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.complex64, device=dev
        ),
        bullet_time=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.float32, device=dev
        ),
        bullet_active=torch.zeros(
            (num_envs, max_ships, max_bullets), dtype=torch.bool, device=dev
        ),
        bullet_cursor=torch.zeros((num_envs, max_ships), dtype=torch.long, device=dev),
    )

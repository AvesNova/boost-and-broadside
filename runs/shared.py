"""Shared config constants used across multiple run profiles.

Import these into individual profiles rather than duplicating values.
Override in a profile only when the run genuinely needs a different value.
"""

from boost_and_broadside.config import ModelConfig, RewardConfig, ShipConfig

SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_heads=4,
    n_fourier_freqs=10,
    n_transformer_blocks=2,
)

# Reward weights shared by all training profiles.
# True/important/aux group scales live in each profile's TrainingSchedule
# since they differ between BC (aux_scale=1.0) and RL (aux_scale=0.0).
REWARDS = RewardConfig(
    # Outcome rewards — ally/enemy split so the critic distinguishes symmetric
    # from asymmetric outcomes (e.g. mutual damage vs no damage, standoff vs
    # close fight).
    ally_damage_weight=0.1,
    enemy_damage_weight=0.1,
    ally_death_weight=0.5,
    enemy_death_weight=0.5,
    ally_win_weight=1.0,
    enemy_win_weight=1.0,
    # Dense shaping rewards — prevent passive collapse during early RL.
    facing_weight=0.000001,
    closing_speed_weight=0.000001,
    shoot_quality_weight=0.000001,
    # Geometry params
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    # Lambda configuration:
    #   enemy_neg_lambda_components → enemy ships get lambda=-1
    #   ally_zero_components        → ally ships get lambda=0 (enemy-perspective only)
    enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
)

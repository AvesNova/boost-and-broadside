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
# true_reward_scale/global_scale/local_scale live in each profile's TrainingSchedule.
REWARDS = RewardConfig(
    # Outcome rewards — ally/enemy split so the critic distinguishes symmetric
    # from asymmetric outcomes (e.g. mutual damage vs no damage, standoff vs
    # close fight).
    ally_damage_weight=0.1,
    enemy_damage_weight=0.1,
    ally_death_weight=0.1,
    enemy_death_weight=0.1,
    ally_win_weight=1.0,
    enemy_win_weight=1.0,
    # Dense shaping rewards — prevent passive collapse during early RL.
    facing_weight=0.1,
    closing_speed_weight=0.1,
    shoot_quality_weight=0.1,
    # Per-ship kill credit — self-only (lambda=0 for all other ships).
    # kill_shot: winner-take-all to the top-damage dealer in the fatal step.
    # kill_assist: proportional share based on cumulative episode damage dealt.
    kill_shot_weight=1.0,
    kill_assist_weight=1.0,
    # Per-ship local combat credit — self-only (lambda=0 for all other ships).
    # damage_taken: negative reward proportional to health lost this step.
    # damage_dealt: positive reward proportional to enemy health removed this step.
    damage_taken_weight=1.0,
    damage_dealt_enemy_weight=1.0,
    damage_dealt_ally_weight=1.0,
    death_weight=1.0,
    # Geometry params
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    # Lambda configuration:
    #   enemy_neg_lambda_components → enemy ships get lambda=-1
    #   ally_zero_components        → ally ships get lambda=0 (enemy-perspective only)
    enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
)

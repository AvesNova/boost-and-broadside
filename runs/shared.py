"""Shared config constants used across multiple run profiles.

Import these into individual profiles rather than duplicating values.
Override in a profile only when the run genuinely needs a different value.
"""

from boost_and_broadside.config import ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.config.obs_spec import (
    ObsConfig, FeatureSpec,
    Normalize, Symlog, Clamp, AsFloat, Bucketize,
    Fourier, OneHot, VecMag, SymlogVec, FourierAngle,
)

SHIP_CONFIG = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

MODEL_CONFIG = ModelConfig(
    d_model=128,
    n_heads=4,
    n_transformer_blocks=2,
)

_W = SHIP_CONFIG.world_size[0]
_H = SHIP_CONFIG.world_size[1]

# Main experiment: no attitude, Fourier vel direction + symlog speed, one-hot health.
OBS_CONFIG = ObsConfig(features=(
    FeatureSpec("pos_x",     ("pos", 0),   (Fourier(10, _W),)),
    FeatureSpec("pos_y",     ("pos", 1),   (Fourier(10, _H),)),
    FeatureSpec("att",       "att",        (FourierAngle(10),)),
    FeatureSpec("vel_dir",   "vel",        (FourierAngle(10),)),
    FeatureSpec("vel_speed", "vel",        (VecMag(), Symlog())),
    FeatureSpec("ang_vel",   "ang_vel",    (Symlog(),)),
    FeatureSpec("health",    "health",     (Normalize(100.0), Bucketize(11), OneHot(11))),
    FeatureSpec("power",     "power",      (Normalize(100.0),)),
    FeatureSpec("cooldown",  "cooldown",   (Normalize(0.1), Clamp(0.0, 1.0))),
    FeatureSpec("team",      "team_id",    (OneHot(3),)),
    FeatureSpec("alive",     "alive",      (AsFloat(),)),
    FeatureSpec("act_power", "prev_power", (OneHot(3),)),
    FeatureSpec("act_turn",  "prev_turn",  (OneHot(7),)),
    FeatureSpec("act_shoot", "prev_shoot", (OneHot(2),)),
    FeatureSpec("radius",    "radius",     (Normalize(40.0),)),
))
# raw_dim = 20+20 + 20+20+1 + 1 + 11 + 1 + 1 + 3 + 1 + 3+7+2 + 1 = 112

# Minimal ablation: 1-freq pos, raw symlog vel, drop att/ang_vel/radius.
OBS_CONFIG_MINIMAL = ObsConfig(features=(
    FeatureSpec("pos_x",     ("pos", 0),   (Fourier(4, _W),)),
    FeatureSpec("pos_y",     ("pos", 1),   (Fourier(4, _H),)),
    FeatureSpec("att",       "att",        (FourierAngle(4),)),
    FeatureSpec("vel",       "vel",        (SymlogVec(),)),
    FeatureSpec("health",    "health",     (Normalize(100.0),)),
    FeatureSpec("power",     "power",      (Normalize(100.0),)),
    FeatureSpec("cooldown",  "cooldown",   (Normalize(0.1), Clamp(0.0, 1.0))),
    FeatureSpec("team",      "team_id",    (OneHot(3),)),
    FeatureSpec("alive",     "alive",      (AsFloat(),)),
    FeatureSpec("act_power", "prev_power", (OneHot(3),)),
    FeatureSpec("act_turn",  "prev_turn",  (OneHot(7),)),
    FeatureSpec("act_shoot", "prev_shoot", (OneHot(2),)),
))
# raw_dim = 2+2 + 2 + 11 + 1 + 1 + 3 + 1 + 3+7+2 = 35

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
    ally_win_weight=4.0,
    enemy_win_weight=4.0,
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
    damage_taken_weight=0.5,
    damage_dealt_enemy_weight=0.5,
    damage_dealt_ally_weight=0.5,
    death_weight=1.0,
    # Geometry params
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    # Lambda configuration:
    #   enemy_neg_lambda_components → enemy ships get lambda=-1
    #   ally_zero_components        → ally ships get lambda=0 (enemy-perspective only)
    enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    # Obstacle rewards
    obstacle_death_weight=0.0,
    obstacle_proximity_weight=0.0,
    obstacle_closing_speed_weight=0.0,
    obstacle_tti_weight=0.0,
    obstacle_proximity_radius=80.0,
    obstacle_tti_max=3.0,
    # Behaviour shaping — off by default in combat mode
    shooting_penalty_weight=0.0,
    speed_weight=1.0,
    speed_penalty_min=10.0,
)

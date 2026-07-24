"""RL training profile for obstacle-only training (no enemies).

Trains purely on obstacle avoidance in isolation. All combat rewards are zeroed;
the only signals are obstacle proximity/collision penalties and a survival reward.

Use this to debug and tune obstacle navigation before mixing in enemy training.
"""

from boost_and_broadside.config import (
    EnvConfig,
    ObstacleCacheConfig,
    RewardConfig,
    ScaleConfig,
    TrainConfig,
    TrainingSchedule,
    constant,
    linear,
    stepped,
)
from runs.shared import COMPONENT_GAMMAS, COMPONENT_LAMBDAS, ELO_EVAL

_MAX_TOKENS = 3840 * 8
_NUM_SHIPS = 8
_NUM_OBSTACLES = 8

OBSTACLES_REWARDS = RewardConfig(
    # All combat rewards zeroed — no enemy interaction in this mode.
    ally_damage_weight=0.0,
    enemy_damage_weight=0.0,
    ally_death_weight=0.0,
    enemy_death_weight=0.0,
    ally_win_weight=0.0,
    enemy_win_weight=0.0,
    facing_weight=0.0,
    closing_speed_weight=0.0,
    shoot_quality_weight=0.0,
    kill_shot_weight=0.0,
    kill_assist_weight=0.0,
    damage_taken_weight=0.0,
    damage_dealt_enemy_weight=0.0,
    damage_dealt_ally_weight=0.0,
    death_weight=1.0,
    # Geometry params (required fields; values unused since combat rewards are zero)
    proximity_radius=400.0,
    shoot_quality_radius=200.0,
    # Lambda config (required fields; no zero-sum needed without combat rewards)
    enemy_neg_lambda_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    ally_zero_components=frozenset({"enemy_damage", "enemy_death", "enemy_win"}),
    # Obstacle rewards — the primary training signal
    obstacle_death_weight=2.0,
    obstacle_proximity_weight=0.1,
    obstacle_closing_speed_weight=0.1,
    obstacle_tti_weight=0.1,
    obstacle_proximity_radius=80.0,
    obstacle_tti_max=3.0,
    # Behaviour shaping
    shooting_penalty_weight=0.1,  # no bullets, never fires
    speed_weight=1.0,
    speed_penalty_min=40.0,
)

RL_OBSTACLES_SCHEDULE = TrainingSchedule(
    learning_rate=linear((0, 1e-7), (5_000_000, 3e-4)),
    policy_gradient_coef=constant(1.0),
    entropy_coef=constant(0.005),
    behavior_cloning_coef=constant(0.0),
    value_function_coef=constant(1.0),
    sigreg_coef=constant(0.00),
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    # No opponents — pure self-navigation, obstacle avoidance only.
    scripted_fraction=stepped((0, 0.0)),
    avg_model_fraction=stepped((0, 0.0)),
    league_fraction=stepped((0, 0.0)),
    checkpoint_interval=stepped((0, 1), (3, 10)),
    num_epochs=constant(4),
    target_kl=constant(None),
    high_elo_threshold=constant(900.0),
    high_elo_target_kl=constant(0.02),
)

RL_OBSTACLES_TRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(
                num_ships=_NUM_SHIPS,
                num_obstacles=_NUM_OBSTACLES,
                max_bullets=0,
                max_episode_steps=1024,
                single_team=True,
            ),
            num_envs=_MAX_TOKENS // (_NUM_SHIPS + _NUM_OBSTACLES),
        ),
    ),
    schedule=RL_OBSTACLES_SCHEDULE,
    rewards=OBSTACLES_REWARDS,
    num_steps=128,
    rollouts_per_update=1,
    num_minibatches=32,
    gamma=0.99,
    gae_lambda=0.95,
    component_gammas=COMPONENT_GAMMAS,
    component_lambdas=COMPONENT_LAMBDAS,
    clip_coef=0.2,
    max_grad_norm=1.0,
    total_timesteps=2_000_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,
    checkpoint_dir="checkpoints",
    league_size=20,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    histogram_interval=10,
    obstacle_cache=ObstacleCacheConfig(
        num_cache_envs=4096,
        cache_size=512,
        max_steps=6000,
    ),
)

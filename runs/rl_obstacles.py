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

_MAX_TOKENS = 3840 * 8
_NUM_SHIPS = 4  # 1 per team — minimum viable for the env's team-based game-over check
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
    shooting_penalty_weight=0.1,
    speed_weight=0.5,
    speed_penalty_min=40.0,
)

RL_OBSTACLES_SCHEDULE = TrainingSchedule(
    learning_rate=linear((0, 1e-7), (5_000_000, 3e-4)),
    policy_gradient_coef=constant(1.0),
    entropy_coef=constant(0.005),
    behavior_cloning_coef=constant(0.0),  # no scripted opponent to clone against
    value_function_coef=constant(1.0),
    sigreg_coef=constant(0.00),
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    # No opponents — pure self-navigation, obstacle avoidance only.
    scripted_fraction=stepped((0, 0.0)),
    avg_model_fraction=stepped((0, 0.0)),
    league_fraction=stepped((0, 0.0)),
    allow_avg_model_updates=stepped((0, False)),
    allow_scripted_in_roster=stepped((0, False)),
    elo_eval_games=stepped((0, 256)),
    elo_eval_interval=stepped((0, 10)),
    checkpoint_interval=stepped((0, 10)),
)

RL_OBSTACLES_TRAIN_CONFIG = TrainConfig(
    scales=(
        ScaleConfig(
            env_config=EnvConfig(
                num_ships=_NUM_SHIPS,
                num_obstacles=_NUM_OBSTACLES,
                max_bullets=20,
                max_episode_steps=1024,
            ),
            num_envs=_MAX_TOKENS // (_NUM_SHIPS * 2 + _NUM_OBSTACLES),
        ),
    ),
    schedule=RL_OBSTACLES_SCHEDULE,
    rewards=OBSTACLES_REWARDS,
    num_steps=128,
    num_epochs=4,
    num_minibatches=32,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    max_grad_norm=1.0,
    total_timesteps=2_000_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,
    checkpoint_dir="checkpoints",
    league_size=20,
    elo_milestone_gap=100.0,
    elo_k_factor=32.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    scripted_roster_min_steps=300_000_000,
    obstacle_cache=ObstacleCacheConfig(
        num_cache_envs=4096,
        cache_size=512,
        max_steps=6000,
    ),
)

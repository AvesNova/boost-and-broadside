"""BC (behaviour cloning) pretraining profile.

LR warms up from 1e-7 → 3e-4 over 6M steps, then holds.
policy_gradient_coef=0.0 throughout — no policy gradient loss.
Scripted agent is queried on all envs for supervised targets only;
no opponents are active (scripted_fraction=0.0).
"""

from boost_and_broadside.config import (
    EnvConfig,
    ScaleConfig,
    TrainConfig,
    TrainingSchedule,
    constant,
    linear,
    stepped,
)
from runs.shared import REWARDS

_MAX_TOKENS = 3840

BC_SCHEDULE = TrainingSchedule(
    # Warmup from 1e-7 to 3e-4 over 6M steps, then hold.
    learning_rate=linear((0, 1e-7), (6_000_000, 3e-4)),
    policy_gradient_coef=constant(0.0),  # BC only — no RL gradient
    entropy_coef=constant(0.01),
    behavior_cloning_coef=constant(1.0),
    value_function_coef=constant(1.0),
    # Group scales — all components active during BC so the value function learns
    # the full reward signal before RL begins.
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    # No opponents during BC — scripted agent only supplies supervised targets.
    scripted_fraction=constant(0.0),
    avg_model_fraction=constant(0.0),
    league_fraction=constant(0.0),
    allow_avg_model_updates=stepped((0, False)),
    allow_scripted_in_roster=stepped((0, True)),
    # ELO evaluation disabled during BC pretraining.
    elo_eval_games=stepped((0, 0)),
    elo_eval_interval=stepped((0, 0)),
    checkpoint_interval=stepped((0, 10)),
)

BC_TRAIN_CONFIG = TrainConfig(
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=1024, num_obstacles=0),
            num_envs=3 * _MAX_TOKENS // 3 // 8,
        ),
    ),
    schedule=BC_SCHEDULE,
    rewards=REWARDS,
    num_steps=128,
    num_epochs=4,
    num_minibatches=4,
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
)

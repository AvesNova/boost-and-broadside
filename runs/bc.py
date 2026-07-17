"""BC (behaviour cloning) pretraining profile.

LR warms up from 1e-7 → 3e-4 over 6M steps, then holds.
policy_gradient_coef=0.0 throughout — no policy gradient loss.
Scripted agent is queried on all envs for supervised targets only;
no opponents are active (opponent_fraction=0.0).
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
from runs.shared import LEAGUE_EVAL, REWARDS

_NUM_BC_ENVS = 480

BC_SCHEDULE = TrainingSchedule(
    # Warmup from 1e-7 to 3e-4 over 6M steps, then hold.
    learning_rate=linear((0, 1e-7), (6_000_000, 3e-4)),
    policy_gradient_coef=constant(0.0),  # BC only — no RL gradient
    entropy_coef=constant(0.01),
    behavior_cloning_coef=constant(1.0),
    value_function_coef=constant(1.0),
    sigreg_coef=constant(0.0),
    # Group scales — all components active during BC so the value function learns
    # the full reward signal before RL begins.
    true_reward_scale=constant(1.0),
    global_scale=constant(1.0),
    local_scale=constant(1.0),
    # No opponents during BC — scripted agent only supplies supervised targets.
    opponent_fraction=constant(0.0),
    checkpoint_interval=stepped((0, 10)),
    num_epochs=constant(4),
    target_kl=constant(None),
    high_winrate_threshold=constant(0.40),
    high_winrate_target_kl=constant(0.02),
)

BC_TRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=1024),
            num_envs=_NUM_BC_ENVS,
        ),
    ),
    schedule=BC_SCHEDULE,
    rewards=REWARDS,
    num_steps=128,
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
    league_k=4,
    league_admission_interval=25,
    opponent_hold_rollouts=4,
    pfsp_mode="variance",
    pfsp_exponent=2.0,
    live_rating_decay=0.9,
    avg_rating_decay=0.995,
    bt_prior_draws=1.0,
    bt_prior_frac=0.02,
    admission_prior_games=10.0,
    league_eval=LEAGUE_EVAL,
    bc_winrate_target=0.5,
    histogram_interval=10,
)

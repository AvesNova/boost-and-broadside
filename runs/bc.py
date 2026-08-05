"""BC (behaviour cloning) pretraining profile.

LR warms up from 1e-7 → 3e-4 over 6M steps, then holds.
policy_gradient_coef=0.0 throughout — no policy gradient loss.
Scripted agent is queried on all envs for supervised targets only;
no opponents are active (league_fraction=0.0).
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
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from runs.shared import ELO_EVAL, REWARDS

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
    # No opponents during BC — the scripted agent only supplies supervised targets.
    league_fraction=constant(0.0),
    checkpoint_interval=stepped((0, 10)),
    num_epochs=constant(4),
    target_kl=constant(None),
    high_elo_threshold=constant(900.0),
    high_elo_target_kl=constant(0.02),
)

BC_TRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(
                num_ships=2,
                max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                max_episode_steps=1024,
            ),
            num_envs=_NUM_BC_ENVS,
        ),
    ),
    schedule=BC_SCHEDULE,
    rewards=REWARDS,
    num_steps=128,
    rollouts_per_update=1,
    num_minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    max_grad_norm=1.0,
    total_timesteps=2_000_000_000,
    return_ema_alpha=0.005,
    return_min_span=1.0,  # see runs/rl.py — not an epsilon; lowering it needs critic re-tuning
    advantage_min_rms=1e-4,
    checkpoint_dir="checkpoints",
    league_size=20,
    league_slots=4,
    # No opponents during BC, so no reference ladder is needed.
    reference_ladder=(),
    random_elo=0.0,
    elo_milestone_gap=200.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    elo_eval=ELO_EVAL,
    bc_winrate_target=0.45,
    histogram_interval=10,
)

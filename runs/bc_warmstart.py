"""BC warm-start profile: run BC pretraining for 50M steps, then switch to RL.

This defines configs for both stages. main.py handles the two-stage pipeline:
  1. Run BC_WARMSTART_PRETRAIN_CONFIG for 50M steps → save weights.
  2. Load those weights into RL_TRAIN_CONFIG → run full RL.

The pretrain stage uses the same schedule as bc.py but with a shorter
total_timesteps — just enough to give the policy a solid BC foundation
before RL takes over.
"""

from boost_and_broadside.config import EnvConfig, ScaleConfig, TrainConfig
from boost_and_broadside.config import constant, stepped
from runs.bc import BC_SCHEDULE
from runs.rl import RL_TRAIN_CONFIG
from runs.shared import REWARDS

# max_tokens = num_envs * num_ships * num_steps = 480 * 4 * 128 = 245_760
_MAX_TOKENS = 245_760

BC_WARMSTART_PRETRAIN_CONFIG = TrainConfig(
    max_tokens=_MAX_TOKENS,
    scales=(
        ScaleConfig(
            env_config=EnvConfig(ally_ship_count=2, enemy_ship_count=2, max_bullets=20, max_episode_steps=1024),
            token_fraction=1.0,
            scripted_fraction=constant(0.0),
            avg_model_fraction=constant(0.0),
            league_fraction=constant(0.0),
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
    total_timesteps=20_000_000,  # short: just enough for a good policy initialisation
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

# The RL stage is identical to a standalone RL run.
BC_WARMSTART_RL_CONFIG = RL_TRAIN_CONFIG

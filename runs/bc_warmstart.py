"""BC warm-start profile: run BC pretraining for 50M steps, then switch to RL.

This defines configs for both stages. main.py handles the two-stage pipeline:
  1. Run BC_WARMSTART_PRETRAIN_CONFIG for 50M steps → save weights.
  2. Load those weights into RL_TRAIN_CONFIG → run full RL.

The pretrain stage uses the same schedule as bc.py but with a shorter
total_timesteps — just enough to give the policy a solid BC foundation
before RL takes over.
"""

from boost_and_broadside.config import EnvConfig, ScaleConfig, TrainConfig
from runs.bc import BC_SCHEDULE, _MAX_TOKENS
from runs.rl import RL_TRAIN_CONFIG
from runs.shared import REWARDS

BC_WARMSTART_PRETRAIN_CONFIG = TrainConfig(
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=1024),
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
    total_timesteps=50_000_000,  # short: just enough for a good policy initialisation
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

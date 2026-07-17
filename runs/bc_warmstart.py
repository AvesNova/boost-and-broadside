"""BC warm-start profile: run a short BC pretraining stage, then switch to RL.

This defines configs for both stages. main.py handles the two-stage pipeline:
  1. Run BC_WARMSTART_PRETRAIN_CONFIG to completion → save weights.
  2. Load those weights into RL_TRAIN_CONFIG → run full RL.

The pretrain stage uses the same schedule as bc.py but with a shorter
total_timesteps — just enough to give the policy a solid BC foundation
before RL takes over.
"""

from boost_and_broadside.config import EnvConfig, ScaleConfig, TrainConfig
from runs.bc import _NUM_BC_ENVS, BC_SCHEDULE
from runs.rl import RL_TRAIN_CONFIG
from runs.shared import LEAGUE_EVAL, REWARDS

BC_WARMSTART_PRETRAIN_CONFIG = TrainConfig(
    paradigm="ego_pass",
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=1024),
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
    total_timesteps=20_000_000,  # short: just enough for a good policy initialisation
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

# The RL stage is identical to a standalone RL run.
BC_WARMSTART_RL_CONFIG = RL_TRAIN_CONFIG

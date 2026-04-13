"""RL training profile for high-performance compute clusters.

Identical schedule to rl.py — only the batch size scales up.
Tune _HPC_NUM_ENVS for your target cluster's VRAM / CPU budget.

Rule of thumb: keep num_envs divisible by num_minibatches (currently 4).
Total ships per update = num_envs * num_ships * num_steps.
"""

from boost_and_broadside.config import EnvConfig, ScaleConfig, TrainConfig
from runs.rl import RL_SCHEDULE
from runs.shared import REWARDS

# 4× the default batch. Adjust to fit your cluster.
# Must be divisible by num_minibatches=4.
_HPC_NUM_ENVS = 1920

RL_HPC_TRAIN_CONFIG = TrainConfig(
    scales=(
        ScaleConfig(
            env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=1024),
            num_envs=_HPC_NUM_ENVS,
        ),
    ),
    schedule=RL_SCHEDULE,   # identical schedule — only scale differs
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

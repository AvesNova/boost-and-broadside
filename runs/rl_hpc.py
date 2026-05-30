"""RL training profile for high-performance compute clusters.

Identical schedule to rl.py — only the batch size scales up.
Tune _HPC_MAX_TOKENS for your target cluster's VRAM / CPU budget.

Rule of thumb: max_tokens ≈ num_envs * num_ships * num_steps.
"""

from boost_and_broadside.config import EnvConfig, ScaleConfig, TrainConfig
from boost_and_broadside.config import stepped
from runs.rl import RL_SCHEDULE
from runs.shared import REWARDS, COMPONENT_GAMMAS, COMPONENT_LAMBDAS

# 4× the default batch. Adjust to fit your cluster.
# Equivalent to num_envs ≈ 1920 with num_ships=2, num_steps=128, num_minibatches=4.
_HPC_MAX_TOKENS = 491_520

RL_HPC_TRAIN_CONFIG = TrainConfig(
    max_tokens=_HPC_MAX_TOKENS,
    scales=(
        ScaleConfig(
            env_config=EnvConfig(ally_ship_count=1, enemy_ship_count=1, max_bullets=20, max_episode_steps=1024),
            token_fraction=1.0,
            scripted_fraction=stepped((0, 0.5), (50_000_000, 0.3)),
            avg_model_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
            league_fraction=stepped((0, 0.0), (50_000_000, 0.2)),
        ),
    ),
    schedule=RL_SCHEDULE,  # identical schedule — only scale differs
    rewards=REWARDS,
    num_steps=128,
    num_minibatches=4,
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
    elo_milestone_gap=100.0,
    elo_k_factor=32.0,
    elo_temperature=200.0,
    league_uniform_sampling=False,
    scripted_roster_min_steps=300_000_000,
)

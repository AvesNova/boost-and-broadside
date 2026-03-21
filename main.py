"""MVP RL training entry point.

All hyperparameters are defined here — no config files, no CLI flags.
To experiment: duplicate this file or edit these values directly.

Run with:
    uv run --no-sync main.py
"""

import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # ------------------------------------------------------------------
    # Physics (reference game values — edit carefully)
    # ------------------------------------------------------------------
    ship_config = ShipConfig()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env_config = EnvConfig(
        num_ships         = 8,    # 4v4
        max_bullets       = 20,
        max_episode_steps = 2000,
    )

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    model_config = ModelConfig(
        d_model       = 128,
        n_heads       = 4,
        n_fourier_freqs = 8,
    )

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    reward_config = RewardConfig(
        damage_weight      = 0.01,   # per HP of damage
        kill_weight        = 0.5,    # per kill
        death_weight       = 0.5,    # per death
        victory_weight     = 1.0,    # win / lose outcome
        positioning_weight = 0.05,   # offensive/defensive geometry
        positioning_radius = 400.0,  # world units (world_size = 1024)
    )

    # ------------------------------------------------------------------
    # PPO hyperparameters
    # ------------------------------------------------------------------
    train_config = TrainConfig(
        num_envs        = 128,
        num_steps       = 512,        # steps per rollout per env
        num_epochs      = 4,
        num_minibatches = 4,          # 128 envs / 4 = 32 envs per batch
        learning_rate   = 3e-4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_coef       = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        total_timesteps = 50_000_000,
    )

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    trainer = PPOTrainer(
        train_config  = train_config,
        model_config  = model_config,
        ship_config   = ship_config,
        env_config    = env_config,
        reward_config = reward_config,
        device        = device,
        use_wandb     = False,   # set True and run `wandb login` to enable logging
    )
    trainer.train()


if __name__ == "__main__":
    main()

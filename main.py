"""Boost and Broadside — entry point.

All hyperparameters are defined here — no config files.
Modes are selected via the --mode CLI flag.

Run with:
    uv run --no-sync main.py                                         # train
    uv run --no-sync main.py --mode play                             # human vs AI
    uv run --no-sync main.py --mode watch --checkpoint <path.pt>    # watch checkpoint
"""

import argparse
import sys

import torch

from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig,
)
from boost_and_broadside.modes.interactive import run_play_mode, run_watch_mode
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.ui.renderer import RenderConfig


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for mode selection.

    Returns:
        Parsed args with .mode and .checkpoint fields.
    """
    parser = argparse.ArgumentParser(description="Boost and Broadside")
    parser.add_argument(
        "--mode",
        choices=["train", "play", "watch"],
        default="train",
        help="Operating mode: train (PPO), play (human + AI), watch (checkpoint self-play).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file. Required for --mode watch.",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Shared configs (all modes)
    # ------------------------------------------------------------------
    ship_config = ShipConfig(bullet_energy_cost=0)

    env_config = EnvConfig(
        num_ships         = 2,    # 4v4
        max_bullets       = 20,
        max_episode_steps = 512,
    )

    model_config = ModelConfig(
        d_model         = 128,
        n_heads         = 4,
        n_fourier_freqs = 8,
    )

    reward_config = RewardConfig(
        # --- Objective rewards
        victory_weight     = 8.0,
        kill_weight        = 2.0,
        death_weight       = 2.0,
        damage_weight      = 0.2,

        # --- Shaping rewards
        positioning_weight = 0.05,
        positioning_radius = 400.0,

        facing_weight      = 0.04,
        exposure_weight    = 0.04,

        approach_weight    = 0.0005,

        proximity_weight   = 0.02,
        proximity_radius   = 400.0,

        # --- Behavioral regularizers ---
        power_range_weight = 0.005,
        power_range_lo     = 0.2,
        power_range_hi     = 0.8,

        speed_range_weight = 0.01,
        speed_range_lo     = 40.0,
        speed_range_hi     = 120.0,

        shoot_quality_weight = 0.05,
        shoot_quality_radius = 100.0,
    )

    render_config = RenderConfig()

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------
    match args.mode:
        case "train":
            train_config = TrainConfig(
                num_envs            = 256 + 128,
                num_steps           = 512,
                num_epochs          = 4,
                num_minibatches     = 4,
                learning_rate       = 3e-4,
                gamma               = 0.99,
                gae_lambda          = 0.95,
                clip_coef           = 0.2,
                ent_coef            = 0.01,
                vf_coef             = 0.5,
                max_grad_norm       = 0.5,
                total_timesteps     = 50_000_000,
                checkpoint_interval = 10,
                checkpoint_dir      = "checkpoints",
            )
            trainer = PPOTrainer(
                train_config  = train_config,
                model_config  = model_config,
                ship_config   = ship_config,
                env_config    = env_config,
                reward_config = reward_config,
                device        = device,
                use_wandb     = True,
            )
            trainer.train()

        case "play":
            run_play_mode(ship_config, env_config, reward_config, model_config,
                          render_config, device)

        case "watch":
            if args.checkpoint is None:
                sys.exit("Error: --checkpoint <path> is required for watch mode.")
            run_watch_mode(args.checkpoint, ship_config, env_config, reward_config,
                           model_config, render_config, device)


if __name__ == "__main__":
    main()

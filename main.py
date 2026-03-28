"""Boost and Broadside — entry point.

All hyperparameters are defined here — no config files.
Modes are selected via the --mode CLI flag.

Run with:
    uv run --no-sync main.py                                                   # train
    uv run --no-sync main.py --mode watch                                      # human vs latest checkpoint
    uv run --no-sync main.py --mode watch --team0 null --team1 scripted        # human vs scripted
    uv run --no-sync main.py --mode watch --team0 latest --team1 latest        # self-play
    uv run --no-sync main.py --mode collect_stats                              # scripted vs random
    uv run --no-sync main.py --mode collect_stats --team0 latest --team1 scripted

Agent specs (--team0 / --team1):
    null        human keyboard (watch only)
    random      uniform random actions
    scripted    stochastic scripted agent
    latest      most recently modified checkpoint
    <path.pt>   specific checkpoint file
"""

import argparse

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import (
    ShipConfig, EnvConfig, ModelConfig, RewardConfig, TrainConfig,
)
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.interactive import run_watch_mode
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.ui.renderer import RenderConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boost and Broadside",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "watch", "collect_stats"],
        default="train",
        help="Operating mode.",
    )
    parser.add_argument(
        "--team0",
        type=str,
        default=None,
        metavar="SPEC",
        help="Agent for team 0: null, random, scripted, latest, or path/to/checkpoint.pt. "
             "Defaults: watch→null, collect_stats→scripted.",
    )
    parser.add_argument(
        "--team1",
        type=str,
        default=None,
        metavar="SPEC",
        help="Agent for team 1: null, random, scripted, latest, or path/to/checkpoint.pt. "
             "Defaults: watch→latest, collect_stats→random.",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Shared configs (all modes)
    # ------------------------------------------------------------------
    ship_config = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

    env_config = EnvConfig(
        num_ships         = 2,
        max_bullets       = 20,
        max_episode_steps = 1024,
    )

    model_config = ModelConfig(
        d_model         = 128,
        n_heads         = 4,
        n_fourier_freqs = 8,
    )

    reward_config = RewardConfig(
        # --- Objective rewards
        victory_weight     = 1.0,
        death_weight       = 1.0,
        damage_weight      = 1.0,

        # Lambda=-1 for these components on enemy ships (outcome rewards)
        enemy_neg_lambda_components = frozenset({"damage", "death", "victory", "exposure"}),

        # --- Shaping rewards
        positioning_weight = 0.0,
        positioning_radius = 400.0,

        facing_weight      = 0.01,
        exposure_weight    = 0.0,

        closing_speed_weight = 0.001,
        turn_rate_weight     = 0.001,

        proximity_weight   = 0.0,
        proximity_radius   = 400.0,

        # --- Behavioral regularizers ---
        power_range_weight = 0.0,
        power_range_lo     = 0.2,
        power_range_hi     = 0.8,

        speed_range_weight = 0.0,
        speed_range_lo     = 40.0,
        speed_range_hi     = 120.0,

        shoot_quality_weight = 0.0,
        shoot_quality_radius = 200.0,

        scripted_agent_weight = 0.0,
    )

    render_config = RenderConfig()

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------
    match args.mode:
        case "train":
            scripted_agent = StochasticScriptedAgent(ship_config, StochasticAgentConfig())
            train_config = TrainConfig(
                num_envs            = 1700,
                num_steps           = 128,
                num_epochs          = 4,
                num_minibatches     = 4,
                learning_rate       = 3e-4,
                gamma               = 0.99,
                gae_lambda          = 0.95,
                clip_coef           = 0.2,
                ent_coef            = 0.01,
                vf_coef             = 0.5,
                max_grad_norm       = 0.5,
                total_timesteps     = 1_000_000_000,
                return_ema_alpha    = 0.005,  # ~200-update memory for percentile EMA
                return_min_span     = 1.0,    # guard against zero-return disabled components
                lr_warmup_steps     = 10_000_000,
                checkpoint_interval = 10,
                checkpoint_dir      = "checkpoints",
                scripted_frac       = 0.0,
                avg_model_frac      = 0.50,
                bc_coef             = 0.1,

                league_frac         = 0.0,
                league_size         = 20,
                elo_eval_games      = 256,
                elo_eval_interval   = 10,
                elo_milestone_gap         = 50.0,
                elo_k_factor              = 32.0,
                elo_temperature           = 200.0,
                scripted_roster_min_steps = 100_000_000,
            )
            trainer = PPOTrainer(
                train_config   = train_config,
                model_config   = model_config,
                ship_config    = ship_config,
                env_config     = env_config,
                reward_config  = reward_config,
                device         = device,
                use_wandb      = True,
                scripted_agent = scripted_agent,
            )
            trainer.train()

        case "watch":
            team0 = args.team0 if args.team0 is not None else "null"
            team1 = args.team1 if args.team1 is not None else "latest"
            run_watch_mode(
                team0_spec    = team0,
                team1_spec    = team1,
                ship_config   = ship_config,
                env_config    = env_config,
                reward_config = reward_config,
                model_config  = model_config,
                render_config = render_config,
                device        = device,
                checkpoint_dir = "checkpoints",
            )

        case "collect_stats":
            collect_stats_num_envs = 4096
            team0 = args.team0 if args.team0 is not None else "scripted"
            team1 = args.team1 if args.team1 is not None else "random"
            run_collect_stats_mode(
                team0_spec    = team0,
                team1_spec    = team1,
                num_envs      = collect_stats_num_envs,
                ship_config   = ship_config,
                env_config    = env_config,
                model_config  = model_config,
                device        = device,
                checkpoint_dir = "checkpoints",
            )


if __name__ == "__main__":
    main()

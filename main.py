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
    ShipConfig,
    EnvConfig,
    ModelConfig,
    PhaseConfig,
    TimelineConfig,
    TrainConfig,
    ScaleConfig,
)
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.elo_stats import run_elo_stats_mode
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
        choices=["train", "watch", "collect_stats", "elo_stats"],
        default="train",
        help="Operating mode.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="latest",
        metavar="RUN",
        help="Run name for elo_stats mode (e.g. 'bright-cloud-219') or 'latest'.",
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
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Shared configs (all modes)
    # ------------------------------------------------------------------
    ship_config = ShipConfig(bullet_energy_cost=2, bullet_min_damage_frac=1.0)

    env_config = EnvConfig(
        num_ships=4,
        max_bullets=20,
        max_episode_steps=1024,
    )

    model_config = ModelConfig(
        d_model=128,
        n_heads=4,
        n_fourier_freqs=10,
        n_transformer_blocks=2,
    )

    timeline = TimelineConfig(
        phases=(
            PhaseConfig(
                step=0,
                # --- Optimization (pretrain: pg_coef=0 zeroes policy gradient) ---
                learning_rate=1e-7,
                pg_coef=1.0,
                ent_coef=0.2,
                bc_coef=1.0,
                vf_coef=1.0,
                # --- Opponents (pretrain: 100% scripted for BC targets) ---
                scripted_frac=0.0,
                avg_model_frac=0.0,
                league_frac=0.0,
                allow_avg_model_updates=False,
                allow_scripted_in_roster=False,
                # --- League / Checkpointing ---
                elo_eval_games=256,
                elo_eval_interval=10,
                checkpoint_interval=10,
                # --- Reward Group Scales ---
                true_reward_scale=1.0,
                important_scale=1.0,
                aux_scale=1.0,
                # --- Reward Individual Weights ---
                victory_weight=1.0,
                death_weight=1.0,
                damage_weight=1.0,
                facing_weight=0.0,
                exposure_weight=0.0,
                closing_speed_weight=0.0,
                turn_rate_weight=0.0,
                proximity_weight=0.0,
                positioning_weight=0.0,
                power_range_weight=0.0,
                speed_range_weight=0.0,
                shoot_quality_weight=0.0,
                # --- Reward Static Params ---
                positioning_radius=400.0,
                proximity_radius=400.0,
                power_range_lo=0.2,
                power_range_hi=0.8,
                speed_range_lo=40.0,
                speed_range_hi=120.0,
                shoot_quality_radius=200.0,
                # --- Set-valued Fields ---
                enemy_neg_lambda_components=frozenset(
                    {"damage", "death", "victory", "exposure"}
                ),
                disabled_rewards=frozenset(),
            ),
            PhaseConfig(
                step=10_000_000,
                learning_rate=3e-4,
            ),
            PhaseConfig(
                step=50_000_000,
            ),
            PhaseConfig(
                step=100_000_000,
                bc_coef=0.1,
            ),
            PhaseConfig(
                step=200_000_000,
                bc_coef=0.0,
            ),
            # PhaseConfig(
            #     step=40_000_000,
            #     ent_coef=0.02,
            #     vf_coef=1.0,
            #     pg_coef=0.1,
            # ),
            # PhaseConfig(
            #     step=100_000_000,
            #     pg_coef=1.0,
            #     bc_coef=0.0,
            #     avg_model_frac=0.30,
            #     league_frac=0.20,
            #     allow_avg_model_updates=True,
            #     allow_scripted_in_roster=True,
            # ),
            # PhaseConfig(
            #     step=150_000_000,
            # ),
            # PhaseConfig(
            #     step=200_000_000,
            #     aux_scale=0.0,
            # ),
            # PhaseConfig(
            #     step=400_000_000,
            #     important_scale=0.0,
            # ),
        )
    )

    render_config = RenderConfig()

    max_tokens = 3840

    # ------------------------------------------------------------------
    # Mode dispatch
    # ------------------------------------------------------------------
    match args.mode:
        case "train":
            scripted_agent = StochasticScriptedAgent(
                ship_config, StochasticAgentConfig()
            )
            train_config = TrainConfig(
                scales=(
                    # ScaleConfig(
                    #     env_config=EnvConfig(
                    #         num_ships=2, max_bullets=20, max_episode_steps=1024
                    #     ),
                    #     num_envs=max_tokens // 3 // 2,
                    # ),
                    # ScaleConfig(
                    #     env_config=EnvConfig(
                    #         num_ships=4, max_bullets=20, max_episode_steps=1024
                    #     ),
                    #     num_envs=max_tokens // 3 // 4,
                    # ),
                    ScaleConfig(
                        env_config=EnvConfig(
                            num_ships=8, max_bullets=20, max_episode_steps=1024
                        ),
                        num_envs=3 * max_tokens // 3 // 8,
                    ),
                ),
                timeline=timeline,
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
                avg_model_min_steps=200_000_000,
                league_size=20,
                league_uniform_sampling=False,
                elo_milestone_gap=100.0,
                elo_k_factor=32.0,
                elo_temperature=200.0,
                scripted_roster_min_steps=300_000_000,
            )
            trainer = PPOTrainer(
                train_config=train_config,
                model_config=model_config,
                ship_config=ship_config,
                device=device,
                use_wandb=True,
                scripted_agent=scripted_agent,
            )
            trainer.train()

        case "watch":
            team0 = args.team0 if args.team0 is not None else "null"
            team1 = args.team1 if args.team1 is not None else "latest"
            run_watch_mode(
                team0_spec=team0,
                team1_spec=team1,
                ship_config=ship_config,
                env_config=env_config,
                phase=base_phase,
                model_config=model_config,
                render_config=render_config,
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "collect_stats":
            collect_stats_num_envs = 1024
            team0 = args.team0 if args.team0 is not None else "scripted"
            team1 = args.team1 if args.team1 is not None else "random"
            run_collect_stats_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_envs=collect_stats_num_envs,
                ship_config=ship_config,
                env_config=env_config,
                model_config=model_config,
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "elo_stats":
            run_elo_stats_mode(
                run_spec=args.run,
                num_envs=1024,
                ship_config=ship_config,
                env_config=env_config,
                model_config=model_config,
                device=device,
                checkpoint_dir="checkpoints",
                elo_k_factor=32.0,
            )


if __name__ == "__main__":
    main()

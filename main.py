"""Boost and Broadside — entry point.

Modes are selected via the --mode CLI flag. All hyperparameters live in runs/.

Run with:
    uv run --no-sync main.py --mode bc                                             # BC pretraining from scratch
    uv run --no-sync main.py --mode rl                                             # RL from scratch (no pretrained weights)
    uv run --no-sync main.py --mode rl --pretrain_from checkpoints/run/best.pt    # RL from pretrained init
    uv run --no-sync main.py --mode bc_warmstart                                  # pretrain 50M → save → RL (one process)
    uv run --no-sync main.py --mode watch                                          # human vs latest checkpoint
    uv run --no-sync main.py --mode watch --team0 null --team1 scripted            # human vs scripted
    uv run --no-sync main.py --mode watch --team0 latest --team1 latest            # self-play
    uv run --no-sync main.py --mode collect_stats                                  # scripted vs random
    uv run --no-sync main.py --mode collect_stats --team0 latest --team1 scripted
    uv run --no-sync main.py --mode elo_stats                                     # all scripted agents
    uv run --no-sync main.py --mode elo_stats --run latest                        # scripted + latest run checkpoints

Agent specs (--team0 / --team1):
    null        human keyboard (watch only)
    random      uniform random actions
    scripted    stochastic scripted agent
    latest      most recently modified checkpoint
    <path.pt>   specific checkpoint file
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.elo_stats import run_elo_stats_mode
from boost_and_broadside.modes.interactive import run_watch_mode
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.ui.renderer import RenderConfig
from runs.bc import BC_TRAIN_CONFIG
from runs.bc_warmstart import BC_WARMSTART_PRETRAIN_CONFIG, BC_WARMSTART_RL_CONFIG
from runs.rl import RL_TRAIN_CONFIG
from runs.rl_obstacles import RL_OBSTACLES_TRAIN_CONFIG
from runs.shared import MODEL_CONFIG, REWARDS, SHIP_CONFIG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boost and Broadside",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["bc", "rl", "rl_obstacles", "bc_warmstart", "watch", "collect_stats", "elo_stats"],
        default="rl",
        help=(
            "Operating mode. "
            "'bc': BC-only pretraining, no RL gradient. "
            "'rl': RL run, optionally loading pretrained weights via --pretrain_from. "
            "'bc_warmstart': pretrain for 50M steps then immediately start RL from those weights."
        ),
    )
    parser.add_argument(
        "--pretrain_from",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a pretrained checkpoint (.pt) to warm-start the rl run. "
        "Loads policy + scaler; optimizer is reset (fresh Adam). "
        "Example: checkpoints/major-serenity-381/best_training.pt",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="none",
        metavar="RUN",
        help="Run name for elo_stats mode (e.g. 'bright-cloud-219'), 'latest', or 'none' "
        "(scripted agents only, no checkpoints).",
    )
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=["none", "reduce-overhead", "default", "max-autotune"],
        default="reduce-overhead",
        metavar="MODE",
        help="torch.compile mode: none (eager), reduce-overhead (default, fast startup), "
        "default, or max-autotune.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke-test mode: tiny batch (4 envs), no W&B, no compile, exits after a few updates.",
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
    parser.add_argument(
        "--fast-cache",
        action="store_true",
        default=False,
        help="(watch mode) Generate obstacle cache headlessly in the background instead of "
        "rendering the convergence animation. Much faster; shows a loading screen instead.",
    )
    return parser.parse_args()


def _apply_smoke(config):
    """Shrink a TrainConfig to the smallest viable size for crash-testing."""
    from boost_and_broadside.config.schedule import stepped
    # num_envs must be divisible by num_minibatches, so use 1 minibatch with 4 envs.
    scales = tuple(replace(s, num_envs=4) for s in config.scales)
    obstacle_cache = config.obstacle_cache
    if obstacle_cache is not None:
        obstacle_cache = replace(obstacle_cache, num_cache_envs=128, cache_size=4, max_steps=6000)
    schedule = replace(config.schedule, checkpoint_interval=stepped((0, 1)))
    return replace(
        config,
        scales=scales,
        schedule=schedule,
        obstacle_cache=obstacle_cache,
        num_minibatches=1,
        total_timesteps=5_000,
        log_interval=1,
    )


def _run_trainer(trainer: PPOTrainer) -> None:
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        trainer._shutdown()


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    use_wandb = not args.smoke
    compile_mode = None if (args.smoke or args.compile_mode == "none") else args.compile_mode

    match args.mode:
        case "bc":
            scripted_agent = StochasticScriptedAgent(
                SHIP_CONFIG, StochasticAgentConfig()
            )
            train_config = _apply_smoke(BC_TRAIN_CONFIG) if args.smoke else BC_TRAIN_CONFIG
            trainer = PPOTrainer(
                train_config=train_config,
                model_config=MODEL_CONFIG,
                ship_config=SHIP_CONFIG,
                device=device,
                use_wandb=use_wandb,
                scripted_agent=scripted_agent,
                compile_mode=compile_mode,
            )
            _run_trainer(trainer)

        case "rl":
            scripted_agent = StochasticScriptedAgent(
                SHIP_CONFIG, StochasticAgentConfig()
            )
            train_config = _apply_smoke(RL_TRAIN_CONFIG) if args.smoke else RL_TRAIN_CONFIG
            trainer = PPOTrainer(
                train_config=train_config,
                model_config=MODEL_CONFIG,
                ship_config=SHIP_CONFIG,
                device=device,
                use_wandb=use_wandb,
                scripted_agent=scripted_agent,
                compile_mode=compile_mode,
            )
            if args.pretrain_from is not None:
                trainer.load_pretrained_weights(args.pretrain_from)
            _run_trainer(trainer)

        case "rl_obstacles":
            train_config = _apply_smoke(RL_OBSTACLES_TRAIN_CONFIG) if args.smoke else RL_OBSTACLES_TRAIN_CONFIG
            scripted_agent = StochasticScriptedAgent(SHIP_CONFIG, StochasticAgentConfig())
            trainer = PPOTrainer(
                train_config=train_config,
                model_config=MODEL_CONFIG,
                ship_config=SHIP_CONFIG,
                device=device,
                use_wandb=use_wandb,
                scripted_agent=scripted_agent,
                compile_mode=compile_mode,
            )
            if args.pretrain_from is not None:
                trainer.load_pretrained_weights(args.pretrain_from)
            _run_trainer(trainer)

        case "bc_warmstart":
            scripted_agent = StochasticScriptedAgent(
                SHIP_CONFIG, StochasticAgentConfig()
            )

            print("=== BC_WARMSTART: starting BC pretraining phase (50M steps) ===")
            pretrain_trainer = PPOTrainer(
                train_config=BC_WARMSTART_PRETRAIN_CONFIG,
                model_config=MODEL_CONFIG,
                ship_config=SHIP_CONFIG,
                device=device,
                use_wandb=True,
                scripted_agent=scripted_agent,
                compile_mode=None if args.compile_mode == "none" else args.compile_mode,
            )
            _run_trainer(pretrain_trainer)

            ckpt_dir = (
                Path(BC_WARMSTART_PRETRAIN_CONFIG.checkpoint_dir)
                / pretrain_trainer._run_name
            )
            pretrain_path = ckpt_dir / "pretrained_for_rl.pt"
            torch.save(pretrain_trainer._checkpoint_payload(update=0), pretrain_path)
            print(f"=== BC_WARMSTART: pretrained weights saved to {pretrain_path} ===")
            pretrain_trainer._shutdown()
            del pretrain_trainer

            print("=== BC_WARMSTART: starting RL phase ===")
            rl_trainer = PPOTrainer(
                train_config=BC_WARMSTART_RL_CONFIG,
                model_config=MODEL_CONFIG,
                ship_config=SHIP_CONFIG,
                device=device,
                use_wandb=True,
                scripted_agent=scripted_agent,
                compile_mode=None if args.compile_mode == "none" else args.compile_mode,
            )
            rl_trainer.load_pretrained_weights(str(pretrain_path))
            _run_trainer(rl_trainer)

        case "watch":
            team0 = args.team0 if args.team0 is not None else "null"
            team1 = args.team1 if args.team1 is not None else "latest"
            run_watch_mode(
                team0_spec=team0,
                team1_spec=team1,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=16, max_bullets=20, max_episode_steps=1024, num_obstacles=16,
                ),
                rewards=REWARDS,
                model_config=MODEL_CONFIG,
                render_config=RenderConfig(),
                device=device,
                checkpoint_dir="checkpoints",
                fast_cache=args.fast_cache,
            )

        case "collect_stats":
            team0 = args.team0 if args.team0 is not None else "scripted"
            team1 = args.team1 if args.team1 is not None else "random"
            run_collect_stats_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_envs=1024,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=4, max_bullets=20, max_episode_steps=1024
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "elo_stats":
            run_elo_stats_mode(
                run_spec=args.run,
                num_envs=110000,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=4, max_bullets=20, max_episode_steps=1024
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                elo_k_factor=32.0,
            )


if __name__ == "__main__":
    main()

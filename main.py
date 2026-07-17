"""Boost and Broadside — entry point.

Select a mode with --mode. All hyperparameters live in runs/.

Training:
    uv run --no-sync main.py --mode rl                          # RL from scratch
    uv run --no-sync main.py --mode rl --smoke                  # crash-test (tiny batch, no W&B)
    uv run --no-sync main.py --mode rl --compile max-autotune   # RL with max-autotune
    uv run --no-sync main.py --mode rl --pretrain_from checkpoints/run/best.pt
    uv run --no-sync main.py --mode rl --resume                 # resume latest checkpoint
    uv run --no-sync main.py --mode rl --resume checkpoints/run/step_000001000.pt
    uv run --no-sync main.py --mode rl_obstacles                # RL with dynamic obstacles
    uv run --no-sync main.py --mode bc                          # BC pretraining from scratch
    uv run --no-sync main.py --mode bc_warmstart                # BC pretrain → RL (one process)

Watch / play:
    uv run --no-sync main.py --mode watch                       # human vs latest checkpoint
    uv run --no-sync main.py --mode watch --team0 null --team1 scripted
    uv run --no-sync main.py --mode watch --team0 latest --team1 latest
    uv run --no-sync main.py --mode watch --fast-cache          # skip convergence animation

Evaluation:
    uv run --no-sync main.py --mode collect_stats               # scripted vs random
    uv run --no-sync main.py --mode collect_stats --team0 latest --team1 scripted
    uv run --no-sync main.py --mode feature_stats               # label-scale calibration stats
    uv run --no-sync main.py --mode elo_stats                   # ELO across scripted agents
    uv run --no-sync main.py --mode elo_stats --run latest      # + checkpoints from latest run
    uv run --no-sync main.py --mode ar_report                   # autoregressive prediction report
    uv run --no-sync main.py --mode noise_calibration           # NextStateHead error statistics

Agent specs (--team0 / --team1):
    null        human keyboard (watch only)
    random      uniform random actions
    scripted    stochastic scripted agent
    latest      most recently modified checkpoint in checkpoints/
    <path.pt>   specific checkpoint file
    plus named scripted agents: scripted_team, jouster, team_jouster, boom_zoom,
    abreast, reverse_turret, run_away, spiral_evader, jinking

Compile profiles (--compile):
    reduce-overhead    default; fast startup, good throughput
    default            slower startup, slightly higher throughput
    max-autotune       slowest startup, best throughput
    none               eager mode (useful for debugging)
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig, TrainConfig
from boost_and_broadside.modes.ar_report import run_ar_report_mode
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.elo_stats import run_elo_stats_mode
from boost_and_broadside.modes.feature_stats import run_feature_stats_mode
from boost_and_broadside.modes.interactive import run_watch_mode
from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode
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
        choices=[
            "bc",
            "rl",
            "rl_obstacles",
            "bc_warmstart",
            "watch",
            "collect_stats",
            "feature_stats",
            "elo_stats",
            "ar_report",
            "noise_calibration",
        ],
        default="rl",
        help=(
            "Operating mode. "
            "'bc': BC-only pretraining, no RL gradient. "
            "'rl': RL run, optionally loading pretrained weights via --pretrain_from. "
            "'bc_warmstart': run BC pretraining then immediately start RL from those weights."
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
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Resume training from a checkpoint. "
        "No PATH → finds the latest step_*.pt across all runs in checkpoints/. "
        "PATH can be a .pt file or a run directory (latest step_*.pt in dir is used). "
        "Restores optimizer, scheduler state, ELO, and W&B graph.",
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
        help="Smoke-test mode: tiny training or elo_stats run, with no W&B or compilation.",
    )
    parser.add_argument(
        "--team0",
        type=str,
        default=None,
        metavar="SPEC",
        help="Agent for team 0: null, random, scripted, latest, path/to/checkpoint.pt, "
        "or a named scripted agent (see module docstring). "
        "Defaults: watch→null, collect_stats→scripted.",
    )
    parser.add_argument(
        "--team1",
        type=str,
        default=None,
        metavar="SPEC",
        help="Agent for team 1: null, random, scripted, latest, path/to/checkpoint.pt, "
        "or a named scripted agent (see module docstring). "
        "Defaults: watch→latest, collect_stats→random.",
    )
    parser.add_argument(
        "--fast-cache",
        action="store_true",
        default=False,
        help="(watch mode) Generate obstacle cache headlessly in the background instead of "
        "rendering the convergence animation. Much faster; shows a loading screen instead.",
    )
    parser.add_argument(
        "--matchups",
        nargs="+",
        default=["2v2"],
        help="List of matchups to evaluate sequentially (e.g. 1v1 2v2 3v4 32v32). Defaults to 2v2.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="List of specific agents to evaluate. Overrides default agent discovery in elo_stats.",
    )
    return parser.parse_args()


def _find_resume_checkpoint(
    path_hint: str, checkpoint_dir: str = "checkpoints"
) -> tuple[str, str | None]:
    """Resolve a resume hint to (checkpoint_path, wandb_run_id_or_None).

    path_hint == ""       → find latest step_*.pt across all subdirs of checkpoint_dir
    path_hint ends .pt    → use directly
    path_hint is a dir    → find latest step_*.pt inside it
    """
    hint = Path(path_hint) if path_hint else None

    if hint is None or (hint.exists() and hint.is_dir()):
        search_root = hint if hint is not None else Path(checkpoint_dir)
        candidates = sorted(
            search_root.glob("*/step_*.pt" if hint is None else "step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No step_*.pt checkpoints found under {search_root}")
        ckpt_path = candidates[-1]
    else:
        ckpt_path = hint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_id_file = ckpt_path.parent / "wandb_run_id.txt"
    wandb_run_id = run_id_file.read_text().strip() if run_id_file.exists() else None
    return str(ckpt_path), wandb_run_id


def _apply_smoke(config: TrainConfig) -> TrainConfig:
    """Shrink a TrainConfig to the smallest viable size for crash-testing."""
    from boost_and_broadside.config.schedule import stepped

    # num_envs must be divisible by num_minibatches, so use 1 minibatch with 4 envs.
    scales = tuple(replace(s, num_envs=4) for s in config.scales)
    obstacle_cache = config.obstacle_cache
    if obstacle_cache is not None:
        obstacle_cache = replace(obstacle_cache, num_cache_envs=128, cache_size=4, max_steps=6000)
    schedule = replace(config.schedule, checkpoint_interval=stepped((0, 1)))
    league_eval = replace(
        config.league_eval,
        eval_num_envs=4,
        eval_pairs=4,
        eval_slots=1,
        eval_block_rollouts=2,
        # Blocks hard-reset the eval envs, so episodes must fit in one block:
        # 2 rollouts x 128 steps at 1:1 stride = 2 episodes per block.
        max_episode_steps=128,
    )
    return replace(
        config,
        scales=scales,
        schedule=schedule,
        league_eval=league_eval,
        obstacle_cache=obstacle_cache,
        num_minibatches=1,
        total_timesteps=5_000,
        league_admission_interval=1,
        admission_min_interval=1,
        log_interval=1,
    )


def _make_trainer(
    train_config: TrainConfig,
    args: argparse.Namespace,
    *,
    resume_wandb_run_id: str | None = None,
) -> PPOTrainer:
    """Build a trainer with the shared entry-point settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compile_mode = None if (args.smoke or args.compile_mode == "none") else args.compile_mode
    return PPOTrainer(
        train_config=train_config,
        model_config=MODEL_CONFIG,
        ship_config=SHIP_CONFIG,
        device=device,
        use_wandb=not args.smoke,
        scripted_agent=StochasticScriptedAgent(SHIP_CONFIG, StochasticAgentConfig()),
        compile_mode=compile_mode,
        resume_wandb_run_id=resume_wandb_run_id,
    )


def _run_trainer(trainer: PPOTrainer) -> None:
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        trainer.shutdown()


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    match args.mode:
        case "bc":
            train_config = _apply_smoke(BC_TRAIN_CONFIG) if args.smoke else BC_TRAIN_CONFIG
            _run_trainer(_make_trainer(train_config, args))

        case "rl":
            train_config = _apply_smoke(RL_TRAIN_CONFIG) if args.smoke else RL_TRAIN_CONFIG
            resume_ckpt, resume_wandb_id = (
                _find_resume_checkpoint(args.resume) if args.resume is not None else (None, None)
            )
            trainer = _make_trainer(
                train_config,
                args,
                resume_wandb_run_id=resume_wandb_id,
            )
            if resume_ckpt is not None:
                trainer.load_checkpoint(resume_ckpt)
            elif args.pretrain_from is not None:
                trainer.load_pretrained_weights(args.pretrain_from)
            _run_trainer(trainer)

        case "rl_obstacles":
            train_config = (
                _apply_smoke(RL_OBSTACLES_TRAIN_CONFIG) if args.smoke else RL_OBSTACLES_TRAIN_CONFIG
            )
            resume_ckpt, resume_wandb_id = (
                _find_resume_checkpoint(args.resume) if args.resume is not None else (None, None)
            )
            trainer = _make_trainer(
                train_config,
                args,
                resume_wandb_run_id=resume_wandb_id,
            )
            if resume_ckpt is not None:
                trainer.load_checkpoint(resume_ckpt)
            elif args.pretrain_from is not None:
                trainer.load_pretrained_weights(args.pretrain_from)
            _run_trainer(trainer)

        case "bc_warmstart":
            warmstart_args = argparse.Namespace(**vars(args), smoke=False)

            print("=== BC_WARMSTART: starting BC pretraining phase ===")
            pretrain_trainer = _make_trainer(BC_WARMSTART_PRETRAIN_CONFIG, warmstart_args)
            _run_trainer(pretrain_trainer)

            ckpt_dir = Path(BC_WARMSTART_PRETRAIN_CONFIG.checkpoint_dir) / pretrain_trainer.run_name
            pretrain_path = ckpt_dir / "pretrained_for_rl.pt"
            torch.save(pretrain_trainer.checkpoint_payload(update=0), pretrain_path)
            print(f"=== BC_WARMSTART: pretrained weights saved to {pretrain_path} ===")
            pretrain_trainer.shutdown()
            del pretrain_trainer

            print("=== BC_WARMSTART: starting RL phase ===")
            rl_trainer = _make_trainer(BC_WARMSTART_RL_CONFIG, warmstart_args)
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
                    num_ships=8,
                    max_bullets=20,
                    max_episode_steps=1024,
                    num_obstacles=0,
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
                env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=1024),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                matchups=args.matchups,
            )

        case "feature_stats":
            team0 = args.team0 if args.team0 is not None else "scripted"
            team1 = args.team1 if args.team1 is not None else "scripted"
            run_feature_stats_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_envs=128,
                num_steps=1024,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=1024),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "elo_stats":
            elo_num_envs = 128 if args.smoke else 1024 * 4
            elo_max_steps = 128 if args.smoke else 1024
            run_elo_stats_mode(
                run_spec=args.run,
                num_envs=elo_num_envs,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=4,
                    max_bullets=20,
                    max_episode_steps=elo_max_steps,
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                matchups=args.matchups,
                custom_agents=args.agents,
            )

        case "ar_report":
            team0 = args.team0 if args.team0 is not None else "latest"
            team1 = args.team1 if args.team1 is not None else "latest"

            print("\n" + "=" * 40)
            print("--- Running 2v2 Scenario ---")
            print("=" * 40)
            run_ar_report_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_steps=512,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=512),
                rewards=REWARDS,
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                out_dir="docs/ar_report/2v2",
            )

            print("\n" + "=" * 40)
            print("--- Running 1v1 Scenario ---")
            print("=" * 40)
            run_ar_report_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_steps=512,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(num_ships=2, max_bullets=20, max_episode_steps=512),
                rewards=REWARDS,
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                out_dir="docs/ar_report/1v1",
            )

        case "noise_calibration":
            team0 = (
                args.team0
                if args.team0 is not None
                else "checkpoints/dulcet-dragon-570/recent_avg.pt"
            )
            team1 = (
                args.team1
                if args.team1 is not None
                else "checkpoints/dulcet-dragon-570/recent_avg.pt"
            )
            run_noise_calibration_mode(
                team0_spec=team0,
                team1_spec=team1,
                num_envs=512,
                num_steps=512,
                num_ar_envs=256,
                num_ar_windows=20,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(num_ships=4, max_bullets=20, max_episode_steps=1024),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                output_dir="docs/noise_calibration",
            )


if __name__ == "__main__":
    main()

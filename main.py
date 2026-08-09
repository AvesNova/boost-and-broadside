"""Boost and Broadside — entry point.

Select a mode with --mode. All hyperparameters live in runs/.

Training:
    uv run main.py --mode rl                          # RL from scratch
    uv run main.py --mode rl --smoke                  # crash-test (tiny batch, no W&B)
    uv run main.py --mode rl --no-wandb               # full run, no W&B logging
    uv run main.py --mode rl --compile max-autotune   # RL with max-autotune
    uv run main.py --mode rl --pretrain_from checkpoints/run/best.pt
    uv run main.py --mode rl --resume                 # resume latest checkpoint
    uv run main.py --mode rl --resume checkpoints/run/step_000001000.pt
    uv run main.py --mode rl_fields                   # RL with refractive fields
    uv run main.py --mode bc                          # BC pretraining from scratch
    uv run main.py --mode bc_warmstart                # BC pretrain → RL (one process)

Watch / play:
    uv run main.py --mode play                        # 1v1 player vs null, four fields
    uv run main.py --mode watch                       # human vs latest checkpoint
    uv run main.py --mode watch --team0 null --team1 scripted
    uv run main.py --mode watch --team0 latest --team1 latest

Evaluation:
    uv run main.py --mode collect_stats               # scripted vs random
    uv run main.py --mode collect_stats --team0 latest --team1 scripted
    uv run main.py --mode feature_stats               # label-scale calibration stats
    uv run main.py --mode elo_stats                   # Elo across scripted agents
    uv run main.py --mode elo_stats --run latest      # + checkpoints from latest run
    uv run main.py --mode elo_calibrate --run latest  # post-hoc calibrated Elo + plots
    uv run main.py --mode elo_calibrate --run vague-lion-678 --target-stderr 5
        writes checkpoints/<run>/elo_calibrated.json and elo_calibration/*.png
    uv run main.py --mode elo_scale --run resilient-resonance-682
        rates frozen checkpoints at 1v1 through 64v64 and writes the selected Elo view
    uv run main.py --mode semi_random --run resilient-resonance-682
        evaluates a random-to-scripted action-mixture ladder across fleet sizes
    uv run main.py --mode ar_report                   # autoregressive prediction report
    uv run main.py --mode noise_calibration           # NextStateHead error statistics
    uv run main.py --mode crossover                   # scripted count to beat T trained
    uv run main.py --mode crossover --trained-counts 4,8 --eval-envs 512
        writes docs/crossover/crossover.json and prints a crossover table

Gameplay video (headless mp4 capture of a run's final checkpoint):
    uv run main.py --mode capture                     # 682, self + vs_scripted, seeds 0-7
    uv run main.py --mode capture --run latest --scenarios self --seeds 0-15
    uv run main.py --mode capture --scenarios self --sizes 1v1 4v4 16v16 64v64
        writes <out>/<scenario>_<AvA>_seed<NN>.mp4 (default out: gameplay_clips/)

Agent specs (--team0 / --team1):
    null        human keyboard (watch only)
    random      uniform random actions
    scripted    stochastic scripted agent
    semi_scripted:P  scripted full action with probability P, otherwise random
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
from boost_and_broadside.config.defaults import (
    ELO_CALIBRATE,
    MODEL_CONFIG,
    REWARDS,
    SHIP_CONFIG,
)
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from boost_and_broadside.modes.ar_report import run_ar_report_mode
from boost_and_broadside.modes.capture import run_capture_mode
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.crossover import parse_counts, run_crossover_mode
from boost_and_broadside.modes.elo_calibrate import run_elo_calibrate_mode
from boost_and_broadside.modes.elo_scale import run_elo_scale_mode
from boost_and_broadside.modes.elo_stats import run_elo_stats_mode
from boost_and_broadside.modes.feature_stats import run_feature_stats_mode
from boost_and_broadside.modes.interactive import run_play_mode, run_watch_mode
from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode
from boost_and_broadside.modes.semi_random_tournament import (
    parse_probabilities,
    run_semi_random_tournament,
)
from boost_and_broadside.profiles import (
    BC_TRAIN_CONFIG,
    BC_WARMSTART_PRETRAIN_CONFIG,
    BC_WARMSTART_RL_CONFIG,
    RL_FIELDS_TRAIN_CONFIG,
    RL_TRAIN_CONFIG,
)
from boost_and_broadside.train.rl.policy_io import set_config_drift_allowed
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.ui.renderer import RenderConfig

# Training profiles addressable by name, for modes that need a run's environment
# config before any such run exists.
_TRAIN_PROFILES: dict[str, TrainConfig] = {
    "rl": RL_TRAIN_CONFIG,
    "rl_fields": RL_FIELDS_TRAIN_CONFIG,
    "bc": BC_TRAIN_CONFIG,
}


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
            "rl_fields",
            "bc_warmstart",
            "play",
            "watch",
            "collect_stats",
            "feature_stats",
            "elo_stats",
            "elo_calibrate",
            "elo_scale",
            "semi_random",
            "ar_report",
            "noise_calibration",
            "capture",
            "crossover",
        ],
        default="rl",
        help=(
            "Operating mode. "
            "'bc': BC-only pretraining, no RL gradient. "
            "'rl': RL run, optionally loading pretrained weights via --pretrain_from. "
            "'bc_warmstart': run BC pretraining then immediately start RL from those weights. "
            "'play': control one ship against a null ship in four refractive fields."
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
        "Restores optimizer, scheduler state, Elo, and W&B graph.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="none",
        metavar="RUN",
        help="Run name for Elo modes (e.g. 'bright-cloud-219'), "
        "'latest', or 'none' (scripted agents only, no checkpoints).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="rl",
        choices=sorted(_TRAIN_PROFILES),
        help="(semi_random) Training profile whose environment the reference "
        "ladder is rated under. Rung ratings are a property of the environment "
        "(tick rate, fields, ship config), so a run must use a ladder fitted "
        "under its own profile. Ignored when --run names a finished run.",
    )
    # elo_calibrate defaults live in config/defaults.py (ELO_CALIBRATE); these flags
    # are ad-hoc overrides, so they default to None and fall back to the config.
    parser.add_argument(
        "--target-stderr",
        type=float,
        default=None,
        metavar="ELO",
        help="(Elo calibration modes) Override ELO_CALIBRATE.target_stderr: stop once every "
        "rating is pinned to within this standard error, in Elo points.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        metavar="N",
        help="(Elo calibration modes) Override ELO_CALIBRATE.max_batches, the cap on adaptive "
        "tournament batches.",
    )
    parser.add_argument(
        "--calib-envs",
        type=int,
        default=None,
        metavar="N",
        help="(Elo calibration modes) Override ELO_CALIBRATE.num_envs: parallel envs per batch, "
        "which is also the number of games each batch plays.",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        default=True,
        help="(Elo calibration modes) Skip PNG rendering.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        default=False,
        help="(elo_calibrate) Refit from the stored win/tie matrices in elo_calibrated.json "
        "instead of playing a tournament. Cheap CPU-only path for a change of anchor or "
        "draw convention; requires a previous full calibration.",
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
        "--no-wandb",
        dest="no_wandb",
        action="store_true",
        default=False,
        help="Disable W&B logging for a full run (keeps batch size and compile). "
        "Implied by --smoke.",
    )
    parser.add_argument(
        "--allow-config-drift",
        dest="allow_config_drift",
        action="store_true",
        default=False,
        help="Load checkpoints whose recorded physics constants differ from the current "
        "ShipConfig. Refused by default: those weights were fitted to differently-scaled "
        "inputs, so results are not comparable with a matching run's.",
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
        help="Deprecated no-op retained for watch-mode CLI compatibility; field maps are static.",
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
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["self", "vs_scripted"],
        help="(capture) Match scenarios to record: 'self' (final policy vs itself) and/or "
        "'vs_scripted' (final policy vs the scripted agent).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-7",
        help="(capture) Seeds to record, as a range '0-7' or a list '0,3,9'. One clip per "
        "scenario per size per seed.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="(capture) Team sizes to record zero-shot, e.g. 1v1 2v2 4v4 8v8 16v16 32v32 64v64. "
        "Omit to use the run's native training size.",
    )
    parser.add_argument(
        "--team-counts",
        type=str,
        default="1,2,4,8,16,32,64",
        help="(elo_scale/semi_random) Symmetric ships-per-team sizes: a list '1,2,4' "
        "or ranges '1-8'.",
    )
    parser.add_argument(
        "--scripted-probs",
        type=str,
        default="0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1",
        help="(semi_random) Probability that each ship uses its complete scripted action. "
        "The comma-separated ladder must include 0 and 1.",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=128,
        help="(semi_random) Side-balanced games for every unordered agent pair and fleet size.",
    )
    parser.add_argument(
        "--trained-counts",
        type=str,
        default="1,2,4,8,16,32,64",
        help="(crossover) Trained-team sizes to sweep: a list '1,2,4' or ranges '1-64' (mixable). "
        "Swept ascending so each size warm-starts from the previous crossover.",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=256,
        help="(crossover) Parallel games per matchup used to estimate each win rate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gameplay_clips"),
        help="(capture) Output directory for the mp4 clips.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="(capture) Playback frame rate of the recorded clips.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="(capture) Match length before it's called a tie (steps at 60Hz ≈ seconds×60); "
        "most matches end far sooner when a team is eliminated.",
    )
    parser.add_argument(
        "--hold-ms",
        type=int,
        default=1500,
        help="(capture) Keep playing this many ms after the outcome is decided so the winner "
        "is clear before the clip ends.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        default=False,
        help="(capture) Also write a downscaled .gif beside each .mp4.",
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
    field_map = config.field_map
    if field_map is not None:
        field_map = replace(field_map, cache_size=4, max_generation_attempts=256)
    schedule = replace(config.schedule, checkpoint_interval=stepped((0, 1)))
    elo_eval = replace(config.elo_eval, envs_per_matchup=4)
    return replace(
        config,
        scales=scales,
        schedule=schedule,
        elo_eval=elo_eval,
        field_map=field_map,
        num_minibatches=1,
        total_timesteps=5_000,
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
        use_wandb=not (args.smoke or args.no_wandb),
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


def _run_training_mode(base_config: TrainConfig, args: argparse.Namespace) -> None:
    """Apply --smoke, resolve --resume/--pretrain_from, build the trainer, and run it."""
    train_config = _apply_smoke(base_config) if args.smoke else base_config
    resume_ckpt, resume_wandb_id = (
        _find_resume_checkpoint(args.resume) if args.resume is not None else (None, None)
    )
    trainer = _make_trainer(train_config, args, resume_wandb_run_id=resume_wandb_id)
    if resume_ckpt is not None:
        trainer.load_checkpoint(resume_ckpt)
    elif args.pretrain_from is not None:
        trainer.load_pretrained_weights(args.pretrain_from)
    _run_trainer(trainer)


def main() -> None:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.allow_config_drift:
        set_config_drift_allowed(True)
        print("Config drift allowed: checkpoints may load across a physics change.")

    match args.mode:
        case "bc":
            train_config = _apply_smoke(BC_TRAIN_CONFIG) if args.smoke else BC_TRAIN_CONFIG
            _run_trainer(_make_trainer(train_config, args))

        case "rl":
            _run_training_mode(RL_TRAIN_CONFIG, args)

        case "rl_fields":
            _run_training_mode(RL_FIELDS_TRAIN_CONFIG, args)

        case "bc_warmstart":
            # --smoke must shrink BOTH stages; forcing it off here (as an earlier
            # version did) launched a full multi-day run under a crash-test flag.
            pretrain_config = (
                _apply_smoke(BC_WARMSTART_PRETRAIN_CONFIG)
                if args.smoke
                else BC_WARMSTART_PRETRAIN_CONFIG
            )
            rl_config = (
                _apply_smoke(BC_WARMSTART_RL_CONFIG) if args.smoke else BC_WARMSTART_RL_CONFIG
            )

            print("=== BC_WARMSTART: starting BC pretraining phase ===")
            pretrain_trainer = _make_trainer(pretrain_config, args)
            _run_trainer(pretrain_trainer)

            ckpt_dir = Path(pretrain_config.checkpoint_dir) / pretrain_trainer.run_name
            pretrain_path = ckpt_dir / "pretrained_for_rl.pt"
            torch.save(pretrain_trainer.checkpoint_payload(update=0), pretrain_path)
            print(f"=== BC_WARMSTART: pretrained weights saved to {pretrain_path} ===")
            pretrain_trainer.shutdown()
            del pretrain_trainer

            print("=== BC_WARMSTART: starting RL phase ===")
            rl_trainer = _make_trainer(rl_config, args)
            rl_trainer.load_pretrained_weights(str(pretrain_path))
            _run_trainer(rl_trainer)

        case "play":
            run_play_mode(
                ship_config=SHIP_CONFIG,
                rewards=REWARDS,
                model_config=MODEL_CONFIG,
                render_config=RenderConfig(),
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "watch":
            team0 = args.team0 if args.team0 is not None else "null"
            team1 = args.team1 if args.team1 is not None else "latest"
            run_watch_mode(
                team0_spec=team0,
                team1_spec=team1,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=8,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=1024,
                    num_fields=4,
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
                    num_ships=4,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=1024,
                ),
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
                env_config=EnvConfig(
                    num_ships=4,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=1024,
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
            )

        case "elo_stats":
            run_elo_stats_mode(
                run_spec=args.run,
                num_envs=1024 * 4,
                ship_config=SHIP_CONFIG,
                env_config=EnvConfig(
                    num_ships=4,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=1024,
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                elo_k_factor=32.0,
                matchups=args.matchups,
                custom_agents=args.agents,
            )

        case "elo_calibrate":
            calibrate_config = replace(
                ELO_CALIBRATE,
                **{
                    field: value
                    for field, value in (
                        ("num_envs", args.calib_envs),
                        ("target_stderr", args.target_stderr),
                        ("max_batches", args.max_batches),
                    )
                    if value is not None
                },
            )
            run_elo_calibrate_mode(
                run_spec=args.run if args.run != "none" else "latest",
                ship_config=SHIP_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                config=calibrate_config,
                plot=args.plots,
                refit=args.refit,
            )

        case "elo_scale":
            calibrate_config = replace(
                ELO_CALIBRATE,
                **{
                    field: value
                    for field, value in (
                        ("num_envs", args.calib_envs),
                        ("target_stderr", args.target_stderr),
                        ("max_batches", args.max_batches),
                    )
                    if value is not None
                },
            )
            run_elo_scale_mode(
                run_spec=(args.run if args.run != "none" else "resilient-resonance-682"),
                team_sizes=parse_counts(args.team_counts),
                ship_config=SHIP_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                config=calibrate_config,
                plot=args.plots,
            )

        case "semi_random":
            # Rungs are rated under a *profile* by default: their ratings are a
            # property of the environment they play in, and the run that will use
            # them as fixed references need not exist yet. --run overrides, to
            # re-rate the ladder a finished run actually trained against.
            profile_config = None if args.run != "none" else _TRAIN_PROFILES[args.profile]
            run_semi_random_tournament(
                run_spec=(args.profile if profile_config is not None else args.run),
                train_config=profile_config,
                team_sizes=parse_counts(args.team_counts),
                probabilities=parse_probabilities(args.scripted_probs),
                games_per_pair=args.games_per_pair,
                max_parallel_envs=(
                    args.calib_envs if args.calib_envs is not None else ELO_CALIBRATE.num_envs
                ),
                ship_config=SHIP_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                plot=args.plots,
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
                env_config=EnvConfig(
                    num_ships=4,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=512,
                ),
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
                env_config=EnvConfig(
                    num_ships=2,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=512,
                ),
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
                env_config=EnvConfig(
                    num_ships=4,
                    max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
                    max_episode_steps=1024,
                ),
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                output_dir="docs/noise_calibration",
            )

        case "capture":
            run_capture_mode(
                run_spec=args.run if args.run != "none" else "resilient-resonance-682",
                scenarios=args.scenarios,
                seeds=args.seeds,
                ship_config=SHIP_CONFIG,
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                out_dir=args.out,
                sizes=args.sizes,
                fps=args.fps,
                max_steps=args.max_steps,
                hold_ms=args.hold_ms,
                gif=args.gif,
            )

        case "crossover":
            run_crossover_mode(
                run_spec=args.run if args.run != "none" else "resilient-resonance-682",
                trained_counts=parse_counts(args.trained_counts),
                ship_config=SHIP_CONFIG,
                model_config=MODEL_CONFIG,
                device=device,
                checkpoint_dir="checkpoints",
                num_envs=args.eval_envs,
            )


if __name__ == "__main__":
    main()

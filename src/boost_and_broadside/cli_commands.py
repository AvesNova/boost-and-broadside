"""Runtime adapters for the lightweight :mod:`boost_and_broadside.cli` parser."""

from __future__ import annotations

import argparse
from dataclasses import replace

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.config.defaults import ELO_CALIBRATE, MODEL_CONFIG, REWARDS, SHIP_CONFIG
from boost_and_broadside.config.resolve import LaunchOverrides
from boost_and_broadside.config.schema import ResolvedTrainConfig
from boost_and_broadside.config.service import resolved_profile_document
from boost_and_broadside.constants import DEFAULT_MAX_BULLETS_PER_SHIP
from boost_and_broadside.evaluation.run_catalog import (
    resolve_exact_run,
    resolve_explicit_checkpoint,
    select_latest_resumable_checkpoint,
)
from boost_and_broadside.execution import initialize_execution, resolve_execution_settings
from boost_and_broadside.modes.ar_report import run_canonical_ar_report_mode
from boost_and_broadside.modes.capture import run_capture_mode
from boost_and_broadside.modes.collect import run_collect_stats_mode
from boost_and_broadside.modes.crossover import parse_counts, run_crossover_mode
from boost_and_broadside.modes.elo_calibrate import run_elo_calibrate_mode
from boost_and_broadside.modes.elo_scale import run_elo_scale_mode
from boost_and_broadside.modes.feature_stats import run_feature_stats_mode
from boost_and_broadside.modes.interactive import run_play_mode, run_watch_mode
from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode
from boost_and_broadside.modes.semi_random_tournament import (
    parse_probabilities,
    run_semi_random_tournament,
)
from boost_and_broadside.profiles import resolve_named_profile
from boost_and_broadside.train.rl.policy_io import set_config_drift_allowed
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.ui.renderer import RenderConfig


def _prepare_execution(args: argparse.Namespace) -> str:
    requested_compile = getattr(args, "compile_mode", None)
    compile_mode = None if requested_compile == "none" else requested_compile
    settings = resolve_execution_settings(
        device=args.device,
        seed=args.seed,
        compile_mode=compile_mode,
        wandb=not getattr(args, "no_wandb", False),
        allow_config_drift=getattr(args, "allow_config_drift", False),
    )
    args.seed = settings.seed
    initialize_execution(settings)
    set_config_drift_allowed(getattr(args, "allow_config_drift", False))
    return settings.device


def _calibration_config(args: argparse.Namespace):
    return replace(
        ELO_CALIBRATE,
        **{
            field: value
            for field, value in (
                ("num_envs", args.games_per_batch),
                ("target_stderr", args.target_stderr),
                ("max_batches", args.max_batches),
            )
            if value is not None
        },
    )


def _resume_checkpoint(subject: str, checkpoint_dir: str = "checkpoints") -> tuple[str, str | None]:
    if subject.endswith(".pt"):
        checkpoint = resolve_explicit_checkpoint(subject).path
    else:
        run = resolve_exact_run(subject, checkpoint_dir)
        checkpoint = select_latest_resumable_checkpoint(run).path
    run_id_path = checkpoint.parent / "wandb_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.is_file() else None
    return str(checkpoint), run_id


def _make_trainer(
    resolved: ResolvedTrainConfig,
    args: argparse.Namespace,
    device: str,
    *,
    resume_wandb_run_id: str | None = None,
) -> PPOTrainer:
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    return PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=device,
        use_wandb=not args.no_wandb,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=compile_mode,
        resume_wandb_run_id=resume_wandb_run_id,
        resolved_config_document=resolved_profile_document(resolved),
        launch_provenance={
            "device": device,
            "seed": args.seed,
            "compile_mode": compile_mode,
            "wandb": not args.no_wandb,
            "allow_config_drift": args.allow_config_drift,
        },
    )


def _run_trainer(trainer: PPOTrainer) -> None:
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        trainer.shutdown()


def _train(args: argparse.Namespace) -> None:
    resume_path, run_id = (
        _resume_checkpoint(args.resume) if args.resume is not None else (None, None)
    )
    pretrain_path = (
        str(resolve_explicit_checkpoint(args.pretrain_from).path)
        if args.pretrain_from is not None
        else None
    )
    resolved = resolve_named_profile(
        args.profile,
        LaunchOverrides(num_envs=args.num_envs, microbatch_tokens=args.microbatch_tokens),
    )
    device = _prepare_execution(args)
    trainer = _make_trainer(resolved, args, device, resume_wandb_run_id=run_id)
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    elif pretrain_path is not None:
        trainer.load_pretrained_weights(pretrain_path)
    _run_trainer(trainer)


def _play(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_play_mode(
        ship_config=SHIP_CONFIG,
        rewards=REWARDS,
        model_config=MODEL_CONFIG,
        render_config=RenderConfig(),
        device=device,
        checkpoint_dir="checkpoints",
    )


def _watch(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_watch_mode(
        team0_spec=args.team0,
        team1_spec=args.team1,
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
    )


def _capture(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_capture_mode(
        run_spec=args.run,
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


def _collect_stats(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_collect_stats_mode(
        team0_spec=args.team0,
        team1_spec=args.team1,
        num_envs=args.games,
        ship_config=SHIP_CONFIG,
        env_config=EnvConfig(
            num_ships=8,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            max_episode_steps=1024,
        ),
        model_config=MODEL_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        matchups=args.sizes,
    )


def _crossover(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_crossover_mode(
        run_spec=args.run,
        trained_counts=parse_counts(args.sizes),
        ship_config=SHIP_CONFIG,
        model_config=MODEL_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        num_envs=args.games_per_matchup,
    )


def _elo_calibrate(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_elo_calibrate_mode(
        run_spec=args.run,
        agent_specs=args.agents,
        ship_config=SHIP_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        config=_calibration_config(args),
        model_config=MODEL_CONFIG,
        env_config=EnvConfig(
            num_ships=8,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            max_episode_steps=1024,
        ),
        plot=True,
        refit=False,
    )


def _elo_scale(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_elo_scale_mode(
        run_spec=args.run,
        team_sizes=parse_counts(args.sizes),
        ship_config=SHIP_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        config=_calibration_config(args),
        plot=True,
    )


def _semi_random(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    resolved = resolve_named_profile(args.profile) if args.profile is not None else None
    run_spec = args.profile if resolved is not None else args.run
    run_semi_random_tournament(
        run_spec=run_spec,
        train_config=None if resolved is None else resolved.train_config,
        team_sizes=parse_counts(args.sizes),
        probabilities=parse_probabilities(args.scripted_probabilities),
        games_per_pair=args.games_per_pair,
        max_parallel_envs=(
            args.max_parallel_games
            if args.max_parallel_games is not None
            else ELO_CALIBRATE.num_envs
        ),
        ship_config=SHIP_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        plot=True,
    )


def _ar_report(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_canonical_ar_report_mode(
        team0_spec=args.team0,
        team1_spec=args.team1,
        num_steps=args.decision_steps,
        ship_config=SHIP_CONFIG,
        rewards=REWARDS,
        model_config=MODEL_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
    )


def _noise_calibration(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_noise_calibration_mode(
        team0_spec=args.team0,
        team1_spec=args.team1,
        num_envs=512,
        num_steps=512,
        num_ar_envs=256,
        num_ar_windows=20,
        ship_config=SHIP_CONFIG,
        env_config=EnvConfig(
            num_ships=8,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            max_episode_steps=1024,
        ),
        model_config=MODEL_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
        output_dir="docs/noise_calibration",
    )


def _feature_stats(args: argparse.Namespace) -> None:
    device = _prepare_execution(args)
    run_feature_stats_mode(
        team0_spec=args.team0,
        team1_spec=args.team1,
        num_envs=args.games,
        num_steps=args.decision_steps,
        ship_config=SHIP_CONFIG,
        env_config=EnvConfig(
            num_ships=8,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            max_episode_steps=1024,
        ),
        model_config=MODEL_CONFIG,
        device=device,
        checkpoint_dir="checkpoints",
    )


def _smoke(args: argparse.Namespace) -> None:
    from boost_and_broadside.smoke import run_smoke_matrix

    run_smoke_matrix(args.case)


_HANDLERS = {
    "train": _train,
    "play": _play,
    "watch": _watch,
    "capture": _capture,
    "collect-stats": _collect_stats,
    "crossover": _crossover,
    "elo-calibrate": _elo_calibrate,
    "elo-scale": _elo_scale,
    "semi-random": _semi_random,
    "ar-report": _ar_report,
    "noise-calibration": _noise_calibration,
    "feature-stats": _feature_stats,
}


def runtime_command_names() -> tuple[str, ...]:
    """Commands that own runtime smoke cases (excluding orchestration/publication)."""

    return tuple(_HANDLERS)


def execute(command: str, args: argparse.Namespace) -> None:
    """Execute one completely parsed command."""
    if command == "smoke":
        _smoke(args)
        return
    if command == "publish":
        raise ValueError("bnb publish is registered but unavailable until S09")
    try:
        handler = _HANDLERS[command]
    except KeyError as error:
        raise ValueError(f"no handler registered for command {command!r}") from error
    handler(args)

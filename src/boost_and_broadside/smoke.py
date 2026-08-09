"""Hermetic, sequential subprocess smoke coverage for every runtime mode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO
from unittest.mock import patch

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import EnvConfig
from boost_and_broadside.config.defaults import ELO_CALIBRATE, REWARDS
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.config.schema import LaunchSizingSpec, ResolvedTrainConfig
from boost_and_broadside.config.service import resolved_profile_document
from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES, build_reward_components
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.buffer import AdvantageScaler, ReturnScaler
from boost_and_broadside.train.rl.checkpoint import (
    build_policy_checkpoint_payload,
    build_training_checkpoint_payload,
    clone_to_cpu,
    write_checkpoint_payload,
)
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.roster import EloRoster

_RUN_NAME = "synthetic-run"
_MANAGED_ROOT_NAMES = frozenset({"checkpoints", "artifacts", "out", "tmp"})


@dataclass(frozen=True)
class SmokeCase:
    """One registry-owned bounded runtime invocation."""

    name: str
    command: str
    profile: str | None = None
    timeout_seconds: int = 90


SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase("train-rl", "train", "rl", 120),
    SmokeCase("train-rl-fields", "train", "rl-fields", 180),
    SmokeCase("train-bc", "train", "bc", 120),
    SmokeCase("play", "play"),
    SmokeCase("watch", "watch"),
    SmokeCase("capture", "capture", timeout_seconds=120),
    SmokeCase("collect-stats", "collect-stats"),
    SmokeCase("crossover", "crossover", timeout_seconds=120),
    SmokeCase("elo-calibrate", "elo-calibrate", timeout_seconds=120),
    SmokeCase("elo-scale", "elo-scale", timeout_seconds=120),
    SmokeCase("semi-random", "semi-random"),
    SmokeCase("ar-report", "ar-report", timeout_seconds=120),
    SmokeCase("noise-calibration", "noise-calibration", timeout_seconds=120),
    SmokeCase("feature-stats", "feature-stats"),
)
_CASES_BY_NAME = {case.name: case for case in SMOKE_CASES}


class SmokeError(RuntimeError):
    """Base error for the smoke system."""


class SmokeIsolationError(SmokeError):
    """A case wrote outside its managed root layout or changed the checkout."""


class SmokeCaseError(SmokeError):
    """One or more case subprocesses failed."""


@dataclass(frozen=True)
class SmokeRoots:
    root: Path
    checkpoints: Path
    artifacts: Path
    out: Path
    tmp: Path

    @classmethod
    def create(cls, root: str | Path) -> SmokeRoots:
        base = Path(root).resolve()
        base.mkdir(parents=True, exist_ok=True)
        roots = cls(
            root=base,
            checkpoints=base / "checkpoints",
            artifacts=base / "artifacts",
            out=base / "out",
            tmp=base / "tmp",
        )
        for path in (roots.checkpoints, roots.artifacts, roots.out, roots.tmp):
            path.mkdir(parents=True, exist_ok=True)
        return roots


@dataclass(frozen=True)
class SyntheticRun:
    run_dir: Path
    checkpoint: Path
    roster: Path
    elo_history: Path
    resolved: ResolvedTrainConfig


def _smoke_resolved_profile(
    profile_name: str,
    checkpoint_root: Path,
    *,
    resolved_name: str | None = None,
) -> ResolvedTrainConfig:
    """Resolve a registry profile's objective under a fixed one-update smoke shape."""

    base = PROFILES[profile_name]
    num_fields = base.environment.num_fields
    entity_tokens = 2 + num_fields
    num_steps = 2
    environment = replace(
        base.environment,
        num_ships=2,
        max_bullets=2,
        max_episode_steps=2,
    )
    rollout = replace(
        base.rollout,
        logical_batch_tokens=entity_tokens * num_steps,
        num_steps=num_steps,
        num_minibatches=1,
    )
    launch = LaunchSizingSpec(num_envs=1)
    optimizer = replace(
        base.optimizer,
        total_timesteps=num_steps,
        checkpoint_dir=str(checkpoint_root),
        histogram_interval=100,
        log_interval=1,
    )
    league = replace(
        base.league,
        league_slots=1,
        reference_ladder=(),
        elo_milestone_gap=0.0,
        elo_eval=replace(
            base.league.elo_eval,
            envs_per_matchup=1,
            step_interval=1,
            window_size=2,
            min_games_to_freeze=0,
        ),
    )
    field_map = (
        replace(base.field_map, cache_size=1, max_generation_attempts=256)
        if base.field_map is not None
        else None
    )
    spec = replace(
        base,
        name=resolved_name or f"smoke-{profile_name}",
        environment=environment,
        rollout=rollout,
        launch_defaults=launch,
        optimizer=optimizer,
        league=league,
        field_map=field_map,
    )
    return resolve_profile(spec)


def _active_value_layout(resolved: ResolvedTrainConfig) -> tuple[int, tuple[int, ...]]:
    components = build_reward_components(resolved.train_config.rewards, resolved.ship_config)
    by_name = {component.name: component for component in components}
    active = [
        name
        for name in REWARD_COMPONENT_NAMES
        if name in by_name and by_name[name].weight != 0
    ]
    team_pma_k = tuple(
        index for index, name in enumerate(active) if name in {"ally_win", "enemy_win"}
    )
    return len(active), team_pma_k


def build_synthetic_run(
    checkpoint_root: str | Path,
    *,
    seed: int = 7,
    run_name: str = _RUN_NAME,
) -> SyntheticRun:
    """Create the smallest current-schema run through production serializers."""

    root = Path(checkpoint_root).resolve()
    resolved = _smoke_resolved_profile("rl", root, resolved_name="smoke-fixture")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    num_components, team_pma_k = _active_value_layout(resolved)
    policy = build_policy(
        resolved.model_config,
        resolved.ship_config,
        num_value_components=num_components,
        num_ships=resolved.env_config.num_ships,
        team_pma_k=team_pma_k,
    )
    policy_payload = build_policy_checkpoint_payload(
        policy_state_dict=policy.state_dict(),
        num_value_components=num_components,
        team_pma_k=team_pma_k,
        global_step=1,
        training_elo=0.0,
        model_config=resolved.model_config,
        env_config=resolved.env_config,
        ship_config=resolved.ship_config,
        paradigm=resolved.train_config.paradigm,
        resolved_config=resolved_profile_document(resolved),
        launch={
            "device": "cpu",
            "seed": seed,
            "compile_mode": None,
            "wandb": False,
            "allow_config_drift": False,
        },
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    scaler = ReturnScaler(
        num_components,
        torch.device("cpu"),
        ema_alpha=resolved.train_config.return_ema_alpha,
        min_span=resolved.train_config.return_min_span,
    )
    adv_scaler = AdvantageScaler(
        num_components,
        torch.device("cpu"),
        min_rms=resolved.train_config.advantage_min_rms,
    )
    payload = build_training_checkpoint_payload(
        policy_payload=policy_payload,
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        adv_scaler_state_dict=adv_scaler.state_dict(),
        avg_policy_state_dict=policy.state_dict(),
        avg_param_cumsum=[torch.zeros_like(parameter) for parameter in policy.parameters()],
        avg_update_count=0,
        update=0,
        ship_steps=0,
        grad_tokens=0,
        elapsed_train_time=0.0,
        avg_training_elo=0.0,
        scripted_elo=resolved.train_config.elo_eval.scripted_elo_init,
        floating_games=0,
        eval_window_rand=[],
        eval_window_sc=[],
        eval_window_ladder=[],
        eval_window_floating=[],
        eval_window_live_vs_avg=[],
        elo_milestone=0.0,
        train_config=resolved.train_config,
    )
    checkpoint = write_checkpoint_payload(
        run_dir / "step_000000000001.pt",
        clone_to_cpu(payload),
    )
    ladder_checkpoint = write_checkpoint_payload(
        run_dir / "ladder_step_000000000001.pt",
        clone_to_cpu(payload),
    )

    roster_model = EloRoster(random_elo=resolved.train_config.random_elo)
    roster_model.add_special(
        "scripted",
        global_step=0,
        update=0,
        initial_elo=resolved.train_config.elo_eval.scripted_elo_init,
    )
    roster_model.add_checkpoint(
        str(ladder_checkpoint),
        global_step=1,
        update=0,
        initial_elo=0.0,
    )
    roster = run_dir / "roster.json"
    roster_model.save_json(roster)

    elo_history = run_dir / "elo_history.jsonl"
    record = {
        "update": 0,
        "global_step": 1,
        "live": 0.0,
        "avg": 0.0,
        "scripted": resolved.train_config.elo_eval.scripted_elo_init,
        "counts": {"random": [0, 0, 1], "scripted": [0, 0, 1]},
        "entries": [
            {"label": entry.label, "elo": entry.elo, "fixed": entry.fixed}
            for entry in roster_model.entries
        ],
    }
    elo_history.write_text(json.dumps(record, sort_keys=True) + "\n")
    return SyntheticRun(run_dir, checkpoint, roster, elo_history, resolved)


def validate_case_root(root: str | Path) -> None:
    """Reject any write outside the four registry-managed case roots."""

    path = Path(root)
    unexpected = sorted(
        child.name for child in path.iterdir() if child.name not in _MANAGED_ROOT_NAMES
    )
    if unexpected:
        raise SmokeIsolationError(
            f"smoke case wrote outside managed roots in {path}: {', '.join(unexpected)}"
        )
    rendered = sorted(
        str(candidate.relative_to(path))
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".md", ".png"}
    )
    if rendered:
        raise SmokeIsolationError(
            "smoke cases must disable report/publication rendering: " + ", ".join(rendered)
        )


def _tiny_elo_config():
    return replace(
        ELO_CALIBRATE,
        num_envs=2,
        target_stderr=1_000_000.0,
        max_batches=1,
        reference_probabilities=(),
        plot_decisive=False,
    )


def _basic_env(*, num_fields: int = 0) -> EnvConfig:
    return EnvConfig(
        num_ships=2,
        num_fields=num_fields,
        max_bullets=2,
        max_episode_steps=2,
    )


def _run_training_case(case: SmokeCase, roots: SmokeRoots) -> None:
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    assert case.profile is not None
    resolved = _smoke_resolved_profile(case.profile, roots.checkpoints)
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(
            resolved.ship_config,
            StochasticAgentConfig(),
        ),
        compile_mode=None,
        resolved_config_document=resolved_profile_document(resolved),
        launch_provenance={
            "device": "cpu",
            "seed": 7,
            "compile_mode": None,
            "wandb": False,
            "allow_config_drift": False,
        },
    )
    trainer.train()


def _queue_quit_event() -> None:
    import pygame

    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))


def _run_mode_case(case: SmokeCase, roots: SmokeRoots) -> None:
    if case.command == "play":
        from boost_and_broadside.modes.interactive import run_play_mode
        from boost_and_broadside.ui.renderer import RenderConfig

        resolved = _smoke_resolved_profile("rl", roots.checkpoints)
        _queue_quit_event()
        run_play_mode(
            ship_config=resolved.ship_config,
            rewards=REWARDS,
            model_config=resolved.model_config,
            render_config=RenderConfig(window_size=64, fps=1),
            device="cpu",
            checkpoint_dir=str(roots.checkpoints),
        )
        return
    if case.command == "watch":
        from boost_and_broadside.modes.interactive import run_watch_mode
        from boost_and_broadside.ui.renderer import RenderConfig

        resolved = _smoke_resolved_profile("rl", roots.checkpoints)
        _queue_quit_event()
        run_watch_mode(
            team0_spec="scripted",
            team1_spec="random",
            ship_config=resolved.ship_config,
            env_config=_basic_env(),
            rewards=REWARDS,
            model_config=resolved.model_config,
            render_config=RenderConfig(window_size=64, fps=1),
            device="cpu",
            checkpoint_dir=str(roots.checkpoints),
        )
        return

    fixture = build_synthetic_run(roots.checkpoints)
    checkpoint = str(fixture.checkpoint)
    ship_config = fixture.resolved.ship_config
    model_config = fixture.resolved.model_config
    checkpoint_root = str(roots.checkpoints)

    if case.command == "capture":
        from boost_and_broadside.modes.capture import run_capture_mode

        run_capture_mode(
            run_spec=_RUN_NAME,
            scenarios=["vs_scripted"],
            seeds="0",
            ship_config=ship_config,
            model_config=model_config,
            device="cpu",
            checkpoint_dir=checkpoint_root,
            out_dir=roots.out,
            sizes=["1v1"],
            fps=1,
            max_steps=1,
            window=64,
            hold_ms=1,
        )
    elif case.command == "collect-stats":
        from boost_and_broadside.modes.collect import run_collect_stats_mode

        run_collect_stats_mode(
            "scripted",
            "random",
            1,
            ship_config,
            _basic_env(),
            model_config,
            "cpu",
            checkpoint_root,
            ["1v1"],
        )
    elif case.command == "crossover":
        from boost_and_broadside.modes.crossover import run_crossover_mode

        run_crossover_mode(
            _RUN_NAME,
            [1],
            ship_config,
            model_config,
            "cpu",
            checkpoint_root,
            num_envs=1,
            max_total_ships=2,
            output_dir=str(roots.artifacts / "crossover"),
        )
    elif case.command == "elo-calibrate":
        from boost_and_broadside.modes.elo_calibrate import run_elo_calibrate_mode

        run_elo_calibrate_mode(
            _RUN_NAME,
            ship_config,
            "cpu",
            _tiny_elo_config(),
            checkpoint_root,
            plot=False,
        )
    elif case.command == "elo-scale":
        from boost_and_broadside.modes.elo_scale import run_elo_scale_mode

        run_elo_scale_mode(
            _RUN_NAME,
            [1],
            ship_config,
            "cpu",
            _tiny_elo_config(),
            checkpoint_root,
            plot_dir=str(roots.artifacts / "elo-scale"),
            plot=False,
        )
    elif case.command == "semi-random":
        from boost_and_broadside.modes.semi_random_tournament import (
            run_semi_random_tournament,
        )

        run_semi_random_tournament(
            _RUN_NAME,
            [1],
            [0.0, 1.0],
            1,
            1,
            ship_config,
            "cpu",
            checkpoint_root,
            plot_dir=str(roots.artifacts / "semi-random"),
            plot=False,
        )
    elif case.command == "ar-report":
        from boost_and_broadside.modes.ar_report import run_ar_report_mode

        with patch("boost_and_broadside.modes.ar_report._generate_report"):
            run_ar_report_mode(
                checkpoint,
                checkpoint,
                2,
                ship_config,
                _basic_env(),
                REWARDS,
                model_config,
                "cpu",
                checkpoint_root,
                str(roots.artifacts / "ar-report"),
            )
    elif case.command == "noise-calibration":
        from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode

        with patch("boost_and_broadside.modes.noise_calibration._write_outputs"):
            run_noise_calibration_mode(
                checkpoint,
                "scripted",
                1,
                2,
                1,
                1,
                ship_config,
                _basic_env(),
                model_config,
                "cpu",
                checkpoint_root,
                str(roots.artifacts / "noise-calibration"),
            )
    elif case.command == "feature-stats":
        from boost_and_broadside.modes.feature_stats import run_feature_stats_mode

        run_feature_stats_mode(
            checkpoint,
            "scripted",
            1,
            2,
            ship_config,
            _basic_env(),
            model_config,
            "cpu",
            checkpoint_root,
        )
    else:  # pragma: no cover - registry coverage makes this unreachable
        raise SmokeError(f"no smoke implementation for command {case.command!r}")


def execute_case(case_name: str, root: str | Path) -> None:
    """Execute one case inside an already isolated child process."""

    try:
        case = _CASES_BY_NAME[case_name]
    except KeyError as error:
        raise SmokeError(f"unknown smoke case {case_name!r}") from error
    roots = SmokeRoots.create(root)
    os.chdir(roots.root)
    torch.manual_seed(7)
    if case.command == "train":
        _run_training_case(case, roots)
    else:
        _run_mode_case(case, roots)
    validate_case_root(roots.root)


def _case_environment(roots: SmokeRoots) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HEADLESS": "1",
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(roots.tmp / "matplotlib"),
            "XDG_CACHE_HOME": str(roots.tmp / "cache"),
            "TMPDIR": str(roots.tmp),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def run_case_subprocess(
    case: SmokeCase | str,
    root: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run one registry case in a fresh interpreter with a fixed timeout."""

    selected = _CASES_BY_NAME[case] if isinstance(case, str) else case
    roots = SmokeRoots.create(root)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "boost_and_broadside.smoke",
            "--run-case",
            selected.name,
            "--root",
            str(roots.root),
        ],
        cwd=roots.root,
        env=_case_environment(roots),
        capture_output=True,
        text=True,
        timeout=selected.timeout_seconds,
        check=False,
    )


def _checkout_snapshot(repository: Path) -> bytes:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    return status + b"\0DIFF\0" + diff


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SmokeIsolationError("bnb smoke must run from inside a Git checkout")
    return Path(result.stdout.strip()).resolve()


def _failure_tail(result: subprocess.CompletedProcess[str], lines: int = 12) -> str:
    output = (result.stdout + "\n" + result.stderr).strip().splitlines()
    return "\n".join(output[-lines:])


def run_smoke_matrix(selected_case: str | None = None, *, file: TextIO | None = None) -> None:
    """Run the full sequential matrix, or one focused diagnostic case."""

    stream = file or sys.stdout
    if selected_case is None:
        cases = SMOKE_CASES
    else:
        try:
            cases = (_CASES_BY_NAME[selected_case],)
        except KeyError as error:
            choices = ", ".join(_CASES_BY_NAME)
            raise ValueError(
                f"unknown smoke case {selected_case!r}; choose from {choices}"
            ) from error

    repository = _repository_root()
    checkout_before = _checkout_snapshot(repository)
    failures: list[str] = []
    print(f"bnb smoke: {len(cases)} isolated case(s), sequential", file=stream, flush=True)
    with tempfile.TemporaryDirectory(prefix="bnb-smoke-") as temporary:
        matrix_root = Path(temporary)
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.name} ... ", end="", file=stream, flush=True)
            case_root = matrix_root / case.name
            try:
                result = run_case_subprocess(case, case_root)
                validate_case_root(case_root)
            except (SmokeIsolationError, subprocess.TimeoutExpired) as error:
                failures.append(f"{case.name}: {error}")
                print("FAIL", file=stream, flush=True)
                continue
            checkout_after = _checkout_snapshot(repository)
            if checkout_after != checkout_before:
                raise SmokeIsolationError(f"case {case.name!r} changed the source checkout")
            if result.returncode:
                failures.append(
                    f"{case.name}: exit {result.returncode}\n{_failure_tail(result)}"
                )
                print("FAIL", file=stream, flush=True)
            else:
                print("PASS", file=stream, flush=True)

    if failures:
        raise SmokeCaseError("smoke matrix failed:\n" + "\n\n".join(failures))
    print(f"bnb smoke: all {len(cases)} case(s) passed; checkout unchanged", file=stream)


def _module_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal child runner for bnb smoke")
    parser.add_argument("--run-case", required=True, choices=tuple(_CASES_BY_NAME))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    execute_case(args.run_case, args.root)
    print(f"SMOKE PASS {args.run_case}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(_module_main())

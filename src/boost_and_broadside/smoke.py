"""Hermetic, sequential subprocess smoke coverage for every runtime mode."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO
from unittest.mock import patch

import torch

from boost_and_broadside.config import EnvConfig
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
    # Gradient-diagnostic level this training case launches at. Off is the
    # default every non-diagnostic case runs, and the one shipping runs use.
    gradient_diagnostics: str = "off"


SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase("train-rl", "train", "rl", 180),
    SmokeCase("train-bc", "train", "bc", 180),
    # One case per diagnostic level, because each level adds a distinct code
    # path: decomposing nothing, the loss terms, the policy by reward, and the
    # critic by reward. What is being exercised is the decomposition, not the
    # environment, so they all run the one training profile.
    SmokeCase("train-grad-top-level", "train", "rl", 210, gradient_diagnostics="top_level"),
    SmokeCase("train-grad-reward-policy", "train", "rl", 240, gradient_diagnostics="reward_policy"),
    SmokeCase("train-grad-reward-full", "train", "rl", 300, gradient_diagnostics="reward_full"),
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
    ladder_checkpoint: Path
    roster: Path
    elo_history: Path
    resolved: ResolvedTrainConfig


def _smoke_resolved_profile(
    profile_name: str,
    checkpoint_root: Path,
    *,
    resolved_name: str | None = None,
    num_fields: int | None = None,
) -> ResolvedTrainConfig:
    """Resolve a registry profile's objective under a fixed one-update smoke shape.

    ``num_fields`` overrides the profile's field count. It is a parameter rather
    than a second profile because the network does not change with it -- zero
    fields is the configuration run 682 trained under, and keeping a fixture at
    that width is how the field-free path stays exercised.
    """

    base = PROFILES[profile_name]
    if num_fields is None:
        num_fields = base.environment.num_fields
    entity_tokens = 2 + num_fields
    num_steps = 2
    environment = replace(
        base.environment,
        num_ships=2,
        num_fields=num_fields,
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
        live_reference_probabilities=(),
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
        if num_fields and base.field_map is not None
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
    profile: str = "rl",
    num_fields: int | None = None,
) -> SyntheticRun:
    """Create the smallest current-schema run through production serializers.

    ``num_fields`` overrides the profile's field count, and it is what the
    evaluation modes need to be varied across: a field-free fixture cannot catch
    a mode that fails to read a run's field distribution, and a fielded one
    cannot catch a mode that has stopped handling the width run 682 trained at.
    """

    root = Path(checkpoint_root).resolve()
    resolved = _smoke_resolved_profile(
        profile, root, resolved_name="smoke-fixture", num_fields=num_fields
    )
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
        live_elo=0.0,
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
        avg_live_elo=0.0,
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
    ladder_payload = {**policy_payload, "global_step": 0}
    ladder_checkpoint = write_checkpoint_payload(
        run_dir / "ladder_step_000000000000.pt",
        clone_to_cpu(ladder_payload),
    )

    roster_model = EloRoster()
    roster_model.add_special(
        "scripted",
        global_step=0,
        update=0,
        initial_elo=resolved.train_config.elo_eval.scripted_live_elo,
    )
    roster_model.add_checkpoint(
        str(ladder_checkpoint),
        global_step=0,
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
        "scripted": resolved.train_config.elo_eval.scripted_live_elo,
        "counts": {"random": [0, 0, 1], "scripted": [0, 0, 1]},
        "entries": [
            {"label": entry.label, "elo": entry.elo, "fixed": entry.fixed}
            for entry in roster_model.entries
        ],
    }
    elo_history.write_text(json.dumps(record, sort_keys=True) + "\n")
    return SyntheticRun(run_dir, checkpoint, ladder_checkpoint, roster, elo_history, resolved)


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


def _basic_env(*, num_fields: int = 0) -> EnvConfig:
    return EnvConfig(
        num_ships=2,
        num_fields=num_fields,
        max_bullets=2,
        max_episode_steps=2,
    )


def _run_training_case(case: SmokeCase, roots: SmokeRoots) -> None:
    from boost_and_broadside import cli_commands
    from boost_and_broadside.cli import main

    assert case.profile is not None
    resolved = _smoke_resolved_profile(case.profile, roots.checkpoints)
    with patch.object(
        cli_commands,
        "resolve_named_profile",
        side_effect=lambda profile, overrides=None: resolved,
    ):
        result = main(
            [
                "train",
                "--profile",
                case.profile,
                "--device",
                "cpu",
                "--seed",
                "7",
                "--compile",
                "none",
                "--no-wandb",
                "--gradient-diagnostics",
                case.gradient_diagnostics,
            ]
        )
    if result:
        raise SmokeError(f"CLI returned {result} for smoke case {case.name!r}")


def _queue_quit_event() -> None:
    import pygame

    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))


def _run_mode_case(case: SmokeCase, roots: SmokeRoots) -> None:
    from boost_and_broadside import cli_commands
    from boost_and_broadside.cli import main

    if case.command == "play":
        _queue_quit_event()
        result = main(["play", "--device", "cpu", "--seed", "7"])
        if result:
            raise SmokeError(f"CLI returned {result} for smoke case {case.name!r}")
        return
    if case.command == "watch":
        _queue_quit_event()
        result = main(
            [
                "watch",
                "--team0",
                "scripted",
                "--team1",
                "random",
                "--device",
                "cpu",
                "--seed",
                "7",
            ]
        )
        if result:
            raise SmokeError(f"CLI returned {result} for smoke case {case.name!r}")
        return

    fixture = build_synthetic_run(roots.checkpoints)
    checkpoint = str(fixture.checkpoint)

    if case.command == "capture":
        from boost_and_broadside.modes.capture import run_capture_mode

        def bounded_capture(**kwargs):
            return run_capture_mode(**kwargs, window=64)

        with patch.object(cli_commands, "run_capture_mode", side_effect=bounded_capture):
            result = main(
                [
                    "capture",
                    "--run",
                    _RUN_NAME,
                    "--scenarios",
                    "vs_scripted",
                    "--seeds",
                    "0",
                    "--sizes",
                    "1v1",
                    "--fps",
                    "1",
                    "--max-steps",
                    "1",
                    "--hold-ms",
                    "1",
                    "--out",
                    str(roots.out),
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "collect-stats":
        from boost_and_broadside.modes.collect import run_collect_stats_mode

        def bounded_collect(**kwargs):
            kwargs["env_config"] = replace(kwargs["env_config"], max_episode_steps=2)
            return run_collect_stats_mode(**kwargs)

        with patch.object(cli_commands, "run_collect_stats_mode", side_effect=bounded_collect):
            result = main(
                [
                    "collect-stats",
                    "--team0",
                    "scripted",
                    "--team1",
                    "random",
                    "--sizes",
                    "1v1",
                    "--games",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "crossover":
        from boost_and_broadside.modes.crossover import run_crossover_mode

        def bounded_crossover(**kwargs):
            return run_crossover_mode(**kwargs, max_total_ships=2)

        with patch.object(cli_commands, "run_crossover_mode", side_effect=bounded_crossover):
            result = main(
                [
                    "crossover",
                    "--run",
                    _RUN_NAME,
                    "--sizes",
                    "1",
                    "--games-per-matchup",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "elo-calibrate":
        from boost_and_broadside.modes.elo_calibrate import run_elo_calibrate_mode

        def bounded_calibrate(**kwargs):
            return run_elo_calibrate_mode(**kwargs)

        with patch.object(cli_commands, "run_elo_calibrate_mode", side_effect=bounded_calibrate):
            result = main(
                [
                    "elo-calibrate",
                    "--run",
                    _RUN_NAME,
                    "--games-per-batch",
                    "2",
                    "--target-stderr",
                    "1000000",
                    "--max-batches",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "elo-scale":
        from boost_and_broadside.modes.elo_scale import run_elo_scale_mode

        def bounded_scale(**kwargs):
            return run_elo_scale_mode(**kwargs)

        with patch.object(cli_commands, "run_elo_scale_mode", side_effect=bounded_scale):
            result = main(
                [
                    "elo-scale",
                    "--run",
                    _RUN_NAME,
                    "--sizes",
                    "1",
                    "--games-per-batch",
                    "2",
                    "--target-stderr",
                    "1000000",
                    "--max-batches",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "semi-random":
        from boost_and_broadside.modes.semi_random_tournament import (
            run_semi_random_tournament,
        )

        def bounded_semi_random(**kwargs):
            return run_semi_random_tournament(**kwargs)

        with patch.object(
            cli_commands,
            "run_semi_random_tournament",
            side_effect=bounded_semi_random,
        ):
            result = main(
                [
                    "semi-random",
                    "--run",
                    _RUN_NAME,
                    "--sizes",
                    "1",
                    "--scripted-probabilities",
                    "0,1",
                    "--games-per-pair",
                    "1",
                    "--max-parallel-games",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "ar-report":
        from boost_and_broadside.modes.ar_report import run_ar_report_mode

        calls = 0

        def bounded_ar(**kwargs):
            nonlocal calls
            calls += 1
            kwargs["num_steps"] = 2
            kwargs["env_config"] = _basic_env()
            return run_ar_report_mode(**kwargs)

        with patch.object(cli_commands, "run_canonical_ar_report_mode", side_effect=bounded_ar):
            result = main(
                [
                    "ar-report",
                    "--team0",
                    checkpoint,
                    "--team1",
                    checkpoint,
                    "--decision-steps",
                    "2",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "noise-calibration":
        from boost_and_broadside.modes.noise_calibration import run_noise_calibration_mode

        def bounded_noise(**kwargs):
            kwargs.update(
                num_envs=1,
                num_steps=2,
                num_ar_envs=1,
                num_ar_windows=1,
                env_config=_basic_env(),
            )
            return run_noise_calibration_mode(**kwargs)

        with patch.object(
            cli_commands,
            "run_noise_calibration_mode",
            side_effect=bounded_noise,
        ):
            result = main(
                [
                    "noise-calibration",
                    "--team0",
                    checkpoint,
                    "--team1",
                    "scripted",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    elif case.command == "feature-stats":
        from boost_and_broadside.modes.feature_stats import run_feature_stats_mode

        def bounded_feature_stats(**kwargs):
            kwargs["env_config"] = _basic_env()
            return run_feature_stats_mode(**kwargs)

        with patch.object(
            cli_commands,
            "run_feature_stats_mode",
            side_effect=bounded_feature_stats,
        ):
            result = main(
                [
                    "feature-stats",
                    "--team0",
                    checkpoint,
                    "--team1",
                    "scripted",
                    "--games",
                    "1",
                    "--decision-steps",
                    "2",
                    "--device",
                    "cpu",
                    "--seed",
                    "7",
                ]
            )
    else:  # pragma: no cover - registry coverage makes this unreachable
        raise SmokeError(f"no smoke implementation for command {case.command!r}")

    if result:
        raise SmokeError(f"CLI returned {result} for smoke case {case.name!r}")


def execute_case(case_name: str, root: str | Path) -> None:
    """Execute one case inside an already isolated child process."""

    from boost_and_broadside import cli_commands

    try:
        case = _CASES_BY_NAME[case_name]
    except KeyError as error:
        raise SmokeError(f"unknown smoke case {case_name!r}") from error
    roots = SmokeRoots.create(root)
    os.chdir(roots.root)
    torch.manual_seed(7)
    dispatched: list[str] = []
    execute = cli_commands.execute

    def tracked_execute(
        command: str, args: argparse.Namespace, argv: Sequence[str] | None = None
    ) -> None:
        dispatched.append(command)
        execute(command, args, argv)

    with patch.object(cli_commands, "execute", side_effect=tracked_execute):
        if case.command == "train":
            _run_training_case(case, roots)
        else:
            _run_mode_case(case, roots)
    if dispatched != [case.command]:
        raise SmokeError(
            f"case {case.name!r} did not traverse its registered CLI handler exactly once: "
            f"{dispatched!r}"
        )
    validate_case_root(roots.root)


def _case_environment(roots: SmokeRoots) -> dict[str, str]:
    home = roots.tmp / "home"
    cache = roots.tmp / "cache"
    config = roots.tmp / "config"
    data = roots.tmp / "data"
    state = roots.tmp / "state"
    for path in (home, cache, config, data, state):
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "HEADLESS": "1",
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(roots.tmp / "matplotlib"),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
            "XDG_STATE_HOME": str(state),
            "TORCH_HOME": str(cache / "torch"),
            "TMPDIR": str(roots.tmp),
            "WANDB_DIR": str(roots.tmp / "wandb"),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def _tree_snapshot(path: Path) -> tuple[tuple[object, ...], ...]:
    """Snapshot names and metadata without following symlinks or hashing large artifacts."""

    if not path.exists() and not path.is_symlink():
        return (("<missing>",),)
    candidates = [path]
    if path.is_dir() and not path.is_symlink():
        candidates.extend(path.rglob("*"))
    records = []
    for candidate in sorted(candidates, key=lambda item: str(item)):
        stat = candidate.lstat()
        records.append(
            (
                str(candidate.relative_to(path.parent)),
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                os.readlink(candidate) if candidate.is_symlink() else None,
            )
        )
    return tuple(records)


def _outside_case_snapshot(root: Path) -> tuple[tuple[str, tuple], ...]:
    """Snapshot siblings so a case cannot escape into its private matrix parent."""

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    return tuple(
        (child.name, _tree_snapshot(child))
        for child in sorted(parent.iterdir(), key=lambda item: item.name)
        if child != root
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate the fresh process group, including renderer/ffmpeg descendants."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def run_case_subprocess(
    case: SmokeCase | str,
    root: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Run one registry case in a fresh interpreter with a fixed timeout."""

    selected = _CASES_BY_NAME[case] if isinstance(case, str) else case
    root_path = Path(root).resolve()
    outside_before = _outside_case_snapshot(root_path)
    roots = SmokeRoots.create(root_path)
    command = [
        sys.executable,
        "-m",
        "boost_and_broadside.smoke",
        "--run-case",
        selected.name,
        "--root",
        str(roots.root),
    ]
    process = subprocess.Popen(
        command,
        cwd=roots.root,
        env=_case_environment(roots),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timeout: subprocess.TimeoutExpired | None = None
    try:
        stdout, stderr = process.communicate(timeout=selected.timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_tree(process)
        timeout = subprocess.TimeoutExpired(
            command,
            selected.timeout_seconds,
            output=stdout,
            stderr=stderr,
        )

    outside_after = _outside_case_snapshot(root_path)
    if outside_after != outside_before:
        raise SmokeIsolationError(f"case {selected.name!r} wrote outside {root_path}")
    if timeout is not None:
        raise timeout
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


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


def _repository_output_snapshot(repository: Path) -> tuple[tuple[str, tuple], ...]:
    """Observe real mutable roots, including files ignored by Git."""

    return tuple(
        (name, _tree_snapshot(repository / name))
        for name in ("artifacts", "checkpoints", "docs", "out")
    )


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
    outputs_before = _repository_output_snapshot(repository)
    failures: list[str] = []
    print(f"bnb smoke: {len(cases)} isolated case(s), sequential", file=stream, flush=True)
    with tempfile.TemporaryDirectory(prefix="bnb-smoke-") as temporary:
        matrix_root = Path(temporary)
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.name} ... ", end="", file=stream, flush=True)
            case_root = matrix_root / case.name
            result: subprocess.CompletedProcess[str] | None = None
            case_error: SmokeIsolationError | subprocess.TimeoutExpired | None = None
            try:
                result = run_case_subprocess(case, case_root)
                validate_case_root(case_root)
            except (SmokeIsolationError, subprocess.TimeoutExpired) as error:
                case_error = error
            finally:
                checkout_after = _checkout_snapshot(repository)
                outputs_after = _repository_output_snapshot(repository)
            if checkout_after != checkout_before:
                raise SmokeIsolationError(f"case {case.name!r} changed the source checkout")
            if outputs_after != outputs_before:
                raise SmokeIsolationError(
                    f"case {case.name!r} changed a real checkout output root"
                )
            if case_error is not None:
                failures.append(f"{case.name}: {case_error}")
                print("FAIL", file=stream, flush=True)
                continue
            assert result is not None
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

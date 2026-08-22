"""Synthetic-run fidelity and isolated subprocess smoke contracts."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest
import torch

from boost_and_broadside import cli, cli_commands
from boost_and_broadside.config.diagnostics import GRADIENT_DIAGNOSTICS_LEVELS
from boost_and_broadside.smoke import (
    SMOKE_CASES,
    SmokeCase,
    SmokeCaseError,
    SmokeIsolationError,
    SmokeRoots,
    _case_environment,
    _repository_output_snapshot,
    build_synthetic_run,
    run_case_subprocess,
    run_smoke_matrix,
    validate_case_root,
)
from boost_and_broadside.train.rl.checkpoint_schema import OBSERVATION_SCHEMA
from boost_and_broadside.train.rl.policy_io import load_policy_bundle


def _parse(argv: list[str]):
    return cli.parse_args(argv)[1]


def test_registry_covers_every_runtime_command_and_training_profile() -> None:
    by_command: dict[str, list] = {}
    for case in SMOKE_CASES:
        by_command.setdefault(case.command, []).append(case)

    assert set(by_command) == set(cli_commands.runtime_command_names())
    assert {case.profile for case in by_command["train"]} == {"bc", "rl"}
    assert len({case.name for case in SMOKE_CASES}) == len(SMOKE_CASES)
    assert all(case.timeout_seconds > 0 for case in SMOKE_CASES)


def test_every_gradient_diagnostic_level_is_launched_by_a_training_case() -> None:
    """Each level is a distinct code path, so each gets its own bounded launch."""
    launched = {case.gradient_diagnostics for case in SMOKE_CASES if case.command == "train"}
    assert launched == set(GRADIENT_DIAGNOSTICS_LEVELS)
    # Nothing that is not a training run has a gradient to decompose.
    assert all(
        case.gradient_diagnostics == "off" for case in SMOKE_CASES if case.command != "train"
    )


def test_synthetic_run_uses_current_loadable_checkpoint_schema(tmp_path: Path) -> None:
    fixture = build_synthetic_run(tmp_path / "checkpoints", seed=11)
    checkpoint = torch.load(fixture.checkpoint, map_location="cpu", weights_only=False)
    ladder = torch.load(fixture.ladder_checkpoint, map_location="cpu", weights_only=False)

    assert checkpoint["observation_schema"] == OBSERVATION_SCHEMA
    assert checkpoint["global_step"] == 1
    assert checkpoint["model_config"]
    assert checkpoint["ship_config"]
    assert checkpoint["env_config"]["num_ships"] == 2
    assert checkpoint["resolved_config"]["profile"] == "smoke-fixture"
    assert checkpoint["resolved_config"]["resolved_config_fingerprint"]
    assert checkpoint["optimizer_state_dict"]["param_groups"]
    assert checkpoint["avg_policy_state_dict"]
    assert checkpoint["launch"] == {
        "allow_config_drift": False,
        "compile_mode": None,
        "device": "cpu",
        "seed": 11,
        "wandb": False,
    }
    assert ladder["global_step"] == 0
    assert "optimizer_state_dict" not in ladder
    assert "avg_policy_state_dict" not in ladder
    for key in (
        "policy_state_dict",
        "observation_schema",
        "model_config",
        "ship_config",
        "env_config",
        "resolved_config",
        "launch",
    ):
        assert key in ladder

    roster = json.loads(fixture.roster.read_text())
    assert [entry["kind"] for entry in roster["entries"]] == [
        "random",
        "scripted",
        "checkpoint",
    ]
    assert roster["entries"][-1]["global_step"] == 0
    assert Path(roster["entries"][-1]["path"]) == fixture.ladder_checkpoint
    history = [json.loads(line) for line in fixture.elo_history.read_text().splitlines()]
    assert len(history) == 1
    assert history[0]["global_step"] == 1

    bundle = load_policy_bundle(
        str(fixture.checkpoint),
        device="cpu",
        num_ships=2,
        ship_config=fixture.resolved.ship_config,
    )
    assert bundle.global_step == 1
    assert bundle.num_value_components == checkpoint["num_value_components"]
    ladder_bundle = load_policy_bundle(
        str(fixture.ladder_checkpoint),
        device="cpu",
        num_ships=2,
        ship_config=fixture.resolved.ship_config,
    )
    assert ladder_bundle.global_step == 0

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.config.service import resolved_profile_document
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    trainer = PPOTrainer(
        fixture.resolved.train_config,
        fixture.resolved.model_config,
        fixture.resolved.ship_config,
        device="cpu",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(
            fixture.resolved.ship_config,
            StochasticAgentConfig(),
        ),
        compile_mode=None,
        resolved_config_document=resolved_profile_document(fixture.resolved),
        launch_provenance={"allow_config_drift": False},
    )
    assert trainer.load_checkpoint(str(fixture.checkpoint)) == 0
    trainer.shutdown()


def test_case_root_rejects_writes_outside_managed_roots(tmp_path: Path) -> None:
    for managed in ("checkpoints", "artifacts", "out", "tmp"):
        (tmp_path / managed).mkdir()
    validate_case_root(tmp_path)

    (tmp_path / "docs").mkdir()
    with pytest.raises(SmokeIsolationError, match="docs"):
        validate_case_root(tmp_path)


def test_case_root_rejects_rendered_publication_outputs(tmp_path: Path) -> None:
    for managed in ("checkpoints", "artifacts", "out", "tmp"):
        (tmp_path / managed).mkdir()
    (tmp_path / "artifacts" / "unexpected.png").touch()
    with pytest.raises(SmokeIsolationError, match="rendering"):
        validate_case_root(tmp_path)


def test_case_environment_redirects_mutable_home_and_cache_state(tmp_path: Path) -> None:
    roots = SmokeRoots.create(tmp_path / "case")
    environment = _case_environment(roots)

    for name in (
        "HOME",
        "MPLCONFIGDIR",
        "TMPDIR",
        "TORCH_HOME",
        "WANDB_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ):
        assert Path(environment[name]).is_relative_to(roots.root)


def test_subprocess_detects_a_sibling_escape_and_starts_a_process_group(
    tmp_path: Path, monkeypatch
) -> None:
    case_root = tmp_path / "case"
    captured = {}

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        (tmp_path / "escaped.txt").write_text("escape")
        return FakeProcess()

    monkeypatch.setattr("boost_and_broadside.smoke.subprocess.Popen", fake_popen)
    with pytest.raises(SmokeIsolationError, match="wrote outside"):
        run_case_subprocess(SmokeCase("play", "play"), case_root)
    assert captured["start_new_session"] is True


def test_timeout_terminates_the_complete_process_group(tmp_path: Path, monkeypatch) -> None:
    signals = []
    calls = 0

    class TimedOutProcess:
        pid = 456
        returncode = -signal.SIGTERM

        def communicate(self, timeout=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return "partial stdout", "partial stderr"

    monkeypatch.setattr(
        "boost_and_broadside.smoke.subprocess.Popen",
        lambda *args, **kwargs: TimedOutProcess(),
    )
    monkeypatch.setattr(
        "boost_and_broadside.smoke.os.killpg",
        lambda process_group, sent_signal: signals.append((process_group, sent_signal)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_case_subprocess(SmokeCase("play", "play", timeout_seconds=1), tmp_path / "case")
    assert signals == [(456, signal.SIGTERM)]


def test_real_output_snapshot_observes_ignored_style_writes(tmp_path: Path) -> None:
    before = _repository_output_snapshot(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "escaped.pt").touch()
    assert _repository_output_snapshot(tmp_path) != before


def test_timeout_path_still_compares_the_checkout(tmp_path: Path, monkeypatch) -> None:
    snapshots = []
    monkeypatch.setattr("boost_and_broadside.smoke._repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        "boost_and_broadside.smoke._checkout_snapshot",
        lambda repository: snapshots.append(repository) or b"same",
    )
    monkeypatch.setattr(
        "boost_and_broadside.smoke._repository_output_snapshot",
        lambda repository: (("same", ()),),
    )
    monkeypatch.setattr(
        "boost_and_broadside.smoke.run_case_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(["child"], 1)),
    )

    with pytest.raises(SmokeCaseError):
        run_smoke_matrix("play")
    assert snapshots == [tmp_path, tmp_path]


def test_cli_focused_case_selection_dispatches_one_case(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "boost_and_broadside.smoke.run_smoke_matrix",
        lambda selected_case=None: captured.setdefault("selected", selected_case),
    )
    cli_commands.execute("smoke", _parse(["smoke", "--case", "collect-stats"]))
    assert captured == {"selected": "collect-stats"}

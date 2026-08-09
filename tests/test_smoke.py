"""Synthetic-run fidelity and isolated subprocess smoke contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from boost_and_broadside import cli, cli_commands
from boost_and_broadside.smoke import (
    SMOKE_CASES,
    SmokeIsolationError,
    build_synthetic_run,
    run_case_subprocess,
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
    assert {case.profile for case in by_command["train"]} == {"bc", "rl", "rl-fields"}
    assert len({case.name for case in SMOKE_CASES}) == len(SMOKE_CASES)
    assert all(case.timeout_seconds > 0 for case in SMOKE_CASES)


def test_synthetic_run_uses_current_loadable_checkpoint_schema(tmp_path: Path) -> None:
    fixture = build_synthetic_run(tmp_path / "checkpoints", seed=11)
    checkpoint = torch.load(fixture.checkpoint, map_location="cpu", weights_only=False)

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

    roster = json.loads(fixture.roster.read_text())
    assert [entry["kind"] for entry in roster["entries"]] == [
        "random",
        "scripted",
        "checkpoint",
    ]
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


def test_cli_focused_case_selection_dispatches_one_case(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "boost_and_broadside.smoke.run_smoke_matrix",
        lambda selected_case=None: captured.setdefault("selected", selected_case),
    )
    cli_commands.execute("smoke", _parse(["smoke", "--case", "collect-stats"]))
    assert captured == {"selected": "collect-stats"}


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda case: case.name)
def test_every_smoke_case_passes_in_a_fresh_subprocess(case, tmp_path: Path) -> None:
    result = run_case_subprocess(case, tmp_path / case.name)
    assert result.returncode == 0, result.stderr
    assert f"SMOKE PASS {case.name}" in result.stdout
    validate_case_root(tmp_path / case.name)

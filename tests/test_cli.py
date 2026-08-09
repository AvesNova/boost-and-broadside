"""Generated parser, dispatch, and installed-entry-point contracts for ``bnb``."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from boost_and_broadside import cli, cli_commands

EXPECTED_COMMANDS = (
    "train",
    "play",
    "watch",
    "capture",
    "collect-stats",
    "crossover",
    "elo-calibrate",
    "elo-scale",
    "semi-random",
    "ar-report",
    "noise-calibration",
    "feature-stats",
    "publish",
    "smoke",
)

VALID_ARGV = {
    "train": ["--profile", "rl"],
    "play": [],
    "watch": ["--team0", "scripted", "--team1", "random"],
    "capture": ["--run", "exact-run"],
    "collect-stats": ["--team0", "scripted", "--team1", "random"],
    "crossover": ["--run", "exact-run"],
    "elo-calibrate": ["--run", "exact-run"],
    "elo-scale": ["--run", "exact-run"],
    "semi-random": ["--profile", "rl"],
    "ar-report": ["--team0", "scripted", "--team1", "random"],
    "noise-calibration": ["--team0", "model.pt", "--team1", "scripted"],
    "feature-stats": ["--team0", "scripted", "--team1", "random"],
    "publish": [],
    "smoke": [],
}


def _parse(argv: list[str]):
    return cli.parse_args(argv)[1]


def test_registry_is_the_exact_final_hyphenated_command_list() -> None:
    assert tuple(command.name for command in cli.COMMANDS) == EXPECTED_COMMANDS
    for command in cli.COMMANDS:
        assert "_" not in command.name
        assert all("_" not in flag for option in command.options for flag in option.flags)


def test_modifier_ownership_matches_command_contract() -> None:
    owners: dict[str, set[str]] = {}
    for command in cli.COMMANDS:
        for option in command.options:
            for flag in option.flags:
                owners.setdefault(flag, set()).add(command.name)

    assert owners["--resume"] == {"train"}
    assert owners["--pretrain-from"] == {"train"}
    assert owners["--compile"] == {"train"}
    assert owners["--no-wandb"] == {"train"}
    assert owners["--print-config"] == {"train"}
    assert owners["--out"] == {"capture"}
    assert owners["--target-stderr"] == {"elo-calibrate", "elo-scale"}
    assert owners["--max-batches"] == {"elo-calibrate", "elo-scale"}
    assert owners["--team0"] == {
        "watch",
        "collect-stats",
        "ar-report",
        "noise-calibration",
        "feature-stats",
    }


def test_legacy_entrypoint_and_reader_facing_commands_are_gone() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "main.py").exists()
    excluded = {
        root / "docs" / "internal" / "mode-refactor-plan.md",
        root / "docs" / "internal" / "mode-refactor-status.md",
    }
    documents = [root / "README.md", root / "STYLE_GUIDE.md"]
    documents.extend(path for path in (root / "docs").rglob("*.md") if path not in excluded)
    offenders = {
        str(path.relative_to(root)): token
        for path in documents
        for token in ("uv run main.py", "--mode ")
        if token in path.read_text()
    }
    assert offenders == {}


@pytest.mark.parametrize("command", EXPECTED_COMMANDS)
def test_every_registered_command_has_generated_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse([command, "--help"])
    assert exit_info.value.code == 0
    assert f"usage: bnb {command}" in capsys.readouterr().out


@pytest.mark.parametrize("command", EXPECTED_COMMANDS)
def test_every_registered_command_rejects_an_irrelevant_option(command) -> None:
    foreign = "--run" if command == "train" else "--team0"
    if command in {"watch", "collect-stats", "ar-report", "noise-calibration", "feature-stats"}:
        foreign = "--profile"
    if command == "semi-random":
        foreign = "--team0"
    with pytest.raises(SystemExit) as exit_info:
        _parse([command, *VALID_ARGV[command], foreign, "irrelevant"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["capture"],
        ["watch", "--team0", "scripted"],
        ["collect-stats", "--team0", "scripted"],
        ["train"],
        ["semi-random"],
    ],
)
def test_required_subjects_fail_during_parsing(argv) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(argv)
    assert exit_info.value.code == 2


@pytest.mark.parametrize("sentinel", ["latest", "none"])
def test_magic_run_and_agent_sentinels_are_rejected(sentinel) -> None:
    with pytest.raises(SystemExit):
        _parse(["capture", "--run", sentinel])
    with pytest.raises(SystemExit):
        _parse(["watch", "--team0", sentinel, "--team1", "random"])


def test_resume_requires_a_value_and_is_exclusive_with_pretraining() -> None:
    with pytest.raises(SystemExit):
        _parse(["train", "--profile", "rl", "--resume"])
    with pytest.raises(SystemExit):
        _parse(
            [
                "train",
                "--profile",
                "rl",
                "--resume",
                "exact-run",
                "--pretrain-from",
                "weights.pt",
            ]
        )


@pytest.mark.parametrize("subject", ["latest", "none", "nested/run"])
def test_resume_rejects_magic_or_path_like_run_subjects(subject) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(["train", "--profile", "rl", "--resume", subject, "--print-config"])
    assert exit_info.value.code == 2


def test_pretraining_subject_must_be_an_explicit_checkpoint_path() -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(["train", "--profile", "rl", "--pretrain-from", "exact-run"])
    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["collect_stats", "--team0", "scripted", "--team1", "random"],
        ["train", "--profile", "rl_fields"],
        ["train", "--profile", "rl", "--pretrain_from", "weights.pt"],
        ["train", "--profile", "rl", "--smoke"],
    ],
)
def test_legacy_command_and_option_aliases_do_not_parse(argv) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(argv)
    assert exit_info.value.code == 2


def test_no_subcommand_prints_help_without_dispatch(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_dispatch_command", lambda *_: pytest.fail("dispatched"))
    assert cli.main([]) == 0
    assert "usage: bnb" in capsys.readouterr().out


def test_runtime_argument_errors_are_translated_to_cli_errors(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["play", "--device", "not-a-device"])
    assert exit_info.value.code == 2
    assert "invalid --device value" in capsys.readouterr().err


def test_invalid_print_config_is_a_concise_cli_error(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", "--profile", "rl", "--num-envs", "3872", "--print-config"])
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "cannot preserve the fixed logical batch" in error
    assert "Traceback" not in error


def test_print_config_rejects_an_unavailable_execution_backend(capsys, monkeypatch) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", "--profile", "rl", "--device", "mps", "--print-config"])
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "MPS is unavailable" in error
    assert "Traceback" not in error


def test_publish_remains_explicitly_future_owned(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["publish"])
    assert exit_info.value.code == 2
    assert "unavailable until S09" in capsys.readouterr().err


def test_print_config_bypasses_runtime_dispatch_and_records_cli_sources(
    capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_dispatch_command", lambda *_: pytest.fail("dispatched"))
    assert (
        cli.main(
            [
                "train",
                "--profile",
                "rl",
                "--num-envs",
                "1952",
                "--microbatch-tokens",
                "20000",
                "--device",
                "cpu",
                "--seed",
                "17",
                "--compile",
                "none",
                "--no-wandb",
                "--allow-config-drift",
                "--print-config",
            ]
        )
        == 0
    )
    document = json.loads(capsys.readouterr().out)
    assert document["profile"] == "rl"
    assert document["sources"]["train_config.scales.0.num_envs"] == "cli"
    assert document["sources"]["train_config.microbatch_tokens"] == "cli"
    assert document["launch"] == {
        "allow_config_drift": True,
        "compile_mode": None,
        "device": "cpu",
        "seed": 17,
        "wandb": False,
    }


class _StubTrainer:
    def __init__(self) -> None:
        self.loaded_checkpoint = None
        self.loaded_pretrained = None

    def load_checkpoint(self, path: str) -> None:
        self.loaded_checkpoint = path

    def load_pretrained_weights(self, path: str) -> None:
        self.loaded_pretrained = path

    def train(self) -> None:
        pass


def test_train_resume_selects_greatest_numeric_step_within_exact_run(tmp_path, monkeypatch) -> None:
    run = tmp_path / "checkpoints" / "exact-run"
    run.mkdir(parents=True)
    low = run / "step_9.pt"
    high = run / "step_100.pt"
    low.touch()
    high.touch()
    (run / "wandb_run_id.txt").write_text("wandb-id\n")
    captured = {}
    trainer = _StubTrainer()

    def make_trainer(resolved, args, device, *, resume_wandb_run_id=None):
        captured["run_id"] = resume_wandb_run_id
        return trainer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_commands, "_prepare_execution", lambda args: "cpu")
    monkeypatch.setattr(cli_commands, "_make_trainer", make_trainer)
    cli_commands.execute("train", _parse(["train", "--profile", "rl", "--resume", "exact-run"]))

    assert Path(trainer.loaded_checkpoint).resolve() == high
    assert trainer.loaded_pretrained is None
    assert captured["run_id"] == "wandb-id"


def test_train_pretraining_requires_and_loads_an_explicit_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.touch()
    trainer = _StubTrainer()
    monkeypatch.setattr(cli_commands, "_prepare_execution", lambda args: "cpu")
    monkeypatch.setattr(cli_commands, "_make_trainer", lambda *args, **kwargs: trainer)
    cli_commands.execute(
        "train",
        _parse(["train", "--profile", "bc", "--pretrain-from", str(checkpoint)]),
    )
    assert trainer.loaded_pretrained == str(checkpoint)
    assert trainer.loaded_checkpoint is None


def test_train_validates_pretraining_before_execution_or_trainer_allocation(
    tmp_path, monkeypatch
) -> None:
    missing = tmp_path / "missing.pt"
    monkeypatch.setattr(
        cli_commands,
        "_prepare_execution",
        lambda args: pytest.fail("execution prepared before subject validation"),
    )
    monkeypatch.setattr(
        cli_commands,
        "_make_trainer",
        lambda *args, **kwargs: pytest.fail("trainer allocated before subject validation"),
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        cli_commands.execute(
            "train",
            _parse(["train", "--profile", "rl", "--pretrain-from", str(missing)]),
        )


def test_corrupt_checkpoint_is_a_concise_cli_error(tmp_path, capsys, monkeypatch) -> None:
    from boost_and_broadside.train.rl.checkpoint import CheckpointMixin

    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    loader = CheckpointMixin()
    loader.device = "cpu"
    monkeypatch.setattr(cli_commands, "_make_trainer", lambda *args, **kwargs: loader)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "train",
                "--profile",
                "rl",
                "--pretrain-from",
                str(checkpoint),
                "--device",
                "cpu",
            ]
        )
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "could not read checkpoint" in error
    assert "Traceback" not in error


def test_incompatible_checkpoint_weights_are_a_concise_cli_error(
    tmp_path, capsys, monkeypatch
) -> None:
    from boost_and_broadside.train.rl.checkpoint import CheckpointMixin
    from boost_and_broadside.train.rl.checkpoint_schema import OBSERVATION_SCHEMA

    checkpoint = tmp_path / "incompatible.pt"
    import torch

    torch.save(
        {
            "observation_schema": OBSERVATION_SCHEMA,
            "policy_state_dict": {"wrong": torch.zeros(1)},
        },
        checkpoint,
    )

    class IncompatibleModule:
        def load_state_dict(self, state):
            raise RuntimeError("missing and unexpected tensor keys")

    loader = CheckpointMixin()
    loader.device = "cpu"
    loader._policy_module = IncompatibleModule()
    monkeypatch.setattr(cli_commands, "_make_trainer", lambda *args, **kwargs: loader)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "train",
                "--profile",
                "rl",
                "--pretrain-from",
                str(checkpoint),
                "--device",
                "cpu",
            ]
        )
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "incompatible policy weights" in error
    assert "Traceback" not in error


def test_non_mapping_checkpoint_weights_are_a_concise_cli_error(
    tmp_path, capsys, monkeypatch
) -> None:
    import torch

    from boost_and_broadside.train.rl.checkpoint import CheckpointMixin
    from boost_and_broadside.train.rl.checkpoint_schema import OBSERVATION_SCHEMA

    checkpoint = tmp_path / "non-mapping.pt"
    torch.save(
        {
            "observation_schema": OBSERVATION_SCHEMA,
            "policy_state_dict": None,
        },
        checkpoint,
    )

    class UnusedModule:
        def load_state_dict(self, state):
            pytest.fail("non-mapping state reached the model loader")

    loader = CheckpointMixin()
    loader.device = "cpu"
    loader._policy_module = UnusedModule()
    monkeypatch.setattr(cli_commands, "_make_trainer", lambda *args, **kwargs: loader)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "train",
                "--profile",
                "rl",
                "--pretrain-from",
                str(checkpoint),
                "--device",
                "cpu",
            ]
        )
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "invalid policy weights: expected a mapping, got NoneType" in error
    assert "Traceback" not in error


def test_malformed_matchup_is_rejected_during_parsing() -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(
            [
                "collect-stats",
                "--team0",
                "scripted",
                "--team1",
                "random",
                "--sizes",
                "0v4",
            ]
        )
    assert exit_info.value.code == 2


def test_collect_stats_adapter_uses_the_locked_4v4_default(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli_commands, "_prepare_execution", lambda args: "cpu")
    monkeypatch.setattr(
        cli_commands,
        "run_collect_stats_mode",
        lambda **kwargs: captured.update(kwargs),
    )
    cli_commands.execute(
        "collect-stats",
        _parse(["collect-stats", "--team0", "scripted", "--team1", "random"]),
    )
    assert captured["matchups"] == ["4v4"]
    assert captured["env_config"].num_ships == 8
    assert captured["num_envs"] == 1024


@pytest.mark.parametrize(
    ("command", "runtime_name"),
    [
        ("noise-calibration", "run_noise_calibration_mode"),
        ("feature-stats", "run_feature_stats_mode"),
    ],
)
def test_analysis_adapters_use_the_locked_4v4_default(command, runtime_name, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli_commands, "_prepare_execution", lambda args: "cpu")
    monkeypatch.setattr(cli_commands, runtime_name, lambda **kwargs: captured.update(kwargs))
    cli_commands.execute(command, _parse([command, "--team0", "model.pt", "--team1", "scripted"]))
    assert captured["env_config"].num_ships == 8


def test_trainer_receives_complete_resolved_and_launch_provenance(monkeypatch) -> None:
    captured = {}

    class CaptureTrainer(_StubTrainer):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(cli_commands, "PPOTrainer", CaptureTrainer)
    resolved = cli_commands.resolve_named_profile("rl")
    args = _parse(["train", "--profile", "rl", "--no-wandb", "--seed", "0"])
    cli_commands._make_trainer(resolved, args, "cpu")

    document = captured["resolved_config_document"]
    assert document["profile"] == "rl"
    assert document["resolved_config_fingerprint"] == resolved.resolved_config_fingerprint
    assert captured["launch_provenance"] == {
        "device": "cpu",
        "seed": 0,
        "compile_mode": "reduce-overhead",
        "wandb": False,
        "allow_config_drift": False,
    }


def test_project_registers_installed_bnb_entrypoint(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "--no-sync", "bnb", "--help"],
        cwd=root,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: bnb")

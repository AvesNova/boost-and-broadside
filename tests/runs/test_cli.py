"""`bnb train` and `bnb runs` (runs spec 5, 6 and acceptance 4, 8, 9, 10): the run record
across a new run, a stop by Ctrl-C, a resume with an override, a resume on another
commit, a rewind, a fork and a failure; the refusals; the listing."""

import dataclasses
import json
import os
import pathlib
import signal
import subprocess
import sys

import pytest
import runs_cases as cases
import torch

from bnb import cli
from bnb.learn.ppo import WARM_START_ENTROPY
from bnb.models.yemong.model import YemongConfig
from bnb.runs import checkpoint, probe, provenance
from bnb.runs.records import Run
from bnb.runs.serialise import from_json
from bnb.runtime.train import TrainConfig, Trainer, build_model, model_build

STEPS_PER_UPDATE = 16 * 16  # environments x segment length in TRAIN_OPTIONS


def bnb(monkeypatch, runs_directory: pathlib.Path, *arguments: str) -> None:
    """`bnb ARGUMENTS --runs-directory RUNS` in this process, its seeding kept local."""
    monkeypatch.setattr(sys, "argv", ["bnb", *arguments, "--runs-directory", str(runs_directory)])
    with torch.random.fork_rng(devices=[]):
        cli.main()


def new_run(monkeypatch, runs_directory, name: str = "a", updates: int = 2) -> Run:
    bnb(
        monkeypatch,
        runs_directory,
        "train",
        *cases.TRAIN_OPTIONS,
        "--updates",
        str(updates),
        "--name",
        name,
    )
    return Run.open(runs_directory, name)


def test_a_new_run_writes_its_record_config_and_final_checkpoint(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path)
    record = run.record
    assert record["status"] == "finished" and record["final_checkpoint"] == 2 * STEPS_PER_UPDATE
    assert run.checkpoint_steps() == [2 * STEPS_PER_UPDATE]
    assert record["probe"] is None  # the CPU records nothing
    assert record["command"][:2] == ["bnb", "train"]
    (segment,) = record["segments"]
    assert (segment["start_step"], segment["end_step"]) == (0, 2 * STEPS_PER_UPDATE)
    assert segment["commit"] == provenance.commit()[0]
    config = json.loads((run.directory / "config.json").read_text())
    assert config["environments"] == 16 and config["length"] == 16
    for part in ("league", "artifacts", "metrics.jsonl"):
        assert (run.directory / part).exists()


def test_a_resume_with_an_override_records_it_in_a_new_segment(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    bnb(
        monkeypatch,
        tmp_path,
        "train",
        "--resume",
        "a",
        "--device",
        "cpu",
        "--updates",
        "3",
        "--set",
        "learner.learning_rate=0.0001",
    )
    run = Run.open(tmp_path, "a")
    first, second = run.record["segments"]
    assert second["start_step"] == first["end_step"] == 2 * STEPS_PER_UPDATE
    assert second["end_step"] == 3 * STEPS_PER_UPDATE
    assert second["overrides"] == {"updates": 3, "learner.learning_rate": 0.0001}
    assert checkpoint.training_config(run.checkpoint_at())[0].learner.learning_rate == 0.0001


def test_the_micro_batch_rows_and_the_precision_may_change_on_resume(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path, updates=1)
    bnb(
        monkeypatch,
        tmp_path,
        "train",
        "--resume",
        "a",
        "--device",
        "cpu",
        "--updates",
        "2",
        "--micro-batch-rows",
        "32",
        "--precision",
        "tf32",
        "--set",
        "reward.damage=0.5",
    )
    run = Run.open(tmp_path, "a")
    assert run.record["segments"][1]["overrides"] == {
        "updates": 2,
        "precision": "tf32",
        "learner.micro_batch_rows": 32,
        "reward.damage": 0.5,
    }
    config, _ = checkpoint.training_config(run.checkpoint_at())
    assert (config.precision, config.learner.micro_batch_rows, config.reward.damage) == (
        "tf32",
        32,
        0.5,
    )


def test_a_resume_of_a_live_run_is_refused(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path)
    run.start_segment(2 * STEPS_PER_UPDATE)  # running, as this (live) process
    with pytest.raises(
        SystemExit, match=f"run a is running as process {os.getpid()} on this host; stop it"
    ):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu")


def test_a_resume_carries_the_runs_wandb_id(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path, updates=1).update(wandb_id="abc123")
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2")
    assert Run.open(tmp_path, "a").record["wandb_id"] == "abc123"


def test_a_checkpoint_with_an_unknown_config_field_is_a_one_line_refusal(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path)
    path = run.checkpoint_at() / checkpoint.RECORD
    record = json.loads(path.read_text())
    record["train_config"]["learner"]["dropout"] = 0.1
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="unknown field learner.dropout") as raised:
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu")
    assert "\n" not in str(raised.value)


def test_a_probe_over_the_budget_is_a_one_line_refusal(monkeypatch, tmp_path):
    over = dict(allocated=2, reserved=3, total=4, budget=1)
    monkeypatch.setattr(probe, "measure", lambda device, margin=probe.MARGIN: over)
    with pytest.raises(
        SystemExit, match="over the budget .* at 16 environments, 160 micro-batch rows: lower them"
    ):
        bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--probe")


def test_seed_and_set_seed_are_one_setting(monkeypatch, tmp_path, capsys):
    printed = []
    for option in (("--seed", "7"), ("--set", "seed=7")):
        bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, *option, "--print-config")
        printed.append(json.loads(capsys.readouterr().out))
    assert printed[0] == printed[1] and printed[0]["seed"] == 7
    assert "seed" not in printed[0]["learner"]


def test_the_runs_directory_is_runs_at_the_repository_root(monkeypatch, tmp_path):
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert cli.runs_directory_of(None) == pathlib.Path(root) / "runs"
    monkeypatch.chdir(tmp_path)  # outside a repository: the current directory
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert cli.runs_directory_of(None) == pathlib.Path(".") / "runs"
    assert cli.runs_directory_of("elsewhere") == pathlib.Path("elsewhere")


def test_a_structural_option_on_resume_is_refused_pointing_to_a_fork(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    with pytest.raises(
        SystemExit,
        match="a resume takes --environments from the checkpoint and "
        "cannot change it: start a new run from it with "
        "--from a",
    ):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--environments", "32")
    with pytest.raises(SystemExit, match="a resume cannot --set length"):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--set", "length=32")


def test_a_resume_that_sets_a_model_setting_is_refused_with_its_reason(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="a resume cannot --set relational.hidden"):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--set", "relational.hidden=32")


def test_set_yemong_width_reaches_the_built_model(monkeypatch, tmp_path, capsys):
    options = ("--model", "yemong", "--set", "yemong.width=256", "--print-config")
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, *options)
    config, _ = from_json(TrainConfig, json.loads(capsys.readouterr().out))
    assert config.yemong.width == 256 and config.model == "yemong"
    with torch.random.fork_rng(devices=[]):  # the models' initialisation draws
        wide, default = (
            build_model(model_build(c), cases.CPU).state_dict()
            for c in (config, dataclasses.replace(config, yemong=YemongConfig()))
        )
    shapes = {name: (tuple(default[name].shape), tuple(wide[name].shape)) for name in default}
    widened = [
        name
        for name, (before, after) in shapes.items()
        if after == tuple(256 if size == 128 else size for size in before) != before
    ]
    assert widened  # the width is in the weights


def test_set_yemong_forecast_feedback_reaches_the_built_model(monkeypatch, tmp_path, capsys):
    options = ("--model", "yemong", "--set", "yemong.forecast_feedback=false", "--print-config")
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, *options)
    printed = json.loads(capsys.readouterr().out)
    assert printed["yemong"]["forecast_feedback"] is False
    config, _ = from_json(TrainConfig, printed)
    with torch.random.fork_rng(devices=[]):  # the model's initialisation draws
        assert not build_model(model_build(config), cases.CPU).config.forecast_feedback


def test_a_fork_that_sets_no_model_setting_takes_the_parents(monkeypatch, tmp_path):
    """The parent's width (16) though the code's default (256) is what the command gives."""
    new_run(monkeypatch, tmp_path)
    options = list(cases.TRAIN_OPTIONS)
    at = options.index("relational.hidden=16")
    del options[at - 1 : at + 1]
    bnb(monkeypatch, tmp_path, "train", "--from", "a", *options, "--updates", "1", "--name", "b")
    config = json.loads((tmp_path / "b" / "config.json").read_text())
    assert config["relational"]["hidden"] == 16


def test_a_resume_on_another_commit_warns_and_records_it(monkeypatch, tmp_path, capsys):
    new_run(monkeypatch, tmp_path)
    head = provenance.commit()[0]
    monkeypatch.setattr(provenance, "commit", lambda: ("0123abcd", True))
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "3")
    assert (
        f"warning: the run continues on commit 0123abcd; its last segment ran on {head}"
        in capsys.readouterr().out
    )
    first, second = Run.open(tmp_path, "a").record["segments"]
    assert (first["commit"], second["commit"], second["dirty"]) == (head, "0123abcd", True)


def test_a_rewind_keeps_the_newer_checkpoints_aside(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path, updates=1)
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2")
    bnb(
        monkeypatch,
        tmp_path,
        "train",
        "--resume",
        "a",
        "--at",
        str(STEPS_PER_UPDATE),
        "--device",
        "cpu",
        "--updates",
        "2",
    )
    run = Run.open(tmp_path, "a")
    third = run.record["segments"][2]
    assert (third["start_step"], third["rewound_from"]) == (STEPS_PER_UPDATE, 2 * STEPS_PER_UPDATE)
    assert (
        run.directory
        / "checkpoints"
        / "rewound_2"
        / f"step_{2 * STEPS_PER_UPDATE}"
        / checkpoint.RECORD
    ).is_file()
    assert run.checkpoint_steps() == [STEPS_PER_UPDATE, 2 * STEPS_PER_UPDATE]


def test_a_fork_is_a_new_run_naming_its_parent_that_trains_its_own_updates(monkeypatch, tmp_path):
    """The guide's fork example, from a parent with more updates than the fork's budget."""
    parent = new_run(monkeypatch, tmp_path, updates=3)
    bnb(
        monkeypatch,
        tmp_path,
        "train",
        "--from",
        "a",
        *cases.TRAIN_OPTIONS,
        "--set",
        "learner.entropy_shoot=-0.005",
        "--updates",
        "2",
        "--name",
        "b",
    )
    fork = Run.open(tmp_path, "b")
    assert fork.record["parent"] == dict(run="a", step=3 * STEPS_PER_UPDATE)
    (segment,) = fork.record["segments"]
    assert (segment["start_step"], segment["end_step"]) == (
        3 * STEPS_PER_UPDATE,
        5 * STEPS_PER_UPDATE,
    )
    assert fork.record["status"] == "finished"
    assert checkpoint.read_policy(fork.checkpoint_at()).steps["updates"] == 2
    assert parent.checkpoint_steps() == [3 * STEPS_PER_UPDATE]  # the parent is untouched


def test_a_fork_whose_build_differs_is_refused(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="the fork refused: .* config.hidden"):
        bnb(
            monkeypatch,
            tmp_path,
            "train",
            "--from",
            "a",
            *cases.TRAIN_OPTIONS,
            "--set",
            "relational.hidden=32",
        )


def test_a_warm_start_from_a_policy_file_starts_a_new_run(monkeypatch, tmp_path):
    """Acceptance 9 through the command line: BC's policy file starts RL."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer = Trainer(cases.tiny_train_config(), cases.CPU)
    policy = checkpoint.save_policy(
        tmp_path / "bc", trainer.model, return_scale=trainer.ppo.return_scale
    )
    bnb(
        monkeypatch,
        tmp_path / "runs",
        "train",
        "--init-from",
        str(policy),
        *cases.TRAIN_OPTIONS,
        "--updates",
        "1",
        "--name",
        "rl",
    )
    run = Run.open(tmp_path / "runs", "rl")
    assert run.record["parent"] == dict(policy=str(policy.resolve()))
    assert run.record["segments"][0]["start_step"] == 0


@pytest.mark.parametrize(
    "setting, shoot",
    [((), WARM_START_ENTROPY), (("--set", "learner.entropy_shoot=-0.005"), -0.005)],
)
def test_a_warm_start_takes_its_own_entropy_coefficient(
    monkeypatch, tmp_path, capsys, setting, shoot
):
    """The BC -> RL hand-off uses WARM_START_ENTROPY on every factor --set does not
    choose."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer = Trainer(cases.tiny_train_config(), cases.CPU)
    policy = checkpoint.save_policy(
        tmp_path / "bc", trainer.model, return_scale=trainer.ppo.return_scale
    )
    capsys.readouterr()
    bnb(
        monkeypatch,
        tmp_path / "runs",
        "train",
        "--init-from",
        str(policy),
        *cases.TRAIN_OPTIONS,
        *setting,
        "--print-config",
    )
    learner = json.loads(capsys.readouterr().out)["learner"]
    assert learner["entropy_power"] == learner["entropy_stick"] == WARM_START_ENTROPY
    assert learner["entropy_shoot"] == shoot


def test_a_failure_marks_the_run_failed_with_the_message(monkeypatch, tmp_path):
    calls = []
    update = Trainer.update

    def failing(self):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the second update broke")
        return update(self)

    monkeypatch.setattr(Trainer, "update", failing)
    with pytest.raises(RuntimeError, match="the second update broke"):
        new_run(monkeypatch, tmp_path, updates=3)
    run = Run.open(tmp_path, "a")
    assert run.record["status"] == "failed"
    assert run.record["message"] == "RuntimeError: the second update broke"
    assert run.record["segments"][0]["end_step"] == STEPS_PER_UPDATE


def test_an_interrupt_finishes_the_update_writes_a_checkpoint_and_stops(monkeypatch, tmp_path):
    """The quick variant of the next test: SIGINT raised in this process during an update."""
    update = Trainer.update

    def interrupted(self):
        os.kill(os.getpid(), signal.SIGINT)  # the session's handler: stop after this update
        return update(self)

    monkeypatch.setattr(Trainer, "update", interrupted)
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "1000", "--name", "a")
    run = Run.open(tmp_path, "a")
    step = run.record["segments"][0]["end_step"]
    assert run.record["status"] == "stopped" and run.checkpoint_steps() == [step]
    assert step == STEPS_PER_UPDATE and run.record["final_checkpoint"] is None


@pytest.mark.slow  # over 10 s: a fresh process imports torch and trains
def test_ctrl_c_finishes_the_update_writes_a_checkpoint_and_stops(tmp_path):
    arguments = [
        "bnb",
        "train",
        *cases.TRAIN_OPTIONS,
        "--updates",
        "1000",
        "--name",
        "a",
        "--runs-directory",
        str(tmp_path),
    ]
    code = f"import sys; sys.argv = {arguments!r}; from bnb.cli import main; main()"
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
    for line in process.stdout:
        if line.startswith("update"):
            break
    process.send_signal(signal.SIGINT)
    output = process.communicate(timeout=300)[0]
    assert process.returncode == 0, output
    run = Run.open(tmp_path, "a")
    step = run.record["segments"][0]["end_step"]
    assert run.record["status"] == "stopped" and run.checkpoint_steps() == [step]
    assert 0 < step < 1000 * STEPS_PER_UPDATE and run.record["final_checkpoint"] is None


def test_bnb_runs_lists_and_shows_runs_with_a_stale_one(monkeypatch, tmp_path, capsys):
    run = new_run(monkeypatch, tmp_path)
    stale = Run.create(tmp_path, "b", cases.tiny_train_config(), ["bnb", "train"], 64)
    stale.start_segment(0)
    stale.update(pid=2**22 + 12345)
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["bnb", "runs", "--runs-directory", str(tmp_path)])
    cli.main()
    listing = capsys.readouterr().out.splitlines()
    assert listing[1].split()[:3] == ["a", "finished", str(2 * STEPS_PER_UPDATE)]
    assert listing[2].split()[:3] == ["b", "running", "(stale)"]
    assert provenance.commit()[0][:10] in listing[1]
    monkeypatch.setattr(
        sys, "argv", ["bnb", "runs", "show", "a", "--runs-directory", str(tmp_path)]
    )
    cli.main()
    shown = capsys.readouterr().out
    assert shown.startswith("run a: finished")
    assert f"0: steps 0 to {2 * STEPS_PER_UPDATE} at {run.record['segments'][0]['commit']}" in shown


def test_print_config_prints_the_resolved_configuration(monkeypatch, tmp_path, capsys):
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--print-config")
    config = json.loads(capsys.readouterr().out)
    assert config["environments"] == 16 and config["learner"]["minibatches"] == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())  # no run written


def test_the_probe_on_the_cpu_measures_nothing_and_writes_no_run(monkeypatch, tmp_path, capsys):
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--probe")
    assert "memory probe: CPU run, nothing measured" in capsys.readouterr().out
    assert not tmp_path.exists() or not any(tmp_path.iterdir())

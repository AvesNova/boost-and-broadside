"""`bnb train` and `bnb runs` (runs spec 5, 6 and acceptance 4, 8, 9, 10): the run record
across a new run, a stop by Ctrl-C, a resume with an override, a resume on another
commit, a rewind, a fork and a failure; the refusals; the listing."""

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
from bnb.runs import checkpoint, provenance
from bnb.runs.records import Run
from bnb.runtime.train import Trainer

STEPS_PER_UPDATE = 16 * 16  # environments x segment length in TRAIN_OPTIONS


def bnb(monkeypatch, runs_directory: pathlib.Path, *arguments: str) -> None:
    """`bnb ARGUMENTS --runs-directory RUNS` in this process, its seeding kept local."""
    monkeypatch.setattr(sys, "argv", ["bnb", *arguments, "--runs-directory",
                                      str(runs_directory)])
    with torch.random.fork_rng():
        cli.main()


def new_run(monkeypatch, runs_directory, name: str = "a", updates: int = 2) -> Run:
    bnb(monkeypatch, runs_directory, "train", *cases.TRAIN_OPTIONS, "--updates", str(updates),
        "--name", name)
    return Run.open(runs_directory, name)


def test_a_new_run_writes_its_record_config_and_final_checkpoint(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path)
    record = run.record
    assert record["status"] == "finished" and record["final_checkpoint"] == 2 * STEPS_PER_UPDATE
    assert run.checkpoint_steps() == [2 * STEPS_PER_UPDATE]
    assert record["probe"] is None  # the CPU records nothing
    assert record["command"][:2] == ["bnb", "train"]
    segment, = record["segments"]
    assert (segment["start_step"], segment["end_step"]) == (0, 2 * STEPS_PER_UPDATE)
    assert segment["commit"] == provenance.commit()[0]
    config = json.loads((run.directory / "config.json").read_text())
    assert config["num_envs"] == 16 and config["length"] == 16
    for part in ("league", "artifacts", "metrics.jsonl"):
        assert (run.directory / part).exists()


def test_a_resume_with_an_override_records_it_in_a_new_segment(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "3",
        "--set", "learner.learning_rate=0.0001")
    run = Run.open(tmp_path, "a")
    first, second = run.record["segments"]
    assert second["start_step"] == first["end_step"] == 2 * STEPS_PER_UPDATE
    assert second["end_step"] == 3 * STEPS_PER_UPDATE
    assert second["overrides"] == {"updates": 3, "learner.learning_rate": 0.0001}
    assert checkpoint.training_config(run.checkpoint_at())[0].learner.learning_rate == 0.0001


def test_a_structural_option_on_resume_is_refused_pointing_to_a_fork(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="a resume takes --num-envs from the checkpoint and "
                                         "cannot change it: start a new run from it with "
                                         "--from a"):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--num-envs", "32")
    with pytest.raises(SystemExit, match="a resume cannot --set length"):
        bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--set", "length=32")


def test_a_resume_on_another_commit_warns_and_records_it(monkeypatch, tmp_path, capsys):
    new_run(monkeypatch, tmp_path)
    head = provenance.commit()[0]
    monkeypatch.setattr(provenance, "commit", lambda: ("0123abcd", True))
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "3")
    assert f"warning: the run continues on commit 0123abcd; its last segment ran on {head}" \
        in capsys.readouterr().out
    first, second = Run.open(tmp_path, "a").record["segments"]
    assert (first["commit"], second["commit"], second["dirty"]) == (head, "0123abcd", True)


def test_a_rewind_keeps_the_newer_checkpoints_aside(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path, updates=1)
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2")
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--at", str(STEPS_PER_UPDATE),
        "--device", "cpu", "--updates", "2")
    run = Run.open(tmp_path, "a")
    third = run.record["segments"][2]
    assert (third["start_step"], third["rewound_from"]) == (STEPS_PER_UPDATE,
                                                            2 * STEPS_PER_UPDATE)
    assert (run.directory / "checkpoints" / "rewound_2" / f"step_{2 * STEPS_PER_UPDATE}"
            / checkpoint.RECORD).is_file()
    assert run.checkpoint_steps() == [STEPS_PER_UPDATE, 2 * STEPS_PER_UPDATE]


def test_a_fork_is_a_new_run_naming_its_parent(monkeypatch, tmp_path):
    parent = new_run(monkeypatch, tmp_path)
    bnb(monkeypatch, tmp_path, "train", "--from", "a", *cases.TRAIN_OPTIONS, "--updates", "3",
        "--name", "b")
    fork = Run.open(tmp_path, "b")
    assert fork.record["parent"] == dict(run="a", step=2 * STEPS_PER_UPDATE)
    assert fork.record["segments"][0]["start_step"] == 2 * STEPS_PER_UPDATE
    assert parent.checkpoint_steps() == [2 * STEPS_PER_UPDATE]  # the parent is untouched


def test_a_fork_whose_build_differs_is_refused(monkeypatch, tmp_path):
    new_run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="the fork refused: .* config.hidden"):
        bnb(monkeypatch, tmp_path, "train", "--from", "a", *cases.TRAIN_OPTIONS,
            "--set", "relational_hidden=32")


def test_a_warm_start_from_a_policy_file_starts_a_new_run(monkeypatch, tmp_path):
    """Acceptance 9 through the command line: BC's policy file starts RL."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        trainer = Trainer(cases.tiny_train_config(), cases.CPU)
    policy = checkpoint.save_policy(tmp_path / "bc", trainer.model,
                                    return_scale=trainer.ppo.return_scale)
    bnb(monkeypatch, tmp_path / "runs", "train", "--init-from", str(policy),
        *cases.TRAIN_OPTIONS, "--updates", "1", "--name", "rl")
    run = Run.open(tmp_path / "runs", "rl")
    assert run.record["parent"] == dict(policy=str(policy.resolve()))
    assert run.record["segments"][0]["start_step"] == 0


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


def test_ctrl_c_finishes_the_update_writes_a_checkpoint_and_stops(tmp_path):
    arguments = ["bnb", "train", *cases.TRAIN_OPTIONS, "--updates", "1000", "--name", "a",
                 "--runs-directory", str(tmp_path)]
    code = f"import sys; sys.argv = {arguments!r}; from bnb.cli import main; main()"
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               env=dict(os.environ, PYTHONUNBUFFERED="1"))
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
    monkeypatch.setattr(sys, "argv", ["bnb", "runs", "show", "a", "--runs-directory",
                                      str(tmp_path)])
    cli.main()
    shown = capsys.readouterr().out
    assert shown.startswith("run a: finished")
    assert f"0: steps 0 to {2 * STEPS_PER_UPDATE} at {run.record['segments'][0]['commit']}" \
        in shown


def test_print_config_prints_the_resolved_configuration(monkeypatch, tmp_path, capsys):
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--print-config")
    config = json.loads(capsys.readouterr().out)
    assert config["num_envs"] == 16 and config["learner"]["minibatches"] == 2
    assert not tmp_path.exists() or not any(tmp_path.iterdir())  # no run written


def test_the_probe_on_the_cpu_measures_nothing_and_writes_no_run(monkeypatch, tmp_path, capsys):
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--probe")
    assert "memory probe: CPU run, nothing measured" in capsys.readouterr().out
    assert not tmp_path.exists() or not any(tmp_path.iterdir())

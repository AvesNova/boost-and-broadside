"""A graceful stop (runs spec 7): a STOP file in the run directory, SIGTERM or SIGINT
finishes the update in progress, writes a checkpoint at its step, records the request in
run.json and returns cleanly; `--resume` then continues the run as if it had never
stopped (bitwise on the CPU), for PPO and BC."""

import os
import signal
import sys

import pytest
import runs_cases as cases
import torch

from bnb import cli
from bnb.runs import checkpoint
from bnb.runs.records import Run
from bnb.runs.session import STOP_FILE
from bnb.runtime.train import Trainer

STEPS_PER_UPDATE = 16 * 16  # environments x segment length in TRAIN_OPTIONS
BC_OPTIONS = (
    *cases.TRAIN_OPTIONS,
    "--objective",
    "bc",
    "--set",
    "lanes.environments=4",
)


def bnb(monkeypatch, runs_directory, *arguments: str) -> None:
    """`bnb ARGUMENTS --runs-directory RUNS` in this process, its seeding kept local."""
    monkeypatch.setattr(sys, "argv", ["bnb", *arguments, "--runs-directory", str(runs_directory)])
    with torch.random.fork_rng(devices=[]):
        cli.main()


def stopping_at(monkeypatch, update: int, request) -> None:
    """Make `request(trainer)` happen during the trainer's update number `update`."""
    original = Trainer.update

    def update_and_request(self):
        if self.updates_done + 1 == update:
            request(self)
        return original(self)

    monkeypatch.setattr(Trainer, "update", update_and_request)


def stop_file(runs_directory, name):
    return lambda trainer: (runs_directory / name / STOP_FILE).touch()


def sigterm(trainer) -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def budget(objective: str, updates: int) -> tuple[str, ...]:
    if objective == "bc":
        return ("--bc-budget-steps", str(updates * STEPS_PER_UPDATE))
    return ("--updates", str(updates))


def options(objective: str) -> tuple[str, ...]:
    return BC_OPTIONS if objective == "bc" else cases.TRAIN_OPTIONS


@pytest.mark.parametrize("request_kind", ["stop_file", "sigterm"])
@pytest.mark.parametrize("objective", ["ppo", "bc"])
def test_a_stop_request_after_update_k_checkpoints_at_k_and_returns(
    monkeypatch, tmp_path, request_kind, objective
):
    request = stop_file(tmp_path, "a") if request_kind == "stop_file" else sigterm
    stopping_at(monkeypatch, 2, request)
    bnb(monkeypatch, tmp_path, "train", *options(objective), *budget(objective, 10), "--name", "a")
    run = Run.open(tmp_path, "a")
    step = 2 * STEPS_PER_UPDATE
    assert run.record["status"] == "stopped" and run.checkpoint_steps() == [step]
    assert run.record["segments"][0]["end_step"] == step
    assert run.record["final_checkpoint"] is None
    by = "STOP file" if request_kind == "stop_file" else "SIGTERM"
    assert run.record["stop_request"] == dict(by=by, step=step)
    assert not (run.directory / STOP_FILE).exists()  # honoured, so removed
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL  # the handlers are put back
    if objective == "bc":
        assert run.record["stopped_by"] == "interrupted"


@pytest.mark.parametrize("objective", ["ppo", "bc"])
def test_a_stopped_and_resumed_run_equals_one_straight_through(monkeypatch, tmp_path, objective):
    bnb(monkeypatch, tmp_path, "train", *options(objective), *budget(objective, 4), "--name", "s")
    with monkeypatch.context() as patched:
        stopping_at(patched, 2, stop_file(tmp_path, "r"))
        bnb(patched, tmp_path, "train", *options(objective), *budget(objective, 4), "--name", "r")
    resumed = Run.open(tmp_path, "r")
    assert resumed.record["status"] == "stopped"
    bnb(monkeypatch, tmp_path, "train", "--resume", "r", "--device", "cpu")
    resumed, straight = Run.open(tmp_path, "r"), Run.open(tmp_path, "s")
    assert resumed.record["status"] == straight.record["status"] == "finished"
    assert resumed.record["stop_request"] is None  # the new segment cleared it
    assert resumed.checkpoint_steps()[-1] == straight.checkpoint_steps()[-1] == 4 * STEPS_PER_UPDATE
    with torch.random.fork_rng(devices=[]):  # a model's constructor draws its weights
        got = checkpoint.load_policy(resumed.checkpoint_at(), cases.CPU).state_dict()
        want = checkpoint.load_policy(straight.checkpoint_at(), cases.CPU).state_dict()
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name


def test_a_stale_stop_file_is_removed_when_a_run_starts(monkeypatch, tmp_path, capsys):
    bnb(monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "1", "--name", "a")
    (tmp_path / "a" / STOP_FILE).touch()
    bnb(monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2")
    run = Run.open(tmp_path, "a")
    assert run.record["status"] == "finished"
    assert run.checkpoint_steps()[-1] == 2 * STEPS_PER_UPDATE
    assert "left by an earlier process" in capsys.readouterr().out

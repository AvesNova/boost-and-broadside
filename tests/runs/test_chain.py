"""`bnb train --objective chain` (`bnb.runs.chain`): BC until its stop rule fires, the
handover to RL from the clone's exponential average (a fresh optimizer, its warm-up and
the warm start's entropy), the combined budget, the records, and a resume in BC, at the
handover and in RL. CPU, tiny configurations; the plateau is replaced by a rule that fires
at a chosen step, so the handover falls where a test puts it."""

import json
import os
import pathlib
import signal
import sys

import pytest
import runs_cases as cases
import safetensors.torch
import torch

from bnb import cli
from bnb.learn import bc
from bnb.learn.ppo import WARM_START_ENTROPY, LearnerConfig
from bnb.runs import checkpoint
from bnb.runs.chain import Chain, reinforcement_updates
from bnb.runs.records import Run
from bnb.runtime.train import Trainer

STEPS = 16 * 16  # environments x segment length in TRAIN_OPTIONS
CHAIN_OPTIONS = (
    *cases.TRAIN_OPTIONS,
    "--objective",
    "chain",
    "--bc-set",
    "lanes.environments=4",
    "--bc-set",
    "league.uniform_from=0",
)


def bnb(monkeypatch, runs_directory: pathlib.Path, *arguments: str) -> None:
    """`bnb ARGUMENTS --runs-directory RUNS` in this process, its seeding kept local."""
    monkeypatch.setattr(sys, "argv", ["bnb", *arguments, "--runs-directory", str(runs_directory)])
    with torch.random.fork_rng(devices=[]):
        cli.main()


def plateau_at(monkeypatch, step: int) -> None:
    """BC's plateau fires at the first update at or past `step`."""
    monkeypatch.setattr(bc.Plateau, "observe", lambda self, steps, kl: steps >= step)


def chain(monkeypatch, runs, budget: int, *extra: str, name: str = "c") -> Chain:
    bnb(
        monkeypatch, runs, "train", *CHAIN_OPTIONS, "--chain-budget-steps", str(budget),
        "--name", name, *extra,
    )  # fmt: skip
    return Chain.open(runs, name)


def resume(monkeypatch, runs, *extra: str, name: str = "c") -> Chain:
    bnb(monkeypatch, runs, "train", "--resume", name, *extra)
    return Chain.open(runs, name)


def rows(run: Run) -> list[dict]:
    return [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().split("\n")
            if line]  # fmt: skip


def stop_at_update(monkeypatch, count: int) -> None:
    """SIGINT during the `count`-th update from now: the run stops after it."""
    calls, update = [], Trainer.update

    def interrupted(self):
        calls.append(1)
        if len(calls) == count:
            os.kill(os.getpid(), signal.SIGINT)
        return update(self)

    monkeypatch.setattr(Trainer, "update", interrupted)


def test_the_reinforcement_budget_is_the_whole_updates_left():
    assert reinforcement_updates(1000, 300, 100) == 7
    assert reinforcement_updates(1000, 350, 100) == 6  # a part update is not trained
    assert reinforcement_updates(1000, 1000, 100) == 0
    assert reinforcement_updates(1000, 1100, 100) == 0  # BC's last update overshot


def test_a_chain_clones_hands_over_and_reinforces(monkeypatch, tmp_path):
    """BC stops on its plateau after two updates; RL starts from the clone's exponential
    average with a fresh optimizer and step count (so its warm-up starts again) and the
    warm start's entropy, and trains the three updates the budget of five leaves."""
    plateau_at(monkeypatch, 2 * STEPS)
    chained = chain(monkeypatch, tmp_path, 5 * STEPS)
    handover = chained.record["handover"]
    bc_run, rl_run = Run.open(tmp_path, "c-bc"), Run.open(tmp_path, "c-rl")
    policy = bc_run.variant_policy_path("exponential")
    assert handover == dict(step=2 * STEPS, stopped_by="plateau", policy=str(policy))
    assert bc_run.record["chain"] == dict(name="c", stage="bc", start_step=0)
    assert rl_run.record["chain"] == dict(name="c", stage="rl", start_step=2 * STEPS)
    assert bc_run.record["status"] == rl_run.record["status"] == "finished"
    assert bc_run.record["stopped_by"] == "plateau"
    assert rl_run.record["parent"] == dict(policy=str(policy.resolve()))
    # each stage starts with a checkpoint at its step 0
    assert bc_run.checkpoint_steps() == [0, 2 * STEPS]
    assert rl_run.checkpoint_steps() == [0, 3 * STEPS]
    bc_config, _ = checkpoint.training_config(bc_run.checkpoint_at())
    rl_config, _ = checkpoint.training_config(rl_run.checkpoint_at())
    assert bc_config.objective == "bc" and bc_config.stop.budget_steps == 5 * STEPS
    assert bc_config.lanes.environments == 4 and rl_config.lanes.environments == 0
    assert rl_config.objective == "ppo" and rl_config.updates == 3
    learner = rl_config.learner
    assert learner.entropy_power == learner.entropy_stick == WARM_START_ENTROPY
    assert learner.policy == LearnerConfig().policy  # PPO's coefficients, not BC's
    # the RL stage's start: the exponential average's weights, no optimizer state, step 0
    start = rl_run.checkpoint_path(0)
    clone = safetensors.torch.load_file(policy / checkpoint.WEIGHTS)
    started = safetensors.torch.load_file(start / checkpoint.WEIGHTS)
    assert clone.keys() == started.keys()
    for name, value in clone.items():  # copied, not computed
        assert torch.equal(started[name], value), name
    training = safetensors.torch.load_file(start / checkpoint.TRAINING)
    assert not any(name.startswith("optimizer.") for name in training)
    assert checkpoint.read_record(start)["steps"] == dict(environment_steps=0, updates=0)
    first = rows(rl_run)[0]
    assert first["schedule/learning_rate"] < learner.learning_rate  # in its warm-up
    # every record on the chain's axis
    for run, offset in ((bc_run, 0), (rl_run, 2 * STEPS)):
        for row in rows(run):
            chained_steps = row["counters/chain_environment_steps"]
            assert chained_steps == offset + row["counters/environment_steps"]
    assert [row["counters/chain_environment_steps"] for row in rows(rl_run)] == [
        step * STEPS for step in (3, 4, 5)
    ]


def test_the_combined_budget_caps_bc(monkeypatch, tmp_path):
    """A BC stage that never plateaus stops at the combined budget; the RL stage then has
    no update left and ends where it began."""
    plateau_at(monkeypatch, 10**12)
    chained = chain(monkeypatch, tmp_path, 2 * STEPS)
    assert chained.record["handover"]["stopped_by"] == "budget"
    assert chained.record["handover"]["step"] == 2 * STEPS
    rl_run = Run.open(tmp_path, "c-rl")
    assert rl_run.record["status"] == "finished" and rl_run.checkpoint_steps() == [0]
    assert rows(rl_run) == []


def test_a_chain_resumes_in_bc(monkeypatch, tmp_path):
    """Stopped during BC's first update, the chain has no handover and no RL stage; its
    resume continues the BC stage from its checkpoint, then hands over."""
    plateau_at(monkeypatch, 2 * STEPS)
    stop_at_update(monkeypatch, 1)
    chained = chain(monkeypatch, tmp_path, 4 * STEPS)
    bc_run = Run.open(tmp_path, "c-bc")
    assert bc_run.record["status"] == "stopped" and chained.record["handover"] is None
    assert not (tmp_path / "c-rl").exists()
    monkeypatch.undo()
    plateau_at(monkeypatch, 2 * STEPS)
    chained = resume(monkeypatch, tmp_path, "--device", "cpu")
    bc_run = Run.open(tmp_path, "c-bc")
    first, second = bc_run.record["segments"]
    assert (first["end_step"], second["start_step"], second["end_step"]) == (
        STEPS,
        STEPS,
        2 * STEPS,
    )
    assert chained.record["handover"]["step"] == 2 * STEPS
    assert Run.open(tmp_path, "c-rl").record["status"] == "finished"
    assert [row["counters/updates"] for row in rows(bc_run)] == [1, 2]


def test_a_chain_resumes_at_the_handover(monkeypatch, tmp_path):
    """The BC stage finished and the handover recorded, but the RL stage failed before its
    first update: the resume takes the RL stage up from its step-0 checkpoint."""
    plateau_at(monkeypatch, STEPS)
    update = Trainer.update

    def failing(self):
        if self.config.objective == "ppo":
            raise RuntimeError("the first RL update broke")
        return update(self)

    monkeypatch.setattr(Trainer, "update", failing)
    with pytest.raises(RuntimeError, match="the first RL update broke"):
        chain(monkeypatch, tmp_path, 3 * STEPS)
    rl_run = Run.open(tmp_path, "c-rl")
    assert rl_run.record["status"] == "failed" and rl_run.checkpoint_steps() == [0]
    monkeypatch.undo()
    chained = resume(monkeypatch, tmp_path)  # the device from the chain's record
    rl_run = Run.open(tmp_path, "c-rl")
    assert rl_run.record["status"] == "finished"
    assert rl_run.record["segments"][-1]["start_step"] == 0
    assert rl_run.checkpoint_steps()[-1] == 2 * STEPS
    assert len(Run.open(tmp_path, "c-bc").record["segments"]) == 1  # BC was not resumed
    assert chained.record["handover"]["step"] == STEPS


def test_a_chain_resumes_in_rl_with_a_larger_budget(monkeypatch, tmp_path):
    """Stopped in its second RL update, the chain resumes in RL; a larger combined budget
    gives the RL stage more updates."""
    plateau_at(monkeypatch, STEPS)
    stop_at_update(monkeypatch, 3)  # BC's one update, then RL's second
    chain(monkeypatch, tmp_path, 4 * STEPS)
    rl_run = Run.open(tmp_path, "c-rl")
    assert rl_run.record["status"] == "stopped"
    assert rl_run.record["segments"][0]["end_step"] == 2 * STEPS
    monkeypatch.undo()
    chained = resume(monkeypatch, tmp_path, "--chain-budget-steps", str(6 * STEPS))
    assert chained.record["budget_steps"] == 6 * STEPS
    rl_run = Run.open(tmp_path, "c-rl")
    assert rl_run.record["status"] == "finished"
    assert rl_run.record["segments"][-1]["overrides"] == {"updates": 5}
    assert rl_run.record["segments"][-1]["end_step"] == 5 * STEPS
    assert [row["counters/chain_environment_steps"] for row in rows(rl_run)] == [
        step * STEPS for step in range(2, 7)
    ]


def test_print_config_shows_both_stages(monkeypatch, tmp_path, capsys):
    """--set reaches both stages, --bc-set and --rl-set one each; BC's budget is the
    combined budget, RL takes the warm start's entropy."""
    capsys.readouterr()
    bnb(
        monkeypatch, tmp_path, "train", *CHAIN_OPTIONS, "--chain-budget-steps", "1000",
        "--rl-set", "learner.learning_rate=0.001", "--print-config",
    )  # fmt: skip
    configs = json.loads(capsys.readouterr().out)
    bc_config, rl_config = configs["bc"], configs["rl"]
    assert bc_config["objective"] == "bc" and rl_config["objective"] == "ppo"
    assert bc_config["stop"]["budget_steps"] == 1000
    assert bc_config["lanes"]["environments"] == 4 and rl_config["lanes"]["environments"] == 0
    assert rl_config["learner"]["learning_rate"] == 0.001
    assert bc_config["learner"]["learning_rate"] == bc.config().learning_rate
    assert rl_config["learner"]["entropy_shoot"] == WARM_START_ENTROPY
    assert bc_config["relational"] == rl_config["relational"]  # the shared --set
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments, message",
    [
        (CHAIN_OPTIONS, "a chain needs --chain-budget-steps"),
        ((*CHAIN_OPTIONS, "--chain-budget-steps", "9", "--updates", "3"), "--updates itself"),
        ((*CHAIN_OPTIONS, "--chain-budget-steps", "9", "--init-from", "x"), "--init-from"),
    ],
)
def test_a_chain_refuses_what_it_sets_itself(monkeypatch, tmp_path, arguments, message):
    with pytest.raises(SystemExit, match=message):
        bnb(monkeypatch, tmp_path, "train", *arguments)


def test_a_chains_resume_takes_its_settings_from_its_record(monkeypatch, tmp_path):
    plateau_at(monkeypatch, STEPS)
    chain(monkeypatch, tmp_path, STEPS)
    with pytest.raises(SystemExit, match="--environments from the chain's record"):
        resume(monkeypatch, tmp_path, "--environments", "32")

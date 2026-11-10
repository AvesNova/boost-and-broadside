"""Every update's policy files (`policies/`), the offline EMA built from them, and the
retention's newest two."""

import json

import pytest
import runs_cases as cases
import safetensors.torch
import torch
from test_cli import STEPS_PER_UPDATE, bnb, new_run

from bnb.runs import checkpoint, ema
from bnb.runs.records import Run


def test_a_ppo_run_writes_the_live_and_both_averages_after_every_update(monkeypatch, tmp_path):
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "4", "--name", "a",
        "--checkpoint-every", str(STEPS_PER_UPDATE),
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    assert run.record["update_policies"] is True
    steps = [STEPS_PER_UPDATE * n for n in (1, 2, 3, 4)]
    assert ema.live_steps(run) == steps
    for step in steps:
        names = {p.name for p in run.update_policies_path(step).iterdir()}
        assert (
            {"live", "exponential_average"}
            <= names
            <= {
                "live",
                "exponential_average",
                "uniform_average",
            }
        )
        for name in names:
            checkpoint.read_policy(run.update_policies_path(step) / name)  # complete, hashed
    # the training states: the newest two (the final is the newest)
    assert run.checkpoint_steps() == [STEPS_PER_UPDATE * 3, STEPS_PER_UPDATE * 4]


def test_each_averages_policy_file_holds_its_own_average(monkeypatch, tmp_path):
    """The uniform average file is the mean of the live weights so far and the exponential
    average file is a different average (2026-11-10: both names held the uniform average,
    saved after loading both into one scratch model)."""
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "3", "--name", "a",
        "--set", "league.uniform_from=0",
    )  # fmt: skip
    run = Run.open(tmp_path, "a")

    def weights(step: int, name: str) -> dict:
        return safetensors.torch.load_file(
            run.update_policies_path(step) / name / "weights.safetensors"
        )

    newest = 3 * STEPS_PER_UPDATE
    uniform = weights(newest, "uniform_average")
    exponential = weights(newest, "exponential_average")
    floats = [name for name, value in uniform.items() if value.is_floating_point()]
    assert any(not torch.equal(uniform[name], exponential[name]) for name in floats)
    lives = [weights(STEPS_PER_UPDATE * n, "live") for n in (1, 2, 3)]
    for name in floats:
        mean = sum(live[name].double() for live in lives) / len(lives)
        assert torch.allclose(uniform[name].double(), mean, atol=1e-5)


def test_the_last_updates_live_policy_is_the_final_checkpoints_weights(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path, updates=2)
    live = run.update_policies_path(2 * STEPS_PER_UPDATE) / "live" / checkpoint.WEIGHTS
    final = run.checkpoint_at() / checkpoint.WEIGHTS
    saved, kept = safetensors.torch.load_file(live), safetensors.torch.load_file(final)
    assert saved.keys() == kept.keys()
    assert all(torch.equal(saved[name], kept[name]) for name in saved)


def test_the_option_turns_the_per_update_policies_off_and_a_resume_may_change_it(
    monkeypatch, tmp_path
):
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "1", "--name", "a",
        "--update-policies", "False",
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    # off: only the snapshots of the cadence (policy_snapshot_every) and the segment's end
    assert run.record["update_policies"] is False
    assert ema.live_steps(run) == [STEPS_PER_UPDATE]
    bnb(
        monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2",
        "--update-policies", "True",
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    assert ema.live_steps(run) == [STEPS_PER_UPDATE, 2 * STEPS_PER_UPDATE]


def test_an_ema_of_decay_zero_is_the_newest_live_and_the_run_s_own_matches_at_its_start(
    monkeypatch, tmp_path
):
    run = new_run(monkeypatch, tmp_path, updates=3)
    newest = 3 * STEPS_PER_UPDATE
    weights, end, count = ema.average(run, 0.0)
    assert (end, count) == (newest, 3)
    live = safetensors.torch.load_file(
        run.update_policies_path(newest) / "live" / "weights.safetensors"
    )
    assert all(torch.equal(weights[name], live[name]) for name in live)
    # an EMA by hand over the three saved updates, and `--at` truncating it
    files = [
        safetensors.torch.load_file(
            run.update_policies_path(STEPS_PER_UPDATE * n) / "live" / "weights.safetensors"
        )
        for n in (1, 2, 3)
    ]
    by_hand, _, _ = ema.average(run, 0.5, at=2 * STEPS_PER_UPDATE)
    for name, value in files[0].items():
        if value.is_floating_point():
            expected = 0.5 * files[0][name] + 0.5 * files[1][name]
            assert torch.allclose(by_hand[name], expected)


def test_bnb_runs_ema_writes_a_policy_file_that_loads(monkeypatch, tmp_path, capsys):
    run = new_run(monkeypatch, tmp_path, updates=3)
    bnb(
        monkeypatch,
        tmp_path,
        "runs",
        "ema",
        "a",
        "--decay",
        "0.9",
        "--at",
        str(2 * STEPS_PER_UPDATE),
    )
    path = capsys.readouterr().out.strip().splitlines()[-1]
    assert path.endswith(f"decay_0.9_step_{2 * STEPS_PER_UPDATE}")
    with torch.random.fork_rng(devices=[]):  # the model's constructor draws its weights
        model = checkpoint.load_policy(path, torch.device("cpu"))
    record = json.loads((run.directory / path / checkpoint.RECORD).read_text())
    assert record["steps"]["ema_decay"] == 0.9 and record["steps"]["ema_updates"] == 2
    del model


def test_an_ema_needs_saved_live_policies_and_a_decay_below_one(monkeypatch, tmp_path):
    run = new_run(monkeypatch, tmp_path, updates=1)
    with pytest.raises(ValueError, match="need 0 <= decay < 1"):
        ema.average(run, 1.0)
    with pytest.raises(FileNotFoundError, match="no saved live policies up to step 1"):
        ema.average(run, 0.5, at=1)


def test_a_rewind_sets_aside_the_newer_updates_policy_files(monkeypatch, tmp_path):
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "3", "--name", "a",
        "--checkpoint-every", str(STEPS_PER_UPDATE),
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    middle = 2 * STEPS_PER_UPDATE
    assert run.checkpoint_steps() == [middle, 3 * STEPS_PER_UPDATE]  # the newest two
    bnb(
        monkeypatch, tmp_path, "train", "--resume", "a", "--at", str(middle), "--device", "cpu",
        "--updates", "3",
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    assert ema.live_steps(run) == [STEPS_PER_UPDATE, middle, 3 * STEPS_PER_UPDATE]
    aside = list((run.directory / "policies").glob("rewound_*/step_*"))
    assert [p.name for p in aside] == [f"step_{3 * STEPS_PER_UPDATE}"]


BC_OPTIONS = (
    *cases.TRAIN_OPTIONS,
    "--objective",
    "bc",
    "--set",
    "lanes.environments=4",
    "--set",
    "league.uniform_from=0",
    "--set",
    f"policy_snapshot_every={2 * STEPS_PER_UPDATE}",
)


def test_a_bc_run_keeps_policy_snapshots_on_its_cadence_and_at_each_end(monkeypatch, tmp_path):
    """BC writes no per-update policies by default, but keeps the live clone and both
    averages every `policy_snapshot_every` environment steps and at each segment's end, so
    a run continued past its budget keeps its earlier averages; `bnb ratings --with` takes
    them by their path in the runs directory."""
    from bnb.league.matches import player

    budget = ("--bc-budget-steps", str(3 * STEPS_PER_UPDATE))
    bnb(monkeypatch, tmp_path, "train", *BC_OPTIONS, *budget, "--name", "clone")
    run = Run.open(tmp_path, "clone")
    assert run.record["update_policies"] is False
    assert ema.live_steps(run) == [2 * STEPS_PER_UPDATE, 3 * STEPS_PER_UPDATE]
    earlier = run.update_policies_path(2 * STEPS_PER_UPDATE)
    files = {
        name: (earlier / name / checkpoint.WEIGHTS).read_bytes()
        for name in ("live", "exponential_average", "uniform_average")
    }
    bnb(
        monkeypatch, tmp_path, "train", "--resume", "clone", "--device", "cpu",
        "--bc-budget-steps", str(5 * STEPS_PER_UPDATE),
    )  # fmt: skip
    run = Run.open(tmp_path, "clone")
    assert ema.live_steps(run) == [step * STEPS_PER_UPDATE for step in (2, 3, 4, 5)]
    for name, content in files.items():  # the earlier averages are kept as they were
        assert (earlier / name / checkpoint.WEIGHTS).read_bytes() == content, name
    record = checkpoint.read_record(run.update_policies_path(5 * STEPS_PER_UPDATE) / "live")
    assert record["steps"] == dict(environment_steps=5 * STEPS_PER_UPDATE, updates=5)
    name = f"clone/policies/step_{2 * STEPS_PER_UPDATE}/uniform_average"
    assert player(name, tmp_path).policy == tmp_path / name

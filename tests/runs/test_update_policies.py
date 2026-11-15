"""Every update's policy files (`policies/`) and the retention's newest two; the averages
built from them are in test_averaging."""

import runs_cases as cases
import safetensors.torch
import torch
from test_cli import STEPS_PER_UPDATE, bnb, new_run

from bnb.runs import checkpoint
from bnb.runs.averaging import saved_steps
from bnb.runs.records import Run


def test_a_ppo_run_writes_the_live_and_both_averages_after_every_update_and_at_its_start(
    monkeypatch, tmp_path
):
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "4", "--name", "a",
        "--update-policies", "True",
        "--checkpoint-every", str(STEPS_PER_UPDATE),
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    assert run.record["update_policies"] is True
    steps = [STEPS_PER_UPDATE * n for n in (0, 1, 2, 3, 4)]
    assert saved_steps(run) == steps
    start = checkpoint.read_record(run.update_policies_path(0) / "live")
    assert start["steps"] == dict(environment_steps=0, updates=0)  # the starting weights
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
        "--update-policies", "True",
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
    run = new_run(monkeypatch, tmp_path, updates=2, update_policies=True)
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
    # off: only the scheduled snapshots (policy_snapshots) and the segment's end
    assert run.record["update_policies"] is False
    assert saved_steps(run) == [STEPS_PER_UPDATE]
    bnb(
        monkeypatch, tmp_path, "train", "--resume", "a", "--device", "cpu", "--updates", "2",
        "--update-policies", "True",
    )  # fmt: skip
    run = Run.open(tmp_path, "a")
    assert saved_steps(run) == [STEPS_PER_UPDATE, 2 * STEPS_PER_UPDATE]


def test_a_rewind_sets_aside_the_newer_updates_policy_files(monkeypatch, tmp_path):
    bnb(
        monkeypatch, tmp_path, "train", *cases.TRAIN_OPTIONS, "--updates", "3", "--name", "a",
        "--update-policies", "True",
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
    assert saved_steps(run) == [0, STEPS_PER_UPDATE, middle, 3 * STEPS_PER_UPDATE]
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
    "policy_snapshots.steps=[]",
    "--set",
    f"policy_snapshots.every={2 * STEPS_PER_UPDATE}",
)


def test_a_bc_run_keeps_policy_snapshots_on_its_cadence_and_at_each_end(monkeypatch, tmp_path):
    """BC writes no per-update policies by default, but keeps the live clone and both
    averages on its schedule (`policy_snapshots`, here every two updates) and at each
    segment's end, so a run continued past its budget keeps its earlier averages; `bnb
    ratings --with` takes them by their path in the runs directory."""
    from bnb.league.matches import player

    budget = ("--bc-budget-steps", str(3 * STEPS_PER_UPDATE))
    bnb(monkeypatch, tmp_path, "train", *BC_OPTIONS, *budget, "--name", "clone")
    run = Run.open(tmp_path, "clone")
    assert run.record["update_policies"] is False
    assert saved_steps(run) == [2 * STEPS_PER_UPDATE, 3 * STEPS_PER_UPDATE]
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
    assert saved_steps(run) == [step * STEPS_PER_UPDATE for step in (2, 3, 4, 5)]
    for name, content in files.items():  # the earlier averages are kept as they were
        assert (earlier / name / checkpoint.WEIGHTS).read_bytes() == content, name
    record = checkpoint.read_record(run.update_policies_path(5 * STEPS_PER_UPDATE) / "live")
    assert record["steps"] == dict(environment_steps=5 * STEPS_PER_UPDATE, updates=5)
    name = f"clone/policies/step_{2 * STEPS_PER_UPDATE}/uniform_average"
    assert player(name, tmp_path).policy == tmp_path / name

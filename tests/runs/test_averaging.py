"""The averages built after training from a run's per-update live policies
(`bnb.runs.averaging`, `bnb average`): the run's own two averages rebuilt within rounding,
the window and start boundaries, the written files as players, and one pass over the files."""

import pathlib
import weakref

import pytest
import runs_cases as cases
import safetensors.torch
import torch
from test_cli import STEPS_PER_UPDATE, bnb

from bnb.league.matches import player
from bnb.runs import averaging, checkpoint
from bnb.runs.averaging import AverageSpec, RunFacts, Spec
from bnb.runs.records import Run

UPDATES = 5


@pytest.fixture(scope="module")
def runs_directory(tmp_path_factory) -> pathlib.Path:
    """A tiny PPO run of five updates with every update's policy files, its uniform
    average starting at the second update."""
    directory = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as monkeypatch:
        bnb(
            monkeypatch, directory, "train", *cases.TRAIN_OPTIONS, "--updates", str(UPDATES),
            "--name", "a", "--set", "update_policies=rl",
            "--set", f"league.uniform_from={2 * STEPS_PER_UPDATE}",
            # no warm-up and a large step, so the weights move between updates
            "--set", "learner.warmup_steps=0", "--set", "learner.learning_rate=0.01",
        )  # fmt: skip
    return directory


def _copy(runs_directory: pathlib.Path, tmp_path: pathlib.Path) -> Run:
    """A fresh copy of the run, so each test writes its own averages."""
    import shutil

    shutil.copytree(runs_directory / "a", tmp_path / "a")
    return Run.open(tmp_path, "a")


def _live(run: Run, update: int) -> dict[str, torch.Tensor]:
    return safetensors.torch.load_file(
        run.update_policies_path(update * STEPS_PER_UPDATE) / "live" / checkpoint.WEIGHTS
    )


def _floats(weights: dict) -> list[str]:
    return [name for name, value in weights.items() if value.is_floating_point()]


def _written(run: Run, name: str, update: int) -> dict[str, torch.Tensor]:
    path = run.directory / "average" / name / f"step_{update * STEPS_PER_UPDATE}"
    return safetensors.torch.load_file(path / checkpoint.WEIGHTS)


def test_the_run_s_own_averages_are_rebuilt_within_rounding_at_every_update(
    runs_directory, tmp_path
):
    run = _copy(runs_directory, tmp_path)
    every = tuple(update * STEPS_PER_UPDATE for update in range(1, UPDATES + 1))
    spec = Spec((AverageSpec("training_exponential"), AverageSpec("training_uniform")), steps=every)
    names, checks = averaging.build(run, spec, report=lambda line: None)
    # the exponential one at every update; the uniform one where the run had started its own
    assert [(check.name.split("_")[1], check.step) for check in checks] == sorted(
        [("exponential", step) for step in every]
        + [("uniform", step) for step in every if step >= 2 * STEPS_PER_UPDATE],
        key=lambda pair: (pair[1], pair[0]),
    )
    assert all(check.within_rounding for check in checks), checks
    assert len(names) == 2 * UPDATES - 1  # no uniform average before its start
    # and the uniform one is the mean of the live weights from its start
    lives = [_live(run, update) for update in range(2, UPDATES + 1)]
    built = _written(run, "training_uniform_from_step_" + str(2 * STEPS_PER_UPDATE), UPDATES)
    for name in _floats(built):
        mean = sum(live[name].double() for live in lives) / len(lives)
        assert torch.allclose(built[name].double(), mean, atol=1e-6)


def test_a_differing_rebuild_fails_the_check(runs_directory, tmp_path):
    run = _copy(runs_directory, tmp_path)
    path = run.update_policies_path(UPDATES * STEPS_PER_UPDATE) / "exponential_average"
    weights = safetensors.torch.load_file(path / checkpoint.WEIGHTS)
    name = _floats(weights)[0]
    weights[name] = weights[name] + 1.0
    safetensors.torch.save_file(weights, path / checkpoint.WEIGHTS)
    _, (check,) = averaging.build(
        run, Spec((AverageSpec("training_exponential"),)), report=lambda line: None
    )
    assert not check.within_rounding and check.largest_difference == pytest.approx(1.0)


def test_the_run_s_weights_move_between_updates(runs_directory):
    """Else every comparison here would hold trivially."""
    run = Run.open(runs_directory, "a")
    lives = [_live(run, update) for update in range(UPDATES + 1)]
    for earlier, later in zip(lives, lives[1:], strict=False):
        moved = max(float((later[name] - earlier[name]).abs().max()) for name in _floats(later))
        assert moved > 1e-3


def test_uniform_exponential_and_window_averages_match_their_definitions(runs_directory, tmp_path):
    run = _copy(runs_directory, tmp_path)
    halflife = 2.0
    spec = Spec(
        (
            AverageSpec("uniform", from_step=3 * STEPS_PER_UPDATE),
            AverageSpec("exponential", halflife_steps=int(halflife * STEPS_PER_UPDATE)),
            AverageSpec("window", last_updates=2),
            AverageSpec("live"),
        )
    )
    names, _ = averaging.build(run, spec, report=lambda line: None)
    final = UPDATES * STEPS_PER_UPDATE
    assert names == [
        f"a/average/uniform_from_step_{3 * STEPS_PER_UPDATE}/step_{final}",
        f"a/average/exponential_halflife_{2 * STEPS_PER_UPDATE}steps/step_{final}",
        f"a/average/window_last_2updates/step_{final}",
        f"a/policies/step_{final}/live",
    ]
    lives = [_live(run, update) for update in range(UPDATES + 1)]  # 0: the starting weights
    uniform = _written(run, f"uniform_from_step_{3 * STEPS_PER_UPDATE}", UPDATES)
    exponential = _written(run, f"exponential_halflife_{2 * STEPS_PER_UPDATE}steps", UPDATES)
    window = _written(run, "window_last_2updates", UPDATES)
    decay = 0.5 ** (1.0 / halflife)
    for name in _floats(uniform):
        values = [live[name].double() for live in lives]
        expected = values[0]
        for value in values[1:]:
            expected = decay * expected + (1.0 - decay) * value
        assert torch.allclose(exponential[name].double(), expected, atol=1e-6)
        assert torch.allclose(uniform[name].double(), sum(values[3:]) / 3, atol=1e-6)
        assert torch.allclose(window[name].double(), sum(values[4:]) / 2, atol=1e-6)


def test_the_written_averages_load_as_players_and_record_what_they_average(
    runs_directory, tmp_path, monkeypatch, capsys
):
    run = _copy(runs_directory, tmp_path)
    spec = tmp_path / "spec.toml"
    spec.write_text(
        "[evaluation]\nfractions = [0.6, 1.0]\n\n"
        '[[average]]\nfamily = "exponential"\nhalflife_updates = 1\n'
        'why = "a comment the tool ignores"\n'
    )
    bnb(monkeypatch, tmp_path, "average", "a", "--spec", str(spec))
    captured = capsys.readouterr()
    names = captured.out.split()
    # 0.6 of five updates is step 3; the progress goes to stderr
    assert names == [
        f"a/average/exponential_halflife_1updates/step_{update * STEPS_PER_UPDATE}"
        for update in (3, 5)
    ]
    assert "exponential_halflife_1updates at step" in captured.err
    for name in names:
        policy = player(name, tmp_path).policy
        with torch.random.fork_rng(devices=[]):  # the model's constructor draws its weights
            checkpoint.load_policy(policy, torch.device("cpu"))
        description = checkpoint.read_record(policy)["steps"]["average"]
        assert description["family"] == "exponential" and description["decay"] == 0.5
        assert (
            description["first_step"] == 0
            and description["files"] == int(name.rsplit("_", 1)[1]) // STEPS_PER_UPDATE + 1
        )
    # a second build keeps what is there and reads nothing
    reads = []
    monkeypatch.setattr(averaging, "load_weights", lambda path: reads.append(path))
    again, _ = averaging.build(run, averaging.read_spec(spec), report=lambda line: None)
    assert again == names and reads == []


def test_one_pass_reads_each_file_once_oldest_first_and_holds_one_at_a_time(
    runs_directory, tmp_path, monkeypatch
):
    run = _copy(runs_directory, tmp_path)
    loaded = averaging.load_weights
    reads, held = [], []

    def watched(path):
        if path.name == "live":  # every earlier file is let go before the next is read
            for tensors in held:
                assert all(reference() is None for reference in tensors), path
        weights = loaded(path)
        reads.append((averaging.step_of(path.parent), path.name))
        held.append([weakref.ref(value) for value in weights.values()])
        return weights

    monkeypatch.setattr(averaging, "load_weights", watched)
    spec = Spec(
        (
            AverageSpec("uniform", from_fraction=0.0),
            AverageSpec("exponential", halflife_updates=4),
            AverageSpec("window", last_updates=2),
            AverageSpec("training_exponential"),
        ),
        fractions=(0.4, 1.0),
    )
    averaging.build(run, spec, report=lambda line: None)
    lives = [step for step, name in reads if name == "live"]
    assert lives == [update * STEPS_PER_UPDATE for update in range(UPDATES + 1)]
    # and the run's own exponential average at the two points, once each for the check
    assert [step for step, name in reads if name != "live"] == [
        2 * STEPS_PER_UPDATE,
        UPDATES * STEPS_PER_UPDATE,
    ]


FACTS = RunFacts(
    steps=tuple(range(0, 101, 10)),
    start=0,
    steps_per_update=10,
    training_decay=0.9,
    training_uniform_from=20,
)


def _folded(average: AverageSpec, **evaluation) -> dict[int, tuple[int, ...]]:
    accumulators, lives = averaging.plan(Spec((average,), **evaluation), FACTS)
    outputs = [
        output for item in accumulators for group in item.outputs.values() for output in group
    ]
    return {output.end: output.folded for output in outputs + lives}


def test_the_windows_and_starts_take_the_right_updates():
    # uniform from a fraction (inclusive); the starting weights are never an update
    assert _folded(AverageSpec("uniform", from_fraction=0.25), fractions=(0.5, 1.0)) == {
        50: (30, 40, 50),
        100: (30, 40, 50, 60, 70, 80, 90, 100),
    }
    assert _folded(AverageSpec("uniform", from_step=0)) == {100: tuple(range(10, 101, 10))}
    assert _folded(AverageSpec("training_uniform")) == {100: tuple(range(20, 101, 10))}
    # windows: the last updates, steps or fraction of the span before each point
    assert _folded(AverageSpec("window", last_updates=3), steps=(55, 100)) == {
        50: (30, 40, 50),
        100: (80, 90, 100),
    }
    assert _folded(AverageSpec("window", last_steps=30)) == {100: (80, 90, 100)}
    assert _folded(AverageSpec("window", last_fraction=0.2)) == {100: (90, 100)}
    # exponential: from the starting weights by default, or the first file at its start
    assert _folded(AverageSpec("exponential", halflife_updates=2)) == {100: FACTS.steps}
    assert _folded(AverageSpec("exponential", halflife_updates=2, from_step=75)) == {
        100: (80, 90, 100)
    }
    assert _folded(AverageSpec("live"), fractions=(0.1, 0.33)) == {10: (10,), 30: (30,)}


def test_half_lives_names_and_spec_errors():
    assert AverageSpec("exponential", halflife_updates=4).decay(FACTS) == pytest.approx(0.5**0.25)
    assert AverageSpec("exponential", halflife_steps=40).decay(FACTS) == pytest.approx(0.5**0.25)
    assert AverageSpec("training_exponential").decay(FACTS) == 0.9
    names = [
        AverageSpec("uniform", from_fraction=0.25),
        AverageSpec("exponential", halflife_updates=8, from_fraction=0.5),
        AverageSpec("window", last_fraction=0.05),
        AverageSpec("training_exponential"),
        AverageSpec("training_uniform"),
    ]
    assert [average.name(FACTS) for average in names] == [
        "uniform_from_25percent",
        "exponential_halflife_8updates_from_50percent",
        "window_last_5percent",
        "training_exponential_decay_0.9",
        "training_uniform_from_step_20",
    ]
    with pytest.raises(ValueError, match="needs one of"):
        AverageSpec("exponential")
    with pytest.raises(ValueError, match="halflife_updates and halflife_steps"):
        AverageSpec("exponential", halflife_updates=2, halflife_steps=20)
    with pytest.raises(ValueError, match="takes no last_updates"):
        AverageSpec("uniform", from_step=0, last_updates=2)
    with pytest.raises(ValueError, match="family"):
        AverageSpec("median")
    without_start = RunFacts(FACTS.steps[1:], None, 10, 0.9, 20)
    with pytest.raises(ValueError, match="starting weights were not saved"):
        averaging.plan(Spec((AverageSpec("training_exponential"),)), without_start)


def test_the_committed_suite_reads_and_plans_one_tournament_on_a_100m_run():
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = averaging.read_spec(root / "docs/rewrite/sections/averaging/suite.toml")
    per_update = 1_048_576
    facts = RunFacts(
        steps=tuple(range(0, 96 * per_update + 1, per_update)),
        start=0,
        steps_per_update=per_update,
        training_decay=0.95,
        training_uniform_from=10_000_000,
    )
    accumulators, lives = averaging.plan(spec, facts)
    names = {average.name(facts) for average in spec.averages}
    assert 12 <= len(names) == len(spec.averages) <= 16
    assert len(lives) == 1 and len(accumulators) == len(names) - 1
    window = next(item for item in accumulators if item.family.last_fraction == 0.2)
    assert len(window.folded) == 20  # 19.2 updates: those after the cut

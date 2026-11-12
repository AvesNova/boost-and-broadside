"""BC's plateau rule (`bnb.learn.bc.Plateau`, scripted spec B4): hand-computed cases, its
hold, its state across a resume, its bounded memory, and its replay on the hub's BC runs
(`golden/bc_plateau_hub_kl.json`, written by `bench/bc_plateau.py --fixture`)."""

import dataclasses
import json
import pathlib

import pytest

from bnb.learn import bc

# one environment step an update: from step 2, the trailing mean of the last two updates,
# a window of two steps, a quarter of the improvement, held one step
SMALL = bc.StopConfig(
    plateau_fraction=0.25,
    plateau_from_steps=2,
    plateau_smoothing_steps=2,
    plateau_window_steps=2,
    plateau_hold_steps=1,
)
KL = (9.0, 5.0, 3.0, 2.0, 1.6, 1.5, 1.45, 1.44)
HUB = pathlib.Path(__file__).parent / "golden" / "bc_plateau_hub_kl.json"


def _feed(plateau: bc.Plateau, values, first_step: int = 1) -> list[bool]:
    return [plateau.observe(first_step + index, value) for index, value in enumerate(values)]


def test_the_defaults_are_the_offline_rules_with_a_hold_of_one_window():
    config = bc.StopConfig()
    assert config.plateau_fraction == 0.05
    assert config.plateau_window_steps == 10_000_000
    assert config.plateau_from_steps == 5_000_000
    assert config.plateau_smoothing_steps == 5_000_000
    assert config.plateau_hold_steps == 10_000_000


def test_a_hand_computed_plateau():
    """KL 9, 5, 3, 2, 1.6, 1.5, 1.45, 1.44 at steps 1-8. The baseline, the updates within
    one step of step 2 (steps 2 and 3), is 4, fixed at step 3. The trailing means: 9, 7, 4,
    2.5, 1.8, 1.55, 1.475, 1.445. The recent improvement against the improvement since the
    baseline: step 4 (7 - 2.5) / (4 - 2.5) = 3; step 5 (4 - 1.8) / 2.2 = 1; step 6
    (2.5 - 1.55) / 2.45; step 7 (1.8 - 1.475) / 2.525 = 0.129, under a quarter: the hold
    begins; step 8 (1.55 - 1.445) / 2.555 = 0.041: held one step, it fires."""
    plateau = bc.Plateau(SMALL)
    fractions = []
    for step, value in enumerate(KL, 1):
        fired = plateau.observe(step, value)
        fractions.append(plateau.fraction)
        assert fired == (step == 8), step
    assert plateau.baseline == pytest.approx(4.0)
    assert plateau.mean == pytest.approx(1.445)
    assert fractions[:3] == [None, None, None]  # no baseline, then no improvement since it
    expected = (3.0, 1.0, 0.95 / 2.45, 0.325 / 2.525, 0.105 / 2.555)
    assert fractions[3:] == pytest.approx(expected)
    assert plateau.held_steps(8) == 1


def test_the_hold_starts_again_when_the_condition_breaks():
    """At a fraction of 0.15 the condition holds from step 7 (0.129). A second fall at step
    8 (KL 0.7: (1.55 - 1.075) / (4 - 1.075) = 0.162) breaks it, and it holds again from step
    10 (0.114), so the rule fires at step 11; with the KL flat at 1.45 instead, at step 8."""
    config = dataclasses.replace(SMALL, plateau_fraction=0.15)
    second_fall = bc.Plateau(config)
    fired = _feed(second_fall, (*KL[:7], 0.7, 0.7, 0.7, 0.7))
    assert fired.index(True) + 1 == 11
    flat = bc.Plateau(config)
    assert _feed(flat, (*KL[:7], 1.45)).index(True) + 1 == 8


def test_a_temporary_plateau_shorter_than_the_hold_does_not_fire():
    """A fast fall, a stretch where the KL rises back and stalls (the hub runs' 20-30M
    steps), a long steady fall, then flat: the stall holds the condition for three steps
    (8-11), under the hold of six, and the rule fires six steps into the flat end (it holds
    from step 29, when the trailing mean stops falling)."""
    config = bc.StopConfig(
        plateau_from_steps=0,
        plateau_smoothing_steps=2,
        plateau_window_steps=4,
        plateau_hold_steps=6,
    )
    fall, stall = [9.0, 6.0, 4.0, 3.0], [3.4] * 4
    steady, flat = [3.4 - 0.1 * index for index in range(1, 21)], [1.4] * 8
    plateau, held = bc.Plateau(config), []
    for step, value in enumerate(fall + stall + steady + flat, 1):
        fired = plateau.observe(step, value)
        held.append(plateau.held_steps(step))
        assert fired == (step == 35), step
        if fired:
            break
    assert max(held[:12]) == 3 and held[12:28] == [0] * 16


def test_no_fraction_turns_the_plateau_off():
    config = bc.StopConfig(plateau_fraction=None, **{
        name: getattr(SMALL, name)
        for name in ("plateau_from_steps", "plateau_smoothing_steps", "plateau_window_steps")
    })  # fmt: skip
    plateau = bc.Plateau(config)
    assert not any(_feed(plateau, (*KL, *[1.44] * 20)))
    assert plateau.fraction == pytest.approx(0.0)  # still read and logged


def test_a_restored_plateau_decides_as_the_original_would():
    original = bc.Plateau(SMALL)
    _feed(original, KL[:5])
    restored = bc.Plateau(SMALL, json.loads(json.dumps(original.state)))
    assert restored.state == original.state
    tail = (*KL[5:], 1.43)
    assert _feed(original, tail, 6) == _feed(restored, tail, 6)
    assert original.state == restored.state


def test_it_keeps_only_what_it_still_reads():
    """Of the trailing means, the newest one window back or more and the newer; of the
    updates, the last smoothing's (the baseline's too until it is fixed)."""
    config = bc.StopConfig(
        plateau_from_steps=10,
        plateau_smoothing_steps=4,
        plateau_window_steps=10,
        plateau_hold_steps=10**9,
    )
    plateau = bc.Plateau(config)
    _feed(plateau, [10.0 - 0.01 * step for step in range(200)])
    assert [steps for steps, _ in plateau.values] == [197, 198, 199, 200]
    assert [steps for steps, _ in plateau.means] == list(range(190, 201))
    early = bc.Plateau(config)
    _feed(early, [1.0] * 11)  # step 11: the baseline (steps 9-12) is not fixed yet
    assert early.baseline is None and [steps for steps, _ in early.values] == [8, 9, 10, 11]


@pytest.mark.parametrize(
    "run, fires",
    [
        ("2026-11-09-bc-absolute", 98_566_144),  # the null arm; offline 86M, Elo 80.7M
        ("2026-11-09-bc-deadreckon-split-hiddenphysics", 81_788_928),  # offline 69M, Elo 70.3M
        ("2026-11-09-bc-deadreckon-split", None),  # stopped at 60.8M, still improving
        ("2026-11-10-bc-nearest", 80_740_352),
    ],
)
def test_on_the_hub_runs_it_fires_after_the_offline_plateau(run, fires):
    """The default rule replayed on the hub's BC KL: one hold (10M steps) and half a
    smoothing (2.5M) after the offline rule's centred, "stays below" reading; never in the
    temporary plateau at 20-30M steps; never on the run that stopped at 60M while still
    improving."""
    series = json.loads(HUB.read_text())[run]
    plateau = bc.Plateau(bc.StopConfig())
    fired = next((steps for steps, kl in series if plateau.observe(steps, kl)), None)
    assert fired == fires

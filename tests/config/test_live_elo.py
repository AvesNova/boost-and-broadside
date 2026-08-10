"""The live Elo gauge: what it assigns, what it rejects, and what it costs.

The gauge is defined rather than measured, so these tests do two jobs. They pin
the definition (random 0, scripted 1000, rung 1000·p) and its validation, and
they keep the *evidence* for the approximation executable: the fitted ladders
S12 removed from ``config/defaults`` are recorded here with the per-rung error
accepting them costs, so a future edit to the gauge has to face the same numbers
the decision was made on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boost_and_broadside.config.defaults import LIVE_REFERENCE_PROBABILITIES
from boost_and_broadside.config.live_elo import (
    LIVE_RANDOM_ELO,
    LIVE_SCRIPTED_ELO,
    live_reference_elo,
    live_reference_ladder,
    validate_live_reference_probabilities,
)
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.profiles import PROFILES

_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS = _ROOT / "tests" / "fixtures" / "mode_refactor"

# The ladders the semi-random tournament fitted for the two shipped
# environments, on the scripted-anchored gauge, with the rating it gave the
# uniform-random agent. S12 stopped shipping these as configuration; they stay
# here as the measurement the linear gauge is checked against.
_FITTED_ZERO_FIELD = {
    "random": -363.9,
    "rungs": {
        0.2: -236.0,
        0.3: -96.2,
        0.4: 114.9,
        0.5: 270.7,
        0.6: 461.1,
        0.7: 589.2,
        0.8: 733.0,
        0.9: 861.3,
        0.95: 942.5,
    },
}
_FITTED_FIELD = {
    "random": 132.3,
    "rungs": {
        0.2: 238.8,
        0.3: 322.6,
        0.4: 435.0,
        0.5: 550.4,
        0.6: 656.9,
        0.7: 753.6,
        0.8: 824.3,
        0.9: 939.9,
        0.95: 988.9,
    },
}

# Accepted per-rung error of the linear gauge against each fitted ladder
# regauged to the same two endpoints, as (zero-field, field) Elo points.
# Oriented as the gauge's own error, live minus fitted, so positive means the
# gauge rates the rung above where it actually plays. The table in
# docs/internal/mode-refactor-plan.md §1 lists the same numbers negated, as
# fitted minus linear.
_ACCEPTED_ERROR = {
    0.2: (106.2, 77.3),
    0.3: (103.7, 80.7),
    0.4: (48.9, 51.1),
    0.5: (34.7, 18.2),
    0.6: (-4.9, -4.6),
    0.7: (1.2, -16.0),
    0.8: (-4.2, 2.5),
    0.9: (1.7, -30.7),
    0.95: (-7.8, -37.2),
}


def _regauged(fitted: dict, probability: float) -> float:
    """A fitted rung on the gauge where random reads 0 and scripted reads 1000."""

    random_elo = fitted["random"]
    return 1000.0 * (fitted["rungs"][probability] - random_elo) / (1000.0 - random_elo)


def test_the_gauge_pins_random_at_zero_and_scripted_at_one_thousand() -> None:
    assert LIVE_RANDOM_ELO == 0.0
    assert LIVE_SCRIPTED_ELO == 1000.0
    assert live_reference_elo(0.0) == LIVE_RANDOM_ELO
    assert live_reference_elo(1.0) == LIVE_SCRIPTED_ELO


@pytest.mark.parametrize("probability", LIVE_REFERENCE_PROBABILITIES)
def test_a_rung_is_one_thousand_times_its_scripted_probability(probability: float) -> None:
    assert live_reference_elo(probability) == pytest.approx(1000.0 * probability)


def test_the_ladder_is_derived_once_for_the_whole_probability_set() -> None:
    ladder = live_reference_ladder(LIVE_REFERENCE_PROBABILITIES)

    assert [probability for probability, _ in ladder] == list(LIVE_REFERENCE_PROBABILITIES)
    assert [elo for _, elo in ladder] == [
        live_reference_elo(probability) for probability in LIVE_REFERENCE_PROBABILITIES
    ]


def test_a_regauged_scripted_anchor_rescales_the_whole_ladder() -> None:
    """The rungs are stated relative to the unit, not as absolute numbers."""

    ladder = live_reference_ladder((0.25, 0.5), scripted_elo=2000.0)
    assert ladder == ((0.25, 500.0), (0.5, 1000.0))


@pytest.mark.parametrize(
    "probabilities",
    [
        (0.0,),  # the random agent, already on the ladder
        (1.0,),  # the scripted controller, already on the ladder
        (-0.1,),
        (1.5,),
        (0.5, 0.5),  # not strictly increasing
        (0.6, 0.4),  # out of order
        (float("nan"),),
        (True,),  # a bool is not a probability
    ],
)
def test_invalid_rung_probabilities_are_rejected(probabilities: tuple) -> None:
    with pytest.raises(ValueError):
        validate_live_reference_probabilities(probabilities)


def test_a_profile_with_invalid_rungs_fails_to_resolve() -> None:
    from dataclasses import replace

    profile = PROFILES["rl"]
    broken = replace(
        profile,
        league=replace(profile.league, live_reference_probabilities=(0.5, 0.4)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_profile(broken)


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_profile_rates_on_the_same_derived_gauge(name: str) -> None:
    """No profile carries an environment-fitted ladder or random rating.

    Before S12 the zero-field and field environments shipped different fitted
    gauges. The live gauge is a definition, so there is now one, and the fields
    that used to hold a fitted rating must not come back.
    """

    resolved = resolve_profile(PROFILES[name])
    train_config = resolved.train_config

    assert train_config.live_reference_probabilities == LIVE_REFERENCE_PROBABILITIES
    assert train_config.elo_eval.scripted_live_elo == LIVE_SCRIPTED_ELO
    assert not hasattr(train_config, "reference_ladder")
    assert not hasattr(train_config, "random_elo")
    assert not hasattr(PROFILES[name].league, "reference_ladder")
    assert not hasattr(PROFILES[name].league, "random_elo")


@pytest.mark.parametrize("name", ("rl", "rl-fields", "bc"))
def test_no_resolved_snapshot_stores_a_fitted_rating(name: str) -> None:
    """The recorded snapshots are the resolved-config diff S12 is judged on."""

    train_config = json.loads((_SNAPSHOTS / f"{name}.json").read_text())["train_config"]

    assert "reference_ladder" not in train_config
    assert "random_elo" not in train_config
    assert train_config["live_reference_probabilities"] == list(LIVE_REFERENCE_PROBABILITIES)
    assert train_config["elo_eval"]["scripted_live_elo"] == 1000.0


@pytest.mark.parametrize("probability", sorted(_ACCEPTED_ERROR))
def test_the_linear_gauge_error_is_the_one_that_was_reviewed(probability: float) -> None:
    """Pin the cost of the approximation against both fitted environments.

    Rough below the halfway rung — the gauge rates p=0.2 about 106 Elo above
    where it plays — and within ~40 points from 0.6 upward, which is the range
    the live rating actually has to discriminate in once training is under way.
    Anything that moves these numbers is changing what the live curve means.
    """

    derived = live_reference_elo(probability)
    zero_field, field = _ACCEPTED_ERROR[probability]

    assert derived - _regauged(_FITTED_ZERO_FIELD, probability) == pytest.approx(
        zero_field, abs=0.1
    )
    assert derived - _regauged(_FITTED_FIELD, probability) == pytest.approx(field, abs=0.1)


def test_the_gauge_is_accurate_where_the_live_rating_has_to_discriminate() -> None:
    """The rungs from 0.6 up sit within 40 Elo of both fitted ladders."""

    for probability in (0.6, 0.7, 0.8, 0.9, 0.95):
        for fitted in (_FITTED_ZERO_FIELD, _FITTED_FIELD):
            assert abs(live_reference_elo(probability) - _regauged(fitted, probability)) < 40.0

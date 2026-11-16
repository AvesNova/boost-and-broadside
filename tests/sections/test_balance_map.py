"""bench.balance_map: the search space's decode and encode, the time-to-kill repair, the
measures and the objective (no game, no solver)."""

import numpy
import pytest

from bench import balance_map


@pytest.fixture(scope="module")
def jujak():
    return dict(balance_map.reference_ships()["jujak"].handles)


def test_encode_decode_round_trip(jujak):
    coordinates = balance_map.encode(jujak)
    handles = balance_map.decode(coordinates, jujak)
    for name in balance_map.STORED:
        assert handles[name] == pytest.approx(jujak[name], rel=1e-6), name


def test_every_decoded_card_keeps_the_time_to_kill_band(jujak):
    rng = numpy.random.default_rng(0)
    low, high = balance_map.TIME_TO_KILL_BAND
    for _ in range(500):
        handles = balance_map.decode(rng.random(len(balance_map.SPACE)), jujak)
        shield, damage = handles["max_shield"], handles["hit_damage"]
        for time in (
            balance_map.time_to_kill(shield, damage),  # its mirror
            balance_map.time_to_kill(shield, jujak["hit_damage"]),  # the reference on it
            balance_map.time_to_kill(jujak["max_shield"], damage),  # it on the reference
        ):
            assert low - 1e-9 <= time <= high + 1e-9
        assert handles["fire_rate"] == 30.0


def test_reference_time_to_kill(jujak):
    assert balance_map.time_to_kill(jujak["max_shield"], jujak["hit_damage"]) == pytest.approx(
        0.7495
    )


def test_measures_in_the_unit_square(jujak):
    rng = numpy.random.default_rng(1)
    for _ in range(100):
        agility, toughness = balance_map.measures(rng.random(len(balance_map.SPACE)))
        assert 0.0 <= agility <= 1.0 and 0.0 <= toughness <= 1.0
    corner = numpy.zeros(len(balance_map.SPACE))
    corner[balance_map.INDEX["nose_reversal_time"]] = 1.0
    assert balance_map.measures(corner) == (0.0, 0.0)


def test_objective_even_and_worst_condition():
    even = {c: (0.5, 0.01, 128) for c in balance_map.SEARCHED}
    assert balance_map.objective_of(even)["objective"] == pytest.approx(50.0)
    skewed = dict(even, **{"deathmatch-5v5": (0.52, 0.01, 128)})
    score = balance_map.objective_of(skewed)
    assert score["worst"] == "deathmatch-5v5"
    assert score["objective"] < 50.0 and score["lower"] < score["objective"]
    assert balance_map.objective_of({})["objective"] is None


def test_pooled_inverse_variance():
    rows = [
        dict(condition="deathmatch-5v5", kill_share=(0.4, 0.02), games=128),
        dict(condition="deathmatch-5v5", kill_share=(0.5, 0.01), games=384),
    ]
    share, error, games = balance_map.pooled(rows)["deathmatch-5v5"]
    assert share == pytest.approx((0.4 / 4e-4 + 0.5 / 1e-4) / (1 / 4e-4 + 1 / 1e-4))
    assert error < 0.01 and games == 512

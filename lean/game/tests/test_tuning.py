"""lean.game.tuning: the card's rows are the tuning's output, the P-51D's conversion reproduces
the October fit, the handles route recovers a row from its flown handles, and another time scale
flies the same aircraft (docs/rewrite/p51d-model.md)."""

import math

import pytest
import torch

from lean.game import card, core, tuning

torch.set_num_threads(1)
MPH, FEET_PER_MINUTE = 0.44704, 0.00508
# The October fit's sea-level point mass on the fitted airframe (docs/rewrite/physics-checks/
# real_fit.py; decisions.md 2026-10-23): the tested 375 mph and 3,600 ft/min, and its best
# sustained turn 18.6 deg/s at 167 mph, radius 231 m.
OCTOBER = dict(
    top_speed=375 * MPH,
    best_climb=3600 * FEET_PER_MINUTE,
    sustained_rate=math.radians(18.6),
    sustained_speed=167 * MPH,
    sustained_radius=231.0,
)


def test_card_rows_are_the_tuning():
    """Every card row is exactly what lean.game.tuning makes."""
    expected = {"p51d": tuning.airframe_row(tuning.P51D, tuning.GAME_FIELDS)}
    assert card.TYPE_NAMES == tuple(expected)
    for name, row in expected.items():
        assert set(row) == set(card.CARD), name
        held = tuning.row_of(name)
        for field, value in row.items():
            assert held[field] == pytest.approx(value, rel=1e-12), (name, field)


def test_p51d_point_mass_is_the_october_fit():
    """With the game terms off (drag rise, separation drag, soft minima), the P-51D card flies
    the October fit's point mass in real units: top speed and climb within 0.5%, the best
    sustained turn within 1%, its speed and radius within 2% (a 0.05 m/s grid)."""
    p = tuning.in_real_units(tuning.performance(tuning.row_of("p51d"), plain=True))
    assert p["top_speed"] == pytest.approx(OCTOBER["top_speed"], rel=0.005)
    assert p["best_climb"] == pytest.approx(OCTOBER["best_climb"], rel=0.005)
    assert p["sustained_rate"] == pytest.approx(OCTOBER["sustained_rate"], rel=0.01)
    assert p["sustained_speed"] == pytest.approx(OCTOBER["sustained_speed"], rel=0.02)
    assert p["sustained_radius"] == pytest.approx(OCTOBER["sustained_radius"], rel=0.02)


def test_p51d_flies_its_point_mass_in_the_game():
    """The P-51D's row has the game terms off (tuning.POINT_MASS_TERMS), so in the game it flies
    its point mass: the October fit, and the same numbers as plain=True."""
    row = tuning.row_of("p51d")
    assert tuning.performance(row) == tuning.performance(row, plain=True)


def test_handles_route_recovers_a_row_from_its_flown_handles():
    """The handles route: the P-51D's own handles (its direct fields, and its cruise, low-speed
    acceleration and settled full-stick turn as flown), from a base whose induced drag is 30%
    off, solve to a row that flies them to 1e-6, with the P-51D's engine power and induced drag.
    (Its static thrust is not recovered: at its stall speed the P-51D's thrust is power-limited,
    so the low-speed acceleration barely depends on it.)"""
    row = tuning.row_of("p51d")
    flown = tuning.performance(row)
    handles = tuning.handles_of(row)
    base = {**row, "induced_drag_scale": 1.3 * row["induced_drag_scale"]}
    solved = tuning.handles_row(handles, base)
    again = tuning.performance(solved)
    for name in ("top_speed", "low_speed_acceleration", "settled_turn_rate"):
        assert again[name] == pytest.approx(flown[name], rel=1e-6), name
    for name in ("engine_power", "induced_drag_scale"):
        assert solved[name] == pytest.approx(row[name], rel=1e-4), name
    assert solved["exchange_acceleration"] == row["exchange_acceleration"]
    assert set(solved) == set(card.CARD)


@pytest.mark.parametrize("factor", [1.0, 1.5, 3.0])
def test_time_scale_is_the_same_flight_faster_or_slower(factor):
    """At another time factor the card flies the same real aircraft: gravity factor^2 x 9.81,
    and cruise and corner speed (game units) scaled by factor / 2; the card is restored after."""
    config = core.GameConfig(lineup=card.TYPE_NAMES)
    types = torch.arange(len(card.TYPE_NAMES))
    game = core.per_type_constants(config, types)
    saved = card.CARD
    with tuning.time_scale(factor):
        assert card.GRAVITY == pytest.approx(9.81 * factor**2, rel=1e-12)
        scaled = core.per_type_constants(config, types)
    assert card.CARD is saved and card.GRAVITY == 39.24
    ratio = factor / tuning.TIME_FACTOR
    for name in ("cruise_speed", "corner_speed"):
        assert torch.allclose(getattr(scaled, name), getattr(game, name) * ratio, rtol=1e-5), name
    with tuning.time_scale(tuning.TIME_FACTOR):
        assert card.CARD is saved


def test_scripted_references_are_the_flown_numbers():
    """The game's derived references for the scripted agent (docs/rewrite/scripted-bfm-design.md
    B2) are what the quasi-steady analysis flies: the best sustained turn's speed within
    tuning.performance's 0.05 m/s grid, its rate and radius within 0.1%, the corner rate; the
    quarter-turn time and the store's turn follow from them (the P-51D: 2.4 s and 2.5 rad)."""
    config = core.GameConfig(lineup=card.TYPE_NAMES)
    game = core.per_type_constants(config, torch.arange(len(card.TYPE_NAMES)))
    for index, name in enumerate(card.TYPE_NAMES):
        flown = tuning.performance(tuning.row_of(name), config=config)
        for name, tolerance in (
            ("sustained_speed", dict(abs=0.06)),
            ("sustained_rate", dict(rel=1e-3)),
            ("sustained_radius", dict(rel=1e-3)),
            ("corner_rate", dict(rel=1e-4)),
        ):
            value = float(getattr(game, name)[index])
            assert value == pytest.approx(flown[name], **tolerance), name
        rate = float(game.sustained_rate[index])
        assert float(game.quarter_turn_time[index]) == pytest.approx(0.5 * math.pi / rate, rel=1e-6)
        # the store's turn: corner rate over sustained rate for as long as a full store holds
        # corner speed at full stick
        c = core.per_type_constants(config, torch.tensor([index]))
        corner = c.corner_speed
        _, _, energy_rate = core.quasi_steady(c, corner, torch.ones_like(corner), 0, c.spawn_energy)
        held = float(c.max_energy[0] / energy_rate.abs()[0])
        expected = (float(game.corner_rate[index]) - rate) * held
        assert float(game.store_turn[index]) == pytest.approx(expected, rel=1e-4)
    p51d = card.TYPE_NAMES.index("p51d")
    assert float(game.quarter_turn_time[p51d]) == pytest.approx(2.41, abs=0.01)
    assert float(game.store_turn[p51d]) == pytest.approx(2.5, abs=0.02)

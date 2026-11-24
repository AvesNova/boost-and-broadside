"""lean.game.tuning: the card's rows are derived from their targets (cached by a hash of the
targets and the flight model's source), the P-51D's point mass flies its flight test, its game
row (the game terms on) flies its point mass, the handles route recovers a row from its flown
handles, and another time scale flies the same aircraft (docs/rewrite/p51d-model.md)."""

import dataclasses
import math

import pytest
import torch

from lean.game import card, core, tuning

torch.set_num_threads(1)


def test_p51d_point_mass_flies_the_flight_test():
    """The P-51D's point mass, fitted when the card is derived (real_row's first step), flies
    the flight test at 67 in Hg: 375 mph and 3,600 ft/min at sea level (to 1e-6)."""
    aircraft = tuning.REAL_AIRCRAFT["p51d"]
    fitted = tuning.fit_point_mass(aircraft.airframe, aircraft.top_speed, aircraft.best_climb)
    p = tuning.in_real_units(tuning.performance(tuning.airframe_row(fitted, tuning.GAME_FIELDS)))
    assert p["top_speed"] == pytest.approx(tuning.P51D_TOP_SPEED, rel=1e-6)
    assert p["best_climb"] == pytest.approx(tuning.P51D_BEST_CLIMB, rel=1e-6)


def test_card_rows_meet_their_targets():
    """Every card row, derived from its targets when the card is first used (tuning.load_card,
    from the disk cache when its key is there), flies them with the game terms on: the flight
    test's top speed and best climb and its point mass's best sustained rate (to 1e-6); the best
    sustained turn's speed within 12% of the point mass's (not solved: the separation drag that
    sets the tightest turn apart moves it up, P-51D +9.7%), and the same stall and corner speeds
    (the peak full-stick rate within 0.1%); the solved coefficients physically plausible (Oswald
    e at most 1, propeller efficiency 0.7 to 0.9). The game terms are on, not switched off."""
    assert set(card.CARD) == set(card.CARD_FIELDS)
    for name in card.TYPE_NAMES:
        row = tuning.row_of(name)
        aircraft = tuning.REAL_AIRCRAFT[name]
        fitted = tuning.fit_point_mass(aircraft.airframe, aircraft.top_speed, aircraft.best_climb)
        game = tuning.performance(row)
        point = tuning.performance(tuning.airframe_row(fitted, tuning.GAME_FIELDS))
        real = tuning.in_real_units(game)
        assert real["top_speed"] == pytest.approx(aircraft.top_speed, rel=1e-6), name
        assert real["best_climb"] == pytest.approx(aircraft.best_climb, rel=1e-6), name
        assert game["sustained_rate"] == pytest.approx(point["sustained_rate"], rel=1e-6), name
        assert game["sustained_speed"] == pytest.approx(point["sustained_speed"], rel=0.12), name
        a = aircraft.airframe
        induced = card.INDUCED_DRAG_FACTOR * row["induced_drag_scale"]
        oswald = 1.0 / (math.pi * a.aspect_ratio * a.per_metre * induced)
        efficiency = tuning.to_real(row["engine_power"], -3) * a.mass / a.power
        assert 0.5 < oswald <= 1.0 and 0.7 <= efficiency <= 0.9, (name, oswald, efficiency)
        assert game["stall_speed"] == point["stall_speed"]
        assert game["corner_speed"] == point["corner_speed"]
        assert game["corner_speed_flown"] == pytest.approx(point["corner_speed"], rel=1e-3)
        assert game != tuning.performance(row, plain=True)
        for field, value in aircraft.terms.items():
            assert row[field] == value, (name, field)


def test_derived_rows_are_cached_by_their_targets_and_the_source(tmp_path, monkeypatch):
    """A derived row is kept on disk under a key of its targets and the flight model's source
    (tuning.row_key): a second load reads it without solving; another target or another source
    is another key, so a physics change re-solves by itself."""
    monkeypatch.setenv("BNB_CARD_CACHE", str(tmp_path))
    solved = []
    monkeypatch.setattr(tuning, "real_row", lambda aircraft: solved.append(aircraft) or {"x": 1.0})
    assert tuning.cached_real_row("p51d") == {"x": 1.0} and len(solved) == 1
    assert tuning.cached_real_row("p51d") == {"x": 1.0} and len(solved) == 1  # from the disk
    assert len(list(tmp_path.glob("p51d-*.json"))) == 1
    aircraft = tuning.REAL_AIRCRAFT["p51d"]
    key = tuning.row_key(aircraft)
    other = dataclasses.replace(aircraft, top_speed=aircraft.top_speed + 1.0)
    assert tuning.row_key(other) != key
    source = tmp_path / "source.py"
    source.write_text("a = 1\n")
    monkeypatch.setattr(tuning, "FLIGHT_SOURCES", (*tuning.FLIGHT_SOURCES, source))
    with_source = tuning.row_key(aircraft)
    source.write_text("a = 2\n")
    assert tuning.row_key(aircraft) != with_source


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
    quarter-turn time and the store's turn follow from them (the P-51D: 2.37 s and 2.40 rad)."""
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
    assert float(game.quarter_turn_time[p51d]) == pytest.approx(2.37, abs=0.01)
    assert float(game.store_turn[p51d]) == pytest.approx(2.40, abs=0.02)


def test_a_card_built_inside_with_row_does_not_outlive_it():
    """Review: like `flying` and `time_scale`, `_with_row` drops the core's built ship cards
    on entry and exit, so a card built from the changed table is never served after it."""
    config = core.GameConfig(arena_radius=4321.0)  # a key no other test builds
    row = {name: 2.0 * values[0] for name, values in card.CARD.items()}
    with tuning._with_row(row):
        core.ship_card(config)  # built from the card with the extra row
    after = core.ship_card(config)
    core._SHIP_CARDS.clear()
    fresh = core.ship_card(config)
    assert after.uniform == fresh.uniform and torch.equal(after.table, fresh.table)

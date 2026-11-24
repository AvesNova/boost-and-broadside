"""lean.game.tuning: the card's rows are derived from their references (cached by a hash of the
aircraft and the flight model's source); the P-51D's point mass flies its flight test and its
row flies its reference design; moved designs fly their inputs after the solve; the measures
are smooth; implausible designs warn; another time scale flies the same aircraft
(docs/rewrite/p51d-model.md, "Design inputs")."""

import dataclasses
import math

import pytest
import torch

from lean.game import card, core, tuning

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def p51d():
    return tuning.reference("p51d")


def assert_flies(row, design, rel, label):
    """The row flies every design input (design_of) to `rel`; the gun range at least."""
    flown = tuning.design_of(row)
    for name, value in design.items():
        if name == "gun_range":
            assert flown[name] >= value - 1e-9, label
        else:
            assert flown[name] == pytest.approx(value, rel=rel, abs=1e-12), (label, name)


def test_design_inputs_set_every_card_field_once():
    """One ordered table (DESIGN_INPUTS) sets every card column exactly once; the solved ones
    are the five flown targets."""
    fields = [spec.sets for spec in tuning.DESIGN_INPUTS]
    assert sorted(fields) == sorted(card.CARD_FIELDS)
    assert [spec.name for spec in tuning.SOLVED] == [
        "top_speed", "best_climb", "best_climb_speed", "sustained_rate", "gap"
    ]
    assert "reference_bullet_speed" not in card.CARD_FIELDS


def test_p51d_point_mass_flies_the_flight_test(p51d):
    """The P-51D's point mass, fitted when its reference is made (reference_of), flies the
    flight test at 67 in Hg: 375 mph and 3,600 ft/min at sea level (to 1e-6)."""
    aircraft = tuning.REAL_AIRCRAFT["p51d"]
    point = tuning.airframe_row(p51d.airframe, tuning.row_of("p51d"))
    p = tuning.in_real_units(tuning.performance(point))
    assert p["top_speed"] == pytest.approx(aircraft.top_speed, rel=1e-6)
    assert p["best_climb"] == pytest.approx(aircraft.best_climb, rel=1e-6)
    assert p51d.design["sustained_rate"] == p51d.point_mass["sustained_rate"]


def test_card_rows_fly_their_reference_design(p51d):
    """Every card row, derived when the card is first used (tuning.load_card, from the disk
    cache when its key is there), flies its reference's design inputs (to 1e-6): the flight
    test's top speed, best climb and its speed (165 mph), the point mass's best sustained rate,
    the design gap 0.85, and the closed inputs; the same stall and corner speeds as the point
    mass; the best sustained turn's speed within 12% of the point mass's. The game terms are on."""
    assert set(card.CARD) == set(card.CARD_FIELDS)
    for name in card.TYPE_NAMES:
        reference = tuning.reference(name)
        row = tuning.row_of(name)
        assert_flies(row, reference.design, 1e-6, name)
        game, point = tuning.performance(row), reference.point_mass
        assert game["stall_speed"] == pytest.approx(point["stall_speed"], rel=1e-12)
        assert game["corner_speed"] == pytest.approx(point["corner_speed"], rel=1e-12)
        assert game["sustained_speed"] == pytest.approx(point["sustained_speed"], rel=0.12)
        assert game != tuning.performance(row, plain=True)
    real = tuning.in_real_units(tuning.performance(tuning.row_of("p51d")))
    assert real["best_climb_speed"] == pytest.approx(165.0 * tuning.MPH, rel=1e-6)
    assert real["gap"] == pytest.approx(0.85, rel=1e-6)


@pytest.mark.parametrize(
    "change",
    [
        dict(stall_speed=0.97, g_limit=7.5, top_speed=1.03, best_climb=1.05, sustained_rate=1.05, gap=0.86),
        dict(stall_speed=0.9, g_limit=7.0, top_speed=0.93, sustained_rate=1.10, gap=0.87),
        dict(best_climb_speed=1.05, gap=0.83, muzzle_speed=1.2),
        dict(flap_lift=0.15),
    ],
    ids=["faster", "slow-turner", "late-climber", "flaps"],
)
def test_moved_designs_fly_their_inputs(p51d, change):
    """Moved designs (speeds and rates as multiples of the reference's; the g limit, gap and
    flaps as values) fly every design input after the solve: from the P-51D's row to 1e-6,
    and in the web tool's fast path (Broyden) to 1e-4."""
    design = {
        name: value if name in ("g_limit", "gap", "flap_lift") else value * p51d.design[name]
        for name, value in change.items()
    }
    row = tuning.row_of("p51d")
    solved = tuning.design_row(p51d, design, start=row)
    assert solved.residual < 1e-7 and solved.condition < 100.0
    assert_flies(solved.row, {**p51d.design, **design}, 1e-6, "solved")
    fast = tuning.design_row(p51d, design, start=row, fast=True)
    assert_flies(fast.row, {**p51d.design, **design}, 1e-4, "fast")


def test_a_design_from_the_point_mass_guess_flies_its_inputs(p51d):
    """With no start row the solve starts from the point-mass first guess (hard limits, closed
    forms) and still flies every input."""
    design = dict(top_speed=1.05 * p51d.design["top_speed"], g_limit=6.5)
    assert_flies(tuning.design_row(p51d, design).row, {**p51d.design, **design}, 1e-6, "cold")


def test_an_unknown_design_input_is_refused(p51d):
    with pytest.raises(ValueError, match="cruise_speed"):
        tuning.design_row(p51d, {"cruise_speed": 300.0})


def test_the_measures_are_smooth_and_grid_free():
    """performance refines every extremum below its grid (_extremum): the numbers agree between
    a 0.05 and a 1 m/s grid (1e-6), and the gap's finite differences in the separation drag
    agree at two step sizes (the solve's Jacobian sees slopes, not grid steps)."""
    row = tuning.row_of("p51d")
    fine, coarse = tuning.performance(row, speed_step=0.05), tuning.performance(row, speed_step=1.0)
    for name, value in fine.items():
        assert coarse[name] == pytest.approx(value, rel=1e-6, abs=1e-8), name  # fall: rounding
    slopes = []
    for step in (1e-5, 1e-3):
        moved = [
            tuning.performance({**row, "separation_drag": row["separation_drag"] * math.exp(sign * step)})
            for sign in (1.0, -1.0)
        ]
        slopes.append(math.log(moved[0]["gap"] / moved[1]["gap"]) / (2.0 * step))
    assert slopes[0] == pytest.approx(slopes[1], rel=0.02) and slopes[0] < 0.0


def test_implausible_designs_warn(p51d):
    """The plausibility checks (PLAUSIBILITY) warn, never refuse: a hand-made row with too
    little induced drag (Oswald e above 1), more thrust power than shaft power, too little
    zero-lift drag, too much lift and a softer thrust warns on each. The P-51D's own row warns
    on two: e 0.965 against WWII fighters' 0.78-0.93, and the 8 g limit above the structure's
    6.6 g at the tested weight."""
    row = tuning.row_of("p51d")
    own = {(w["quantity"], w["severity"]) for w in tuning.warnings(row, p51d)}
    assert own == {("oswald", "range"), ("structural_load", "range")}
    bad = {
        **row,
        "induced_drag_scale": 0.7 * row["induced_drag_scale"],
        "engine_power": 1.25 * row["engine_power"],
        "zero_lift_drag": 0.5 * row["zero_lift_drag"],
        "max_lift_coefficient": 1.3 * row["max_lift_coefficient"],
        "thrust_softmin_exponent": 1.5,
    }
    found = {(w["quantity"], w["severity"]) for w in tuning.warnings(bad, p51d)}
    for expected in [
        ("oswald", "limit"),
        ("propeller_efficiency", "limit"),
        ("zero_lift_drag_coefficient", "range"),
        ("max_lift_coefficient", "range"),
        ("thrust_loss", "range"),
    ]:
        assert expected in found, (expected, found)
    implied = tuning.implied(row, p51d.airframe)
    assert implied["figure_of_merit"] == pytest.approx(tuning.FIGURE_OF_MERIT, rel=1e-9)
    assert implied["max_lift_coefficient"] == pytest.approx(1.6, rel=1e-9)


def test_derived_rows_are_cached_by_their_aircraft_and_the_source(tmp_path, monkeypatch):
    """A derived row is kept on disk under a key of its aircraft and the flight model's source
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
    assert tuning.row_key(dataclasses.replace(aircraft, best_climb_speed=70.0)) != key
    other = dataclasses.replace(aircraft, choices={**aircraft.choices, "gap": 0.8})
    assert tuning.row_key(other) != key
    source = tmp_path / "source.py"
    source.write_text("a = 1\n")
    monkeypatch.setattr(tuning, "FLIGHT_SOURCES", (*tuning.FLIGHT_SOURCES, source))
    with_source = tuning.row_key(aircraft)
    source.write_text("a = 2\n")
    assert tuning.row_key(aircraft) != with_source


def test_the_damage_reference_speed_is_derived():
    """The bullet's damage reference speed is not a card column: the game derives it
    (core._derived_characteristics), a bullet fired at cruise, DAMAGE_REFERENCE_DISTANCE out."""
    c = core.per_type_constants(core.GameConfig(), torch.arange(len(card.TYPE_NAMES)))
    launch = c.cruise_speed + c.muzzle_speed
    expected = launch * torch.exp(-c.bullet_drag * card.DAMAGE_REFERENCE_DISTANCE)
    assert torch.allclose(1.0 / c.inverse_reference_bullet_speed, expected, rtol=1e-6)



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

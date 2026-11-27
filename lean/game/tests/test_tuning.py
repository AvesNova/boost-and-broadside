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
    """The card's reference (the airframe with the game's engine, tuning.P51D_SHAFT_POWER,
    "airframe_and_engine")."""
    return tuning.reference("p51d")


@pytest.fixture(scope="module")
def documents():
    """The P-51D under its documents (top speed and climb solved for the flight test's engine):
    the design system's solve; its start_row is the documents row."""
    return tuning.reference("p51d", "documents")


def assert_flies(row, design, rel, label, inputs=tuning.DESIGN_INPUTS):
    """The row flies every design input (design_of under `inputs`) to `rel`; the gun range at
    least."""
    flown = tuning.design_of(row, inputs)
    for name, value in design.items():
        if name == "gun_range":
            assert flown[name] >= value - 1e-9, label
        else:
            assert flown[name] == pytest.approx(value, rel=rel, abs=1e-12), (label, name)


def test_design_inputs_set_every_card_field_once():
    """One ordered table (DESIGN_INPUTS) sets every card column exactly once; the solved ones
    are the two flown targets."""
    fields = [spec.sets for spec in tuning.DESIGN_INPUTS]
    assert sorted(fields) == sorted(card.CARD_FIELDS)
    assert [spec.name for spec in tuning.SOLVED] == ["top_speed", "best_climb"]
    assert "reference_bullet_speed" not in card.CARD_FIELDS


def test_p51d_point_mass_flies_the_flight_test(p51d):
    """The P-51D's point mass, fitted when its reference is made (reference_of), flies the
    flight test at 67 in Hg (tuning.P51D_TOP_SPEED and P51D_BEST_CLIMB) at sea level (to 1e-6)."""
    aircraft = tuning.REAL_AIRCRAFT["p51d"]
    point = tuning.airframe_row(p51d.airframe, tuning.row_of("p51d"))
    p = tuning.in_real_units(tuning.performance(point))
    assert p["top_speed"] == pytest.approx(aircraft.top_speed, rel=1e-6)
    assert p["best_climb"] == pytest.approx(aircraft.best_climb, rel=1e-6)


def test_card_rows_fly_their_reference_design(p51d):
    """Every card row, derived when the card is first used (tuning.load_card, from the disk
    cache when its key is there), flies its reference's design inputs under its profile (to
    1e-6): the P-51D's documented airframe's C_D0 and the card's shaft power, and the closed
    inputs (tuning.P51D_VALUES: e, the separation drag and exponent, the thrust softness); the
    same stall and corner speeds as its own point mass; the best sustained turn's speed within
    12% of the point mass's. The game terms are on."""
    assert set(card.CARD) == set(card.CARD_FIELDS)
    for name in card.TYPE_NAMES:
        reference = tuning.reference(name)
        row = tuning.row_of(name)
        assert_flies(row, reference.design, 1e-6, name, reference.inputs)
        game, point = tuning.performance(row), tuning.performance(tuning.point_mass(row))
        assert game["stall_speed"] == pytest.approx(point["stall_speed"], rel=1e-12)
        assert game["corner_speed"] == pytest.approx(point["corner_speed"], rel=1e-12)
        assert game["sustained_speed"] == pytest.approx(point["sustained_speed"], rel=0.12)
        assert game != tuning.performance(row, plain=True)
    row, values = tuning.row_of("p51d"), tuning.P51D_VALUES
    assert tuning.implied(row, p51d.airframe)["oswald"] == pytest.approx(
        values["oswald"], rel=1e-12
    )
    assert (row["separation_drag"], row["separation_exponent"]) == (
        values["separation_drag"],
        values["separation_exponent"],
    )


VALUES = ("g_limit", "oswald", "separation_drag", "thrust_softmin_exponent")  # not multiples


@pytest.mark.parametrize(
    "change",
    [
        dict(stall_speed=0.97, g_limit=7.5, top_speed=1.03, best_climb=1.05, oswald=0.9),
        dict(stall_speed=0.9, g_limit=7.0, top_speed=0.93, separation_drag=0.5),
        dict(best_climb=0.95, thrust_softmin_exponent=4.0, muzzle_speed=1.2),
    ],
    ids=["faster", "slow-turner", "late-climber"],
)
def test_moved_designs_fly_their_inputs(documents, change):
    """Moved designs (speeds and rates as multiples of the reference's; the g limit, e, the
    separation drag and the thrust softness as values) fly every design input after the solve: from
    the P-51D's row to 1e-6,
    and in the web tool's fast path (Broyden) to 1e-4."""
    design = {
        name: value if name in VALUES else value * documents.design[name]
        for name, value in change.items()
    }
    # A bullet lives exactly BULLET_SLOTS ticks: a moved muzzle speed moves the gun range.
    design["gun_range"] = tuning.ring_gun_range({**documents.design, **design})
    row = documents.start_row
    solved = tuning.design_row(documents, design, start=row)
    assert solved.residual < 1e-7 and solved.condition < 100.0
    assert_flies(solved.row, {**documents.design, **design}, 1e-6, "solved")
    fast = tuning.design_row(documents, design, start=row, fast=True)
    assert_flies(fast.row, {**documents.design, **design}, 1e-4, "fast")


@pytest.mark.parametrize(
    "before, after",
    [
        (dict(top_speed=1.03), dict(top_speed=1.05)),
        (dict(oswald=0.87, best_climb=1.02), dict(oswald=0.83, best_climb=1.04, top_speed=0.98)),
    ],
    ids=["one-input", "several"],
)
def test_a_warm_start_agrees_with_a_cold_solve(documents, before, after, monkeypatch):
    """The web tool's warm start (design_row from the last solve's row and Jacobian) reaches the
    cold solve's row (both from the fast path, to its 1e-5) with fewer flight evaluations."""

    def design(change):
        return {
            name: value if name in VALUES else value * documents.design[name]
            for name, value in change.items()
        }

    row = documents.start_row
    last = tuning.design_row(documents, design(before), start=row, fast=True)
    assert last.jacobian is not None
    evaluations = []
    performance = tuning.performance
    monkeypatch.setattr(
        tuning, "performance", lambda *a, **k: evaluations.append(1) or performance(*a, **k)
    )
    warm = tuning.design_row(
        documents, design(after), start=last.row, fast=True, jacobian=last.jacobian
    )
    warm_evaluations = len(evaluations)
    cold = tuning.design_row(documents, design(after), start=row, fast=True)
    assert warm_evaluations < len(evaluations) - warm_evaluations
    monkeypatch.undo()
    assert warm.residual < 1e-5 and cold.residual < 1e-5
    assert_flies(warm.row, {**documents.design, **design(after)}, 2e-5, "warm")
    for name, value in cold.row.items():
        assert warm.row[name] == pytest.approx(value, rel=1e-3), name


def test_a_design_from_the_point_mass_guess_flies_its_inputs(documents):
    """With no start row the solve starts from the reference's point mass (and the table's first
    guesses) and still flies every input."""
    design = dict(top_speed=1.05 * documents.design["top_speed"], g_limit=6.5)
    assert_flies(
        tuning.design_row(documents, design).row, {**documents.design, **design}, 1e-6, "cold"
    )


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
            tuning.performance(
                {**row, "separation_drag": row["separation_drag"] * math.exp(sign * step)}
            )
            for sign in (1.0, -1.0)
        ]
        slopes.append(math.log(moved[0]["gap"] / moved[1]["gap"]) / (2.0 * step))
    assert slopes[0] == pytest.approx(slopes[1], rel=0.02) and slopes[0] < 0.0


def structural_check() -> tuning.Check:
    """PLAUSIBILITY's structural-load check (the g limit over the structure's at this mass)."""
    return next(c for c in tuning.PLAUSIBILITY if c.quantity == "structural_load")


def test_implausible_designs_warn(p51d, documents):
    """The plausibility checks (PLAUSIBILITY) warn, never refuse: a hand-made row under the
    documents (the shaft power held at the flight test's) with too little induced drag
    (Oswald e above 1), more thrust power than shaft power, too little zero-lift drag, too much
    lift and a softer thrust warns on each. The structural check follows the rule, not today's
    numbers: the P-51D's documents row and its card row (the game's engine) warn on the
    structure exactly when their g limit is above the structure's at the tested weight, and
    each, its g limit moved to a tenth above or below the check's bound, warns or does not."""
    check = structural_check()
    for reference, row in ((documents, documents.start_row), (p51d, tuning.row_of("p51d"))):
        load = tuning.implied(row, tuning.airframe_of(row, reference))["structural_load"]
        warned = {w["quantity"] for w in tuning.warnings(row, reference)}
        assert ("structural_load" in warned) == (load > check.high)
        for factor, warns in ((1.1, True), (0.9, False)):
            scale = factor * check.high / load
            moved = {**row, "max_acceleration": row["max_acceleration"] * scale}
            warned = {w["quantity"] for w in tuning.warnings(moved, reference)}
            assert ("structural_load" in warned) == warns, (reference.profile, factor)
    row = documents.start_row
    bad = {
        **row,
        "induced_drag_scale": 0.7 * row["induced_drag_scale"],
        "engine_power": 1.25 * row["engine_power"],
        "zero_lift_drag": 0.5 * row["zero_lift_drag"],
        "max_lift_coefficient": 1.3 * row["max_lift_coefficient"],
        "thrust_softmin_exponent": 1.5,
    }
    found = {(w["quantity"], w["severity"]) for w in tuning.warnings(bad, documents)}
    for expected in [
        ("oswald", "limit"),
        ("propeller_efficiency", "limit"),
        ("zero_lift_drag_coefficient", "range"),
        ("max_lift_coefficient", "range"),
        ("thrust_loss", "range"),
    ]:
        assert expected in found, (expected, found)
    implied = tuning.implied(row, documents.airframe)
    assert implied["figure_of_merit"] == pytest.approx(tuning.FIGURE_OF_MERIT, rel=1e-9)
    assert implied["max_lift_coefficient"] == pytest.approx(
        tuning.P51D.max_lift_coefficient, rel=1e-9
    )


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
    climbs = dataclasses.replace(aircraft, best_climb=aircraft.best_climb + 1.0)
    assert tuning.row_key(climbs) != key
    other = dataclasses.replace(aircraft, values={**aircraft.values, "oswald": 0.8})
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
    """At another time factor the card flies the same real aircraft: gravity factor^2 x g,
    and cruise and corner speed (game units) scaled by factor / 2; the card is restored after."""
    config = core.GameConfig(lineup=card.TYPE_NAMES)
    types = torch.arange(len(card.TYPE_NAMES))
    game = core.per_type_constants(config, types)
    saved, gravity = card.CARD, card.GRAVITY
    with tuning.time_scale(factor):
        assert card.GRAVITY == pytest.approx(tuning.REAL_GRAVITY * factor**2, rel=1e-12)
        scaled = core.per_type_constants(config, types)
    assert card.CARD is saved and card.GRAVITY == gravity
    ratio = factor / tuning.TIME_FACTOR
    for name in ("cruise_speed", "corner_speed"):
        assert torch.allclose(getattr(scaled, name), getattr(game, name) * ratio, rtol=1e-5), name
    with tuning.time_scale(tuning.TIME_FACTOR):
        assert card.CARD is saved


def test_scripted_references_are_the_flown_numbers():
    """The game's derived references for the scripted agent (docs/rewrite/scripted-bfm-design.md
    B2) are what the quasi-steady analysis flies: the best sustained turn's speed within
    tuning.performance's 0.05 m/s grid, its rate and radius within 0.1%, the corner rate; the
    quarter-turn time and the store's turn follow from them."""
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


def test_a_card_built_inside_with_row_does_not_outlive_it():
    """Review: like `flying` and `time_scale`, `_with_row` drops the core's built ship cards
    on entry and exit, so a card built from the changed table is never served after it."""
    config = core.GameConfig(arena_radius=4321.0)  # a key no other test builds
    row = {name: 2.0 * values[0] for name, values in card.CARD.items()}
    row["bullet_life"] = card.CARD["bullet_life"][0]  # the only life the game allows
    with tuning._with_row(row):
        core.ship_card(config)  # built from the card with the extra row
    after = core.ship_card(config)
    core._SHIP_CARDS.clear()
    fresh = core.ship_card(config)
    assert after.uniform == fresh.uniform and torch.equal(after.table, fresh.table)


def test_an_input_is_swapped_by_editing_the_table_only():
    """Changing the design inputs is an edit of the table: a copy of DESIGN_INPUTS with the
    corner speed (solved for the maximum acceleration) in place of the g limit gives a reference
    whose corner speed is the point mass's, and a moved corner speed is met by the same generic
    solve; nothing else is told about the change."""
    corner = tuning.flown(
        "corner_speed", "corner speed", "Performance", "max_acceleration", kind="speed"
    )
    inputs = tuple(corner if spec.name == "g_limit" else spec for spec in tuning.DESIGN_INPUTS)
    reference = tuning.reference_of(tuning.REAL_AIRCRAFT["p51d"], inputs)
    assert "g_limit" not in reference.design
    assert reference.design["corner_speed"] == pytest.approx(
        reference.point_mass["corner_speed"], rel=1e-12
    )
    assert tuning.inputs_version(inputs) != tuning.inputs_version()
    target = 0.95 * reference.design["corner_speed"]
    solved = tuning.design_row(reference, {"corner_speed": target}, start=tuning.row_of("p51d"))
    flown = tuning.design_of(solved.row, inputs)
    assert flown["corner_speed"] == pytest.approx(target, rel=1e-6)
    assert flown["top_speed"] == pytest.approx(reference.design["top_speed"], rel=1e-6)
    with pytest.raises(ValueError, match="each card field once"):
        tuning.check_inputs(inputs[1:])


# --------------------------------------------------------------------------- profiles


def test_profiles_are_named_tables():
    """Each profile (tuning.PROFILES) is a design-input table that sets every card field once;
    the documents (DESIGN_INPUTS) are the default table and the base of "airframe and engine",
    and the P-51D's card is its airframe and engine (the game's engine)."""
    for profile in tuning.PROFILES.values():
        tuning.check_inputs(profile.inputs)
        assert profile.base is None or profile.base in tuning.PROFILES
    assert list(tuning.PROFILES) == ["documents", "airframe_and_engine"]
    assert tuning.DEFAULT_PROFILE == "documents"
    assert tuning.PROFILES[tuning.DEFAULT_PROFILE].inputs is tuning.DESIGN_INPUTS
    assert tuning.PROFILES["airframe_and_engine"].base == "documents"
    assert tuning.REAL_AIRCRAFT["p51d"].profile == "airframe_and_engine"


def test_a_profile_with_a_base_starts_from_the_base_row(tmp_path, monkeypatch):
    """A profile with a base (here a copy of the documents' table) takes its reference values
    from the base's solved row, so its own design is that row with no solve; a preset solved
    under it is another cache key, and its row is the documents row again (1e-6)."""
    monkeypatch.setenv("BNB_CARD_CACHE", str(tmp_path))
    copy = tuning.Profile("copy", "copy", tuning.DESIGN_INPUTS, "documents")
    monkeypatch.setitem(tuning.PROFILES, "copy", copy)
    documents = tuning.reference("p51d", "documents").start_row
    reference = tuning.reference_in(tuning.REAL_AIRCRAFT["p51d"], "copy")
    assert reference.profile == "copy"
    own = tuning.design_row(reference, start=reference.start_row)
    assert own.iterations == 0
    aircraft = tuning.REAL_AIRCRAFT["p51d"]
    other = dataclasses.replace(aircraft, profile="copy")
    assert tuning.row_key(other) != tuning.row_key(aircraft)
    row = tuning.real_row(other)
    for name, value in documents.items():
        assert row[name] == pytest.approx(value, rel=1e-6), name


def test_airframe_and_engine_is_closed_and_more_power_flies_faster():
    """Under "airframe and engine" nothing is solved: the documents row reads back as its
    documented shaft power (tuning.P51D.power) and the airframe's C_D0; the card's reference is
    that airframe with the game's engine (P51D_SHAFT_POWER), and its row is the card's (1e-6); a
    design solves in no iterations,
    and more shaft power raises top speed and climb."""
    reference = tuning.reference("p51d", "airframe_and_engine")
    assert all(spec.apply is not None for spec in reference.inputs)
    documents = tuning.design_of(tuning.reference("p51d", "documents").start_row, reference.inputs)
    documented, engine = tuning.P51D.power, tuning.P51D_SHAFT_POWER
    assert documents["shaft_power"] == pytest.approx(tuning.to_game(documented, -3), rel=1e-6)
    assert reference.design["shaft_power"] == pytest.approx(tuning.to_game(engine, -3), rel=1e-12)
    for name in ("zero_lift_drag_coefficient", "stall_speed", "g_limit", "oswald"):
        assert reference.design[name] == pytest.approx(documents[name], rel=1e-6), name
    for name, value in tuning.row_of("p51d").items():
        assert reference.start_row[name] == pytest.approx(value, rel=1e-6), name
    power = reference.design["shaft_power"]
    stronger = tuning.design_row(reference, dict(shaft_power=1.5 * power))
    assert stronger.iterations == 0
    before, after = tuning.performance(reference.start_row), tuning.performance(stronger.row)
    assert after["top_speed"] > before["top_speed"]
    assert after["best_climb"] > before["best_climb"]


def test_warnings_point_at_the_profiles_input(p51d):
    """A plausibility check names a card field; its warning points at the input that sets that
    field in the reference's table: the structure's limit at the g limit, and an e outside the
    WWII fighters' range (a hand-made row) at e."""
    row = tuning.row_of("p51d")
    found = {w["quantity"]: w for w in tuning.warnings(row, p51d)}
    assert found["structural_load"]["input"] == "g_limit"
    wide = {**row, "induced_drag_scale": 1.2 * row["induced_drag_scale"]}
    found = {w["quantity"]: w for w in tuning.warnings(wide, p51d)}
    assert found["oswald"]["field"] == "induced_drag_scale"
    assert found["oswald"]["input"] == "oswald"

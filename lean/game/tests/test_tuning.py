"""lean.game.tuning: v2's closed forms reproduce v2's cards, the card's lean-only rows are the
tuning's output, the P-51D's conversion reproduces the October fit, and the handles solve
meets its handles under the core model (docs/rewrite/p51d-model.md)."""

import math
import pathlib
import tomllib

import pytest
import torch

from lean.game import card, tuning

torch.set_num_threads(1)
ROOT = pathlib.Path(__file__).resolve().parents[3]
V2_P51D = ROOT / "src/bnb/reference/real-plane-types/p51d.toml"  # game v16, x2 units
V2_VIZIA = ROOT / "src/bnb/game/ships/real-scale/vizia.toml"  # game v20
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


def load(path):
    return tomllib.loads(path.read_text())


@pytest.mark.parametrize("path", [V2_P51D, V2_VIZIA], ids=["p51d", "vizia"])
def test_direct_fields_reproduce_v2_cards(path):
    """v2's closed forms (the direct flight handles and the combat card) give v2's solved card to
    float32 rounding. The P-51D's v16 file keeps a bullet life of 10 ticks that its own handles
    no longer give (11: the file was carried to x2 by the similarity, not re-solved; v2's Vizia,
    re-solved at x2, has 11), so its bullet life is left out."""
    data = load(path)
    handles, solved = data["handles"], data["card"]
    fields = tuning.direct_fields(handles, handles["cruise_speed"])
    if path == V2_P51D:
        del fields["bullet_life"]
    for name, value in fields.items():
        assert value == pytest.approx(solved[name], rel=1e-6, abs=1e-9), name


def test_card_rows_are_the_tuning():
    """The card's lean-only rows are exactly what lean.game.tuning makes."""
    vizia = tuning.row_of("vizia")
    expected = {
        "p51d": tuning.airframe_row(tuning.P51D, vizia),
        "vizia_plain": tuning.unmodified_climb(vizia),
    }
    assert card.TYPE_NAMES[3:] == tuple(expected)
    for name, row in expected.items():
        held = tuning.row_of(name)
        for field, value in row.items():
            assert held[field] == pytest.approx(value, rel=1e-12), (name, field)
    assert set(row) == set(card.CARD)


def test_p51d_point_mass_is_the_october_fit():
    """With the core model's shared terms off (drag rise, separation drag, soft minima), the
    P-51D card flies the October fit's point mass in real units: top speed and climb within
    0.5%, the best sustained turn within 1%, its speed and radius within 2% (a 0.05 m/s grid)."""
    p = tuning.in_real_units(tuning.performance(tuning.row_of("p51d"), plain=True))
    assert p["top_speed"] == pytest.approx(OCTOBER["top_speed"], rel=0.005)
    assert p["best_climb"] == pytest.approx(OCTOBER["best_climb"], rel=0.005)
    assert p["sustained_rate"] == pytest.approx(OCTOBER["sustained_rate"], rel=0.01)
    assert p["sustained_speed"] == pytest.approx(OCTOBER["sustained_speed"], rel=0.02)
    assert p["sustained_radius"] == pytest.approx(OCTOBER["sustained_radius"], rel=0.02)


def test_p51d_flies_its_point_mass_under_the_core_model():
    """The P-51D's row has the game terms off (tuning.POINT_MASS_TERMS), so under the core model
    it flies its point mass: the October fit, and the same numbers as plain=True."""
    row = tuning.row_of("p51d")
    assert tuning.performance(row) == tuning.performance(row, plain=True)


def test_p51d_airframe_with_vizia_terms_against_the_october_fit():
    """With Vizia's game terms (v2's shared values) the P-51D airframe flies below the October
    fit by the measured costs (docs/rewrite/p51d-model.md): top speed -3.7% (the drag rise), climb
    -14% (the thrust's soft minimum), best sustained turn -17% (the separation drag), the stall
    and corner speeds exact. Tolerance: those, each within 1.5 points."""
    row = tuning.airframe_row(tuning.P51D, tuning.row_of("vizia"), tuning.game_terms(tuning.row_of("vizia")))
    p = tuning.in_real_units(tuning.performance(row))
    assert p["top_speed"] / OCTOBER["top_speed"] - 1 == pytest.approx(-0.037, abs=0.015)
    assert p["best_climb"] / OCTOBER["best_climb"] - 1 == pytest.approx(-0.145, abs=0.015)
    assert p["sustained_rate"] / OCTOBER["sustained_rate"] - 1 == pytest.approx(-0.173, abs=0.015)
    stall = math.sqrt(2 * 4427 * 9.81 / (1.225 * 21.83 * 1.50))
    assert p["stall_speed"] == pytest.approx(stall, rel=1e-9)
    assert p["corner_speed"] == pytest.approx(stall * math.sqrt(8.0), rel=1e-9)


@pytest.mark.parametrize("name", ["p51d", "vizia_plain"])
def test_speed_order(name):
    """The required order (core-decisions.md): minimum sustained radius, best sustained turn,
    corner, slowest to fastest."""
    p = tuning.performance(tuning.row_of(name))
    assert p["min_radius_speed"] < p["sustained_speed"] < p["corner_speed"]


def test_vizia_plain_flies_as_vizia_when_coasting():
    """v2's climb modifications are inert under the core model when coasting: reverting them
    changes no coasting number (only the dive and climb trade, 0.98 g to 1 g)."""
    vizia = tuning.performance(tuning.row_of("vizia"))
    plain = tuning.performance(tuning.row_of("vizia_plain"))
    for key, value in vizia.items():
        assert plain[key] == pytest.approx(value, rel=1e-9), key


def test_handles_solve_meets_the_handles_under_the_core_model():
    """The handles route: v2's P-51D handles solved under the core model fly them (cruise,
    low-speed acceleration, settled full-stick turn) to 1e-6, the direct fields as v2's."""
    handles = load(V2_P51D)["handles"]
    row = tuning.handles_row(handles, tuning.row_of("vizia"))
    p = tuning.performance(row)
    assert p["top_speed"] == pytest.approx(handles["cruise_speed"], rel=1e-6)
    assert p["low_speed_acceleration"] == pytest.approx(handles["low_speed_acceleration"], rel=1e-6)
    assert p["settled_turn_rate"] == pytest.approx(handles["sustained_turn_rate"], rel=1e-6)
    assert row["max_lift_coefficient"] == pytest.approx(
        load(V2_P51D)["card"]["max_lift_coefficient"], rel=1e-6
    )


@pytest.mark.parametrize("factor", [1.0, 1.5, 3.0])
def test_time_scale_is_the_same_flight_faster_or_slower(factor):
    """At another time factor the card flies the same real aircraft: gravity factor^2 x 9.81,
    and cruise and corner speed (game units) scaled by factor / 2; the card is restored after."""
    from lean.game import core

    config = core.GameConfig.core(lineup=card.TYPE_NAMES)
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

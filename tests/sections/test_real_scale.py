"""The real-scale variant (`docs/explanation/game/real-scale.md`): its ship set (Yongma,
Jujak and vizia), its rules and its presets; and the reference data of the real planes the
set was researched from (`bnb.reference.real_planes`), no ship types since 2026-11-13.

The set plays at x2 native speed (game v13): its files hold the similarity at length 1 and
time 1 / `real_scale.EQUIVALENT_FACTOR` of the planes' values. So a check of a researched
value reads the reference type at the planes' speed (`plane`)."""

import math
import shutil

import pytest

from bnb.contracts import GameConfig
from bnb.game import api, real_scale, ships
from bnb.game.handling.similarity import HANDLE_UNITS, factor, scale_flight
from bnb.game.rules import FlightConstants
from bnb.reference import real_planes

SET = "real-scale"
SPEED_UP = real_scale.EQUIVALENT_FACTOR
KEPT = ("yongma", "jujak", "vizia")


def plane(handles: dict[str, float]) -> dict[str, float]:
    """A ship file's handles at the planes' speed (speeds / 2, rates / 2, times x 2,
    energies / 4, lengths kept)."""
    return {
        name: value * factor(HANDLE_UNITS[name], 1.0, SPEED_UP) for name, value in handles.items()
    }


def test_the_presets_play_the_real_scale_set_on_a_map_three_times_wider():
    """Frontline and elimination on the 7800 m map; team deathmatch on 4200 m with a 10 s
    respawn wait (the author, 2026-11-13). The random maps are the author's (game v14)."""
    for mode in ("deathmatch", "frontline", "elimination"):
        standard, real = api.PRESETS[mode], api.PRESETS[f"real_scale_{mode}"]
        wide = 4200.0 if mode == "deathmatch" else 3.0 * standard.arena_radius
        assert real.ship_set == SET and real.arena_radius == wide
        rules = api.rules_of(real)
        assert rules.pools[0].arena_radius == wide
        standard_rules = api.rules_of(standard)
        assert rules.pools[0].episode_ticks == 2 * standard_rules.pools[0].episode_ticks
        if mode == "frontline":
            assert rules.pools[0].respawn_ticks == 2 * standard_rules.pools[0].respawn_ticks
        if mode == "deathmatch":
            assert rules.pools[0].respawn_ticks == round(10.0 / rules.tick_length)
        assert rules.pools[0].mode.capture_seconds == 20.0
        assert rules.pools[0].mode.group_radius == 900.0
        assert rules.pools[0].mode.shrink_hold_seconds == 36.0  # a fifth of 180 s
        assert rules.field_band_width == 160.0
        assert (rules.field_radius_min, rules.field_radius_max) == (120.0, 1200.0)
        assert rules.field_radius_distribution == 2
        assert (rules.field_radius_median, rules.field_radius_log_sigma) == (400.0, 0.6)
        assert (rules.boundary_max_damage, rules.boundary_ramp_width) == (40.0, 100.0)


def test_the_rules_are_todays_in_real_units_at_x2_speed():
    rules = api.rules_of(api.PRESETS["real_scale_frontline"])
    assert rules.flight.gravity == pytest.approx(9.81 * SPEED_UP**2, rel=1e-12)
    real = scale_flight(FlightConstants(), real_scale.LENGTH, real_scale.TIME)
    expected = scale_flight(real, 1.0, 1.0 / SPEED_UP)
    assert rules.flight.nose_frequency_floor == pytest.approx(
        expected.nose_frequency_floor, rel=1e-12
    )
    # the exchanges keep their real-unit values: the card's per-type scales carry the x4
    # (`bnb.play.tuning.time_scaled_card`), as no setting holds them
    for name in ("coast_exchange", "climb_exchange", "dive_exchange"):
        assert getattr(rules.flight, name) == pytest.approx(getattr(real, name), rel=1e-12)
    assert rules.longest_bullet_moves == 11  # 0.37 s, the MG 151/20's (its high drag): 0.73 s at x1
    assert rules.bullet_drag == 5e-4  # the .50 cal's, every type's floor
    assert rules.damage_reference_distance == 300.0
    card = api.ship_types(ship_set=SET)
    assert (card.bullet_drag_scale >= 1.0 - 1e-6).all()
    assert rules.vision_range == 3072.0
    # the spawn speed is 1.3 x the fastest stall speed of the types the rules were solved
    # under (the researched ones, now reference data), so above every set type's

    def stall_of(card: dict) -> float:
        return math.sqrt(rules.flight.gravity / card["max_lift_coefficient"])

    fastest = max(stall_of(one.card) for one in real_planes.solved_types().values())
    assert rules.spawn_speed == pytest.approx(1.3 * fastest, rel=1e-4)
    assert all(rules.spawn_speed > 1.3 * stall_of(ship.card) for ship in ships.load(SET))


def test_the_set_is_the_two_archetypes_and_their_hull_and_may_hold_player_types():
    """Yongma and Jujak first (the game's default lineup alternates types 0 and 1), then
    vizia; any type after them is a player's (made with the tuning panel's Add, or by hand)
    and only has to be a valid ship file: a unique key and name, every handle and every card
    field, a fire rate the tick can hold. No researched type is one of them."""
    found = ships.load(SET)
    assert tuple(ship.key for ship in found[: len(KEPT)]) == KEPT
    assert api.type_names(SET)[:2] == ("Yongma", "Jujak")
    assert len({ship.name for ship in found}) == len(found)
    assert len({ship.key for ship in found}) == len(found)
    assert not {ship.key for ship in found} & set(real_planes.solved_types())
    card_fields = set(api.default_card(SET))
    for ship in found:
        assert set(ship.handles) == set(api.HANDLE_NAMES), ship.key
        assert set(ship.card) == card_fields, ship.key
        assert all(math.isfinite(value) for value in ship.card.values()), ship.key
        assert 30.0 % ship.handles["fire_rate"] == 0.0, ship.key


def test_a_player_type_appended_passes_the_pool_rule(tmp_path, monkeypatch):
    """A copy of vizia written after the set's types under another name (as Add writes one)
    keeps the set's types first and in order, and is a valid type."""
    shutil.copytree(ships.SHIPS, tmp_path / "ships")
    monkeypatch.setattr(ships, "SHIPS", tmp_path / "ships")
    vizia = ships.load(SET)[KEPT.index("vizia")]
    ships.append(ships.Ship("test-copy", "Test copy", vizia.handles, vizia.card, {}), SET)
    assert ships.load(SET)[-1].key == "test-copy"
    test_the_set_is_the_two_archetypes_and_their_hull_and_may_hold_player_types()


def test_the_reference_holds_every_researched_plane_with_its_sources():
    """Every plane of the research (18 late-war fighters and 8 invented designs) is kept,
    each quantity with a confidence and a source, and each type as the set last held it,
    with a note for every handle (but the core radius, a rule of game v12, after the
    research) saying where its value came from."""
    kinds = ("researched", "estimated", "derived", "invented", "placeholder", "proposal")
    assert len(real_planes.AIRCRAFT) == 26
    solved = real_planes.solved_types()
    assert list(solved) == [one.key for one in real_planes.AIRCRAFT]
    for one in real_planes.AIRCRAFT:
        assert set(one.data) <= set(real_planes.QUANTITIES), one.key
        for name, datum in one.data.items():
            assert math.isfinite(datum.value) and datum.confidence and datum.source, (one.key, name)
        assert solved[one.key].name == one.name
        for name in set(api.HANDLE_NAMES) - {"gun_range", "climb_time", "core_radius"}:
            assert solved[one.key].notes[name].startswith(kinds), (one.key, name)


def test_the_p51d_reference_is_the_researched_plane_with_a_600_m_gun():
    """The P-51D as researched, its spread the volley's 6 mil 75% cone: sigma = 6e-3 /
    (2 x 1.665) rad times the muzzle speed (the circle of radius sqrt(2 ln 4) sigma holds
    75% of a 2-D normal)."""
    p51 = real_planes.solved_types()["p51d"]
    handles = plane(p51.handles)
    assert p51.name == "P-51D"
    assert handles["stall_speed"] == 44.7 and handles["g_limit"] == 8.0
    assert handles["cruise_speed"] == 162.7 and handles["ceiling"] == pytest.approx(9.81 * 12680)
    assert handles["sustained_turn_rate"] == pytest.approx(math.radians(15.7))
    assert handles["muzzle_speed"] == 856.0 and handles["hit_radius"] == pytest.approx(5.645)
    assert handles["gun_range"] == pytest.approx(600.0, abs=15.0)
    assert "placeholder" in p51.notes["nose_reversal_time"]
    assert p51.notes["stall_speed"].startswith("researched (medium)")
    data = next(one for one in real_planes.AIRCRAFT if one.key == "p51d").data
    assert data["dispersion"].value == 6.0
    sigma = 6e-3 / (2.0 * math.sqrt(2.0 * math.log(4.0)))
    assert handles["bullet_spread"] == pytest.approx(sigma * 856.0)


def test_the_reference_nose_reversal_follows_the_roll_rate():
    solved = real_planes.solved_types()
    fw190, p51 = solved["fw190-a8"].handles, solved["p51d"].handles
    assert fw190["nose_reversal_time"] / p51["nose_reversal_time"] == pytest.approx(95.0 / 150.0)


def test_the_standard_game_is_untouched_by_the_set():
    assert api.rules_of(api.PRESETS["frontline"]).flight == FlightConstants()
    assert api.type_names() == ("Fighter", "Heavy")
    assert GameConfig(1, 1, 1.0).ship_set == "standard"

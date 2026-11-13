"""The real-scale variant (`docs/explanation/game/real-scale.md`): its ship set, its rules
and its presets, and that its cards fly their handles.

The set plays at x2 native speed (game v13): its files hold the similarity at length 1 and
time 1 / `real_scale.EQUIVALENT_FACTOR` of the planes' values, which `real_scale`'s
arithmetic gives. So a test of the files against that arithmetic reads the files at the
planes' speed (`plane`), or scales the arithmetic's values as the build did (`native`)."""

import json
import math
import shutil

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api, real_scale, ships
from bnb.game.handling.similarity import HANDLE_UNITS, factor, scale_flight
from bnb.game.physics.combat import MEAN_CLOSENESS
from bnb.game.rules import FlightConstants, Rules
from bnb.play.tuning import snap_handle

SET = "real-scale"
SPEED_UP = real_scale.EQUIVALENT_FACTOR


def plane(handles: dict[str, float]) -> dict[str, float]:
    """A ship file's handles at the planes' speed (speeds / 2, rates / 2, times x 2,
    energies / 4, lengths kept)."""
    return {
        name: value * factor(HANDLE_UNITS[name], 1.0, SPEED_UP) for name, value in handles.items()
    }


def native(name: str, value: float) -> float:
    """A planes'-speed handle as the set's file holds it: scaled to x2, snapped as the card
    holds it (docs/rewrite/sections/lethality/native_x2.py)."""
    return snap_handle(
        name, value * factor(HANDLE_UNITS[name], 1.0, 1.0 / SPEED_UP), Rules.tick_length
    )


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
    stall = torch.sqrt(rules.flight.gravity / api.ship_types(ship_set=SET).max_lift_coefficient)
    assert rules.spawn_speed == pytest.approx(1.3 * float(stall.max()), rel=1e-4)


def test_the_first_type_is_the_researched_p51d_with_a_600_m_gun():
    p51 = ships.load(SET)[0]
    handles = plane(p51.handles)
    assert p51.name == "P-51D"
    assert handles["stall_speed"] == 44.7 and handles["g_limit"] == 8.0
    assert handles["cruise_speed"] == 162.7 and handles["ceiling"] == pytest.approx(9.81 * 12680)
    assert handles["sustained_turn_rate"] == pytest.approx(math.radians(15.7))
    assert handles["muzzle_speed"] == 856.0 and handles["hit_radius"] == pytest.approx(5.645)
    assert handles["gun_range"] == pytest.approx(600.0, abs=15.0)
    assert "placeholder" in p51.notes["nose_reversal_time"]
    assert p51.notes["stall_speed"].startswith("researched (medium)")


def researched() -> list[ships.Ship]:
    """The set's researched types (`real_scale.AIRCRAFT`): its first, in their order."""
    return ships.load(SET)[: len(real_scale.AIRCRAFT)]


def test_the_set_starts_with_the_researched_aircraft_and_may_hold_player_types():
    """The pool rule: the module's aircraft come first, in order; any type after them is a
    player's (made with the tuning panel's Add, or by hand) and only has to be a valid ship
    file: a unique key and name, every handle and every card field, a fire rate the tick
    can hold."""
    found = ships.load(SET)
    assert [ship.key for ship in found[: len(real_scale.AIRCRAFT)]] == [
        one.key for one in real_scale.AIRCRAFT
    ]
    assert len({ship.name for ship in found}) == len(found)
    assert len({ship.key for ship in found}) == len(found)
    card_fields = set(api.default_card(SET))
    for ship in found[len(real_scale.AIRCRAFT) :]:
        assert set(ship.handles) == set(api.HANDLE_NAMES), ship.key
        assert set(ship.card) == card_fields, ship.key
        assert all(math.isfinite(value) for value in ship.card.values()), ship.key
        assert 30.0 % ship.handles["fire_rate"] == 0.0, ship.key


def test_a_player_type_after_the_researched_ones_passes_the_pool_rule(tmp_path, monkeypatch):
    """A copy of the P-51D written after the researched types under another name (as Add
    writes one) keeps the researched types first and in order, is a valid type, and is
    one of the types a new solve of the set keeps after the researched ones."""
    shutil.copytree(ships.SHIPS, tmp_path / "ships")
    monkeypatch.setattr(ships, "SHIPS", tmp_path / "ships")
    p51 = ships.load(SET)[0]
    ships.append(ships.Ship("test-copy", "Test copy", p51.handles, p51.card, {}), SET)
    found = ships.load(SET)
    assert found[-1].key == "test-copy"
    assert real_scale.player_types()[-1].key == "test-copy"  # kept when the set is solved again
    assert not {one.key for one in real_scale.player_types()} & {
        one.key for one in real_scale.AIRCRAFT
    }
    assert [ship.key for ship in found[: len(real_scale.AIRCRAFT)]] == [
        one.key for one in real_scale.AIRCRAFT
    ]
    test_the_set_starts_with_the_researched_aircraft_and_may_hold_player_types()


def test_every_type_is_written_with_a_note_for_every_handle():
    """Each researched type's ship file's handles are the module's targets (the climb time
    solved), and each note says where its value comes from."""
    found = researched()
    rules = api.rules_of(api.PRESETS["real_scale_frontline"])
    combat = real_scale.combat_constants()
    kinds = ("researched", "estimated", "derived", "invented", "placeholder", "proposal")
    for ship, aircraft in zip(found, real_scale.AIRCRAFT):
        assert set(ship.notes) == set(api.HANDLE_NAMES), ship.key
        targets, notes, _ = real_scale.target_handles(aircraft, rules, combat)
        for name, value in targets.items():
            if name != "climb_time":
                assert ship.handles[name] == pytest.approx(native(name, value), rel=1e-9), (
                    ship.key,
                    name,
                )
            assert ship.notes[name].startswith(notes[name]), (ship.key, name)
        for name in set(api.HANDLE_NAMES) - {"gun_range", "climb_time"}:
            assert ship.notes[name].startswith(kinds), (ship.key, name, ship.notes[name])
        assert 30.0 % ship.handles["fire_rate"] == 0.0, ship.key


def test_the_spread_is_the_volleys_75_percent_cone():
    """A 6 mil cone holding 75% of the rounds: sigma = 6e-3 / (2 x 1.665) rad, times the
    muzzle speed; the circle of radius 1.665 sigma holds 75% of a 2-D normal."""
    assert real_scale.SPREAD_RADIUS_75 == pytest.approx(1.6651, abs=1e-4)
    assert 1.0 - math.exp(-(real_scale.SPREAD_RADIUS_75**2) / 2.0) == pytest.approx(0.75)
    assert real_scale.spread_of(6.0, 856.0) == pytest.approx(6e-3 / 3.3302 * 856.0, rel=1e-4)
    p51 = ships.load(SET)[0]
    assert plane(p51.handles)["bullet_spread"] == pytest.approx(real_scale.spread_of(6.0, 856.0))


def test_the_combat_economy_follows_the_weight_of_fire_and_the_climb_charge():
    """A P-51 shot costs about the standard fighter's 1 s of climb charge, every type's
    continuous fire the same multiple of its own charge, and damage per second follows the
    weight of fire (the P-51's twice the standard fighter's 12 x 15, provisional) only
    weakly: the equal-damage rule keeps it within 5% of the P-51's."""
    power, damage = real_scale.combat_constants()
    standard = {ship.key: ship.handles for ship in ships.load()}["fighter"]
    mean_hit = standard["hit_damage"] * MEAN_CLOSENESS  # the card's is a centre hit's
    assert damage == pytest.approx(2.0 * mean_hit * standard["fire_rate"]) and damage == 360.0
    found = {ship.key: plane(ship.handles) for ship in ships.load(SET)}
    data = {one.key: one.data for one in real_scale.AIRCRAFT}
    charge = standard["ceiling"] / standard["shots_per_store"] / power * standard["fire_rate"]
    assert charge == pytest.approx(real_scale.standard_climb_charges()["fighter"])
    for key in ("p51d", "me262", "yak3", "anvil"):
        handles = found[key]
        cost = handles["ceiling"] / handles["shots_per_store"]
        own_charge = 9.81 * data[key]["climb_rate"].value
        assert cost * handles["fire_rate"] == pytest.approx(power * own_charge), key
        ratio = data[key]["weight_of_fire"].value / data["p51d"]["weight_of_fire"].value
        balanced = real_scale.balanced_damage_per_second(damage * ratio, damage)
        mean = handles["hit_damage"] * MEAN_CLOSENESS * handles["fire_rate"]
        assert mean == pytest.approx(balanced), key
        assert handles["fire_rate"] == data[key]["fire_rate"].value, key
    p51 = found["p51d"]
    assert p51["ceiling"] / p51["shots_per_store"] / (9.81 * 15.2) == pytest.approx(0.96, abs=0.01)


def test_every_type_deals_about_the_p51s_damage_per_second():
    """The equal-damage rule, for the researched types: (w / w_P-51)^0.05 within 5% of 360
    per second, at the type's own fire rate; the note records the value before it."""
    rule = real_scale.balanced_damage_per_second
    assert rule(360.0, 360.0) == 360.0
    assert rule(720.0, 360.0) == pytest.approx(360.0 * 2.0**0.05)
    assert rule(1e6, 360.0) == pytest.approx(378.0) and rule(1.0, 360.0) == pytest.approx(342.0)
    for ship in researched():
        handles = plane(ship.handles)
        per_second = handles["hit_damage"] * MEAN_CLOSENESS * handles["fire_rate"]
        assert 342.0 - 1e-6 <= per_second <= 378.0 + 1e-6, ship.key
        assert "Balanced (round 1)" in ship.notes["hit_damage"], ship.key


def test_a_balance_move_sets_its_handle_and_notes_old_new_and_why(tmp_path, monkeypatch):
    moves = {"p51d": {"max_shield": {"value": 90.0, "round": 2, "reason": "a test."}}}
    path = tmp_path / "balance.json"
    path.write_text(json.dumps(moves))
    monkeypatch.setattr(real_scale, "BALANCE_FILE", path)
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"])
    values, notes, _ = real_scale.target_handles(real_scale.AIRCRAFT[0], rules)
    assert values["max_shield"] == 90.0
    assert notes["max_shield"].endswith("Balanced (round 2): 100 -> 90: a test.")
    assert real_scale.balance_moves(tmp_path / "absent.json") == {}
    moves = {"_rules": {"shield_per_rating": {"value": 0.1, "round": 2, "reason": "a test."}}}
    path.write_text(json.dumps(moves))
    p47 = next(one for one in real_scale.AIRCRAFT if one.key == "p47d")  # survivability 5
    values, notes, _ = real_scale.target_handles(p47, rules)
    assert values["max_shield"] == pytest.approx(130.0)
    assert (
        "Balanced (round 2): 160 -> 130, the step per rating 0.2 -> 0.1: a test."
        in (notes["max_shield"])
    )


def test_the_nose_reversal_follows_the_roll_rate():
    found = {ship.key: ship.handles for ship in ships.load(SET)}
    ratio = found["fw190-a8"]["nose_reversal_time"] / found["p51d"]["nose_reversal_time"]
    assert ratio == pytest.approx(95.0 / 150.0)


def test_the_standard_game_is_untouched_by_the_set():
    assert api.rules_of(api.PRESETS["frontline"]).flight == FlightConstants()
    assert api.type_names() == ("Fighter", "Heavy")
    assert GameConfig(1, 1, 1.0).ship_set == "standard"


@pytest.mark.slow
def test_each_real_scale_card_flies_its_handles():
    """Measured (minutes: a 125 kJ store climbs for about an hour of game time), every
    flown handle is within the solver's 0.5% of the ship file's (but a handle the notes mark
    MISSED), and each climb's initial charge is g times its climb rate."""
    rules = api.rules_of(api.PRESETS["real_scale_frontline"])
    card = api.ship_types(dtype=torch.float64, ship_set=SET)
    flown = api.measure_handles(card, rules)
    for index, ship in enumerate(researched()):
        for name in api.FLOWN_HANDLES:
            if "MISSED" in ship.notes[name]:
                continue
            assert float(getattr(flown, name)[index]) == pytest.approx(
                ship.handles[name], rel=5e-3
            ), (ship.name, name)
    charge = real_scale.initial_climb_charge(rules, card, flown.cruise_speed)
    for index, aircraft in enumerate(real_scale.AIRCRAFT):
        rate = real_scale.target_handles(aircraft, rules)[2]
        assert float(charge[index]) == pytest.approx(9.81 * rate, rel=real_scale.CLIMB_TOLERANCE), (
            aircraft.key
        )

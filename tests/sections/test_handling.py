"""The ship's characteristics (scripted spec A2, A7.5): each measured quantity against an
independent run of the game, and their response to the card."""

import dataclasses
import json
import math
import pathlib

import pytest
import torch

from bnb.contracts import Action, GameConfig
from bnb.game import api
from bnb.game.handling.characteristics import (
    characteristics,
    measure_characteristics,
    time_of_flight,
)
from bnb.game.handling.manoeuvres import TURNING_STORE
from bnb.game.handling.shown import SETTLE_TIME
from bnb.game.physics.flight import airframe_of
from bnb.game.state import replace

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "reference" / "golden"
RULES = api.rules_of(GameConfig(1, 1, 1e6, mode="sandbox"))
TYPES = api.ship_types()
MEASURED = characteristics(TYPES, RULES)


def _one_ship(ship_type: int, speed: float, energy: float):
    """A one-a-side sandbox game of this ship type: ship 0 flies along +x at the origin,
    n = 1 (the field far away), and ship 1 is far off."""
    rules = dataclasses.replace(RULES, team_types=(ship_type,))
    state = api.reset(
        rules,
        TYPES,
        api.uniforms(rules, 1, generator=torch.Generator().manual_seed(0)),
        torch.zeros(1, dtype=torch.bool),
    )
    fields = replace(
        state.fields,
        x=torch.full_like(state.fields.x, 1e5),
        y=torch.full_like(state.fields.y, 1e5),
        radius=torch.ones_like(state.fields.x),
    )
    ships = state.ships

    def at(values, first, rest):
        return torch.tensor([[first, rest]], dtype=values.dtype)

    zeros = torch.zeros_like(ships.x)
    ships = replace(
        ships,
        x=at(ships.x, 0.0, 5e4),
        y=zeros,
        heading_cos=torch.ones_like(ships.x),
        heading_sin=zeros,
        proper_speed=at(ships.x, speed, 50.0),
        angle_of_attack=zeros,
        angle_of_attack_cos=torch.ones_like(ships.x),
        angle_of_attack_sin=zeros,
        nose_rate=zeros,
        energy=at(ships.x, energy, 0.0),
        log_refractive_index=zeros,
        log_index_gradient_x=zeros,
        log_index_gradient_y=zeros,
        gun_cooldown=torch.zeros_like(ships.gun_cooldown),
    )
    return rules, replace(state, ships=ships, fields=fields)


def _action(power=0, stick=0, shoot=0) -> Action:
    def ship_zero(value):
        return torch.tensor([[value, 0]], dtype=torch.int32)

    return Action(ship_zero(power), ship_zero(stick), ship_zero(shoot))


@pytest.mark.parametrize("ship_type", [0, 1])
def test_gun_range_and_time_of_flight_match_a_bullet_fired_in_the_game(ship_type):
    cruise = float(MEASURED.cruise_speed[ship_type])
    rules, state = _one_ship(ship_type, cruise, float(TYPES.max_energy[ship_type]))
    spread = torch.zeros((1, 2, 2))
    state, _ = api.move(rules, TYPES, state, _action(shoot=1), spread)
    start = torch.tensor([0.0, 0.0])
    for move in range(2, rules.bullet_moves + 1):
        state, _ = api.move(rules, TYPES, state, _action(), spread)
        travelled = float(
            torch.hypot(state.bullets.x[0, 0, 0] - start[0], state.bullets.y[0, 0, 0] - start[1])
        )
        flight = float(time_of_flight(MEASURED, torch.tensor(ship_type), torch.tensor(travelled)))
        assert flight == pytest.approx(move * rules.tick_length, rel=2e-3), move
    assert travelled == pytest.approx(float(MEASURED.gun_range[ship_type]), rel=2e-3)
    # Fired along the flight path, the muzzle speed adds to the ship's.
    assert float(MEASURED.bullet_speed[ship_type]) == pytest.approx(
        cruise + float(TYPES.muzzle_speed[ship_type]), rel=1e-4
    )


def test_cruise_and_top_speed_match_the_reference_maneuvers():
    """The reference flight's maneuvers (tests/reference/golden): coasting with a full store
    for 60 s settles at cruise; a boost from 105 m/s with a full store peaks near top speed
    (it starts slightly above cruise, so the peak is a little higher for the fighter)."""
    stats = json.loads((GOLDEN / "flight_maneuvers.json").read_text())["stats"]
    for ship_type, prefix in ((0, ""), (1, "heavy_")):
        assert float(MEASURED.cruise_speed[ship_type]) == pytest.approx(
            stats[f"{prefix}top_full_store"]["final_speed"], rel=2e-4
        )
        assert float(MEASURED.top_speed[ship_type]) == pytest.approx(
            stats[f"{prefix}dive_full_store"]["peak_speed"], rel=5e-3
        )


@pytest.mark.slow
@pytest.mark.parametrize("ship_type", [0, 1])
def test_turn_rates_match_a_settled_turn_in_the_game(ship_type):
    """The coast datum: from cruise, the stick held and the store held at TURNING_STORE,
    coasting for SETTLE_TIME through the game's move; the last second's path rate."""
    cruise = float(MEASURED.cruise_speed[ship_type])
    store = TURNING_STORE * float(TYPES.max_energy[ship_type])
    second = round(1.0 / RULES.tick_length)
    for level, stick in enumerate((2, 4, 6)):
        rules, state = _one_ship(ship_type, cruise, store)
        angles = []
        for _ in range(round(SETTLE_TIME / rules.tick_length)):
            state, _ = api.move(rules, TYPES, state, _action(stick=stick), torch.zeros((1, 2, 2)))
            energy = state.ships.energy.clone()
            energy[0, 0] = store
            state = replace(state, ships=replace(state.ships, energy=energy))
            ships = state.ships
            path_cos = (
                ships.heading_cos * ships.angle_of_attack_cos
                + ships.heading_sin * ships.angle_of_attack_sin
            )
            path_sin = (
                ships.heading_sin * ships.angle_of_attack_cos
                - ships.heading_cos * ships.angle_of_attack_sin
            )
            angles.append(math.atan2(float(path_sin[0, 0]), float(path_cos[0, 0])))
        turned = torch.tensor(angles[-second - 1 :]).diff()
        turned = torch.remainder(turned + math.pi, 2 * math.pi) - math.pi
        rate = float(turned.sum()) / (second * rules.tick_length)
        assert float(MEASURED.turn_rate[ship_type, level]) == pytest.approx(rate, rel=0.01)


def test_card_values_and_constants_are_read_as_given():
    for name in (
        "max_energy",
        "bullet_energy_cost",
        "max_shield",
        "hit_radius",
        "max_acceleration",
    ):
        assert torch.equal(getattr(MEASURED, name), getattr(TYPES, name)), name
    airframe = airframe_of(TYPES, torch.arange(TYPES.max_energy.shape[0]))
    assert torch.equal(MEASURED.corner_speed, airframe.corner_speed)  # the game's own value
    torch.testing.assert_close(
        MEASURED.stall_angle, RULES.flight.stall_angle * TYPES.stall_angle_scale
    )
    torch.testing.assert_close(MEASURED.vision_range, RULES.vision_range * TYPES.vision_range_scale)
    torch.testing.assert_close(MEASURED.bullet_drag, RULES.bullet_drag * TYPES.bullet_drag_scale)
    assert characteristics(TYPES, RULES) is MEASURED  # cached


@pytest.mark.slow
def test_the_characteristics_move_with_the_card():
    def scaled(name: str, factor: float):
        card = api.ship_types(**{name: getattr(TYPES, name) * factor})
        return measure_characteristics(card, RULES)

    faster_gun = scaled("muzzle_speed", 1.1)
    assert torch.all(faster_gun.gun_range > MEASURED.gun_range)
    assert torch.all(faster_gun.bullet_speed > MEASURED.bullet_speed)
    draggier = scaled("zero_lift_drag", 1.2)
    assert torch.all(draggier.cruise_speed < MEASURED.cruise_speed)
    assert torch.all(draggier.top_speed < MEASURED.top_speed)
    stronger = scaled("max_acceleration", 1.2)
    assert torch.all(stronger.corner_speed > MEASURED.corner_speed)
    liftier = scaled("max_lift_coefficient", 1.2)
    assert torch.all(liftier.corner_speed < MEASURED.corner_speed)
    assert torch.all(liftier.turn_rate[:, 0] > MEASURED.turn_rate[:, 0])

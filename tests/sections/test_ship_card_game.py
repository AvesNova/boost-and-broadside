"""The ship card in the game (ship-card spec 1, 2, 5): the bleed exponent, every per-type
value taking effect for its own type only, the time-native bullet life and weapon
cadence, and energy conservation at each supported tick rate."""

import dataclasses
import math

import pytest
import torch
from ship_card_cases import seconds_between_shots

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.physics import flight
from bnb.game.physics.bullets import fire, move_bullets
from bnb.game.physics.combat import apply_damage, hits
from bnb.game.rules import POWER_MODES, STICK_LEVELS, FlightConstants
from bnb.game.ship_types import SCALED, ship_types, ticks_between_shots
from bnb.game.state import Bullets, Fields, Ships, Sweep, replace

RATES = (25, 30, 60)


def _ships(speed, energy, angle=None, ship_type=None, dtype=torch.float32) -> Ships:
    speed = torch.as_tensor(speed, dtype=dtype)
    zeros = torch.zeros_like(speed)
    angle = zeros if angle is None else torch.as_tensor(angle, dtype=dtype)
    kind = torch.zeros(speed.shape, dtype=torch.int32) if ship_type is None else ship_type
    flags = torch.zeros(speed.shape, dtype=torch.bool)
    integers = torch.zeros_like(kind)
    return Ships(x=zeros, y=zeros, heading_cos=zeros + 1.0, heading_sin=zeros, proper_speed=speed,
                 angle_of_attack=angle, angle_of_attack_cos=torch.cos(angle),
                 angle_of_attack_sin=torch.sin(angle), nose_rate=zeros,
                 energy=torch.as_tensor(energy, dtype=dtype).expand_as(speed).clone(),
                 log_refractive_index=zeros, log_index_gradient_x=zeros,
                 log_index_gradient_y=zeros, shield=zeros, shield_recharge_wait=zeros,
                 alive=~flags, fired=flags, spawned=flags, respawn_wait=integers,
                 gun_cooldown=integers, next_slot=integers, ship_type=kind)


def _no_fields(like):
    ones = torch.ones((like.shape[0], 1), dtype=like.dtype)
    return Fields(x=ones * 1e9, y=ones * 0.0, radius=ones, band_width=ones,
                  level=torch.ones((like.shape[0], 1), dtype=torch.int32))


# ------------------------------------------------------------------ the bleed exponent

def test_bleed_exponent_two_is_exactly_the_old_induced_drag():
    """At e = 2 the new formula returns its argument bit for bit (zero included), so the
    default card's drag is the old (C_D0 + k (C^2 + (z/u^2)^2)) u^2 exactly."""
    generator = torch.Generator().manual_seed(0)
    lift = torch.rand(10_000, generator=generator) * 0.05
    lift[:10] = 0.0
    airframe = flight.airframe_of(ship_types(), torch.zeros(10_000, dtype=torch.int32))
    assert torch.equal(flight.bleed(airframe, lift * lift), lift * lift)


def test_bleed_exponent_sets_the_induced_drag_power():
    """k (C^2 + (z/u^2)^2)^(e/2) for e != 2, against float64, and finite at zero lift."""
    exponent = torch.tensor([1.0, 3.0, 4.5])
    card = ship_types(dtype=torch.float64, bleed_exponent=exponent[:, None].expand(3, 2))
    airframe = flight.airframe_of(card, torch.zeros((3, 4), dtype=torch.int32))
    squared = torch.tensor([0.0, 1e-6, 4e-4, 2e-3], dtype=torch.float64).expand(3, 4)
    want = squared ** (0.5 * exponent.double()[:, None])
    assert torch.allclose(flight.bleed(airframe, squared), want, rtol=1e-12, atol=0.0)


# ------------------------------------------------------------------ per-type values

FLIGHT_SCALES = ("induced_drag_scale", "stall_angle_scale", "coast_exchange_scale",
                 "dive_exchange_scale", "climb_exchange_scale", "bleed_exponent")


@pytest.mark.parametrize("name", FLIGHT_SCALES)
def test_flight_values_are_per_type(name):
    """Changing a flight value for the heavy changes the heavy's flight and not the
    fighter's, in every power mode and a hard turn."""
    power = torch.tensor([0.0, -1.0, 1.0, 0.0]).repeat(2)
    stick = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2)
    ship_type = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32)
    start = _ships(torch.full((1, 8), 80.0), 3000.0, ship_type=ship_type[None])

    def fly(types):
        airframe = flight.airframe_of(types, start.ship_type)
        ships = start
        for _ in range(15):
            ships = flight.flight_tick(FlightConstants(), airframe, 1 / 30, ships, stick[None],
                                       power[None], _no_fields(start.x))
        return torch.stack([ships.proper_speed, ships.energy, ships.angle_of_attack])

    default = ship_types()
    changed = ship_types(**{name: (getattr(default, name)[0], 1.5 * getattr(default, name)[1])})
    one, two = fly(default), fly(changed)
    assert torch.equal(one[..., :4], two[..., :4])
    assert (one[..., 4:] != two[..., 4:]).any()


def test_vision_range_is_the_observers():
    """A heavy with twice the vision sees enemies at 1.5 times the shared range, so its team
    does; with the shared range nobody does. Reset and every tick keep the range."""
    rules = api.rules_of(GameConfig(2, 1, 5000.0, mode="sandbox"))
    types = api.ship_types(vision_range_scale=(1.0, 2.0))
    state = api.reset(rules, types, torch.zeros(1, rules.tick_uniform_count))
    assert state.match.vision_range.tolist() == [[1024.0, 2048.0, 1024.0, 2048.0]]
    state, _ = api.step(rules, api.ship_types(), state,
                        Action.unpack(torch.zeros(1, 4, 3, dtype=torch.int32)),
                        torch.zeros(1, 4, 2), torch.zeros(1, rules.tick_uniform_count))
    assert state.match.vision_range.tolist() == [[1024.0] * 4]
    distance = 1.5 * rules.vision_range
    quiet = torch.zeros(1, 4, dtype=torch.bool)
    ships = replace(state.ships, x=torch.tensor([[0.0, 0.0, distance, distance + 50.0]]),
                    y=torch.tensor([[0.0, 30.0, 0.0, 0.0]]), fired=quiet, spawned=quiet)
    fields = replace(state.fields, x=state.fields.x + 1e6)  # no field blocks a sight line
    visible = SHIP["visible"]
    for vision, seen in (([1024.0, 2048.0] * 2, 1.0), ([1024.0] * 4, 0.0)):
        match = replace(state.match, vision_range=torch.tensor([vision]))
        _, obs, _ = api.perceive(rules, replace(state, ships=ships, fields=fields, match=match))
        assert obs.ships[0, 0, 2:, visible].tolist() == [seen, seen]  # team 0's enemy rows


def test_bullet_drag_and_spread_are_the_shooters():
    """Bullets of a type with doubled drag lose speed faster; spread scales per shooter."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = api.ship_types(bullet_drag_scale=(1.0, 2.0), bullet_spread_scale=(1.0, 3.0))
    ones = torch.ones(1, 2, rules.bullet_slots)
    bullets = Bullets(x=ones * 0.0, y=ones * 0.0, direction_x=ones, direction_y=ones * 0.0,
                      proper_speed=ones * 600.0, log_refractive_index=ones * 0.0,
                      log_index_gradient_x=ones * 0.0, log_index_gradient_y=ones * 0.0,
                      moves_left=ones.int())
    ship_type = torch.tensor([[0, 1]], dtype=torch.int32)
    moved = move_bullets(rules, types, bullets, _no_fields(ones), ship_type)
    lost = 600.0 - moved.proper_speed[0, :, 0]
    assert lost[1] > 1.9 * lost[0] > 0.0
    ships = replace(_ships(torch.full((1, 2), 100.0), 5000.0), ship_type=ship_type)
    spread = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    _, fired = fire(rules, types, ships, replace(bullets, moves_left=ones.int() * 0),
                    torch.ones(1, 2, dtype=torch.bool), spread)
    sideways = fired.direction_y[0, :, 0] * fired.proper_speed[0, :, 0]
    assert torch.allclose(sideways, torch.tensor([12.0, 36.0]))


def test_head_on_fraction_is_the_targets_armour():
    """A head-on hit does the target's head-on fraction of the shooter's damage."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = api.ship_types(head_on_fraction_scale=(1.0, 2.0))
    ships = replace(_ships(torch.zeros(1, 2), 0.0), x=torch.tensor([[0.0, 100.0]]),
                    heading_cos=torch.tensor([[1.0, -1.0]]),
                    ship_type=torch.tensor([[0, 1]], dtype=torch.int32))
    K = rules.bullet_slots
    zeros = torch.zeros(1, 2, K)
    live = torch.zeros(1, 2, K, dtype=torch.int32)
    live[0, :, 0] = 1
    # One bullet each way, through the other ship's nose.
    along = torch.tensor([1.0, -1.0])[None, :, None] + zeros
    bullets = Bullets(x=zeros, y=zeros, direction_x=along,
                      direction_y=zeros, proper_speed=zeros + 500.0, log_refractive_index=zeros,
                      log_index_gradient_x=zeros, log_index_gradient_y=zeros, moves_left=live)
    sweep = Sweep(ship_x=ships.x, ship_y=ships.y, ship_displacement_x=ships.x * 0.0,
                  ship_displacement_y=ships.y * 0.0,
                  bullet_x=torch.tensor([[0.0, 100.0]])[..., None] + zeros, bullet_y=zeros,
                  bullet_displacement_x=torch.tensor([[100.0, -100.0]])[..., None] + zeros,
                  bullet_displacement_y=zeros)
    damage = hits(rules, types, sweep, ships, bullets, torch.ones(1, 2, dtype=torch.bool)).damage
    assert torch.allclose(damage[0, 0, 1], torch.tensor(12.0 * 0.6))  # fighter hits the heavy
    assert torch.allclose(damage[0, 1, 0], torch.tensor(16.0 * 0.3))  # heavy hits the fighter


def test_shield_recharge_rate_and_delay_are_per_type():
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = api.ship_types(shield_recharge_rate_scale=(1.0, 2.0),
                           shield_recharge_delay_scale=(1.0, 0.5))
    ships = replace(_ships(torch.zeros(1, 2), 0.0), shield=torch.tensor([[50.0, 50.0]]),
                    ship_type=torch.tensor([[0, 1]], dtype=torch.int32))
    alive = torch.ones(1, 2, dtype=torch.bool)
    hurt = apply_damage(rules, types, ships, alive, torch.tensor([[1.0, 1.0]]))
    assert hurt.ships.shield_recharge_wait.tolist() == [[6.0, 3.0]]
    healed = apply_damage(rules, types, ships, alive, torch.zeros(1, 2))
    assert torch.allclose(healed.shield_restored, torch.tensor([[12.0, 24.0]]) * rules.tick_length)


def test_every_shared_value_has_its_scale():
    """SCALED names a shared value that exists, for every scale field of the card."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    scales = {f.name for f in dataclasses.fields(api.ShipTypes) if f.name.endswith("_scale")}
    assert scales == set(SCALED)
    for owner, shared in SCALED.values():
        assert isinstance(getattr(rules.flight if owner == "flight" else rules, shared), float)


# ------------------------------------------------------------------ tick rates

@pytest.mark.parametrize("rate", RATES)
def test_bullet_life_is_time_native(rate):
    rules = api.rules_of(GameConfig(1, 1, 1000.0), tick_length=1.0 / rate)
    assert math.isclose(rules.bullet_moves * rules.tick_length, 0.4)


def test_weapon_cadence_is_validated_per_rate():
    """15 Hz is 2 ticks at 30 Hz and 4 at 60 Hz, and refused at 25 Hz naming both rates; the
    game's reset refuses it too."""
    fire_rate = torch.tensor([15.0, 10.0])
    for rate, ticks in ((30, [2, 3]), (60, [4, 6])):
        assert ticks_between_shots(fire_rate, 1.0 / rate).tolist() == ticks
    with pytest.raises(ValueError, match="fire rate of 15 Hz is not a whole number of ticks at "
                                         "25 Hz"):
        ticks_between_shots(fire_rate, 1.0 / 25)
    rules = api.rules_of(GameConfig(1, 1, 1000.0), tick_length=1.0 / 25)
    with pytest.raises(ValueError, match="15 Hz.*25 Hz"):
        api.reset(rules, ship_types(), api.uniforms(rules, 1))


def test_a_fire_rate_is_read_as_a_small_fraction():
    """A rate stored as a float is read as the nearest fraction with a small denominator:
    30/7 Hz is 7 ticks at 30 Hz; 4.3 Hz is not whole."""
    assert ticks_between_shots(torch.tensor([30.0 / 7.0]), 1.0 / 30).tolist() == [7]
    with pytest.raises(ValueError, match="4.3 Hz"):
        ticks_between_shots(torch.tensor([4.3], dtype=torch.float64), 1.0 / 30)


@pytest.mark.parametrize("rate", RATES)
def test_the_time_between_shots_is_the_fire_rates_at_every_tick_rate(rate):
    """The game's fire shoots every 1 / fire rate seconds at 25, 30 and 60 Hz (5 and 2.5 Hz
    are whole at each), and so does the default card at 30 and 60 Hz."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0), tick_length=1.0 / rate)
    cards = [ship_types(fire_rate=(5.0, 2.5))] + ([ship_types()] if rate != 25 else [])
    for types in cards:
        seconds = seconds_between_shots(rules, types)[0]
        assert torch.allclose(seconds, 1.0 / types.fire_rate.double(), rtol=1e-9), seconds


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("exponent", [2.0, 3.0])
def test_energy_is_conserved_at_every_rate(rate, exponent):
    """With drag, engine and fire off, 1/2 w^2 + E stays put over 2 s at 25, 30 and 60 Hz,
    within the race test's bound (1e-4), for any bleed exponent."""
    generator = torch.Generator().manual_seed(rate)
    N = 256
    types = ship_types(zero_lift_drag=0.0, induced_drag_scale=0.0, engine_power=0.0,
                       engine_power_full_store=0.0, static_thrust=1e-6, bleed_exponent=exponent)
    speed = 60.0 + 100.0 * torch.rand(1, N, generator=generator)
    energy = 1000.0 + 8000.0 * torch.rand(1, N, generator=generator)
    ship_type = torch.randint(0, 2, (1, N), generator=generator, dtype=torch.int32)
    ships = _ships(speed, energy, ship_type=ship_type)
    stick = torch.tensor(STICK_LEVELS)[torch.randint(0, 7, (1, N), generator=generator)]
    power = torch.tensor(POWER_MODES)[torch.randint(0, 3, (1, N), generator=generator)]
    airframe = flight.airframe_of(types, ship_type)
    start = 0.5 * speed.double() ** 2 + energy
    for _ in range(2 * rate):
        ships = flight.flight_tick(FlightConstants(), airframe, 1.0 / rate, ships, stick, power,
                                   _no_fields(speed))
    end = 0.5 * ships.proper_speed.double() ** 2 + ships.energy
    keep = (ships.energy > 300) & (ships.energy < types.max_energy.min() - 300)
    assert keep.sum() > N // 4
    assert ((end - start).abs() / start)[keep].max() < 1e-4


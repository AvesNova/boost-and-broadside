"""The ship card in the game (ship-card spec 1, 2, 5): the bleed exponent, every per-type
value taking effect for its own type only, the time-native bullet life (each type's, from
its gun range) and weapon cadence, and energy conservation at each supported tick rate."""

import dataclasses
import math

import pytest
import torch
from ship_card_cases import seconds_between_shots

from bnb.contracts import ASPECTS, Action, GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.physics import flight
from bnb.game.physics.bullets import fire, move_bullets
from bnb.game.physics.combat import apply_damage, hits
from bnb.game.rules import HEAD_ON_ANGLE, POWER_MODES, STICK_LEVELS, FlightConstants
from bnb.game.ship_types import SCALED, ship_types, ticks_between_shots
from bnb.game.state import Bullets, Fields, Ships, Sweep, replace

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


RATES = (25, 30, 60)
BULLET_FLOATS = [field.name for field in dataclasses.fields(Bullets) if field.name != "moves_left"]


def _ships(speed, energy, angle=None, ship_type=None, dtype=torch.float32) -> Ships:
    speed = torch.as_tensor(speed, dtype=dtype)
    zeros = torch.zeros_like(speed)
    angle = zeros if angle is None else torch.as_tensor(angle, dtype=dtype)
    kind = torch.zeros(speed.shape, dtype=torch.int32) if ship_type is None else ship_type
    flags = torch.zeros(speed.shape, dtype=torch.bool)
    integers = torch.zeros_like(kind)
    return Ships(
        x=zeros,
        y=zeros,
        heading_cos=zeros + 1.0,
        heading_sin=zeros,
        proper_speed=speed,
        angle_of_attack=angle,
        angle_of_attack_cos=torch.cos(angle),
        angle_of_attack_sin=torch.sin(angle),
        nose_rate=zeros,
        energy=torch.as_tensor(energy, dtype=dtype).expand_as(speed).clone(),
        log_refractive_index=zeros,
        log_index_gradient_x=zeros,
        log_index_gradient_y=zeros,
        shield=zeros,
        shield_recharge_wait=zeros,
        alive=~flags,
        fired=flags,
        spawned=flags,
        respawn_wait=integers,
        gun_cooldown=integers,
        next_slot=integers,
        ship_type=kind,
    )


def _no_fields(like):
    ones = torch.ones((like.shape[0], 1), dtype=like.dtype)
    return Fields(
        x=ones * 1e9,
        y=ones * 0.0,
        radius=ones,
        band_width=ones,
        level=torch.ones((like.shape[0], 1), dtype=torch.int32),
    )


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

FLIGHT_SCALES = (
    "induced_drag_scale",
    "stall_angle_scale",
    "coast_exchange_scale",
    "dive_exchange_scale",
    "climb_exchange_scale",
    "bleed_exponent",
)


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
            ships = flight.flight_tick(
                FlightConstants(),
                airframe,
                1 / 30,
                ships,
                stick[None],
                power[None],
                _no_fields(start.x),
            )
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
    state, _ = api.step(
        rules,
        api.ship_types(),
        state,
        Action.unpack(torch.zeros(1, 4, 3, dtype=torch.int32)),
        torch.zeros(1, 4, 2),
        torch.zeros(1, rules.tick_uniform_count),
    )
    assert state.match.vision_range.tolist() == [[1024.0] * 4]
    distance = 1.5 * rules.vision_range
    quiet = torch.zeros(1, 4, dtype=torch.bool)
    ships = replace(
        state.ships,
        x=torch.tensor([[0.0, 0.0, distance, distance + 50.0]]),
        y=torch.tensor([[0.0, 30.0, 0.0, 0.0]]),
        fired=quiet,
        spawned=quiet,
    )
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
    bullets = Bullets(
        x=ones * 0.0,
        y=ones * 0.0,
        direction_x=ones,
        direction_y=ones * 0.0,
        proper_speed=ones * 600.0,
        log_refractive_index=ones * 0.0,
        log_index_gradient_x=ones * 0.0,
        log_index_gradient_y=ones * 0.0,
        moves_left=ones.int(),
    )
    ship_type = torch.tensor([[0, 1]], dtype=torch.int32)
    moved = move_bullets(rules, types, bullets, _no_fields(ones), ship_type)
    lost = 600.0 - moved.proper_speed[0, :, 0]
    assert lost[1] > 1.9 * lost[0] > 0.0
    ships = replace(_ships(torch.full((1, 2), 100.0), 5000.0), ship_type=ship_type)
    spread = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    _, fired = fire(
        rules,
        types,
        ships,
        replace(bullets, moves_left=ones.int() * 0),
        torch.ones(1, 2, dtype=torch.bool),
        spread,
    )
    sideways = fired.direction_y[0, :, 0] * fired.proper_speed[0, :, 0]
    assert torch.allclose(sideways, torch.tensor([12.0, 36.0]))


FIGHTER_DAMAGE, HEAVY_DAMAGE = api.default_card()["hit_damage"]  # a centre hit's


def _head_on_damage(types, nose_angles=(0.0, 0.0), moves_left: int = 1) -> torch.Tensor:
    """(B, N, N) damage of one bullet each way between two ships 100 m apart on the x axis,
    each through the other's nose turned by its nose angle (rad): the hit's aspect angle.
    Centre hits, all explosive share (flat in speed): the armour alone."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = dataclasses.replace(types, explosive_share=torch.ones_like(types.explosive_share))
    angles = torch.tensor([nose_angles])
    ships = replace(
        _ships(torch.zeros(1, 2), 0.0),
        x=torch.tensor([[0.0, 100.0]]),
        heading_cos=torch.cos(angles) * torch.tensor([[1.0, -1.0]]),
        heading_sin=torch.sin(angles),
        ship_type=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    K = rules.bullet_slots
    zeros = torch.zeros(1, 2, K)
    live = torch.zeros(1, 2, K, dtype=torch.int32)
    live[0, :, 0] = moves_left
    along = torch.tensor([1.0, -1.0])[None, :, None] + zeros
    bullets = Bullets(
        x=zeros,
        y=zeros,
        direction_x=along,
        direction_y=zeros,
        proper_speed=zeros + 500.0,
        log_refractive_index=zeros,
        log_index_gradient_x=zeros,
        log_index_gradient_y=zeros,
        moves_left=live,
    )
    sweep = Sweep(
        ship_x=ships.x,
        ship_y=ships.y,
        ship_displacement_x=ships.x * 0.0,
        ship_displacement_y=ships.y * 0.0,
        bullet_x=torch.tensor([[0.0, 100.0]])[..., None] + zeros,
        bullet_y=zeros,
        bullet_displacement_x=torch.tensor([[100.0, -100.0]])[..., None] + zeros,
        bullet_displacement_y=zeros,
    )
    return hits(rules, types, sweep, ships, bullets, torch.ones(1, 2, dtype=torch.bool)).damage


def test_head_on_fraction_is_the_targets_armour():
    """A head-on hit does the target's head-on fraction of the shooter's damage."""
    damage = _head_on_damage(api.ship_types(head_on_fraction_scale=(1.0, 2.0)))
    assert torch.allclose(
        damage[0, 0, 1], torch.tensor(FIGHTER_DAMAGE * 0.6)
    )  # fighter hits the heavy
    assert torch.allclose(
        damage[0, 1, 0], torch.tensor(HEAVY_DAMAGE * 0.3)
    )  # heavy hits the fighter


def test_head_on_angle_is_where_the_targets_armour_is_half():
    """At its head-on angle off the nose a target takes half way from its head-on fraction to
    full damage; the default angle (42.27 degrees) is the exp(-4 theta^2 / pi) curve's."""
    assert math.degrees(HEAD_ON_ANGLE) == pytest.approx(42.27, abs=0.01)
    types = api.ship_types(head_on_angle_scale=(1.0, 0.5))
    half = 1.0 - 0.7 * 0.5
    damage = _head_on_damage(types, (HEAD_ON_ANGLE, 0.5 * HEAD_ON_ANGLE))
    assert torch.allclose(
        damage[0, 0, 1], torch.tensor(FIGHTER_DAMAGE * half)
    )  # into the heavy's 21 deg
    assert torch.allclose(
        damage[0, 1, 0], torch.tensor(HEAVY_DAMAGE * half)
    )  # the fighter's 42 deg
    theta = 0.6
    old = 1.0 - 0.7 * math.exp(-4.0 * theta * theta / math.pi)
    damage = _head_on_damage(api.ship_types(), (theta, theta))
    assert torch.allclose(damage[0, 1, 0], torch.tensor(HEAVY_DAMAGE * old))


def test_head_on_sharpness_is_the_exponent_of_the_targets_cone():
    """The cone is exp(-ln2 (theta / theta_h)^p): half strength at theta_h whatever p, a
    higher p keeps the armour nearer full inside theta_h and drops it faster outside."""
    p = 6.0
    types = api.ship_types(head_on_sharpness=(p, p))
    half = 1.0 - 0.7 * 0.5
    damage = _head_on_damage(types, (HEAD_ON_ANGLE, HEAD_ON_ANGLE))
    assert torch.allclose(damage[0, 1, 0], torch.tensor(HEAVY_DAMAGE * half))
    for theta in (0.5 * HEAD_ON_ANGLE, 1.5 * HEAD_ON_ANGLE):
        aspect = 1.0 - 0.7 * math.exp(-math.log(2.0) * (theta / HEAD_ON_ANGLE) ** p)
        damage = _head_on_damage(types, (theta, theta))
        assert torch.allclose(damage[0, 1, 0], torch.tensor(HEAVY_DAMAGE * aspect))


def test_arming_distance_ramps_the_shooters_damage_over_the_first_metres():
    """A bullet's damage is a smoothstep of the distance it has travelled over the shooter's
    arming distance X (its moves so far, this one included, at its speed): full past X, and
    none of the other ship's damage changes."""
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    life = round(api.default_card()["bullet_life"][0] / rules.tick_length)
    step = 500.0 * rules.tick_length  # the test bullets fly at 500 m/s
    X = 4.0 * step
    types = api.ship_types(arming_distance=(X, 0.0))
    full = _head_on_damage(api.ship_types(), moves_left=life)
    for moves, s in ((1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0), (6, 1.0)):
        damage = _head_on_damage(types, moves_left=life - moves + 1)
        ramp = s * s * (3.0 - 2.0 * s)
        assert torch.allclose(damage[0, 0, 1], full[0, 0, 1] * ramp)
        assert torch.allclose(damage[0, 1, 0], full[0, 1, 0])


def _aspect_hits(nose_angles, ships_per_team=1):
    """Hits of one bullet each way between ships 0 and 1, 100 m apart on the x axis, each
    through the other's nose turned by its nose angle (rad); with two ships a team they
    are teammates (the others wait far off, unarmed)."""
    rules = api.rules_of(GameConfig(ships_per_team, 1, 1000.0))
    N, K = 2 * ships_per_team, rules.bullet_slots
    angles = torch.zeros(1, N)
    angles[0, :2] = torch.tensor(nose_angles)
    facing = torch.ones(1, N)
    facing[0, 1] = -1.0
    ships = replace(
        _ships(torch.zeros(1, N), 0.0),
        x=torch.tensor([[0.0, 100.0] + [0.0] * (N - 2)]),
        y=torch.tensor([[0.0, 0.0] + [5000.0] * (N - 2)]),
        heading_cos=torch.cos(angles) * facing,
        heading_sin=torch.sin(angles),
    )
    zeros = torch.zeros(1, N, K)
    live = torch.zeros(1, N, K, dtype=torch.int32)
    live[0, :2, 0] = 1
    along = zeros.clone()
    along[0, :2, 0] = torch.tensor([1.0, -1.0])
    start = zeros.clone()
    start[0, 1, 0] = 100.0
    bullets = Bullets(
        x=start,
        y=zeros,
        direction_x=along,
        direction_y=zeros,
        proper_speed=zeros + 500.0,
        log_refractive_index=zeros,
        log_index_gradient_x=zeros,
        log_index_gradient_y=zeros,
        moves_left=live,
    )
    sweep = Sweep(
        ship_x=ships.x,
        ship_y=ships.y,
        ship_displacement_x=ships.x * 0.0,
        ship_displacement_y=ships.y * 0.0,
        bullet_x=start,
        bullet_y=zeros,
        bullet_displacement_x=along * 100.0,
        bullet_displacement_y=zeros,
    )
    alive = torch.ones(1, N, dtype=torch.bool)
    return hits(rules, api.ship_types(), sweep, ships, bullets, alive)


@pytest.mark.parametrize(
    ("angle", "aspect"),
    [
        (0.0, 0),
        (0.17, 0),
        (0.18, 1),
        (0.7, 1),
        (0.8, 2),
        (math.pi / 2, 2),
        (2.3, 2),
        (2.4, 3),
        (math.pi, 3),
    ],
)
def test_a_hits_aspect_class_is_its_angle_off_the_targets_nose(angle, aspect):
    """A mutual head-on under 10 degrees off the nose, an offset head-on to 45, the flank to
    135, the tail beyond (ASPECTS);
    the enemy damage by class is the hit's damage, and the earliest hit's class is kept."""
    hit = _aspect_hits((0.0, angle))
    expected = torch.zeros(len(ASPECTS))
    expected[aspect] = hit.damage[0, 0, 1]
    assert hit.damage[0, 0, 1] > 0.0
    torch.testing.assert_close(hit.enemy_damage_by_aspect[0, 1], expected)
    assert int(hit.first_aspect[0, 1]) == aspect
    assert int(hit.first_aspect[0, 0]) == 0  # ship 0 is hit nose on


def test_the_aspect_damage_leaves_out_friendly_fire():
    """A teammate's hit is damage, but not enemy damage; its class is still the first hit's."""
    hit = _aspect_hits((0.0, math.pi), ships_per_team=2)
    assert hit.damage[0, 0, 1] > 0.0
    assert float(hit.enemy_damage_by_aspect.abs().sum()) == 0.0
    assert int(hit.first_aspect[0, 1]) == 3
    assert (hit.first_aspect[0, 2:] == -1).all()


def test_shield_recharge_rate_and_delay_are_per_type():
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = api.ship_types(
        shield_recharge_rate_scale=(1.0, 2.0), shield_recharge_delay_scale=(1.0, 0.5)
    )
    ships = replace(
        _ships(torch.zeros(1, 2), 0.0),
        shield=torch.tensor([[50.0, 50.0]]),
        ship_type=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    alive = torch.ones(1, 2, dtype=torch.bool)
    hurt = apply_damage(rules, types, ships, alive, torch.tensor([[1.0, 1.0]]), alive)
    assert hurt.ships.shield_recharge_wait.tolist() == [[6.0, 3.0]]
    healed = apply_damage(rules, types, ships, alive, torch.zeros(1, 2), alive)
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
def test_bullet_life_is_the_shooters_and_time_native(rate):
    """A bullet gets its shooter's type's life, in whole moves at the tick; the rules' longest
    and the bullet slots follow the card."""
    types = api.ship_types(bullet_life=(0.4, 0.6))
    rules = api.rules_of(GameConfig(1, 1, 1000.0), types=types, tick_length=1.0 / rate)
    assert rules.longest_bullet_moves == round(0.6 * rate)
    ship_type = torch.tensor([[0, 1]], dtype=torch.int32)
    ships = replace(_ships(torch.full((1, 2), 100.0), 5000.0), ship_type=ship_type)
    zeros = torch.zeros(1, 2, rules.bullet_slots)
    bullets = Bullets(**{name: zeros for name in BULLET_FLOATS}, moves_left=zeros.int())
    _, fired = fire(
        rules, types, ships, bullets, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 2)
    )
    assert fired.moves_left[0, :, 0].tolist() == [round(0.4 * rate), round(0.6 * rate)]


def test_the_bullet_slots_follow_each_types_life_and_fire_rate():
    """A type's slots are its life over its ticks between shots; a ship gets the most."""
    assert api.bullet_slots_for((10.0, 15.0), (0.6, 0.4), 1.0 / 30.0) == 6  # 18 / 3
    assert api.bullet_slots_for((10.0, 15.0), (0.3, 0.8), 1.0 / 30.0) == 12  # 24 / 2


def test_the_gun_range_sets_the_bullet_life():
    """The life is the whole ticks at or above the time the type's own drag takes to cover
    the gun range: the standard card's life is 0.4 s at the rules' drag; a slower muzzle or
    a higher drag flies longer; the flown range is the handle's to within one tick of
    travel; a range past the longest life is infeasible."""
    from bnb.game.handling import Infeasible
    from bnb.game.handling.weapons import bullet_life, gun_range

    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    launch = torch.tensor([603.7, 300.0, 603.7], dtype=torch.float64)
    drag = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64) * rules.bullet_drag
    life = torch.tensor(0.4, dtype=torch.float64)
    distance = gun_range(drag[0], launch[0], life).expand(3)  # the standard fighter's range
    lives = bullet_life(rules, distance, launch, drag)
    assert lives[0].item() == pytest.approx(0.4)
    for kind in (1, 2):
        k, w = drag[kind].item(), launch[kind].item()
        slow = math.expm1(k * distance[kind].item()) / (k * w)
        assert lives[kind].item() == pytest.approx(math.ceil(slow * 30.0) / 30.0)
    flown = gun_range(drag, launch, lives)
    assert (flown >= distance * (1.0 - 1e-9)).all()
    assert (flown - distance <= launch * rules.tick_length).all()
    with pytest.raises(Infeasible, match="beyond the longest"):
        bullet_life(rules, distance * 10.0, launch, drag)


def test_weapon_cadence_is_validated_per_rate():
    """15 Hz is 2 ticks at 30 Hz and 4 at 60 Hz, and refused at 25 Hz naming both rates; the
    game's reset refuses it too."""
    fire_rate = torch.tensor([15.0, 10.0])
    for rate, ticks in ((30, [2, 3]), (60, [4, 6])):
        assert ticks_between_shots(fire_rate, 1.0 / rate).tolist() == ticks
    with pytest.raises(
        ValueError, match="fire rate of 15 Hz is not a whole number of ticks at 25 Hz"
    ):
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
    types = ship_types(
        zero_lift_drag=0.0,
        induced_drag_scale=0.0,
        engine_power=0.0,
        engine_power_full_store=0.0,
        static_thrust=1e-6,
        bleed_exponent=exponent,
    )
    speed = 60.0 + 100.0 * torch.rand(1, N, generator=generator)
    energy = 1000.0 + 8000.0 * torch.rand(1, N, generator=generator)
    ship_type = torch.randint(0, 2, (1, N), generator=generator, dtype=torch.int32)
    ships = _ships(speed, energy, ship_type=ship_type)
    stick = torch.tensor(STICK_LEVELS)[torch.randint(0, 7, (1, N), generator=generator)]
    power = torch.tensor(POWER_MODES)[torch.randint(0, 3, (1, N), generator=generator)]
    airframe = flight.airframe_of(types, ship_type)
    start = 0.5 * speed.double() ** 2 + energy
    for _ in range(2 * rate):
        ships = flight.flight_tick(
            FlightConstants(), airframe, 1.0 / rate, ships, stick, power, _no_fields(speed)
        )
    end = 0.5 * ships.proper_speed.double() ** 2 + ships.energy
    keep = (ships.energy > 300) & (ships.energy < types.max_energy.min() - 300)
    assert keep.sum() > N // 4
    assert ((end - start).abs() / start)[keep].max() < 1e-4


def _induced_drag_balance(rate: int, exponent: float) -> torch.Tensor:
    """With the zero-lift drag, the engine and fire off but the induced drag on, the
    relative mismatch per ship between the change of 1/2 w^2 + E/m over 2 s and the induced
    drag's work, k (C^2 + (z/u^2)^2)^(e/2) u^3, written out here and integrated by the
    trapezoid rule over each tick. The induced drag is scaled by C_max^(2 - e) so that it
    is as strong at any exponent."""
    generator = torch.Generator().manual_seed(rate)
    N = 256
    constants = FlightConstants()
    lift = ship_types().max_lift_coefficient
    types = ship_types(
        zero_lift_drag=0.0,
        engine_power=0.0,
        engine_power_full_store=0.0,
        static_thrust=1e-6,
        bleed_exponent=exponent,
        induced_drag_scale=lift ** (2.0 - exponent),
    )
    speed = 60.0 + 100.0 * torch.rand(1, N, generator=generator)
    energy = 3000.0 + 4000.0 * torch.rand(1, N, generator=generator)
    ship_type = torch.randint(0, 2, (1, N), generator=generator, dtype=torch.int32)
    ships = _ships(speed, energy, ship_type=ship_type)
    stick = torch.tensor(STICK_LEVELS)[torch.randint(0, 7, (1, N), generator=generator)]
    power = torch.tensor(POWER_MODES)[torch.randint(0, 3, (1, N), generator=generator)]
    airframe = flight.airframe_of(types, ship_type)
    factor = constants.induced_drag_factor * airframe.induced_drag_scale.double()

    def total(ship):
        return 0.5 * ship.proper_speed.double() ** 2 + ship.energy.double() / constants.mass

    def induced_power(ship):  # W/kg
        u = ship.proper_speed.double()
        current = flight.setting(constants, airframe, ship.proper_speed, ship.energy, stick, power)
        coefficient = flight.lift_coefficient(constants, airframe, ship.angle_of_attack).double()
        lifting = coefficient**2 + (current.vertical_lift.double() / u**2) ** 2
        return factor * lifting ** (0.5 * exponent) * u**3

    work, start = torch.zeros_like(speed, dtype=torch.float64), total(ships)
    for _ in range(2 * rate):
        after = flight.flight_tick(
            constants, airframe, 1.0 / rate, ships, stick, power, _no_fields(speed)
        )
        work = work + 0.5 / rate * (induced_power(ships) + induced_power(after))
        ships = after
    keep = (ships.energy > 300) & (ships.energy < types.max_energy.min() - 300)
    assert keep.sum() > N // 4
    return ((total(ships) - start + work).abs() / work)[keep]


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("exponent", [1.5, 3.0])
def test_energy_lost_is_the_induced_drags_work(rate, exponent):
    """With the induced drag on, 1/2 w^2 + E/m falls by the bleed's work, within 1%: the
    trapezoid rule and the float32 state leave at most 0.35% (measured), and a bleed 2% off
    fails."""
    assert _induced_drag_balance(rate, exponent).max() < 1e-2


def test_the_balance_sees_the_bleed_exponent(monkeypatch):
    """A bleed that ignores its exponent breaks the balance above (the test can fail)."""
    monkeypatch.setattr(flight, "bleed", lambda airframe, lift_squared: lift_squared)
    assert _induced_drag_balance(30, 3.0).min() > 0.1


def test_yemong_sizes_its_coding_ranges_from_the_card():
    """The state code's ranges are the card's largest per-type values: twice the widest
    stall angle, the longest recharge delay, and one more bin than the most ticks between
    shots."""
    from bnb.models.yemong.state_code import ShipStateCode, state_code_ranges

    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    types = ship_types(
        stall_angle_scale=(1.0, 1.5), shield_recharge_delay_scale=(2.0, 1.0), fire_rate=(15.0, 5.0)
    )
    code = ShipStateCode(state_code_ranges(rules, types))
    assert math.isclose(code.angle_of_attack_limit, 3.0 * rules.flight.stall_angle)
    assert math.isclose(code.shield_delay, 2.0 * rules.shield_recharge_delay)
    assert code.cooldown_bins == 1 + 6

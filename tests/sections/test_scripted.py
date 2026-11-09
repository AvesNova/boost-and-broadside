"""The scripted agent (scripted spec A7): behaviour on constructed observations, invariance,
its softness (the floor and the temperature), and the controller in the rollout.

Every binary decision is floored at its own epsilon (`EPSILONS`), so "certain" below means
within the floors: a single decision at most 1 - epsilon, a turn at a given level and side
at most about (1 - epsilon)^2 (EPSILON below is the largest)."""

import dataclasses
import inspect
import math
import typing

import pytest
import torch
from scripted_cases import ARENA_RADIUS, CHARACTERISTICS, absent, observation, ship, value, zone
from test_perception import _swap_teams

from bnb.contracts import GameConfig, Obs, Pool, Privileged
from bnb.contracts.action import NUM_JOINT, to_joint
from bnb.contracts.obs import SHIP, SHIP_HIDDEN_ZERO, ZONE
from bnb.controllers import scripted_combat
from bnb.controllers.scripted import Scripted, distribution, marginals
from bnb.controllers.scripted_memory import ScriptedCarry
from bnb.controllers.scripted_parameters import (
    EPSILONS,
    NAMES,
    SCALE_FLOORS,
    Logistic,
    ScriptedParameters,
    from_row,
    logistic,
    pack,
    table,
    unpack,
)
from bnb.game import api
from bnb.game.handling.characteristics import time_of_flight
from bnb.runtime.rollout import Rollout

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


PARAMETERS = ScriptedParameters()
TYPES = (0, 1)
EPSILON = max(EPSILONS.values())
SURE = 1.0 - EPSILON - 0.02  # one decision near its ceiling 1 - EPSILON
SURE_TWICE = (1.0 - EPSILON) ** 2 - 0.03  # two decisions near theirs (a side and a turn)
UNLIKELY = EPSILON + 0.02  # one decision near its floor


def _beyond(shape: Logistic, scales: float) -> float:
    """The input `scales` scales past the logistic's centre, on its high-probability side
    (negative: on its low side)."""
    direction = 1.0 if shape.rising else -1.0
    return shape.centre * math.exp(direction * scales * shape.scale)


def _marginals(obs: Obs, row: int = 0):
    parts = marginals(obs, CHARACTERISTICS, PARAMETERS)
    return tuple(part[0, row] for part in parts)


def _duel(ship_type: int, target: torch.Tensor, own: torch.Tensor | None = None) -> Obs:
    """Own ship 0 (at the origin, nose along +x) against one enemy row; the second own
    and enemy slots are absent."""
    own = ship(ship_type=ship_type) if own is None else own
    return observation([own, absent(team_relative=0)], [target, absent()])


def _ahead(ship_type: int, angle: float, distance_in_gun_range: float, **kwargs) -> torch.Tensor:
    """An enemy at this bearing (positive left) and distance, flying with the own ship's
    velocity, so the lead leaves the aim point on it."""
    distance = distance_in_gun_range * value("gun_range", ship_type)
    cruise = value("cruise_speed", ship_type)
    return ship(
        x=distance * math.cos(angle),
        y=distance * math.sin(angle),
        team_relative=1,
        velocity=(cruise, 0.0),
        **kwargs,
    )


def _levels(stick: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mean stick level, P(left), P(right))."""
    level = torch.tensor([0.0, 1, 1, 2, 2, 3, 3])
    return (stick * level).sum(), stick[2::2].sum(), stick[1::2].sum()


# ------------------------------------------------------------------ the stick


@pytest.mark.parametrize("ship_type", TYPES)
def test_the_stick_level_rises_with_the_angle_error_and_the_side_follows_its_sign(ship_type):
    previous = -1.0
    for degrees in (0.0, 1.0, 4.0, 10.0, 25.0, 60.0, 120.0):
        for sign in (1.0, -1.0):
            target = _ahead(ship_type, sign * math.radians(degrees), 0.75)
            _, stick, _ = _marginals(_duel(ship_type, target))
            level, left, right = _levels(stick)
            if degrees > 0.0:
                turning = 1.0 - stick[0]
                assert (left if sign > 0 else right) >= (1.0 - EPSILONS["side"] - 1e-5) * turning
                assert (right if sign > 0 else left) <= (EPSILONS["side"] + 1e-5) * turning
        assert level >= previous - 1e-6, degrees
        previous = float(level)
    ceiling = (1.0 - EPSILON) * (1.0 + (1.0 - EPSILON) + (1.0 - EPSILON) ** 2)
    assert previous > ceiling - 0.1  # far off the nose: the full stick, within the floors


@pytest.mark.parametrize("ship_type", TYPES)
def test_dead_and_waiting_ships_take_the_no_op(ship_type):
    dead = ship(ship_type=ship_type, alive=False)
    q = distribution(
        _duel(ship_type, _ahead(ship_type, 0.3, 0.5), own=dead), CHARACTERISTICS, PARAMETERS
    )
    assert q[0, 0, 0] == 1.0 and q[0, 1, 0] == 1.0


# ------------------------------------------------------------------ the gun


@pytest.mark.parametrize("ship_type", TYPES)
def test_it_fires_only_when_aligned_in_range_ready_and_paid_for(ship_type):
    cost = value("bullet_energy_cost", ship_type)
    close = 0.5 * PARAMETERS.shoot_in_gun_range.centre
    likely = _marginals(_duel(ship_type, _ahead(ship_type, 0.0, close)))[2][1]
    assert likely > SURE
    far = _beyond(PARAMETERS.shoot_in_gun_range, -5.0)
    cases = dict(
        unaligned=_duel(ship_type, _ahead(ship_type, math.radians(30.0), close)),
        out_of_range=_duel(ship_type, _ahead(ship_type, 0.0, far)),
        cooling=_duel(
            ship_type,
            _ahead(ship_type, 0.0, close),
            own=ship(ship_type=ship_type, gun_cooldown=RULES_TICK),
        ),
        short_of_energy=_duel(
            ship_type,
            _ahead(ship_type, 0.0, close),
            own=ship(ship_type=ship_type, energy=0.9 * cost),
        ),
    )
    for name, obs in cases.items():
        assert _marginals(obs)[2][1] < 0.02, name
        if name in ("cooling", "short_of_energy"):  # the gate: exactly 0
            assert _marginals(obs)[2][1] == 0.0, name


RULES_TICK = 1.0 / 30.0


# ------------------------------------------------------------------ power


@pytest.mark.parametrize("ship_type", TYPES)
def test_it_boosts_when_slow_or_far_with_a_full_store_and_reverses_when_close(ship_type):
    corner = value("corner_speed", ship_type)
    middle = 2.0 * PARAMETERS.reverse_in_gun_range.centre  # neither reversing nor far
    slow = _duel(
        ship_type,
        _ahead(ship_type, 0.0, middle),
        own=ship(
            ship_type=ship_type,
            speed=_beyond(PARAMETERS.boost_slow_in_corner_speed, 5.0) * corner,
        ),
    )
    assert _marginals(slow)[0][1] > SURE_TWICE  # boost and not reverse
    distance = _beyond(PARAMETERS.boost_far_in_store_gun_ranges, 5.0)  # with a full store
    far = _duel(ship_type, _ahead(ship_type, 0.0, distance))
    assert _marginals(far)[0][1] > SURE_TWICE
    empty = _duel(
        ship_type,
        _ahead(ship_type, 0.0, distance),
        own=ship(ship_type=ship_type, energy=1e-3 * value("max_energy", ship_type)),
    )
    assert _marginals(empty)[0][1] < UNLIKELY
    close = _duel(ship_type, _ahead(ship_type, 0.0, _beyond(PARAMETERS.reverse_in_gun_range, 5.0)))
    assert _marginals(close)[0][2] > SURE


# ------------------------------------------------------------------ the lead


@pytest.mark.parametrize("ship_type", TYPES)
@pytest.mark.parametrize("shooter_velocity", [(0.0, 0.0), (80.0, -40.0)])
def test_the_lead_moves_the_aim_point_along_a_crossing_target_by_its_time_of_flight(
    ship_type, shooter_velocity
):
    """The aim point is the displacement at the time of flight under the relative velocity:
    the bullet inherits the shooter's velocity, so a target keeping pace along x stays on
    the line and only the crossing speed relative to the shooter leads."""
    distance, crossing = 0.8 * value("gun_range", ship_type), 60.0
    target_velocity = (shooter_velocity[0], crossing)
    obs = _duel(
        ship_type,
        ship(x=distance, team_relative=1, velocity=target_velocity),
        own=ship(ship_type=ship_type, velocity=shooter_velocity),
    )
    fleet = scripted_combat.context(obs, CHARACTERISTICS, PARAMETERS)
    one = torch.ones(1, 2)
    aim_x, aim_y = scripted_combat.lead(
        fleet,
        CHARACTERISTICS,
        PARAMETERS,
        distance * one,
        0.0 * one,
        target_velocity[0] * one,
        crossing * one,
    )
    flight = time_of_flight(CHARACTERISTICS, fleet.own_type, torch.tensor(distance))
    torch.testing.assert_close(aim_x[0, 0], torch.tensor(distance))
    torch.testing.assert_close(aim_y[0, 0], (crossing - shooter_velocity[1]) * flight[0, 0])
    _, stick, _ = _marginals(obs)
    assert _levels(stick)[1] > 0.5  # it turns toward the lead, to the left


# ------------------------------------------------------------------ the boundary


@pytest.mark.parametrize("ship_type", TYPES)
def test_near_the_boundary_and_heading_out_it_turns_inward(ship_type):
    edge = ARENA_RADIUS - 20.0
    target = ship(
        x=edge + 0.5 * value("gun_range", ship_type),
        team_relative=1,
        velocity=(value("cruise_speed", ship_type), 0.0),
    )
    outward = _marginals(_duel(ship_type, target, own=ship(x=edge, ship_type=ship_type)))[1]
    assert outward[0] < UNLIKELY  # it turns
    target = ship(
        x=0.5 * value("gun_range", ship_type),
        team_relative=1,
        velocity=(value("cruise_speed", ship_type), 0.0),
    )
    centred = _marginals(_duel(ship_type, target))[1]
    assert centred[0] > SURE  # the same chase at the centre flies straight


@pytest.mark.parametrize("ship_type", TYPES)
def test_near_the_boundary_and_heading_inward_it_does_not_turn_back(ship_type):
    """Deep in the margin but flying inward (at 120 degrees, the centre 60 degrees to its
    left), the boundary leaves the chase of a target dead ahead alone."""
    edge, heading = ARENA_RADIUS - 20.0, 2.0 * math.pi / 3.0
    cruise, ahead = value("cruise_speed", ship_type), 0.5 * value("gun_range", ship_type)
    target = ship(
        x=edge + ahead * math.cos(heading),
        y=ahead * math.sin(heading),
        team_relative=1,
        velocity=(cruise * math.cos(heading), cruise * math.sin(heading)),
    )
    own = ship(x=edge, heading=heading, ship_type=ship_type)
    assert _marginals(_duel(ship_type, target, own=own))[1][0] > SURE


@pytest.mark.parametrize("ship_type", TYPES)
def test_with_no_visible_enemy_it_heads_for_the_centre_and_holds_fire(ship_type):
    away = ship(x=1500.0, heading=math.pi / 2, ship_type=ship_type)  # the centre is on its left
    for enemy in (absent(), ship(x=1700.0, y=200.0, team_relative=1, visible=False)):
        power, stick, shoot = _marginals(_duel(ship_type, enemy, own=away))
        assert _levels(stick)[1] > SURE_TWICE
        assert shoot[1] == pytest.approx(EPSILONS["shoot"])
        assert power[2] == pytest.approx(EPSILONS["reverse"])
    near = ship(x=100.0, heading=math.pi / 2, ship_type=ship_type)
    holding = _marginals(_duel(ship_type, absent(), own=near))[1]
    assert holding[0] == pytest.approx(
        1.0 - EPSILONS["turn"], abs=1e-3
    )  # holds its course, within the floor


# ------------------------------------------------------------------ Frontline


def _ring(direction: float = 0.0, progress: float = 0.0) -> list[torch.Tensor]:
    """Own spawn below, own defense on the left, enemy defense on the right."""
    return [
        zone(0.0, -1600.0, 0),
        zone(-1600.0, 0.0, 1, progress, direction),
        zone(0.0, 1600.0, 2),
        zone(1600.0, 0.0, 3),
        zone(0.0, 2600.0, 4),
    ]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_it_defends_a_zone_whose_meter_moves_against_it(ship_type):
    own = [
        ship(heading=math.pi / 2, ship_type=ship_type),
        ship(x=60.0, heading=math.pi / 2, ship_type=ship_type),
    ]
    raiders = [ship(x=-1600.0, y=50.0 * i, team_relative=1) for i in range(2)]
    pressed = observation(own, raiders, _ring(direction=-1.0, progress=0.4))
    for row in (0, 1):  # both turn left, toward the defense zone
        assert _levels(_marginals(pressed, row)[1])[1] > SURE_TWICE
    calm = observation(own, [absent(), absent()], _ring())
    sides = [_levels(_marginals(calm, row)[1])[1:] for row in (0, 1)]
    assert sides[0][0] > SURE_TWICE and sides[1][1] > SURE_TWICE  # one defends, one attacks


def _defending(obs: Obs, parameters: ScriptedParameters = PARAMETERS) -> list[bool]:
    """Per own ship flying north, whether it turns left, toward the own defense zone on the
    left of `_ring`, rather than right, toward the attack zone."""
    stick = marginals(obs, CHARACTERISTICS, parameters).stick[0]
    left, right = stick[:, 2::2].sum(-1), stick[:, 1::2].sum(-1)
    assert bool(((left > SURE_TWICE) | (right > SURE_TWICE)).all()), stick
    return (left > SURE_TWICE).tolist()


def _line(xs: list[float], ship_type: int, y: float = 0.0) -> list[torch.Tensor]:
    return [ship(x=x, y=y, heading=math.pi / 2, ship_type=ship_type) for x in xs]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_a_meter_moving_against_it_calls_the_attacker_back(ship_type):
    """No enemy in sight: with the defense meter still, or moving the viewer's way, the ship
    nearer the defense zone defends and the other attacks; moving the enemy's way, both
    defend (the urgency)."""
    own = _line([-100.0, 100.0], ship_type)
    nobody = [absent(), absent()]
    for direction, progress in ((0.0, 0.0), (1.0, 0.5)):
        assert _defending(observation(own, nobody, _ring(direction, progress))) == [True, False]
    assert _defending(observation(own, nobody, _ring(-1.0, 0.0))) == [True, True]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_the_urgency_grows_with_the_enemys_progress(ship_type):
    """The defense wants floor + urgency (1 + progress) while the meter moves against the
    viewer: with floor 0.5, no margin and urgency 1, four healthy ships send two to defend
    at progress 0 (want 1.5) and three at progress 0.9 (want 2.4)."""
    parameters = dataclasses.replace(
        PARAMETERS, defensive_floor=0.5, defensive_margin=0.0, defense_urgency=1.0
    )
    own = _line([-300.0, -100.0, 100.0, 300.0], ship_type)
    nobody = [absent()] * 4
    early = observation(own, nobody, _ring(-1.0, 0.0))
    late = observation(own, nobody, _ring(-1.0, 0.9))
    assert _defending(early, parameters) == [True, True, False, False]
    assert _defending(late, parameters) == [True, True, True, False]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_equally_placed_ships_split_by_row(ship_type):
    """Two ships exactly as far from both zones: the lower row defends, the other attacks,
    whichever ship holds row 0."""
    north, south = _line([0.0], ship_type, 300.0)[0], _line([0.0], ship_type, -300.0)[0]
    for own in ([north, south], [south, north]):
        assert _defending(observation(own, [absent(), absent()], _ring())) == [True, False]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_hands_over_to_combat_from_the_combat_radius_in(ship_type):
    """A lone defender (the defense zone on its left) and an enemy on its right: beyond the
    combat radius it flies the strategy, inside the shoot ramp's start it fights, and half
    way between it mixes the two."""
    gun_range, cruise = value("gun_range", ship_type), value("cruise_speed", ship_type)
    inner = PARAMETERS.shoot_in_gun_range.centre
    outer = PARAMETERS.combat_radius_in_gun_range
    own = [ship(heading=math.pi / 2, ship_type=ship_type), absent(team_relative=0)]

    def sides(distance_in_gun_range: float) -> tuple[float, float]:
        enemy = ship(x=distance_in_gun_range * gun_range, team_relative=1, velocity=(0.0, cruise))
        _, left, right = _levels(_marginals(observation(own, [enemy, absent()], _ring()))[1])
        return float(left), float(right)

    assert sides(1.2 * outer)[0] > SURE_TWICE  # the strategy: to the defense zone
    assert sides(0.8 * inner)[1] > SURE_TWICE  # combat: at the enemy
    left, right = sides(0.5 * (inner + outer))
    assert 0.25 < left < 0.75 and 0.25 < right < 0.75


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_it_falls_back_to_its_spawn_when_hurt_and_threatened(ship_type):
    gun_range = value("gun_range", ship_type)
    threats = [
        ship(x=1.5 * gun_range, y=(0.5 + 0.2 * i) * gun_range, team_relative=1) for i in range(2)
    ]
    hurt = ship(
        ship_type=ship_type,
        shield=0.5 * PARAMETERS.recovery_shield_fraction * value("max_shield", ship_type),
    )
    healthy = ship(ship_type=ship_type)
    for own, side in ((hurt, 2), (healthy, 1)):  # spawn on the right, the threats on the left
        obs = observation([own, absent(team_relative=0)], threats, _ring())
        assert _levels(_marginals(obs)[1])[side] > SURE_TWICE


# ------------------------------------------------------------------ invariance


def _game_views(mode: str, seed: int = 3):
    """A few ticks of a fogged 3v3 game, scripted on both sides: the state and its views."""
    config = GameConfig(3, 4, 2600.0, mode=mode)
    rules, types = api.rules_of(config), api.ship_types()
    generator = torch.Generator().manual_seed(seed)
    B = 32
    state = api.reset(
        rules, types, api.uniforms(rules, B, generator=generator), torch.ones(B, dtype=torch.bool)
    )
    agent = Scripted(CHARACTERISTICS, config.ships_per_team, PARAMETERS)
    carries = [agent.init_carry(B, torch.device("cpu")) for _ in range(2)]
    torch.manual_seed(seed)
    for _ in range(90):
        state, obs, _ = api.perceive(rules, state)
        actions = []
        for team in (0, 1):
            action, carries[team], _ = agent.step(carries[team], _view(obs, team), None)
            actions.append(action)
        action = type(actions[0])(
            *(torch.cat(parts, 1) for parts in zip(*[(a.power, a.stick, a.shoot) for a in actions]))
        )
        state, _ = api.step(
            rules,
            types,
            state,
            action,
            api.spread(rules, B, generator=generator),
            api.uniforms(rules, B, generator=generator),
        )
    state, obs, _ = api.perceive(rules, state)
    return rules, state, obs


def _view(obs: Obs, team: int) -> Obs:
    return Obs(
        ships=obs.ships[:, team],
        fields=None,
        globals=obs.globals[:, team],
        zones=obs.zones[:, team],
    )


def _q(obs: Obs) -> torch.Tensor:
    return distribution(obs, CHARACTERISTICS, PARAMETERS)


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_the_distribution_does_not_depend_on_the_seat(mode):
    rules, state, obs = _game_views(mode)
    _, swapped, _ = api.perceive(rules, _swap_teams(state, rules.ships_per_team))
    if mode == "team_deathmatch":  # the zones swap with the teams only in team deathmatch
        for team in (0, 1):
            assert torch.equal(_q(_view(obs, team)), _q(_view(swapped, 1 - team)))
    # Mirror: a reflected world reflects the stick and leaves power and shoot.
    view = _view(obs, 0)
    reflected = view.ships.clone()
    for name in ("y", "heading_sin", "velocity_y", "log_index_gradient_y"):
        reflected[..., SHIP[name]] = -reflected[..., SHIP[name]]
    reflected[..., SHIP["angle_of_attack"]] = -reflected[..., SHIP["angle_of_attack"]]
    reflected[..., SHIP["nose_rate"]] = -reflected[..., SHIP["nose_rate"]]
    zones = view.zones.clone()
    zones[..., ZONE["y"]] = -zones[..., ZONE["y"]]
    mirrored = marginals(
        Obs(ships=reflected, fields=None, globals=view.globals, zones=zones),
        CHARACTERISTICS,
        PARAMETERS,
    )
    original = marginals(view, CHARACTERISTICS, PARAMETERS)
    swap_sides = torch.tensor([0, 2, 1, 4, 3, 6, 5])
    torch.testing.assert_close(mirrored.power, original.power, atol=1e-5, rtol=0)
    torch.testing.assert_close(mirrored.shoot, original.shoot, atol=1e-5, rtol=0)
    torch.testing.assert_close(mirrored.stick, original.stick[..., swap_sides], atol=1e-5, rtol=0)


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_the_distribution_follows_row_permutations_and_ignores_hidden_columns(mode):
    rules, _, obs = _game_views(mode)
    view = _view(obs, 0)
    S = rules.ships_per_team
    q = _q(view)
    hidden = view.ships[..., SHIP["visible"]] < 0.5
    assert hidden.any(), "the scenario should hide some enemies"
    noisy = view.ships.clone()
    noise = torch.randn(noisy.shape[:-1] + (SHIP_HIDDEN_ZERO,)) * 1000.0
    noisy[..., :SHIP_HIDDEN_ZERO] = torch.where(
        hidden[..., None], noise, noisy[..., :SHIP_HIDDEN_ZERO]
    )
    assert torch.equal(_q(dataclasses.replace(view, ships=noisy)), q)
    own_order = torch.tensor([2, 0, 1])
    enemy_order = torch.tensor([1, 2, 0]) + S
    permuted = view.ships[:, torch.cat([own_order, enemy_order])]
    torch.testing.assert_close(
        _q(dataclasses.replace(view, ships=permuted)), q[:, own_order], atol=1e-6, rtol=0
    )


# ------------------------------------------------------------------ the controller


def test_the_controller_never_takes_privileged():
    hints = typing.get_type_hints(Scripted.step)
    for name in inspect.signature(Scripted.step).parameters:
        assert hints.get(name) is not Privileged


def _copies(obs: Obs, views: int) -> Obs:
    return Obs(
        ships=obs.ships.expand(views, -1, -1),
        fields=None,
        globals=obs.globals.expand(views, -1),
        zones=obs.zones.expand(views, -1, -1),
    )


def test_a_rung_plays_the_agent_at_its_temperature_and_infinity_is_uniform():
    torch.manual_seed(0)
    obs = _duel(0, _ahead(0, 0.3, 0.5))
    views = 40_000
    many = _copies(obs, views)
    temperature = torch.full((views,), 2.0)
    rung = Scripted(CHARACTERISTICS, 2, PARAMETERS, temperature=temperature)
    expected = rung.action_distribution(many)[0, 0]
    torch.testing.assert_close(expected, distribution(obs, CHARACTERISTICS, PARAMETERS, 2.0)[0, 0])
    # a fresh carry makes every plan choice afresh: the memoryless distribution
    action, carry, _ = rung.step(rung.init_carry(views, many.ships.device), many, None)
    assert isinstance(carry, ScriptedCarry) and not bool(carry.fresh.any())
    counts = torch.bincount(to_joint(action)[:, 0].long(), minlength=NUM_JOINT).float() / views
    assert (counts - expected).abs().max() < 0.01
    temperature.fill_(math.inf)  # the league changes the temperature in place: random
    assert torch.allclose(
        rung.action_distribution(many)[0, 0], torch.full((NUM_JOINT,), 1.0 / NUM_JOINT)
    )
    with pytest.raises(ValueError):
        Scripted(CHARACTERISTICS, 2, temperature=0.0)


def _entropy(q: torch.Tensor) -> torch.Tensor:
    return -torch.special.xlogy(q, q).sum(-1)


@pytest.mark.parametrize("mode", ["team_deathmatch", "frontline"])
def test_no_living_ships_action_is_certain_and_heat_flattens_every_decision(mode):
    """In a played game, at T = 1 every living ship's every power and stick value has a
    probability in (0, 1), each binary decision's at least its epsilon, and the shot is never
    certain (its only exact value is P(shoot) = 0, at its gates: no target, or a gun that
    cannot fire); a higher temperature raises every factor's mean entropy, every logistic
    toward an even choice (the gates stay)."""
    _, _, obs = _game_views(mode)
    view = _view(obs, 0)
    alive = view.ships[:, : view.ships.shape[1] // 2, SHIP["alive"]] > 0.5
    previous = None
    for temperature in (1.0, 2.0, 4.0):
        parts = marginals(view, CHARACTERISTICS, PARAMETERS, temperature)
        if temperature == 1.0:
            for part in (parts.power, parts.stick):
                living = part[alive]
                assert bool(((living > 0.0) & (living < 1.0)).all())
            shoot = parts.shoot[alive][:, 1]
            assert bool((shoot < 1.0).all())
            assert float(parts.power[alive][:, 2].min()) >= EPSILONS["reverse"] - 1e-6  # P(reverse)
        entropies = torch.stack([_entropy(part)[alive].mean() for part in parts])
        if previous is not None:
            assert bool((entropies >= previous).all()), (temperature, entropies, previous)
            assert bool((entropies[:2] > previous[:2]).all())  # the shot may be gated throughout
        previous = entropies
    x = torch.tensor([1e-3, 0.5, 2.0, 50.0])
    for name in SCALE_FLOORS:  # every logistic tends to an even choice
        flat = logistic(x, getattr(PARAMETERS, name), 1e6)
        torch.testing.assert_close(flat, torch.full_like(x, 0.5), atol=1e-4, rtol=0)


def test_the_defaults_respect_the_scale_floors_and_pack_round_trips():
    defaults = ScriptedParameters()
    logistics = [name for name in NAMES if isinstance(getattr(defaults, name), Logistic)]
    assert sorted(logistics) == sorted(SCALE_FLOORS)
    for name in logistics:
        shape = getattr(defaults, name)
        assert shape.scale >= SCALE_FLOORS[name] and shape.centre > 0.0, name
    assert from_row(pack(defaults)) == defaults


def test_per_view_parameters_match_the_single_set():
    obs = _duel(1, _ahead(1, 0.4, 0.7))
    two = _copies(obs, 2)
    other = dataclasses.replace(PARAMETERS, lead_cap_in_gun_range=0.5)
    per_view = distribution(two, CHARACTERISTICS, unpack(table([PARAMETERS, other])))
    torch.testing.assert_close(per_view[0], _q(obs)[0])
    torch.testing.assert_close(per_view[1], distribution(obs, CHARACTERISTICS, other)[0])
    assert len(pack(PARAMETERS)) == table([PARAMETERS]).shape[1]


@pytest.mark.slow
def test_the_scripted_agent_plays_in_the_rollout_against_the_chaser():
    from bnb.controllers.scripted import Chaser

    game = GameConfig(2, 2, 1200.0, mode="frontline")
    pools = (Pool("scripted", 1.0, ("scripted", "chaser")),)
    rollout = Rollout(
        game,
        api.ship_types(),
        pools,
        {"scripted": Scripted(CHARACTERISTICS, 2), "chaser": Chaser(CHARACTERISTICS, 2)},
        "scripted",
        4,
        16,
        torch.device("cpu"),
        compiled=False,
    )
    trajectory = rollout.run()
    assert trajectory.obs.zones is not None and trajectory.log_probability.shape == (16, 4, 2)


NEUTRAL = 2  # the viewer's zone role of Frontline's neutral zone (ZONE_COLUMNS' `role`)
MIRRORED_PAIRS = (
    ("x", "y"),
    ("heading_cos", "heading_sin"),
    ("velocity_x", "velocity_y"),
    ("log_index_gradient_x", "log_index_gradient_y"),
)


def _reflected(view: Obs, angle: torch.Tensor) -> Obs:
    """The view (R, ...) reflected about the line through the arena centre at `angle` (R,):
    positions, directions and gradients mirrored, the angle of attack and the nose rate
    negated."""
    cos2, sin2 = torch.cos(2.0 * angle)[:, None], torch.sin(2.0 * angle)[:, None]

    def mirror(x, y):
        return cos2 * x + sin2 * y, sin2 * x - cos2 * y

    ships, zones = view.ships.clone(), view.zones.clone()
    for first, second in MIRRORED_PAIRS:
        a, b = mirror(view.ships[..., SHIP[first]], view.ships[..., SHIP[second]])
        ships[..., SHIP[first]], ships[..., SHIP[second]] = a, b
    for name in ("angle_of_attack", "nose_rate"):
        ships[..., SHIP[name]] = -view.ships[..., SHIP[name]]
    a, b = mirror(view.zones[..., ZONE["x"]], view.zones[..., ZONE["y"]])
    zones[..., ZONE["x"]], zones[..., ZONE["y"]] = a, b
    return Obs(ships=ships, fields=None, globals=view.globals, zones=zones)


def test_a_view_reflected_about_frontlines_neutral_axis_reflects_the_stick():
    """Frontline's map is mirror-symmetric about the axis through the centre and the neutral
    zone, so the reflected view is a view of a real game: the agent's distribution there is
    the original's with each stick level swapped for its mirror (cleanup spec 8.4)."""
    _, _, obs = _game_views("frontline")
    view = _view(obs, 0)
    neutral = (view.zones[..., ZONE["role"]] == NEUTRAL) & (view.zones[..., ZONE["present"]] > 0.5)
    assert bool(neutral.sum(-1).eq(1).all()), "every Frontline view shows one neutral zone"
    centre_x = torch.where(neutral, view.zones[..., ZONE["x"]], 0.0).sum(-1)
    centre_y = torch.where(neutral, view.zones[..., ZONE["y"]], 0.0).sum(-1)
    mirrored = marginals(
        _reflected(view, torch.atan2(centre_y, centre_x)), CHARACTERISTICS, PARAMETERS
    )
    original = marginals(view, CHARACTERISTICS, PARAMETERS)
    swap_sides = torch.tensor([0, 2, 1, 4, 3, 6, 5])
    # float32 rounding of the reflection itself moves the inputs by about 1e-4 m
    torch.testing.assert_close(mirrored.power, original.power, atol=1e-4, rtol=0)
    torch.testing.assert_close(mirrored.shoot, original.shoot, atol=1e-4, rtol=0)
    torch.testing.assert_close(mirrored.stick, original.stick[..., swap_sides], atol=1e-4, rtol=0)

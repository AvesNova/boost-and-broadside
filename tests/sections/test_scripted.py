"""The scripted agent (scripted spec A7): behaviour on constructed observations, invariance,
the ladder's rungs, and the controller in the rollout."""

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
from bnb.controllers.scripted_parameters import ScriptedParameters, pack, table, unpack
from bnb.game import api
from bnb.game.handling import time_of_flight
from bnb.runtime.rollout import Rollout

PARAMETERS = ScriptedParameters()
TYPES = (0, 1)


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
    return ship(x=distance * math.cos(angle), y=distance * math.sin(angle), team_relative=1,
                velocity=(cruise, 0.0), **kwargs)


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
                assert (left if sign > 0 else right) > 0.99 * (1.0 - stick[0])
                assert (right if sign > 0 else left) == 0.0
        assert level >= previous - 1e-6, degrees
        previous = float(level)
    assert previous > 2.9  # far off the nose: the full stick


@pytest.mark.parametrize("ship_type", TYPES)
def test_dead_and_waiting_ships_take_the_no_op(ship_type):
    dead = ship(ship_type=ship_type, alive=False)
    q = distribution(_duel(ship_type, _ahead(ship_type, 0.3, 0.5), own=dead), CHARACTERISTICS,
                     PARAMETERS)
    assert q[0, 0, 0] == 1.0 and q[0, 1, 0] == 1.0


# ------------------------------------------------------------------ the gun


@pytest.mark.parametrize("ship_type", TYPES)
def test_it_fires_only_when_aligned_in_range_ready_and_paid_for(ship_type):
    cost = value("bullet_energy_cost", ship_type)
    likely = _marginals(_duel(ship_type, _ahead(ship_type, 0.0, 0.4)))[2][1]
    assert likely > 0.95
    cases = dict(
        unaligned=_duel(ship_type, _ahead(ship_type, math.radians(30.0), 0.4)),
        out_of_range=_duel(ship_type, _ahead(ship_type, 0.0, 1.2)),
        cooling=_duel(ship_type, _ahead(ship_type, 0.0, 0.4),
                      own=ship(ship_type=ship_type, gun_cooldown=RULES_TICK)),
        short_of_energy=_duel(ship_type, _ahead(ship_type, 0.0, 0.4),
                              own=ship(ship_type=ship_type, energy=0.9 * cost)))
    for name, obs in cases.items():
        assert _marginals(obs)[2][1] == 0.0, name


RULES_TICK = 1.0 / 30.0


# ------------------------------------------------------------------ power


@pytest.mark.parametrize("ship_type", TYPES)
def test_it_boosts_when_slow_or_far_with_a_full_store_and_reverses_when_close(ship_type):
    corner = value("corner_speed", ship_type)
    slowest = PARAMETERS.boost_slow_ramp_in_corner_speed.low
    slow = _duel(ship_type, _ahead(ship_type, 0.0, 1.0), own=ship(ship_type=ship_type,
                                                                   speed=0.9 * slowest * corner))
    assert _marginals(slow)[0][1] > 0.99
    distance = 1.1 * PARAMETERS.boost_far_ramp_in_store_gun_ranges.high  # with a full store
    far = _duel(ship_type, _ahead(ship_type, 0.0, distance))
    assert _marginals(far)[0][1] > 0.99
    empty = _duel(ship_type, _ahead(ship_type, 0.0, distance),
                  own=ship(ship_type=ship_type, energy=0.1 * value("max_energy", ship_type)))
    assert _marginals(empty)[0][1] < 0.01
    closest = PARAMETERS.reverse_ramp_in_gun_range.low
    close = _duel(ship_type, _ahead(ship_type, 0.0, 0.9 * closest))
    assert _marginals(close)[0][2] > 0.99


# ------------------------------------------------------------------ the lead


@pytest.mark.parametrize("ship_type", TYPES)
def test_the_lead_moves_the_aim_point_along_a_crossing_target_by_its_time_of_flight(ship_type):
    distance, crossing = 0.8 * value("gun_range", ship_type), 60.0
    obs = _duel(ship_type, ship(x=distance, team_relative=1, velocity=(0.0, crossing)),
                own=ship(ship_type=ship_type, velocity=(0.0, 0.0)))
    fleet = scripted_combat.context(obs, CHARACTERISTICS, PARAMETERS)
    one = torch.ones(1, 2)
    aim_x, aim_y = scripted_combat.lead(fleet, CHARACTERISTICS, PARAMETERS, distance * one,
                                        0.0 * one, 0.0 * one, crossing * one)
    flight = time_of_flight(CHARACTERISTICS, fleet.own_type, torch.tensor(distance))
    torch.testing.assert_close(aim_x, distance * one)
    torch.testing.assert_close(aim_y, crossing * flight)
    _, stick, _ = _marginals(obs)
    assert _levels(stick)[1] > 0.5  # it turns toward the lead, to the left


# ------------------------------------------------------------------ the boundary


@pytest.mark.parametrize("ship_type", TYPES)
def test_near_the_boundary_and_heading_out_it_turns_inward(ship_type):
    edge = ARENA_RADIUS - 20.0
    target = ship(x=edge + 0.5 * value("gun_range", ship_type), team_relative=1,
                  velocity=(value("cruise_speed", ship_type), 0.0))
    outward = _marginals(_duel(ship_type, target, own=ship(x=edge, ship_type=ship_type)))[1]
    assert outward[0] < 0.01  # it turns
    target = ship(x=0.5 * value("gun_range", ship_type), team_relative=1,
                  velocity=(value("cruise_speed", ship_type), 0.0))
    centred = _marginals(_duel(ship_type, target))[1]
    assert centred[0] > 0.99  # the same chase at the centre flies straight


@pytest.mark.parametrize("ship_type", TYPES)
def test_with_no_visible_enemy_it_heads_for_the_centre_and_holds_fire(ship_type):
    away = ship(x=1500.0, heading=math.pi / 2, ship_type=ship_type)  # the centre is on its left
    for enemy in (absent(), ship(x=1700.0, y=200.0, team_relative=1, visible=False)):
        power, stick, shoot = _marginals(_duel(ship_type, enemy, own=away))
        assert _levels(stick)[1] > 0.99 and shoot[1] == 0.0 and power[2] == 0.0
    near = ship(x=100.0, heading=math.pi / 2, ship_type=ship_type)
    assert _marginals(_duel(ship_type, absent(), own=near))[1][0] == 1.0  # holds its course


# ------------------------------------------------------------------ Frontline


def _ring(direction: float = 0.0, progress: float = 0.0) -> list[torch.Tensor]:
    """Own spawn below, own defense on the left, enemy defense on the right."""
    return [zone(0.0, -1600.0, 0), zone(-1600.0, 0.0, 1, progress, direction),
            zone(0.0, 1600.0, 2), zone(1600.0, 0.0, 3), zone(0.0, 2600.0, 4)]


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_it_defends_a_zone_whose_meter_moves_against_it(ship_type):
    own = [ship(heading=math.pi / 2, ship_type=ship_type),
           ship(x=60.0, heading=math.pi / 2, ship_type=ship_type)]
    raiders = [ship(x=-1600.0, y=50.0 * i, team_relative=1) for i in range(2)]
    pressed = observation(own, raiders, _ring(direction=-1.0, progress=0.4))
    for row in (0, 1):  # both turn left, toward the defense zone
        assert _levels(_marginals(pressed, row)[1])[1] > 0.99
    calm = observation(own, [absent(), absent()], _ring())
    sides = [_levels(_marginals(calm, row)[1])[1:] for row in (0, 1)]
    assert sides[0][0] > 0.99 and sides[1][1] > 0.99  # one defends, the other attacks


@pytest.mark.parametrize("ship_type", TYPES)
def test_frontline_it_falls_back_to_its_spawn_when_hurt_and_threatened(ship_type):
    gun_range = value("gun_range", ship_type)
    threats = [ship(x=1.5 * gun_range, y=(0.5 + 0.2 * i) * gun_range, team_relative=1)
               for i in range(2)]
    hurt = ship(ship_type=ship_type, shield=0.5 * PARAMETERS.recovery_shield_fraction
                * value("max_shield", ship_type))
    healthy = ship(ship_type=ship_type)
    for own, side in ((hurt, 2), (healthy, 1)):  # spawn on the right, the threats on the left
        obs = observation([own, absent(team_relative=0)], threats, _ring())
        assert _levels(_marginals(obs)[1])[side] > 0.9


# ------------------------------------------------------------------ invariance


def _game_views(mode: str, seed: int = 3):
    """A few ticks of a fogged 3v3 game, scripted on both sides: the state and its views."""
    config = GameConfig(3, 4, 2600.0, mode=mode)
    rules, types = api.rules_of(config), api.ship_types()
    generator = torch.Generator().manual_seed(seed)
    B = 32
    state = api.reset(rules, types, api.uniforms(rules, B, generator=generator),
                      torch.ones(B, dtype=torch.bool))
    agent = Scripted(CHARACTERISTICS)
    torch.manual_seed(seed)
    for _ in range(90):
        state, obs, _ = api.perceive(rules, state)
        actions = [agent.step(None, _view(obs, team), None)[0] for team in (0, 1)]
        action = type(actions[0])(*(torch.cat(parts, 1) for parts in zip(
            *[(a.power, a.stick, a.shoot) for a in actions])))
        state, _ = api.step(rules, types, state, action,
                            api.spread(rules, B, generator=generator),
                            api.uniforms(rules, B, generator=generator))
    state, obs, _ = api.perceive(rules, state)
    return rules, state, obs


def _view(obs: Obs, team: int) -> Obs:
    return Obs(ships=obs.ships[:, team], fields=None, globals=obs.globals[:, team],
               zones=obs.zones[:, team])


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
    mirrored = marginals(Obs(ships=reflected, fields=None, globals=view.globals, zones=zones),
                         CHARACTERISTICS, PARAMETERS)
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
    noisy[..., :SHIP_HIDDEN_ZERO] = torch.where(hidden[..., None], noise,
                                                noisy[..., :SHIP_HIDDEN_ZERO])
    assert torch.equal(_q(dataclasses.replace(view, ships=noisy)), q)
    own_order = torch.tensor([2, 0, 1])
    enemy_order = torch.tensor([1, 2, 0]) + S
    permuted = view.ships[:, torch.cat([own_order, enemy_order])]
    torch.testing.assert_close(_q(dataclasses.replace(view, ships=permuted)), q[:, own_order],
                               atol=1e-6, rtol=0)


# ------------------------------------------------------------------ the controller


def test_the_controller_never_takes_privileged():
    hints = typing.get_type_hints(Scripted.step)
    for name in inspect.signature(Scripted.step).parameters:
        assert hints.get(name) is not Privileged


def test_a_rung_plays_p_scripted_plus_uniform():
    torch.manual_seed(0)
    obs = _duel(0, _ahead(0, 0.3, 0.5))
    views = 40_000
    many = Obs(ships=obs.ships.expand(views, -1, -1), fields=None,
               globals=obs.globals.expand(views, -1), zones=obs.zones.expand(views, -1, -1))
    p = torch.full((views,), 0.6)
    rung = Scripted(CHARACTERISTICS, scripted_probability=p)
    expected = rung.action_distribution(many)[0, 0]
    torch.testing.assert_close(expected, 0.6 * _q(obs)[0, 0] + 0.4 / NUM_JOINT)
    action, carry, _ = rung.step(None, many, None)
    assert carry is None
    counts = torch.bincount(to_joint(action)[:, 0].long(), minlength=NUM_JOINT).float() / views
    assert (counts - expected).abs().max() < 0.01
    p.zero_()  # the league changes p in place: random
    assert torch.allclose(rung.action_distribution(many)[0, 0],
                          torch.full((NUM_JOINT,), 1.0 / NUM_JOINT))


def test_per_view_parameters_match_the_single_set():
    obs = _duel(1, _ahead(1, 0.4, 0.7))
    two = Obs(ships=obs.ships.expand(2, -1, -1), fields=None, globals=obs.globals.expand(2, -1),
              zones=obs.zones.expand(2, -1, -1))
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
    rollout = Rollout(game, pools, {"scripted": Scripted(CHARACTERISTICS), "chaser": Chaser(2)},
                      "scripted", 4, 16, torch.device("cpu"), compile=False)
    trajectory = rollout.run()
    assert trajectory.obs.zones is not None and trajectory.logp.shape == (16, 4, 2)

"""The scripted agent's per-type doctrines (`bnb.controllers.scripted_doctrine`): off, the
agent as without them; each element moves the aim or the power the way its docstring says;
each own ship flies its own type's doctrine."""

import dataclasses
import math

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship, value

from bnb.contracts import Obs
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import decide
from bnb.controllers.scripted_combat import advantage_weight, context
from bnb.controllers.scripted_doctrine import (
    WIDTH,
    Doctrine,
    from_named,
    from_row,
    named,
    pack,
    per_ship,
    table,
)
from bnb.controllers.scripted_parameters import Logistic, ScriptedParameters

pytestmark = pytest.mark.usefixtures("global_generator")
PARAMETERS = ScriptedParameters()
TURN = CHARACTERISTICS.turn_rate[:, 2]
AGILE, STIFF = int(TURN.argmax()), int(TURN.argmin())
TYPES = TURN.shape[0]
ALWAYS = Logistic(1e-3, 0.05, True)  # every advantage takes the doctrine


def _duel(
    own_type: int,
    enemy_type: int,
    bearing: float,
    gun_ranges: float,
    enemy_heading: float = 0.0,
    own_speed: float | None = None,
) -> Obs:
    """Own ship 0 at the origin (nose +x) against one enemy at this bearing (positive left)
    and distance in the own gun range, flying at `enemy_heading`; the other slots absent."""
    distance = gun_ranges * value("gun_range", own_type)
    enemy = ship(
        x=distance * math.cos(bearing),
        y=distance * math.sin(bearing),
        heading=enemy_heading,
        ship_type=enemy_type,
        team_relative=1,
    )
    own = ship(ship_type=own_type, speed=own_speed)
    return observation([own, absent(own_type, 0)], [enemy, absent(enemy_type)])


def _decide(obs: Obs, doctrines=None, seed: int = 0):
    torch.manual_seed(seed)
    return decide(obs, CHARACTERISTICS, PARAMETERS, switching=True, doctrines=doctrines)[0]


def _every_type(doctrine: Doctrine) -> torch.Tensor:
    return table([doctrine] * TYPES)


def _cases() -> list[Obs]:
    return [
        _duel(own, enemy, bearing, distance, heading)
        for own in (AGILE, STIFF)
        for enemy in (AGILE, STIFF)
        for bearing in (0.0, 0.4, 1.6, -2.8)
        for distance in (0.3, 1.0, 4.0)
        for heading in (0.0, math.pi)
    ]


def test_an_off_doctrine_is_the_agent_without_one():
    """Every element at full weight but a strength of 0, per type and per view: the agent's
    marginals, within rounding."""
    moved = Doctrine(extend=1.0, slash=1.0, bleed=1.0, lag=1.0, deny=1.0)
    for obs in _cases():
        agent = _decide(obs)
        for doctrines in (_every_type(moved), _every_type(moved)[None].expand(1, -1, -1)):
            for a, b in zip(agent, _decide(obs, doctrines), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_the_doctrine_weighs_in_by_the_advantage():
    def weight(doctrine, own_type: int, enemy_type: int) -> float:
        obs = _duel(own_type, enemy_type, 0.0, 1.0)
        fleet = context(obs, CHARACTERISTICS, PARAMETERS)

        def target_of(values):  # the one enemy, row 0, is the target
            return values[:, None, :1].sum(-1).expand_as(fleet.x)

        has_target = torch.ones_like(fleet.x, dtype=torch.bool)
        rows = per_ship(table([doctrine]), torch.zeros_like(fleet.own_type))
        return float(advantage_weight(fleet, CHARACTERISTICS, rows, target_of, has_target)[0, 0])

    ratio = float(TURN[AGILE] / TURN[STIFF])
    sharp = Doctrine(strength=1.0, in_advantage=Logistic(1.0, 0.05, True))
    expected = 1.0 / (1.0 + math.exp(-math.log(ratio) / 0.05))
    assert weight(sharp, STIFF, AGILE) == pytest.approx(expected, rel=1e-5)
    assert weight(sharp, AGILE, STIFF) == pytest.approx(1.0 - expected, rel=1e-4)
    assert weight(sharp, STIFF, STIFF) == pytest.approx(0.5)  # an even match: the centre
    flipped = dataclasses.replace(sharp, turn_exponent=1.0)
    assert weight(flipped, AGILE, STIFF) == pytest.approx(expected, rel=1e-5)
    half = dataclasses.replace(sharp, strength=0.5)
    assert weight(half, STIFF, AGILE) == pytest.approx(0.5 * expected, rel=1e-5)
    assert weight(Doctrine(), STIFF, AGILE) == 0.0


def test_extending_it_flies_away_from_a_close_target_off_its_nose():
    """Close and 90 degrees to its left, the agent turns left after it; extending, the ship
    turns right, away, boosts and does not reverse. A target on its nose is attacked as
    before, and the shot is the agent's in either."""
    extend = _every_type(
        Doctrine(
            strength=1.0,
            in_advantage=ALWAYS,
            extend=1.0,
            fight_in_gun_range=Logistic(2.0, 0.1, False),
        )
    )
    close_left = _duel(STIFF, AGILE, math.pi / 2, 0.3)
    agent, extending = _decide(close_left), _decide(close_left, extend)
    assert float(agent.stick[0, 0, 2::2].sum()) > 0.9  # left, after it
    assert float(extending.stick[0, 0, 1::2].sum()) > 0.9  # right, away
    assert float(extending.power[0, 0, 2]) < 0.5 * float(agent.power[0, 0, 2])
    assert float(extending.power[0, 0, 1]) > 0.75
    assert torch.equal(agent.shoot, extending.shoot)
    ahead = _duel(STIFF, STIFF, 0.0, 0.5, enemy_heading=math.pi)  # an even store
    agent, extending = _decide(ahead), _decide(ahead, extend)
    torch.testing.assert_close(agent.stick, extending.stick, atol=0.02, rtol=0.0)


def test_without_the_energy_to_engage_it_extends_from_a_target_on_its_nose():
    """A target on the nose with far more energy (a full store against an empty one): the
    energy rule turns the ship away; with the energy (a type of its own store), it attacks."""
    doctrine = Doctrine(
        strength=1.0,
        in_advantage=ALWAYS,
        extend=1.0,
        fight_in_gun_range=Logistic(2.0, 0.1, False),
        engage_in_energy_ratio=Logistic(1.0, 0.03, True),
    )
    obs = _duel(STIFF, STIFF, 0.0, 0.5, enemy_heading=math.pi)
    poor = obs.ships.clone()
    poor[0, 0, SHIP["energy"]] = 0.0  # the own store: empty
    starved = Obs(ships=poor, fields=obs.fields, globals=obs.globals, zones=obs.zones)
    straight = _decide(obs, _every_type(doctrine))
    away = _decide(starved, _every_type(doctrine))
    assert float(straight.stick[0, 0, 0]) > 0.5  # on the nose: straight at it
    assert float(away.stick[0, 0, 0]) < 0.1  # turning away


def test_slashing_boosts_into_the_attack_and_never_reverses():
    doctrine = Doctrine(strength=1.0, in_advantage=ALWAYS, slash=1.0)
    obs = _duel(STIFF, AGILE, 0.0, 0.3, enemy_heading=math.pi)
    agent, slash = _decide(obs), _decide(obs, _every_type(doctrine))
    assert float(slash.power[0, 0, 1]) > 0.9
    assert float(slash.power[0, 0, 2]) < float(agent.power[0, 0, 2]) + 1e-6
    assert float(slash.power[0, 0, 2]) < 0.05


def test_bleeding_reverses_toward_the_corner_speed_in_a_turning_fight():
    """Fast, close, the target far off the nose: the bleed reverses; at the corner speed it
    does not."""
    doctrine = Doctrine(
        strength=1.0,
        in_advantage=ALWAYS,
        bleed=1.0,
        fight_in_gun_range=Logistic(2.0, 0.1, False),
        bleed_in_corner_speed=Logistic(1.2, 0.031, True),
    )
    corner = float(CHARACTERISTICS.corner_speed[AGILE])
    fast = _duel(AGILE, STIFF, 2.0, 1.5, own_speed=2.0 * corner)
    slow = _duel(AGILE, STIFF, 2.0, 1.5, own_speed=corner)
    assert float(_decide(fast, _every_type(doctrine)).power[0, 0, 2]) > 0.9
    assert float(_decide(fast).power[0, 0, 2]) < 0.5
    assert float(_decide(slow, _every_type(doctrine)).power[0, 0, 2]) < 0.5


def test_denying_turns_off_the_line_of_a_target_whose_nose_is_on_the_ship():
    """A target two gun ranges ahead, nose on, slightly left of the line: the agent flies
    straight at it; denying, the ship turns (to the side it is already off its nose)."""
    doctrine = Doctrine(
        strength=1.0,
        in_advantage=ALWAYS,
        deny=1.0,
        deny_angle=0.6,
        deny_in_gun_range=Logistic(3.0, 0.1, False),
    )
    obs = _duel(AGILE, STIFF, 0.0, 2.0, enemy_heading=math.pi - 0.02)
    agent, deny = _decide(obs), _decide(obs, _every_type(doctrine))
    assert float(agent.stick[0, 0, 0]) > 0.5
    assert float(deny.stick[0, 0, 0]) < 0.1
    # the target flying away (its tail to the ship): no denial
    away = _duel(AGILE, STIFF, 0.0, 2.0, enemy_heading=0.0)
    torch.testing.assert_close(
        _decide(away).stick, _decide(away, _every_type(doctrine)).stick, atol=0.02, rtol=0.0
    )


def test_each_ship_flies_its_own_types_doctrine():
    """A team of the two types: a doctrine on one type moves that ship only."""
    distance = 0.3 * value("gun_range", STIFF)
    enemy = ship(x=0.0, y=distance, ship_type=AGILE, team_relative=1)
    own = [ship(ship_type=STIFF), ship(x=0.0, y=-5.0, ship_type=AGILE)]
    obs = observation(own, [enemy, absent(AGILE)])
    doctrines = [Doctrine()] * TYPES
    doctrines[STIFF] = Doctrine(
        strength=1.0,
        in_advantage=ALWAYS,
        extend=1.0,
        fight_in_gun_range=Logistic(2.0, 0.1, False),
    )
    agent, mixed = _decide(obs), _decide(obs, table(doctrines))
    assert not torch.allclose(agent.stick[0, 0], mixed.stick[0, 0], atol=0.05)
    torch.testing.assert_close(agent.stick[0, 1], mixed.stick[0, 1], rtol=1e-5, atol=1e-6)


def test_a_doctrine_packs_and_names_both_ways():
    doctrine = Doctrine(strength=0.7, extend=0.4, deny_angle=0.3, lag_in_gun_range=0.25)
    assert len(pack(doctrine)) == WIDTH
    assert from_row(pack(doctrine)) == doctrine
    assert from_named(named(doctrine, figures=12)) == doctrine
    assert from_named({}) == Doctrine()
    with pytest.raises(ValueError):
        from_row(pack(doctrine)[:-1])

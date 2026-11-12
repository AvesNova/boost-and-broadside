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


LEFT_LEVELS, RIGHT_LEVELS = (2, 4, 6), (1, 3, 5)


def test_a_lead_turn_turns_before_the_merge_to_the_passing_side_or_away():
    """A target one gun range ahead flying at the ship, offset a little to the left: the
    agent flies at it; the lead turn turns left at full stick (the side it will pass on: the
    two-circle sense), or right in the one-circle sense; far from the merge it does not."""
    lead = Doctrine(lead_turn=1.0, lead_turn_in_seconds=Logistic(3.0, 0.1, False))
    obs = _duel(STIFF, STIFF, 0.05, 1.0, enemy_heading=math.pi)
    agent = _decide(obs)
    two = _decide(obs, _every_type(lead))
    one = _decide(obs, _every_type(dataclasses.replace(lead, lead_turn_one_circle=1.0)))
    assert float(agent.stick[0, 0, 6]) < 0.5
    assert float(two.stick[0, 0, 6]) > 0.9
    assert float(one.stick[0, 0, 5]) > 0.9
    far = _duel(STIFF, STIFF, 0.05, 6.0, enemy_heading=math.pi)
    torch.testing.assert_close(
        _decide(far).stick, _decide(far, _every_type(lead)).stick, atol=0.02, rtol=0.0
    )
    half = _decide(obs, _every_type(dataclasses.replace(lead, lead_turn_stick=0.4)))
    assert float(half.stick[0, 0, 2]) > 0.9


def test_a_yoyo_climbs_with_the_stick_eased_after_a_stalemate_then_dives():
    """A crossing target inside the fight's range, its nose off the ship: once the
    stalemate clock passes
    yoyo_after_seconds the ship climbs (reverse) at the eased stick for
    yoyo_climb_seconds, then dives (boost); a re-read carry gives the same marginals."""
    yoyo = Doctrine(
        yoyo=1.0, yoyo_after_seconds=5.0, yoyo_climb_seconds=0.05, yoyo_dive_seconds=0.05
    )
    doctrines = _every_type(yoyo)
    obs = _duel(STIFF, STIFF, 0.6, 2.0, enemy_heading=0.6 + 0.5 * math.pi)
    _, carry = decide(obs, CHARACTERISTICS, PARAMETERS, switching=True, doctrines=doctrines)
    assert float(carry.stalemate_seconds[0, 0]) == 0.0  # fresh
    _, carry = decide(
        obs, CHARACTERISTICS, PARAMETERS, carry=carry, switching=True, doctrines=doctrines
    )
    assert float(carry.stalemate_seconds[0, 0]) > 0.0
    assert float(carry.yoyo_seconds[0, 0]) < 0.0
    carry = dataclasses.replace(
        carry, stalemate_seconds=torch.full_like(carry.stalemate_seconds, 9.0)
    )
    climb, carry = decide(
        obs, CHARACTERISTICS, PARAMETERS, carry=carry, switching=True, doctrines=doctrines
    )
    assert float(carry.yoyo_seconds[0, 0]) == 0.0
    assert float(climb.power[0, 0, 2]) > 0.9  # reverse: speed into the store
    again = decide(obs, CHARACTERISTICS, PARAMETERS, carry=carry, doctrines=doctrines)[0]
    torch.testing.assert_close(again.power, climb.power)
    for _ in range(2):
        dive, carry = decide(
            obs, CHARACTERISTICS, PARAMETERS, carry=carry, switching=True, doctrines=doctrines
        )
    assert float(dive.power[0, 0, 1]) > 0.9  # boost: the store into speed
    for _ in range(2):
        _, carry = decide(
            obs, CHARACTERISTICS, PARAMETERS, carry=carry, switching=True, doctrines=doctrines
        )
    assert float(carry.yoyo_seconds[0, 0]) < 0.0  # done: the clocks start over
    assert float(carry.stalemate_seconds[0, 0]) < 1.0
    # the low yo-yo dives first; never under the target's guns
    low = dataclasses.replace(yoyo, yoyo_low=1.0)
    stalled = dataclasses.replace(carry, stalemate_seconds=torch.full_like(carry.yoyo_seconds, 9.0))
    dive = decide(
        obs, CHARACTERISTICS, PARAMETERS, carry=stalled, switching=True, doctrines=_every_type(low)
    )[0]
    assert float(dive.power[0, 0, 1]) > 0.9
    head_on = _duel(STIFF, STIFF, 0.6, 2.0, enemy_heading=0.6 + math.pi)
    _, after = decide(
        head_on, CHARACTERISTICS, PARAMETERS, carry=stalled, switching=True, doctrines=doctrines
    )
    assert float(after.yoyo_seconds[0, 0]) < 0.0


def test_a_brake_reverses_with_a_closing_target_on_the_tail():
    """A faster target half a gun range behind, nose on the ship: braking reverses."""
    brake = Doctrine(brake=1.0)
    distance = 0.5 * value("gun_range", STIFF)
    enemy = ship(
        x=-distance,
        heading=0.0,
        speed=2.0 * value("cruise_speed", STIFF),
        ship_type=STIFF,
        team_relative=1,
    )
    obs = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [enemy, absent(STIFF)])
    assert float(_decide(obs).power[0, 0, 2]) < 0.5
    assert float(_decide(obs, _every_type(brake)).power[0, 0, 2]) > 0.9


def test_the_sense_turns_with_or_against_the_targets_rotation_after_the_merge():
    """A target just passed (behind, its tail to the ship's tail) turning left: the
    two-circle sense turns left (with its rotation), the one-circle sense right."""
    distance = 0.5 * value("gun_range", STIFF)
    enemy = ship(x=-distance, y=-30.0, heading=math.pi, ship_type=STIFF, team_relative=1)
    enemy[SHIP["nose_rate"]] = 0.3
    obs = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [enemy, absent(STIFF)])
    two = _decide(obs, _every_type(Doctrine(sense=1.0)))
    one = _decide(obs, _every_type(Doctrine(sense=1.0, sense_one_circle=1.0)))
    assert float(two.stick[0, 0, 6]) > 0.9
    assert float(one.stick[0, 0, 5]) > 0.9


def test_the_sense_gates_by_the_lift_ratio_and_the_mirror():
    """Against a target of the other type: the mirror-only sense does not act, nor does a
    sense whose lift-ratio gate the pair's ratio is below."""
    distance = 0.5 * value("gun_range", STIFF)
    other = AGILE if AGILE != STIFF else STIFF
    enemy = ship(x=-distance, y=-30.0, heading=math.pi, ship_type=other, team_relative=1)
    enemy[SHIP["nose_rate"]] = 0.3
    obs = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [enemy, absent(other)])
    agent = _decide(obs)
    one = Doctrine(sense=1.0, sense_one_circle=1.0)
    mirror_only = _decide(obs, _every_type(dataclasses.replace(one, sense_mirror_only=1.0)))
    torch.testing.assert_close(mirror_only.stick, agent.stick)
    lift = CHARACTERISTICS.max_acceleration / CHARACTERISTICS.corner_speed**2
    ratio = float(lift[STIFF] / lift[other])
    gated = dataclasses.replace(one, sense_in_lift_ratio=Logistic(ratio * 10.0, 0.05, True))
    torch.testing.assert_close(
        _decide(obs, _every_type(gated)).stick, agent.stick, atol=1e-3, rtol=0
    )
    assert float(_decide(obs, _every_type(one)).stick[0, 0, 5]) > 0.9


def test_the_sense_hands_over_to_the_crowd_sense_with_another_enemy_near():
    """The one-circle sense alone with its target; with another enemy near, the crowd
    sense flies straight (stick 0) and boosts."""
    gun = value("gun_range", STIFF)
    enemy = ship(x=-0.5 * gun, y=-30.0, heading=math.pi, ship_type=STIFF, team_relative=1)
    enemy[SHIP["nose_rate"]] = 0.3
    doctrine = Doctrine(
        sense=1.0,
        sense_one_circle=1.0,
        sense_in_crowd=Logistic(0.5, 0.1, False),
        crowd_sense=1.0,
        crowd_sense_boost=1.0,
    )
    alone = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [enemy, absent(STIFF)])
    assert float(_decide(alone, _every_type(doctrine)).stick[0, 0, 5]) > 0.9
    other = ship(x=-2.0 * gun, y=2.0 * gun, heading=0.0, ship_type=STIFF, team_relative=1)
    crowd = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [enemy, other])
    flown = _decide(crowd, _every_type(doctrine))
    assert float(flown.stick[0, 0, 0]) > 0.9
    assert float(flown.power[0, 0, 1]) > 0.8

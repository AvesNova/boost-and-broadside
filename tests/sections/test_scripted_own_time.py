"""The scripted agent's own time and relative tactics (the energy fight): off by default and
then the agent exactly, and when on, a ship out-turned by its target extends from a close
target off its nose rather than turning after it (docs/explanation/game/ship-cycles.md)."""

import dataclasses
import math
import pathlib
import sys

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship, value

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.controllers.scripted import decide
from bnb.controllers.scripted_combat import context, energy_fight
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.controllers.scripted_memory import fresh_carry, remember, time_unit
from bnb.controllers.scripted_parameters import (
    ADDITIVE_BOUNDS,
    RELATIVE_TACTICS,
    Logistic,
    ScriptedParameters,
    Switch,
    in_own_time,
    logistic,
    table,
    unpack,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))

from bench import scripted_search as search  # noqa: E402

pytestmark = pytest.mark.usefixtures("global_generator")
PARAMETERS = ScriptedParameters()
# The standard card's two types by their settled full-stick turn: the better turner and
# the worse.
TURN = CHARACTERISTICS.turn_rate[:, 2]
AGILE, STIFF = int(TURN.argmax()), int(TURN.argmin())
# Every relative-tactics number moved but the strength: still off.
MOVED_OFF = dataclasses.replace(
    PARAMETERS,
    energy_fight_turn_exponent=1.7,
    energy_fight_climb_exponent=0.5,
    energy_fight_speed_exponent=-0.3,
    energy_fight_range_exponent=2.0,
    energy_fight_in_advantage=Logistic(0.3, 0.2, True),
    extend_in_gun_range=Logistic(3.0, 0.4, False),
    extend_in_stall_angle=Logistic(0.5, 0.3, True),
)
ON = dataclasses.replace(PARAMETERS, energy_fight_strength=1.0)


def _duel(own_type: int, enemy_type: int, bearing: float, gun_ranges: float) -> Obs:
    """Own ship 0 at the origin (nose +x) against one enemy at this bearing (positive left)
    and distance in the own gun range, flying parallel; the other slots absent."""
    distance = gun_ranges * value("gun_range", own_type)
    enemy = ship(
        x=distance * math.cos(bearing),
        y=distance * math.sin(bearing),
        ship_type=enemy_type,
        team_relative=1,
        velocity=(value("cruise_speed", own_type), 0.0),
    )
    own = ship(ship_type=own_type)
    return observation([own, absent(own_type, 0)], [enemy, absent(enemy_type)])


def _cases() -> list[Obs]:
    return [
        _duel(own, enemy, bearing, distance)
        for own in (AGILE, STIFF)
        for enemy in (AGILE, STIFF)
        for bearing in (0.0, 0.4, 1.6, -2.8)
        for distance in (0.3, 1.0, 4.0)
    ]


def _decide(obs: Obs, parameters, seed: int = 0):
    torch.manual_seed(seed)
    return decide(obs, CHARACTERISTICS, parameters, switching=True)


def test_off_the_relative_tactics_and_own_time_change_nothing_bitwise():
    """A strength of 0 and an own-time exponent of 0 give exactly the agent's marginals and
    carry, whatever the other new numbers are, for one set and per view (a table's float32
    against a table's)."""
    for obs in _cases():
        for agent, other in (
            (PARAMETERS, MOVED_OFF),
            (unpack(table([PARAMETERS])), unpack(table([MOVED_OFF]))),
        ):
            parts, carry = _decide(obs, agent)
            moved, moved_carry = _decide(obs, other)
            for a, b in zip(parts, moved, strict=True):
                assert torch.equal(a, b)
            for field in dataclasses.fields(carry):
                assert torch.equal(getattr(carry, field.name), getattr(moved_carry, field.name))


def test_the_defaults_are_off_in_every_mode():
    for mode, parameters in MODE_DEFAULTS.sets.items():
        assert parameters.energy_fight_strength == 0.0, mode
        assert parameters.time_unit_exponent == 0.0, mode


def test_the_doctrine_weighs_in_when_out_turned():
    def weight(parameters, own_type: int, enemy_type: int) -> float:
        obs = _duel(own_type, enemy_type, 0.0, 1.0)
        fleet = context(obs, CHARACTERISTICS, parameters)

        def target_of(values):  # the one enemy, row 0, is the target
            return values[:, None, :1].sum(-1).expand_as(fleet.x)

        has_target = torch.ones_like(fleet.x, dtype=torch.bool)
        return float(energy_fight(fleet, CHARACTERISTICS, parameters, target_of, has_target)[0, 0])

    ratio = float(TURN[AGILE] / TURN[STIFF])
    sharp = dataclasses.replace(ON, energy_fight_in_advantage=Logistic(1.0, 0.05, True))
    expected = 1.0 / (1.0 + math.exp(-math.log(ratio) / 0.05))
    assert weight(sharp, STIFF, AGILE) == pytest.approx(expected, rel=1e-5)
    assert weight(sharp, AGILE, STIFF) == pytest.approx(1.0 - expected, rel=1e-4)
    assert weight(sharp, STIFF, STIFF) == pytest.approx(0.5)  # an even match: the centre
    # the turn exponent's sign flips who takes it; a half strength halves it
    flipped = dataclasses.replace(sharp, energy_fight_turn_exponent=1.0)
    assert weight(flipped, AGILE, STIFF) == pytest.approx(expected, rel=1e-5)
    half = dataclasses.replace(sharp, energy_fight_strength=0.5)
    assert weight(half, STIFF, AGILE) == pytest.approx(0.5 * expected, rel=1e-5)
    assert weight(PARAMETERS, STIFF, AGILE) == 0.0


def test_out_turned_it_extends_from_a_close_target_off_its_nose():
    """The stiff ship against the agile one, close and 90 degrees to its left: the agent
    turns left after it; in the doctrine (every turn ratio taken) it turns right, away,
    boosts and does not reverse. A target on its nose is attacked as before, and the shot is
    the agent's in either."""
    always = dataclasses.replace(ON, energy_fight_in_advantage=Logistic(1e-3, 0.05, True))
    close_left = _duel(STIFF, AGILE, math.pi / 2, 0.3)
    agent, _ = _decide(close_left, PARAMETERS)
    energy, _ = _decide(close_left, always)
    stick, energy_stick = agent.stick[0, 0], energy.stick[0, 0]
    assert float(stick[2::2].sum()) > 0.9  # left, after it
    assert float(energy_stick[1::2].sum()) > 0.9  # right, away
    assert float(energy.power[0, 0, 2]) < 0.5 * float(agent.power[0, 0, 2])  # no reverse
    boost = float(energy.power[0, 0, 1])  # 90 degrees off is 1.5 times the extend centre
    assert boost > 0.75 and boost > float(agent.power[0, 0, 1])
    assert torch.equal(agent.shoot, energy.shoot)
    ahead = _duel(STIFF, AGILE, 0.0, 0.5)
    agent, _ = _decide(ahead, PARAMETERS)
    energy, _ = _decide(ahead, always)
    assert torch.allclose(agent.stick, energy.stick, atol=0.02)
    assert torch.equal(agent.shoot, energy.shoot)


def test_own_time_scales_the_switches_and_the_fade():
    own_types = torch.tensor([[AGILE, STIFF]], dtype=torch.int32)
    assert torch.equal(time_unit(CHARACTERISTICS, own_types, PARAMETERS), torch.ones(1, 2))
    whole = dataclasses.replace(PARAMETERS, time_unit_exponent=1.0)
    units = time_unit(CHARACTERISTICS, own_types, whole)
    turn_time = CHARACTERISTICS.corner_speed / CHARACTERISTICS.max_acceleration
    assert torch.allclose(units, (turn_time / turn_time[0])[own_types.long()])
    assert float(time_unit(CHARACTERISTICS, torch.zeros(1, 1, dtype=torch.int32), whole)) == 1.0
    shape = Switch(8.0, 0.3, 0.06)
    assert in_own_time(shape, torch.ones(1)).rate.item() == 8.0
    assert in_own_time(shape, torch.full((1,), 2.0)).rate.item() == 4.0
    # a hidden enemy fades in the team's mean unit
    stiff_unit = float(units[0, 1])
    obs = observation([ship(ship_type=STIFF), absent(STIFF, 0)], [absent(), absent()])
    obs.ships[0, 2, SHIP["alive"]] = 1.0
    obs.ships[0, 2, SHIP["time_since_seen"]] = 1.0
    obs.globals[:, GLOBAL["reset"]] = 0.0
    carry = fresh_carry(1, 2, 2, torch.device("cpu"))
    carry = dataclasses.replace(
        carry,
        known=torch.tensor([[True, False]]),
        enemy_x=torch.tensor([[5e4, 0.0]]),  # far outside the own vision
    )
    reset = torch.zeros(1, dtype=torch.bool)
    _, recall = remember(carry, obs, reset, CHARACTERISTICS, whole, 1.0)
    fade = PARAMETERS.memory_fade_in_seconds
    expected = logistic(torch.tensor(1.0 / stiff_unit), fade)  # one own ship: its unit
    # the absent second own ship has the same type: the mean is the stiff unit
    assert float(recall.weight[0, 0]) == pytest.approx(float(expected), rel=1e-5)


def test_the_search_keeps_the_relative_tactics_unless_asked_and_bounds_them():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        child = search.perturb(PARAMETERS, 1.0, generator)
        for name in RELATIVE_TACTICS:
            assert getattr(child, name) == getattr(PARAMETERS, name), name
    moved = set()
    for _ in range(200):
        child = search.perturb(PARAMETERS, 1.0, generator, frozen=())
        for name, (low, high) in ADDITIVE_BOUNDS.items():
            assert low <= getattr(child, name) <= high, name
            moved |= {name} if getattr(child, name) != getattr(PARAMETERS, name) else set()
        assert child.extend_in_stall_angle.rising and not child.extend_in_gun_range.rising
    assert moved == set(ADDITIVE_BOUNDS)

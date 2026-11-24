"""The scripted agent's assessment (lean.scripted.situation): hand-built pairs give the right
quadrant (bearing and aspect against the 3/9 line), flow sign, closure, line-of-sight rate and
energy advantage."""

import math

import pytest
import torch

from lean.game import core
from lean.scripted import ScriptedConfig
from lean.scripted.opponent import engage
from lean.scripted.situation import assess
from lean.scripted.tests.test_scripted import placed

FLOOR = ScriptedConfig().values["turning_floor"]


def situation(own: dict, target: dict, nose_rates=(0.0, 0.0), stores=(0.5, 0.5)):
    state = placed(own, target)
    state = state.replace(
        nose_rate=torch.tensor([nose_rates]),
        energy=torch.tensor([stores]) * state.constants["max_energy"],
    )
    fight = engage(state, 0)
    return state, assess(state, fight, FLOOR)


def ship(x=0.0, y=0.0, heading=0.0, speed=200.0):
    return dict(x=x, y=y, heading=heading, speed=speed)


# Own ship at the origin heading +x; the target 400 m away. Quadrant by the 3/9 line (pi / 2):
# he is ahead (|bearing| < pi / 2) or behind; I am behind his 3/9 line (aspect < pi / 2) or not.
QUADRANTS = [
    ("tail chase: offensive", ship(400.0), True, True),
    ("head-on: merge", ship(400.0, heading=math.pi), True, False),
    ("he is on my six: defensive", ship(-400.0), False, False),
    ("flying apart after the pass: turning", ship(-400.0, heading=math.pi), False, True),
    (
        "beam, he crosses ahead left to right, nose away: offensive",
        ship(300.0, 200.0, heading=0.3),
        True,
        True,
    ),
]


@pytest.mark.parametrize("name, target, ahead, astern", QUADRANTS, ids=[q[0] for q in QUADRANTS])
def test_quadrants(name, target, ahead, astern):
    _, s = situation(ship(), target)
    assert (float(s.bearing.abs()[0, 0]) < math.pi / 2) == ahead, float(s.bearing[0, 0])
    assert (float(s.aspect[0, 0]) < math.pi / 2) == astern, float(s.aspect[0, 0])
    assert 0.0 <= float(s.aspect[0, 0]) <= math.pi + 1e-6


def test_bearing_is_positive_to_the_left():
    _, s = situation(ship(), ship(0.0, 400.0))
    assert float(s.bearing[0, 0]) == pytest.approx(math.pi / 2, abs=1e-5)


def test_flow_sign():
    """Both turning left: +1 (two-circle); opposite: -1 (one-circle); one straight: 0."""
    state, _ = situation(ship(), ship(400.0))
    rate = state.constants["sustained_rate"]
    for rates, expected in (((rate, rate), 1.0), ((rate, -rate), -1.0), ((rate, 0.0), 0.0)):
        _, s = situation(ship(), ship(400.0), nose_rates=rates)
        assert float(s.flow[0, 0]) == pytest.approx(expected, abs=0.002), rates
        assert float(s.own_sense[0, 0]) == pytest.approx(1.0, abs=0.002)


def test_sight_rate():
    """He is 400 m ahead flying +y and I fly +x, both at v: relative velocity (-v, v): the
    line of sight turning left at v / 400 rad/s."""
    speed = 200.0
    state, s = situation(ship(speed=speed), ship(400.0, heading=math.pi / 2, speed=speed))
    c = state.constants
    assert float(s.sight_rate[0, 0]) == pytest.approx(speed / 400.0 / c["sustained_rate"], rel=1e-4)


def test_energy_advantage():
    """In store fractions: the store difference plus the kinetic difference (v^2 / 2)."""
    _, s = situation(ship(), ship(400.0), stores=(0.8, 0.3))
    assert float(s.energy_advantage[0, 0]) == pytest.approx(0.5, abs=1e-5)
    assert float(s.store[0, 0]) == pytest.approx(0.8, abs=1e-6)
    state, s = situation(ship(speed=263.0), ship(400.0, speed=210.0))
    kinetic = 0.5 * (263.0**2 - 210.0**2) / state.constants["max_energy"]
    assert float(s.energy_advantage[0, 0]) == pytest.approx(kinetic, rel=1e-4)
    # each ship's view is the mirror of the other's
    both = assess(state, engage(state, None), FLOOR)
    assert float(both.energy_advantage[0, 1]) == pytest.approx(-kinetic, rel=1e-4)


def test_the_target_row_is_gathered():
    """The target's nose rate and store come from its row of the enemies' view (under fog its
    last sighting: lean/game/tests/test_fog.py)."""
    state, _ = situation(ship(), ship(400.0), nose_rates=(0.0, 0.5), stores=(0.5, 0.9))
    fight = engage(state, 0, enemies=core.enemy_view(state))
    assert float(fight.target_nose_rate[0, 0]) == pytest.approx(0.5)
    assert float(fight.target_energy[0, 0]) == pytest.approx(0.9 * state.constants["max_energy"])

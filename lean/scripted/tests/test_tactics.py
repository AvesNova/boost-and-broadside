"""The scripted agent's tactics (docs/rewrite/scripted-bfm-design.md B3 to B5, tests B9):
hand-built situations, one per tactic, read through `explain` (the weights, the pursuit time,
the speed target, the stick cap) and the marginals."""

import math

import torch

from lean.scripted import ScriptedConfig
from lean.scripted.opponent import CLIMB, DIVE, explain
from lean.scripted.tests.test_scripted import placed

LEFT, RIGHT = [2, 4, 6], [1, 3, 5]  # stick indices (left_is_even)
FULL_LEFT, FULL_RIGHT = 6, 5


def constants():
    return placed(dict(x=0.0, y=0.0, heading=0.0, speed=100.0),
                  dict(x=1.0, y=0.0, heading=0.0, speed=100.0)).constants


C = constants()
CORNER, GUN, RADIUS = C["corner_speed"], C["gun_range"], C["sustained_radius"]


def ship(x=0.0, y=0.0, heading=0.0, speed=1.0):
    """A ship at (x, y) metres, heading in radians, speed in corner speeds."""
    return dict(x=x, y=y, heading=heading, speed=speed * CORNER)


def decision(own, target, stores=(0.5, 0.5), nose_rates=(0.0, 0.0), memory=None, config=None):
    state = placed(own, target)
    state = state.replace(
        nose_rate=torch.tensor([nose_rates]),
        energy=torch.tensor([stores]) * C["max_energy"],
    )
    memory = memory or {"target": torch.full((1, 2), -1), "side": torch.ones(1, 2)}
    return explain(config or ScriptedConfig(), state, 0, None, memory)


def first(tensor):
    return float(tensor[0, 0])


def weight(explanation, tactic):
    return first(explanation.weights[tactic])


def test_weights_sum_to_one():
    e = decision(ship(), ship(300.0, 200.0, heading=2.0))
    assert abs(sum(weight(e, name) for name in e.weights) - 1.0) < 1e-5


def test_disengaged_coasts_at_merge_speed_and_banks_speed_with_the_level_capped():
    """Disengaged (6 gun ranges): at corner speed coast; at 1.3 corner speeds climb, the full
    stick level folded into the sustained one (above the sustained stick speed no turn is
    lost, and a climb at full stick banks nothing)."""
    target = ship(6.0 * GUN, 0.0)
    at_corner = decision(ship(), target)
    assert weight(at_corner, "disengaged") > 0.95
    merge = CORNER * ScriptedConfig().values["merge_speed"]
    assert abs(first(at_corner.speed_target) / merge - 1.0) < 0.03
    assert int(at_corner.marginals.power[0, 0].argmax()) == 0
    fast = decision(ship(speed=1.3), target)
    assert int(fast.marginals.power[0, 0].argmax()) == CLIMB
    assert first(fast.cap) > 0.75
    turning = decision(ship(speed=1.3), ship(0.0, 6.0 * GUN))  # target abeam: a hard turn
    assert float(turning.marginals.stick[0, 0, FULL_LEFT]) < 0.25


def test_tail_chase_closing_fast_inside_the_control_zone_lags_and_climbs():
    """He is 300 m ahead flying away at 0.6 corner speeds, I close at 0.3 corner speeds: too
    much closure for the nose-tail distance (a high yo-yo): lag pursuit, climb."""
    e = decision(ship(speed=0.9), ship(300.0, speed=0.6))
    assert weight(e, "offensive") > 0.9
    assert first(e.yo_yo) > 0.3
    assert first(e.pursuit_time) < 0.0
    assert int(e.marginals.power[0, 0].argmax()) == CLIMB


def test_tail_chase_falling_behind_out_of_range_leads_and_dives():
    """He is 1.6 gun ranges ahead and faster, turning: a low yo-yo, lead pursuit, dive to
    corner. Flying straight: the engine closes (coast: no dive, no banking)."""
    rate = C["sustained_rate"]
    e = decision(ship(speed=0.6), ship(1.6 * GUN, speed=0.8), nose_rates=(0.0, rate))
    assert weight(e, "offensive") > 0.9
    assert first(e.yo_yo) < -0.9
    assert first(e.pursuit_time) > 0.0
    assert abs(first(e.speed_target) / CORNER - 1.0) < 0.05
    assert int(e.marginals.power[0, 0].argmax()) == DIVE
    straight = decision(ship(speed=0.6), ship(1.6 * GUN, speed=0.8))
    assert int(straight.marginals.power[0, 0].argmax()) == 0


def test_lag_and_lead_aim_behind_and_ahead_of_a_crossing_target():
    """He crosses left to right ahead of me, nose away: falling behind out of gun range I lead
    (aim right of him), closing fast in range I lag (aim left of him); pure pursuit would
    point at him."""
    heading = -math.pi / 2 + 0.6
    lead = decision(ship(speed=0.5), ship(1.5 * GUN, heading=heading, speed=0.8))
    lag = decision(ship(speed=1.2), ship(0.8 * GUN, heading=heading, speed=0.8))
    assert first(lead.pursuit_time) > 0.0 > first(lag.pursuit_time)
    assert first(lead.stick_error) < first(lead.situation.bearing)
    assert first(lag.stick_error) > first(lag.situation.bearing)


def test_defensive_breaks_into_him_and_spends_the_store():
    """He is on my six, in gun range, nose on: defensive; full stick, corner speed (dive)."""
    e = decision(ship(speed=0.6), ship(-0.6 * GUN, 30.0, speed=0.9))
    assert weight(e, "defensive") > 0.9
    stick = e.marginals.stick[0, 0]
    assert float(stick[FULL_LEFT] + stick[FULL_RIGHT]) > 0.9
    assert abs(first(e.speed_target) / CORNER - 1.0) < 0.02
    assert int(e.marginals.power[0, 0].argmax()) == DIVE


def test_merge_and_turning_quadrants():
    """Head-on, both ahead of each other's 3/9 line: merge; after the pass, flying apart:
    turning (the rate fight at the fight speed)."""
    merge = decision(ship(), ship(3.0 * RADIUS, 2.0 * RADIUS, heading=math.pi))
    assert weight(merge, "merge") > 0.9
    turning = decision(ship(), ship(-3.0 * RADIUS, 0.5 * RADIUS, heading=math.pi))
    assert weight(turning, "turning") > 0.9
    fight = ScriptedConfig().values["fight_speed"] * C["sustained_speed"]
    assert abs(first(turning.speed_target) / fight - 1.0) < 0.05


def test_two_circle_banks_at_the_sustained_level_then_cashes():
    """Turning (both behind each other's 3/9 line), both turning left (two-circle): the energy
    plan eases to the sustained level and banks above the fight speed (Shaw p. 110); as his
    angular advantage grows toward 90 deg it cashes (less cap, a faster speed target)."""
    rate = C["sustained_rate"]
    neutral = decision(ship(speed=0.8), ship(-3.0 * RADIUS, 0.5 * RADIUS, heading=math.pi),
                       nose_rates=(rate, rate))
    assert weight(neutral, "turning") > 0.9 and first(neutral.situation.flow) > 0.9
    assert first(neutral.cap) > 0.9
    assert int(neutral.marginals.power[0, 0].argmax()) == CLIMB
    # he has turned toward me: his advantage near 90 deg (angles off about 1.45 pi)
    late = decision(ship(speed=0.8), ship(-3.0 * RADIUS, 0.0, heading=0.47 * math.pi),
                    nose_rates=(rate, rate))
    assert first(late.situation.angles_off) > 1.4 * math.pi
    assert first(late.cap) < first(neutral.cap) - 0.2
    assert first(late.speed_target) > first(neutral.speed_target) * 1.1


def test_one_circle_inside_a_radius_slows_to_the_floor():
    """One-circle (opposite senses) inside one sustained radius: the scissors, slow to the
    speed floor (Shaw p. 83)."""
    rate = C["sustained_rate"]
    e = decision(ship(speed=0.7), ship(-0.5 * RADIUS, 0.3 * RADIUS, heading=math.pi),
                 nose_rates=(rate, -rate))
    assert weight(e, "turning") > 0.8 and first(e.situation.flow) < -0.9
    floor = ScriptedConfig().values["speed_floor"] * C["stall_speed"]
    assert first(e.speed_target) < 1.1 * floor
    assert int(e.marginals.power[0, 0].argmax()) == CLIMB


def test_defence_is_graded_by_range():
    """He is on my six nose on: inside the defend range a break (full stick, corner speed);
    two gun ranges out a hard turn at the sustained level (Shaw p. 132)."""
    close = decision(ship(speed=0.6), ship(-0.5 * GUN, 20.0, speed=0.9))
    far = decision(ship(speed=0.6), ship(-2.0 * GUN, 20.0, speed=0.9))
    assert weight(far, "defensive") > 0.8
    assert first(close.cap) < 0.1 and first(far.cap) > 0.8
    assert first(close.speed_target) > 1.3 * first(far.speed_target)

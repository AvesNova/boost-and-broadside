"""The scripted agent's tactics (docs/rewrite/scripted-bfm-design.md B3 to B5, tests B9):
hand-built situations, one per tactic, read through `explain` (the weights, the pursuit time,
the speed target, the stick cap) and the marginals."""

import math

import torch

from lean.scripted import ScriptedConfig
from lean.scripted.opponent import CLIMB, DIVE, explain
from lean.scripted.parameters import BFM_DEFAULT_2026_11_24
from lean.scripted.tests.test_scripted import memory_of, placed

LEFT, RIGHT = [2, 4, 6], [1, 3, 5]  # stick indices (left_is_even)
FULL_LEFT, FULL_RIGHT = 6, 5


def constants():
    return placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=100.0), dict(x=1.0, y=0.0, heading=0.0, speed=100.0)
    ).constants


C = constants()
CORNER, GUN, RADIUS = C["corner_speed"], C["gun_range"], C["sustained_radius"]
# A persona with the design's B6 values for the rows the yardstick leaves at 0 or widens
# (yo-yo time, lead turn) and Shaw's graded defence (break inside one gun range, relax at 90
# deg of his advantage).
# It stands on the previous defaults, not on BFM-CMA2: the persona does not move with the yardstick.
BFM = (
    ScriptedConfig()
    .replace(**BFM_DEFAULT_2026_11_24)
    .replace(yo_yo_time=0.5, lead_turn_scale=1.0, relax_angle=1.5 * math.pi, defend_range=1.0)
)


def sustained_speeds(multiple):
    """A speed in corner speeds (as `ship` takes it) that is `multiple` sustained speeds: for a
    situation set against the fight speed (a multiple of the sustained speed), which moves
    against the corner when the flight model changes (2026-11-24, the stick levels and the
    separation drag: 0.555 to 0.626 corner speeds)."""
    return multiple * C["sustained_speed"] / CORNER


def ship(x=0.0, y=0.0, heading=0.0, speed=1.0):
    """A ship at (x, y) metres, heading in radians, speed in corner speeds."""
    return dict(x=x, y=y, heading=heading, speed=speed * CORNER)


def decision(own, target, stores=(0.5, 0.5), nose_rates=(0.0, 0.0), memory=None, config=None):
    state = placed(own, target)
    state = state.replace(
        nose_rate=torch.tensor([nose_rates]),
        energy=torch.tensor([stores]) * C["max_energy"],
    )
    memory = memory or memory_of()
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
    stick level folded into the middle one (above the middle stick speed no turn is
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
    e = decision(ship(speed=0.9), ship(300.0, speed=0.6), config=BFM)
    assert weight(e, "offensive") > 0.9
    assert first(e.yo_yo) > 0.3
    assert first(e.pursuit_time) < 0.0
    assert int(e.marginals.power[0, 0].argmax()) == CLIMB


def test_tail_chase_falling_behind_out_of_range_leads_and_dives():
    """He is 1.6 gun ranges ahead and faster, turning: a low yo-yo, lead pursuit, dive to
    corner. Flying straight: the engine closes (coast: no dive, no banking). I fly 1.08
    sustained speeds (0.6 corner speeds on the flight before 2026-11-24's), he 0.8 corner."""
    rate = C["sustained_rate"]
    own = sustained_speeds(1.08)
    e = decision(ship(speed=own), ship(1.6 * GUN, speed=0.8), nose_rates=(0.0, rate), config=BFM)
    assert weight(e, "offensive") > 0.9
    assert first(e.yo_yo) < -0.9
    assert first(e.pursuit_time) > 0.0
    assert abs(first(e.speed_target) / CORNER - 1.0) < 0.05
    assert int(e.marginals.power[0, 0].argmax()) == DIVE
    straight = decision(ship(speed=own), ship(1.6 * GUN, speed=0.8), config=BFM)
    assert int(straight.marginals.power[0, 0].argmax()) == 0


def test_lag_and_lead_aim_behind_and_ahead_of_a_crossing_target():
    """He crosses left to right ahead of me, nose away (aspect 56 deg): falling behind out of
    gun range I lead (aim right of him), closing fast out of gun range I lag (aim left of him);
    pure pursuit would point at him. Closing fast inside gun range the lag is still asked for,
    but the gun owns the nose (docs/rewrite/lead-turn-yo-yo-review.md)."""
    heading = -math.pi / 2 + 0.6
    lead = decision(ship(speed=0.5), ship(1.5 * GUN, heading=heading, speed=0.8), config=BFM)
    lag = decision(ship(speed=1.8), ship(1.2 * GUN, heading=heading, speed=0.3), config=BFM)
    assert first(lag.situation.aspect) < math.pi / 3
    assert first(lead.pursuit_time) > 0.0 > first(lag.pursuit_time)
    assert first(lead.stick_error) < first(lead.situation.bearing)
    assert first(lag.stick_error) > first(lag.situation.bearing)
    in_range = decision(ship(speed=1.2), ship(0.8 * GUN, heading=heading, speed=0.8), config=BFM)
    assert first(in_range.pursuit_time) < 0.0
    assert abs(first(in_range.stick_error) - first(in_range.gun_error)) < 0.05


def test_no_high_yo_yo_lag_at_the_beam():
    """Closing fast with him crossing at the beam (aspect 90 deg, the 3/9 line): the high
    yo-yo's lag needs him in my rear quarter (Shaw p. 71), so no lag pursuit."""
    yo_yo_only = BFM.replace(lead_turn_scale=0.0)
    crossing = ship(1.2 * GUN, heading=-math.pi / 2, speed=0.5)
    beam = decision(ship(speed=1.2), crossing, config=yo_yo_only)
    assert first(beam.yo_yo) > 0.0
    assert abs(first(beam.pursuit_time)) < 0.05


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
    plan eases to the middle level and banks above the fight speed (Shaw p. 110); as his
    angular advantage grows toward 90 deg it cashes (less cap, a faster speed target). I fly
    1.44 sustained speeds (0.8 corner speeds on the flight before 2026-11-24's)."""
    rate = C["sustained_rate"]
    own = sustained_speeds(1.44)
    neutral = decision(
        ship(speed=own), ship(-3.0 * RADIUS, 0.5 * RADIUS, heading=math.pi), nose_rates=(rate, rate)
    )
    assert weight(neutral, "turning") > 0.9 and first(neutral.situation.flow) > 0.9
    assert first(neutral.cap) > 0.9
    assert int(neutral.marginals.power[0, 0].argmax()) == CLIMB
    # he has turned toward me: his advantage near 90 deg (angles off about 1.45 pi)
    late = decision(
        ship(speed=own), ship(-3.0 * RADIUS, 0.0, heading=0.47 * math.pi), nose_rates=(rate, rate)
    )
    assert first(late.situation.angles_off) > 1.4 * math.pi
    assert first(late.cap) < first(neutral.cap) - 0.2
    assert first(late.speed_target) > first(neutral.speed_target) * 1.1


def test_one_circle_inside_a_radius_slows_to_the_floor():
    """One-circle (opposite senses) inside one sustained radius: the scissors, slow to the
    speed floor (Shaw p. 83)."""
    rate = C["sustained_rate"]
    e = decision(
        ship(speed=0.7),
        ship(-0.5 * RADIUS, 0.3 * RADIUS, heading=math.pi),
        nose_rates=(rate, -rate),
    )
    assert weight(e, "turning") > 0.8 and first(e.situation.flow) < -0.9
    floor = ScriptedConfig().values["speed_floor"] * C["stall_speed"]
    assert first(e.speed_target) < 1.1 * floor
    assert int(e.marginals.power[0, 0].argmax()) == CLIMB


def test_defence_is_graded_by_range():
    """He is on my six nose on: inside the defend range a break (full stick, corner speed);
    two gun ranges out a hard turn at the middle level (Shaw p. 132)."""
    close = decision(ship(speed=0.6), ship(-0.5 * GUN, 20.0, speed=0.9), config=BFM)
    far = decision(ship(speed=0.6), ship(-2.0 * GUN, 20.0, speed=0.9), config=BFM)
    assert weight(far, "defensive") > 0.8
    assert first(close.cap) < 0.1 and first(far.cap) > 0.8
    assert first(close.speed_target) > 1.3 * first(far.speed_target)


def committed(own, target, stores=(0.5, 0.5), nose_rates=(0.0, 0.0), memory=None):
    """The sense committed by one decision (memory "sense") and P(left) of the stick."""
    e = decision(own, target, stores, nose_rates, memory)
    return first(e.sense), float(e.marginals.stick[0, 0, LEFT].sum()), e


def test_the_pass_commits_the_plan_sense():
    """Just after an offset head-on pass (he is aft, on my left, flying straight): with more
    energy two-circle, toward his side (left); with less one-circle, away from it (right).
    Turning: two-circle takes his rotation, one-circle the opposite."""
    passed = ship(-1.0 * RADIUS, 2.0 * RADIUS, heading=math.pi)
    sense, left, _ = committed(ship(), passed, stores=(0.8, 0.3))
    assert sense == 1.0 and left > 0.9
    sense, left, _ = committed(ship(), passed, stores=(0.3, 0.8))
    assert sense == -1.0 and left < 0.1
    rate = C["sustained_rate"]
    for his_rate, stores, expected in ((-rate, (0.8, 0.3), -1.0), (-rate, (0.3, 0.8), 1.0)):
        sense, _, _ = committed(ship(), passed, stores, nose_rates=(0.0, his_rate))
        assert sense == expected, (his_rate, stores)


def test_the_held_sense_turns_the_long_way_and_releases_when_he_comes_forward():
    """One-circle held to the right while he is aft on my left: the stick goes right (the long
    way round); once he is forward of the 3/9 line the sense is released (short way)."""
    memory = memory_of()
    memory["target"] = torch.tensor([[1, 0]])
    memory["sense"] = torch.tensor([[-1.0, 0.0]])
    memory["last_pass"] = torch.tensor([[-4.0, 0.0]])
    aft = ship(-1.0 * RADIUS, 1.5 * RADIUS, heading=math.pi)
    sense, left, _ = committed(ship(), aft, memory=memory)
    assert sense == -1.0 and left < 0.1
    ahead = ship(2.0 * RADIUS, 1.0 * RADIUS, heading=math.pi)
    sense, left, _ = committed(ship(), ahead, memory=memory)
    assert sense == 0.0 and left > 0.9


def test_overshoot_reverses_only_when_close_low_crossing_and_he_is_faster():
    """He is close behind, crossing my six from my left to my right (I hold a left turn)
    fast, his track close to mine, and he is faster: he overshoots, so the sense reverses to
    the short way, flown level (no climb). Slowly crossing, or with me the faster: hold
    (Shaw p. 76: if in doubt, do not reverse)."""

    def held_left():
        memory = memory_of()
        memory["target"] = torch.tensor([[1, 0]])
        memory["sense"] = torch.tensor([[1.0, 0.0]])
        memory["last_pass"] = torch.tensor([[4.0, 0.0]])
        return memory

    # he is 60 m behind, just across my six to the right, flying my way at an angle across
    def behind(speed, heading=-0.5):
        return ship(-60.0, -5.0, heading=heading, speed=speed)

    sense, _, e = committed(ship(speed=0.6), behind(1.0), memory=held_left())
    assert bool(e.overshoot[0, 0]) and sense == -1.0
    assert float(e.marginals.power[0, 0, CLIMB]) < 0.05
    slow = committed(ship(speed=0.6), behind(0.62, heading=-0.02), memory=held_left())
    assert not bool(slow[2].overshoot[0, 0]) and slow[0] == 1.0
    faster_me = committed(ship(speed=1.1), behind(1.0), memory=held_left())
    assert not bool(faster_me[2].overshoot[0, 0]) and faster_me[0] == 1.0


def test_a_pass_that_gained_nothing_switches_the_plan():
    """Shaw pp. 112-113: the last pass was one-circle at 4.0 rad angles off; at this pass he
    is no better off for me (4.0 rad again): switch to two-circle. Had it gained 0.5 rad,
    keep one-circle."""
    passed = ship(-1.0 * RADIUS, 2.0 * RADIUS, heading=math.pi)
    _, _, probe = committed(ship(), passed, stores=(0.3, 0.8))
    angles_off = first(probe.situation.angles_off)

    def memory(last):
        m = memory_of()
        m["target"] = torch.tensor([[1, 0]])
        m["last_pass"] = torch.tensor([[last, 0.0]])
        return m

    stalled, _, _ = committed(ship(), passed, (0.3, 0.8), memory=memory(-angles_off))
    assert stalled == 1.0  # two-circle: toward his side (left)
    gaining, _, e = committed(ship(), passed, (0.3, 0.8), memory=memory(-(angles_off + 0.5)))
    assert gaining == -1.0
    assert first(e.last_pass) == -angles_off


def test_lead_turn_at_an_offset_merge():
    """Head-on with a lateral offset of two sustained radii (Shaw Fig 2-9): with more energy
    (two-circle plan) the ship starts its turn toward his track before the pass, the lead
    turn; with less (one-circle) it flies at him (pure pursuit or the gun)."""
    him = ship(6.0 * RADIUS, 2.0 * RADIUS, heading=math.pi)
    two = decision(ship(), him, stores=(0.8, 0.3), config=BFM)
    one = decision(ship(), him, stores=(0.3, 0.8), config=BFM)
    gun = first(two.gun_error)
    assert weight(two, "merge") > 0.7  # 0.74 with the design-system P-51D (0.75 before)
    assert first(two.pursuit_time) > 1.0 and abs(first(one.pursuit_time)) < 0.05
    assert first(two.stick_error) > gun + 0.15
    assert abs(first(one.stick_error) - first(one.gun_error)) < 0.05
    # no offset, no lead turn
    head_on = ship(6.0 * RADIUS, 0.0, heading=math.pi)
    straight = decision(ship(), head_on, stores=(0.8, 0.3), config=BFM)
    assert abs(first(straight.pursuit_time)) < 0.1

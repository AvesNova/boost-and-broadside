"""The scripted agent's tactics (docs/rewrite/scripted-bfm-design.md B3 to B5, tests B9):
hand-built situations, one per tactic, read through `explain` (the weights, the speed target,
the stick cap) and the marginals."""

import math

import torch

from lean.scripted import ScriptedConfig
from lean.scripted.opponent import CLIMB, DIVE, explain
from lean.scripted.parameters import BFM_DEFAULT_2026_11_24
from lean.scripted.tests.test_scripted import LEFT, RIGHT, memory_of, placed

FULL_LEFT, FULL_RIGHT = LEFT[-1], RIGHT[-1]  # the most lateral stick each way


def constants():
    return placed(
        dict(x=0.0, y=0.0, heading=0.0, speed=100.0), dict(x=1.0, y=0.0, heading=0.0, speed=100.0)
    ).constants


C = constants()
CORNER, GUN, RADIUS = C["corner_speed"], C["gun_range"], C["sustained_radius"]
# The previous defaults (fight speed 1), not BFM-CMA2: the chase test is set against them.
BFM = ScriptedConfig().replace(**BFM_DEFAULT_2026_11_24)


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


def test_the_chase_speed_rises_against_a_turning_target():
    """He is 1.6 gun ranges ahead and faster. Turning: the chase speed is the fight speed
    raised by corner over sustained speed (corner speed at fight speed 1), so dive. Flying
    straight: the fight speed, and the engine closes (coast: no dive, no banking). Either
    way the stick points at the gun's lead point. I fly 1.08 sustained speeds, he 0.8
    corner."""
    rate = C["sustained_rate"]
    own = sustained_speeds(1.08)
    e = decision(ship(speed=own), ship(1.6 * GUN, speed=0.8), nose_rates=(0.0, rate), config=BFM)
    assert weight(e, "offensive") > 0.9
    assert abs(first(e.speed_target) / CORNER - 1.0) < 0.05
    assert int(e.marginals.power[0, 0].argmax()) == DIVE
    assert abs(first(e.stick_error) - first(e.gun_error)) < 1e-4
    straight = decision(ship(speed=own), ship(1.6 * GUN, speed=0.8), config=BFM)
    assert abs(first(straight.speed_target) / C["sustained_speed"] - 1.0) < 0.05
    assert int(straight.marginals.power[0, 0].argmax()) == 0


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


def test_the_turn_after_the_pass_is_full_stick_in_either_geometry():
    """Author 2026-11-26: once he passes abeam the ship turns at full stick at once, no wide
    circles. Turning (both behind each other's 3/9 line) with the same rotation (two-circle)
    or opposite rotations (one-circle): no cap, the full level carries the turn. Above the
    fight speed (1.44 sustained speeds) the power still climbs."""
    rate = C["sustained_rate"]
    own = sustained_speeds(1.44)
    for his_rate in (rate, -rate):
        e = decision(
            ship(speed=own),
            ship(-3.0 * RADIUS, 0.5 * RADIUS, heading=math.pi),
            nose_rates=(rate, his_rate),
        )
        assert weight(e, "turning") > 0.9
        assert first(e.cap) < 0.01
        stick = e.marginals.stick[0, 0]
        assert float(stick[FULL_LEFT] + stick[FULL_RIGHT]) > 0.9, his_rate
        assert int(e.marginals.power[0, 0].argmax()) == CLIMB


def test_the_pass_commits_a_full_stick_turn_at_once():
    """The first tick he is aft of my 3/9 line (an offset head-on pass, both straight): the
    sense is committed and the stick is the full level in that sense, for either plan."""
    passed = ship(-0.2 * RADIUS, 0.5 * RADIUS, heading=math.pi)
    for stores, expected in (((0.8, 0.3), 1.0), ((0.1, 0.8), -1.0)):
        sense, _, e = committed(ship(), passed, stores)
        assert sense == expected
        stick = e.marginals.stick[0, 0]
        full = FULL_LEFT if expected > 0 else FULL_RIGHT
        assert float(stick[full]) > 0.9, stores


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


def test_defence_breaks_at_any_range():
    """He is on my six nose on: a break (full stick, corner speed) close in and two gun ranges
    out alike (the graded defence was removed after the reach check)."""
    close = decision(ship(speed=0.6), ship(-0.5 * GUN, 20.0, speed=0.9))
    far = decision(ship(speed=0.6), ship(-2.0 * GUN, 20.0, speed=0.9))
    assert weight(far, "defensive") > 0.8
    assert first(close.cap) < 0.1 and first(far.cap) < 0.1
    assert abs(first(far.speed_target) / CORNER - 1.0) < 0.02


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


def plan_memory(plan, sense=0.0):
    """A memory with the target kept and a held plan (plan x angles off at its choice)."""
    m = memory_of()
    m["target"] = torch.tensor([[1, 0]])
    m["plan"] = torch.tensor([[plan, 0.0]])
    m["sense"] = torch.tensor([[sense, 0.0]])
    return m


def test_the_plan_is_held_across_passes_against_the_energy():
    """Author 2026-11-26: the plan is chosen once per engagement and kept. A one-circle plan
    chosen at this same angles off: at the next pass the energy says two-circle (more store),
    but the plan holds (one-circle: away from his side, right) and its reference is unchanged.
    Released (no plan) only when the target changes."""
    passed = ship(-1.0 * RADIUS, 2.0 * RADIUS, heading=math.pi)
    _, _, probe = committed(ship(), passed, stores=(0.8, 0.3))
    assert first(probe.sense) == 1.0  # fresh: the energy plan, two-circle (left)
    angles_off = first(probe.situation.angles_off)
    sense, left, e = committed(ship(), passed, (0.8, 0.3), memory=plan_memory(-angles_off))
    assert sense == -1.0 and left < 0.1
    assert first(e.plan) == -angles_off
    # The sense released (he came forward) does not release the plan.
    ahead = ship(2.0 * RADIUS, 1.0 * RADIUS, heading=math.pi)
    _, _, forward = committed(ship(), ahead, (0.8, 0.3), memory=plan_memory(-angles_off, -1.0))
    assert first(forward.sense) == 0.0 and first(forward.plan) == -angles_off
    # A new target: the plan is chosen afresh.
    other = plan_memory(-angles_off)
    other["target"] = torch.tensor([[0, 0]])
    sense, _, fresh = committed(ship(), passed, (0.8, 0.3), memory=other)
    assert sense == 1.0 and first(fresh.plan) == angles_off


def test_a_plan_that_is_clearly_losing_switches():
    """The switch condition: at a pass, my angles off have grown by more than the switch loss
    since the plan was chosen (he gained that much angle on me under it): switch, and the
    reference is the angles off now. Grown by less: keep."""
    passed = ship(-1.0 * RADIUS, 2.0 * RADIUS, heading=math.pi)
    _, _, probe = committed(ship(), passed, stores=(0.3, 0.8))
    angles_off = first(probe.situation.angles_off)
    loss = ScriptedConfig().values["switch_loss"]
    for plan, expected in ((-1.0, 1.0), (1.0, -1.0)):
        chosen = plan * (angles_off - 1.1 * loss)
        sense, _, e = committed(ship(), passed, (0.3, 0.8), memory=plan_memory(chosen))
        assert sense == expected
        assert abs(first(e.plan) - expected * angles_off) < 1e-6
        kept = plan * (angles_off - 0.9 * loss)
        sense, _, e = committed(ship(), passed, (0.3, 0.8), memory=plan_memory(kept))
        assert sense == -expected and first(e.plan) == float(torch.tensor(kept))


def test_the_lean_and_the_personas_keep_a_mix_of_plans():
    """Author 2026-11-26: the population keeps a mix of one- and two-circle fights. The same
    offset pass at equal energy over many ships, each with its drawn lean (fresh_memory) and a
    persona at spread 1: both plans are chosen, each by at least a tenth; with more energy
    two-circle is chosen more often."""
    from lean.scripted import fresh_memory, personas

    draws = 300
    generator = torch.Generator().manual_seed(3)
    shares = []
    for stores in ((0.5, 0.5), (0.8, 0.4)):
        state = placed(ship(), ship(-1.0 * RADIUS, 2.0 * RADIUS, heading=math.pi))
        state = state.replace(energy=torch.tensor([stores]) * C["max_energy"])
        two_circle = 0
        for _ in range(draws):
            memory = fresh_memory(1, 2, generator, "cpu")
            persona = personas(ScriptedConfig(), 1, 2, generator, "cpu", 1.0)
            sense = first(explain(ScriptedConfig(), state, 0, persona, memory).sense)
            assert sense != 0.0
            two_circle += sense > 0.0  # two-circle: toward his side (left)
        shares.append(two_circle / draws)
    assert 0.1 < shares[0] < 0.9, shares
    assert shares[1] > shares[0], shares

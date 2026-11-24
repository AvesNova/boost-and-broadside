"""An analysis copy of the opponent's explain with switches (docs/rewrite/scripted-reach.md).

`explain` here is the opponent's `explain` line for line, with two additions: `off`, a set of
mechanism names whose term is forced to its neutral value, and the intermediate quantities
returned in `probe` (gate values, tactic weights, the overshoot conditions). With `off` empty
it gives the opponent's marginals (`reach.py check` compares them). The hot path in lean/ is
not touched. Since the trim (scripted-reach.md "Trim") it reads the frozen copy of the
untrimmed agent in `untrimmed/`, the agent this page measured.

Mechanisms (`OFF`), each switched off as stated:
  overshoot       the overshoot rule never fires
  spend           no spending: the "shot near" product is 0
  spend_need      the spend_need gate always open (1)
  spend_range     the spend_range gate always open (1)
  spend_share     the shot_near gate (centred on spend_share) always open (1)
  switch          the plan switch on gain never switches (the last pass's plan is kept)
  scissors        the scissors gate shut (0): one-circle stays at the fight speed
  cash            the cash gate shut (0): two-circle never cashes
  relax           the relax gate open (1): a break never relaxes to a hard turn
  defend          the defend gate open (1): the break at any range
  boundary        no boundary steer for a ship with a target (centre weight 0)
  speed_floor     the speed target is not clamped at the floor
Row-valued switches (weakest_weight 0, turning_floor near 0, overshoot_crossing widened) are
row overrides in `reach.py`, not names here.
"""

import math

import torch
from untrimmed.opponent import (
    TINY,
    Marginals,
    _cos,
    _gates,
    _last,
    _log,
    _per_other,
    engage,
)
from untrimmed.parameters import GATES, STICK_GATES, ScriptedConfig, value
from untrimmed.situation import assess, signed_angle

from lean.game.core import enemy_view

OFF = (
    "overshoot",
    "spend",
    "spend_need",
    "spend_range",
    "spend_share",
    "switch",
    "scissors",
    "cash",
    "relax",
    "defend",
    "boundary",
    "speed_floor",
)


def explain(config: ScriptedConfig, state, team, persona, memory, off=frozenset()):
    """(Marginals, probe dict): lean.scripted.opponent.explain with mechanisms `off`."""
    unknown = set(off) - set(OFF)
    if unknown:
        raise ValueError(f"unknown switches {sorted(unknown)}")
    probe = {}
    c = state.constants
    enemies = enemy_view(state)
    gun_range = c["gun_range"]

    def read(name):
        return value(config, persona, name)

    fight = engage(
        state,
        team,
        read("weakest_weight"),
        memory["target"],
        read("target_keep_factor"),
        clearance=_per_other(read("ally_clearance") * c["hit_radius"]),
        enemies=enemies,
    )
    has_target = fight.has_target
    safe_distance = fight.distance.clamp_min(TINY)
    range_ratio = safe_distance / gun_range
    situation = assess(state, fight, read("turning_floor"))
    probe["has_target"] = has_target
    probe["target"] = fight.target
    probe["own_sense"] = situation.own_sense
    probe["target_sense"] = situation.target_sense

    lead_distance = range_ratio.clamp_max(read("lead_cap"))
    bullet_drag = c["bullet_drag"]
    flight_time = torch.expm1(bullet_drag * gun_range * lead_distance) * (
        read("lead_gain") / (bullet_drag * c["bullet_launch_speed"])
    )
    if fight.index is not None:
        flight_time = flight_time * fight.index
    aim_x = fight.offset_x + (fight.target_velocity_x - fight.velocity_x) * flight_time
    aim_y = fight.offset_y + (fight.target_velocity_y - fight.velocity_y) * flight_time
    gun_error = signed_angle(state.heading_cos, state.heading_sin, aim_x, aim_y)
    turn_needed = gun_error.abs()

    target_size = (2.0 * torch.atan(c["hit_radius"] / safe_distance)).clamp(
        read("target_size_minimum"), read("target_size_maximum")
    )
    fight_speed = read("fight_speed") * c["sustained_speed"]
    gate = _gates(
        config,
        persona,
        GATES,
        {
            "engaged": range_ratio,
            "front": situation.bearing.abs(),
            "rear": situation.aspect,
            "defend": range_ratio,
            "shot_near": turn_needed / (situation.store * c["store_turn"] + TINY),
            "spend_need": turn_needed,
            "spend_range": range_ratio,
            "cash": situation.angles_off,
            "relax": situation.angles_off,
            "scissors": safe_distance / c["sustained_radius"],
            "level_cap": state.speed / c["middle_stick_speed"],
            "aligned": turn_needed / target_size,
            "in_range": range_ratio,
        },
    )
    gate = dict(gate)
    probe["gate"] = dict(gate)
    if "spend_need" in off:
        gate["spend_need"] = torch.ones_like(gate["spend_need"])
    if "spend_range" in off:
        gate["spend_range"] = torch.ones_like(gate["spend_range"])
    if "spend_share" in off:
        gate["shot_near"] = torch.ones_like(gate["shot_near"])
    if "scissors" in off:
        gate["scissors"] = torch.zeros_like(gate["scissors"])
    if "cash" in off:
        gate["cash"] = torch.zeros_like(gate["cash"])
    if "relax" in off:
        gate["relax"] = torch.ones_like(gate["relax"])
    if "defend" in off:
        gate["defend"] = torch.ones_like(gate["defend"])

    engaged = torch.where(has_target, gate["engaged"], 0.0)
    ahead, astern = engaged * gate["front"], gate["rear"]
    behind = engaged - ahead
    weights = {
        "disengaged": 1.0 - engaged,
        "merge": ahead * (1.0 - astern),
        "offensive": ahead * astern,
        "turning": behind * astern,
        "defensive": behind * (1.0 - astern),
    }
    probe["weights"] = weights

    same_target = has_target & (fight.target == memory["target"])
    last_pass = torch.where(same_target, memory["last_pass"], 0.0)
    energy_plan = torch.where(situation.energy_advantage > read("plan_threshold"), 1.0, -1.0)
    previous_plan = torch.sign(last_pass)
    gained = last_pass.abs() - situation.angles_off
    switched = gained < read("switch_gain")
    if "switch" in off:
        switched = torch.zeros_like(switched)
    kept_plan = torch.where(switched, -previous_plan, previous_plan)
    plan = torch.where(last_pass != 0.0, kept_plan, energy_plan)
    probe["plan_switched"] = (last_pass != 0.0) & switched

    near = gate["shot_near"] * gate["spend_need"] * gate["spend_range"]
    if "spend" in off:
        near = torch.zeros_like(near)
    probe["near"] = near
    log_corner = _log(c["corner_speed"])
    log_fight = _log(fight_speed)
    log_merge = _log(read("merge_speed") * c["corner_speed"])
    log_floor = _log(read("speed_floor") * c["stall_speed"])
    turning_target = situation.target_sense.abs()
    log_chase_speed = log_fight + turning_target * _log(c["corner_speed"] / c["sustained_speed"])
    two_circle = 0.5 * (1.0 + situation.flow)
    probe["two_circle"] = two_circle
    cash, break_turn = gate["cash"], gate["defend"] * gate["relax"]
    log_two_circle = log_fight + cash * (log_corner - log_fight)
    log_one_circle = log_fight + gate["scissors"] * (log_floor - log_fight)
    log_turning = log_one_circle + two_circle * (log_two_circle - log_one_circle)
    log_target = (
        (weights["disengaged"] + weights["merge"]) * log_merge
        + weights["offensive"] * (log_chase_speed + near * (log_corner - log_chase_speed))
        + weights["turning"] * (log_turning + near * (log_corner - log_turning))
        + weights["defensive"] * (log_fight + break_turn * (log_corner - log_fight))
    )
    probe["log_target_raw"] = log_target
    probe["log_floor"] = log_floor
    if "speed_floor" not in off:
        log_target = log_target.clamp_min(log_floor)
    cap = (
        weights["disengaged"] * gate["level_cap"]
        + weights["turning"] * two_circle * (1.0 - cash)
        + weights["defensive"] * (1.0 - break_turn)
    )
    no_bank = weights["offensive"] * (1.0 - turning_target)

    distance_from_centre = torch.sqrt(state.x * state.x + state.y * state.y).clamp_min(TINY)
    in_corner_speeds = state.speed / c["corner_speed"]
    turn_radius = c["sustained_radius"] * (in_corner_speeds * in_corner_speeds).clamp_min(1.0)
    margin = read("boundary_margin") * turn_radius
    depth = (distance_from_centre - (c["arena_radius"] - margin)) / margin
    ramp_low = read("boundary_ramp_low")
    ramp = (depth - ramp_low) / (read("boundary_ramp_high") - ramp_low)
    outward = state.x * fight.velocity_x + state.y * fight.velocity_y > 0.0
    centre_weight = torch.where(outward & has_target, ramp.clamp(0.0, 1.0), 0.0)
    if "boundary" in off:
        centre_weight = torch.zeros_like(centre_weight)
    probe["centre_weight"] = centre_weight
    centre_weight = torch.where(has_target, centre_weight, 1.0)
    point_length = torch.sqrt(aim_x * aim_x + aim_y * aim_y).clamp_min(TINY)
    point_scale = (1.0 - centre_weight) / point_length
    centre_scale = centre_weight / distance_from_centre
    wanted_x = point_scale * aim_x - centre_scale * state.x
    wanted_y = point_scale * aim_y - centre_scale * state.y
    stick_error = signed_angle(state.heading_cos, state.heading_sin, wanted_x, wanted_y)

    line = read("three_nine_line")
    bearing_size = situation.bearing.abs()
    aft = has_target & (bearing_size > line)
    forward = bearing_size < line - read("release_band")
    held = memory["sense"]
    held = torch.where(same_target & ~forward, held, 0.0)
    bearing_rate = situation.sight_rate * c["sustained_rate"] - state.nose_rate
    crossing_cos = (
        state.heading_cos * fight.target_heading_cos + state.heading_sin * fight.target_heading_sin
    )
    conditions = {
        "aft_held": aft & (held != 0.0),
        "from_my_side": bearing_rate * held > 0.0,
        "rate": bearing_rate.abs() > read("overshoot_rate") * c["sustained_rate"],
        "range": fight.distance < read("overshoot_range") * c["sustained_radius"],
        "crossing": crossing_cos > _cos(read("overshoot_crossing")),
        "slower": state.speed <= fight.target_speed,
    }
    overshoot = conditions["aft_held"]
    for name in ("from_my_side", "rate", "range", "crossing", "slower"):
        overshoot = overshoot & conditions[name]
    probe["overshoot_conditions"] = conditions
    probe["overshoot"] = overshoot
    if "overshoot" in off:
        overshoot = torch.zeros_like(overshoot)
    short_way = torch.where(situation.bearing > 0.0, 1.0, -1.0)
    held = torch.where(overshoot, short_way, held)
    commit = aft & (held == 0.0)
    probe["commit"] = commit
    his_turn = torch.where(
        situation.target_sense.abs() > 0.5, torch.sign(situation.target_sense), 0.0
    )
    dead_astern = math.pi - bearing_size < read("tie_band")
    his_side = torch.where(dead_astern, memory["side"], short_way)
    two_circle_sense = torch.where(his_turn != 0.0, his_turn, his_side)
    sense = torch.where(commit, plan * two_circle_sense, held)

    turn_size = stick_error.abs()
    stick_gate = _gates(
        config,
        persona,
        STICK_GATES,
        {"turn": turn_size, "level_two": turn_size, "level_three": turn_size},
    )
    turn, above_one = stick_gate["turn"], stick_gate["level_two"]
    above_two = stick_gate["level_three"]
    turn_above_one = turn * above_one
    full = turn_above_one * above_two * (1.0 - cap)
    levels = torch.stack([turn - turn_above_one, turn_above_one - full, full], -1)
    holding = aft & (sense != 0.0) & (centre_weight <= 0.0)
    turn_left = torch.where(holding, sense > 0.0, stick_error > 0.0)
    side_floor = read("side_floor")
    left = torch.where(turn_left, 1.0 - side_floor, side_floor)[..., None]
    left_levels = left * levels
    sides = (
        (left_levels, levels - left_levels)
        if config.left_is_even
        else (levels - left_levels, left_levels)
    )
    stick = torch.cat(
        [(1.0 - turn)[..., None], torch.stack([sides[1], sides[0]], -1).flatten(-2)], -1
    )

    excess = torch.log(state.speed) - log_target
    band = read("speed_band")
    sharpness = 1.0 / (read("power_scale") * read("temperature"))
    climb_dive = torch.sigmoid(torch.stack([excess - band, -excess - band], -1) * _last(sharpness))
    bottom = state.energy * c["inverse_fade"]
    top = c["inverse_fade"] / c["inverse_max_energy"] - bottom
    room = torch.stack([top, bottom], -1).clamp(0.0, 1.0)
    climb, dive = (climb_dive * room).unbind(-1)
    climb = torch.where(overshoot, 0.0, climb * (1.0 - no_bank))
    dive = dive * (weights["offensive"] + weights["turning"] + weights["defensive"])
    power_floor = _last(read("power_floor"))
    power = power_floor + (1.0 - 3.0 * power_floor) * torch.stack(
        [1.0 - dive - climb, dive, climb], -1
    )

    ready = (state.cooldown <= 0) & ~fight.ally_in_line
    fire = torch.where(has_target & ready, gate["aligned"] * gate["in_range"], 0.0)
    return Marginals(stick, power, fire, fight.controlled, fight.target), probe

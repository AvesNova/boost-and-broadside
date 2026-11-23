"""The scripted opponent: Imugi's combat core without doctrines, memory layers or Frontline.

One decision per living ship per call, read from its team's view of the game (fog on: enemies
as last seen, `lean.game.core.enemy_view`): the nearest remembered enemy kept with a factor,
a drag-law lead in the medium, a logistic stick choice toward the lead point, logistic power
and gun gates, a steer toward the centre near the arena edge, and one ordinary uniform draw
per action head. Every ship characteristic is read from the game's constants
(`state.constants`) by its name. Everything is (E, N): E environments, N ship slots.

Action indices follow `lean.contract.State`: stick 0 straight, then (1, 2) fine, (3, 4)
sustained, (5, 6) full; power 0 coast, 1 dive (store into speed), 2 climb (speed into
store); fire 0 or 1.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lean.contract import random_keywords
from lean.game.core import built_once, enemy_view, slot_indices

from .parameters import GATES, STICK_GATES, Gate, ScriptedConfig, value
from .situation import Situation, assess, signed_angle

COAST, DIVE, CLIMB = 0, 1, 2
TINY = 1e-6  # keeps lengths and ratios away from zero
LOGIT_LIMIT = 30.0


def draw_uniforms(
    envs: int, slots: int, generator: torch.Generator | None, device: torch.device | str
) -> Tensor:
    """The random draw one `act` call needs: (E, N, 3) float32 uniform on [0, 1), one
    per action head [stick, power, fire]."""
    return torch.rand((envs, slots, 3), **random_keywords(generator, device))


def fresh_memory(
    envs: int, slots: int, generator: torch.Generator | None, device: torch.device | str
) -> dict[str, Tensor]:
    """The opponent's memory at the start of a match, which the caller carries from call to
    call (every caller: collect, evaluation, the sandbox), (E, N) each:
      "target" long, the kept target's slot (-1 for none);
      "side" float32, +1 left or -1 right, drawn once per ship per match: the tie-break
        when the plan's turn sense has no side to go by;
      "sense" float32, the committed turn sense while he is aft of my 3/9 line: +1 left,
        -1 right, 0 none;
      "last_pass" float32, the plan and the angles off at the last commit: plan (+1
        two-circle, -1 one-circle) x angles off (radians, > 0); 0 before the first."""
    left = torch.rand((envs, slots), **random_keywords(generator, device)) < 0.5
    zeros = torch.zeros((envs, slots), device=device)
    return {
        "target": torch.full((envs, slots), -1, dtype=torch.long, device=device),
        "side": torch.where(left, 1.0, -1.0),
        "sense": zeros,
        "last_pass": zeros.clone(),
    }


def renew_memory(
    memory: dict[str, Tensor], done: Tensor, generator: torch.Generator | None
) -> dict[str, Tensor]:
    """The memory with the environments in `done` (E,) bool started afresh (a new match):
    no kept target, no sense or pass, a new side. Draws for every environment (no host
    read)."""
    envs, slots = memory["target"].shape
    fresh = fresh_memory(envs, slots, generator, memory["target"].device)
    return {name: torch.where(done[:, None], fresh[name], memory[name]) for name in memory}


def map_velocity(state) -> tuple[Tensor, Tensor, Tensor | None]:
    """Each ship's velocity over the map (E, N), metres per second: the stored proper velocity
    over the refractive index n at the ship; and n (E, N), or None without fields (n = 1)."""
    if state.fields.shape[-1] == 0:
        return state.velocity_x, state.velocity_y, None
    index = torch.exp(state.log_index)
    return state.velocity_x / index, state.velocity_y / index, index


@dataclass
class Engagement:
    """Each ship's target geometry, (E, N) each; values are arbitrary where
    `has_target` is False."""

    controlled: Tensor  # bool: rows of the acting team that exist and are alive
    has_target: Tensor  # bool: controlled and a living enemy is known
    target: Tensor  # long: the target's slot
    offset_x: Tensor  # metres, target minus own position
    offset_y: Tensor
    distance: Tensor  # metres
    target_velocity_x: Tensor  # metres per second over the map
    target_velocity_y: Tensor
    target_heading_cos: Tensor  # the target's nose direction
    target_heading_sin: Tensor
    target_speed: Tensor  # proper speed, metres per second
    target_nose_rate: Tensor  # radians per second, positive to the left
    target_energy: Tensor  # its store
    velocity_x: Tensor  # own velocity over the map
    velocity_y: Tensor
    ally_in_line: Tensor  # bool: a living team-mate is in the line of fire, nearer than the target
    index: Tensor | None  # the refractive index at the ship; None without fields (1)


def engage(
    state,
    team: int | None,
    weakest_weight: Tensor | float = 0.0,
    kept_target: Tensor | None = None,
    keep_factor: float = 1.0,
    clearance: Tensor | float = 24.0,
    enemies=None,
) -> Engagement:
    """Choose each ship's target: the living enemy with the smallest score, where the
    score is the distance times (1 + weakest_weight * the enemy's store fraction; skipped
    when the weight is the number 0). An enemy equal to `kept_target` has its score
    multiplied by `keep_factor` (below 1), so it is kept unless another scores better by that
    factor. A team-mate blocks the shot when it is ahead along the nose, nearer than the
    target, and within `clearance` metres of the line of fire.

    team: the team whose rows act, or None for every ship of both teams (each against the
    other team). Own ships and team-mates are read from `state`; enemies from `enemies`, the
    state with every ship as its enemy team knows it (`lean.game.core.enemy_view`; default
    `state`, as for a team view or with fog off)."""
    enemies = state if enemies is None else enemies
    present = state.exists & state.alive
    controlled = present if team is None else present & (state.team == team)
    same_team = state.team[:, :, None] == state.team[:, None, :]  # (E, own, other)
    enemy = (enemies.exists & enemies.alive)[:, None, :] & ~same_team
    velocity_x, velocity_y, index = map_velocity(state)
    enemy_x = enemies.x[:, None, :] - state.x[:, :, None]  # (E, own, other)
    enemy_y = enemies.y[:, None, :] - state.y[:, :, None]
    if enemies is state:
        offset_x, offset_y = enemy_x, enemy_y
    else:
        offset_x = state.x[:, None, :] - state.x[:, :, None]
        offset_y = state.y[:, None, :] - state.y[:, :, None]
    score = enemy_x * enemy_x + enemy_y * enemy_y
    if not (isinstance(weakest_weight, float) and weakest_weight == 0.0):
        store = (enemies.energy * enemies.constants["inverse_max_energy"])[:, None, :]
        preference = 1.0 + weakest_weight[..., None] * store
        score = score * preference * preference
    slots = slot_indices(state.x.shape[1], state.x.device).ship
    if kept_target is not None:
        score = torch.where(slots == kept_target[..., None], score * keep_factor**2, score)
    score = torch.where(enemy, score, math.inf)
    target = score.argmin(-1)
    if enemies is state:
        enemy_velocity_x, enemy_velocity_y = velocity_x, velocity_y
    else:
        enemy_velocity_x, enemy_velocity_y, _ = map_velocity(enemies)
    columns = (
        enemies.x, enemies.y, enemy_velocity_x, enemy_velocity_y, enemies.heading_cos,
        enemies.heading_sin, enemies.speed, enemies.nose_rate, enemies.energy,
    )
    features = torch.stack(columns, -1)  # (E, N, 9)
    chosen = torch.gather(features, 1, target[..., None].expand(-1, -1, len(columns)))
    (target_x, target_y, target_velocity_x, target_velocity_y, target_heading_cos,
     target_heading_sin, target_speed, target_nose_rate, target_energy) = chosen.unbind(-1)
    target_x = target_x - state.x
    target_y = target_y - state.y
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    along = offset_x * state.heading_cos[..., None] + offset_y * state.heading_sin[..., None]
    across = offset_x * state.heading_sin[..., None] - offset_y * state.heading_cos[..., None]
    ally = present[:, None, :] & same_team & (slots[:, None] != slots[None, :])
    blocking = ally & (along > 0.0) & (along < distance[..., None]) & (across.abs() < clearance)
    return Engagement(
        controlled=controlled,
        has_target=controlled & enemy.any(-1),
        target=target,
        offset_x=target_x,
        offset_y=target_y,
        distance=distance,
        target_velocity_x=target_velocity_x,
        target_velocity_y=target_velocity_y,
        target_heading_cos=target_heading_cos,
        target_heading_sin=target_heading_sin,
        target_speed=target_speed,
        target_nose_rate=target_nose_rate,
        target_energy=target_energy,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        ally_in_line=blocking.any(-1),
        index=index,
    )


@dataclass
class Marginals:
    """The three independent action distributions of one decision; they are also the
    soft labels for behaviour cloning. Rows that are not controlled are arbitrary."""

    stick: Tensor  # (E, N, 7), sums to 1
    power: Tensor  # (E, N, 3), sums to 1: coast, dive, climb
    fire: Tensor  # (E, N), probability of firing
    controlled: Tensor  # (E, N) bool
    target: Tensor  # (E, N) long, the chosen target's slot


TACTICS = ("disengaged", "merge", "offensive", "turning", "defensive")


@dataclass
class Explanation:
    """One decision with its reasons, (E, N) each: what `marginals` returns and what led to
    it (for tests, measurements and the sandbox)."""

    marginals: Marginals
    situation: Situation
    weights: dict[str, Tensor]  # by TACTICS, sum to 1
    yo_yo: Tensor  # y: +1 a full high yo-yo (lag, climb) .. -1 a full low one (lead, dive)
    pursuit_time: Tensor  # tau, seconds: > 0 lead, 0 pure, < 0 lag pursuit
    gun_weight: Tensor  # the share of the wanted point that is the gun's lead point
    speed_target: Tensor  # v*, metres per second
    cap: Tensor  # the share of the full stick level folded into the sustained one
    stick_error: Tensor  # radians to the wanted direction, positive left
    gun_error: Tensor  # radians to the gun's lead point, positive left
    sense: Tensor  # the committed turn sense after this decision (memory "sense")
    last_pass: Tensor  # memory "last_pass" after this decision
    overshoot: Tensor  # bool: he overshot (the sense reversed to the short way)


def _per_other(value: Tensor | float) -> Tensor | float:
    """A per-ship value (E, N) laid along the "other ship" axis of (E, own, other) pairs."""
    return value[:, None, :] if isinstance(value, Tensor) else value


def marginals(
    config: ScriptedConfig,
    state,
    team: int | None,
    persona: dict[str, Tensor] | None,
    memory: dict[str, Tensor],
) -> Marginals:
    """The opponent's action distributions for the rows of `team`, or of every ship of both
    teams when `team` is None (one call for both: each ship decides exactly as in its own
    team's call).

    state: `lean.contract.State`; persona: from `personas` (None: the yardstick); memory:
    from `fresh_memory`, as `act` last returned it. Under fog each ship reads only its team's
    view: its own team as it is, enemies at their last sighting (`enemy_view`), none never
    seen.
    """
    return explain(config, state, team, persona, memory).marginals


def explain(
    config: ScriptedConfig,
    state,
    team: int | None,
    persona: dict[str, Tensor] | None,
    memory: dict[str, Tensor],
) -> Explanation:
    """`marginals` with its reasons. The stages (docs/rewrite/scripted-bfm-design.md B1):
    target, assessment, tactic weights, commands mixed as scalars (pursuit time, speed
    target, stick cap), control laws (stick, power, gun)."""
    c = state.constants
    enemies = enemy_view(state)
    gun_range = c["gun_range"]

    def read(name: str) -> Tensor | float:
        return value(config, persona, name)

    # Target.
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

    # The gun's lead point: where the target is, relative to the shooter, when a bullet fired
    # now arrives. The bullet inherits the shooter's velocity, so the relative velocity
    # carries it. In a medium of index n a bullet's map speed is its proper speed over n, so
    # its time of flight is n times longer (the shooter's index taken for the whole path).
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

    # The fight's gates in one logistic.
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
            "level_cap": state.speed / c["sustained_stick_speed"],
            "yo_yo_cap": state.speed / fight_speed,
            "aligned": turn_needed / target_size,
            "in_range": range_ratio,
        },
    )

    # Tactic weights: engaged or not, then the quadrant by the 3/9 line (he is ahead of me;
    # I am behind his 3/9 line).
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

    # Yo-yo signal: closure against the closure that reaches the nose-tail distance in the
    # closing time (corner speeds); > 0 too much closure, < 0 falling behind.
    wanted_closure = (fight.distance - read("nose_tail") * c["sustained_radius"]) / (
        read("closing_time") * c["quarter_turn_time"] * c["corner_speed"]
    )
    yo_yo = torch.tanh((situation.closure - wanted_closure) / read("yo_yo_scale"))
    high_yo_yo = yo_yo.clamp_min(0.0)

    # The plan for this pass: two-circle (+1) when my energy advantage is above the plan
    # threshold, else one-circle (-1); after a pass, the last pass's plan, switched when that
    # pass gained less than the switch gain in angles off (Shaw pp. 112-113: the stalemate
    # breaker inside the fight). The kept target's last pass only.
    same_target = has_target & (fight.target == memory["target"])
    last_pass = torch.where(same_target, memory["last_pass"], 0.0)
    energy_plan = torch.where(situation.energy_advantage > read("plan_threshold"), 1.0, -1.0)
    previous_plan = torch.sign(last_pass)
    gained = last_pass.abs() - situation.angles_off
    kept_plan = torch.where(gained < read("switch_gain"), -previous_plan, previous_plan)
    plan = torch.where(last_pass != 0.0, kept_plan, energy_plan)
    two_circle_plan = (plan > 0.0).float()

    # Lead turn (Shaw pp. 76-77, Fig 2-9): with a lateral miss distance d at the pass, the
    # turn worth starting early is theta = arccos(1 - min(d / R_st, 2)), taking theta /
    # sustained rate; the two-circle plan leads his position by that time (or the time left
    # to the pass, if less); one-circle passes pure.
    relative_x = fight.target_velocity_x - fight.velocity_x
    relative_y = fight.target_velocity_y - fight.velocity_y
    relative_squared = (relative_x * relative_x + relative_y * relative_y).clamp_min(TINY)
    to_pass = (-(fight.offset_x * relative_x + fight.offset_y * relative_y) / relative_squared)
    miss = (fight.offset_x * relative_y - fight.offset_y * relative_x).abs() * torch.rsqrt(
        relative_squared
    )
    lead_angle = torch.acos(1.0 - (miss / c["sustained_radius"]).clamp(0.0, 2.0))
    lead_time = read("lead_turn_scale") * lead_angle / c["sustained_rate"]
    merge_time = two_circle_plan * torch.minimum(to_pass.clamp_min(0.0), lead_time)

    # Commands, mixed as scalars. Pursuit time: offensive lags (high yo-yo) or leads (low);
    # the merge leads by the lead turn.
    pursuit_time = weights["offensive"] * (
        -yo_yo * (read("yo_yo_time") * c["quarter_turn_time"])
    ) + weights["merge"] * merge_time
    # The wanted point is the gun's lead point while he is ahead within reach: offensive
    # unless a high yo-yo lags the chase, at the merge unless the plan lead-turns.
    gun_weight = gate["in_range"] * (
        weights["offensive"] * (1.0 - high_yo_yo) + weights["merge"] * (1.0 - two_circle_plan)
    )
    # Speed target, in log space, and the stick cap (the share of the full level folded into
    # the sustained one), per tactic:
    # - disengaged and merge: the merge speed; disengaged, the cap above the sustained stick
    #   speed (no turn lost there);
    # - offensive: the fight speed moved by the yo-yo: down (climb, the cap above the fight
    #   speed) in a high yo-yo; up to corner speed in a low one against a turning target
    #   (against a straight one the engine closes: no dive, and no climb while behind);
    # - turning, two-circle (the energy plan, Shaw p. 110): ease to the sustained level and
    #   bank above the fight speed until his angular advantage reaches the cash angle, then
    #   full stick at corner speed (cash the store); one-circle: full stick at the fight
    #   speed, down to the speed floor inside the scissors range (Shaw p. 83);
    # - defensive (Shaw pp. 132-134): break (full stick, corner speed) while he is inside
    #   the defend range with more than the relax angle of advantage, else a hard turn
    #   (sustained level) at the fight speed;
    # - offensive and turning: corner speed when a shot is near (spend for a purpose).
    near = gate["shot_near"] * gate["spend_need"] * gate["spend_range"]
    log_corner = _log(c["corner_speed"])
    log_fight = _log(fight_speed)
    log_merge = _log(read("merge_speed") * c["corner_speed"])
    log_floor = _log(read("speed_floor") * c["stall_speed"])
    low_yo_yo = (-yo_yo).clamp_min(0.0)
    turning_target = situation.target_sense.abs()
    log_yo_yo = log_fight + (
        low_yo_yo * turning_target - high_yo_yo
    ) * _log(c["corner_speed"] / c["sustained_speed"])
    two_circle = 0.5 * (1.0 + situation.flow)
    cash, break_turn = gate["cash"], gate["defend"] * gate["relax"]
    log_two_circle = log_fight + cash * (log_corner - log_fight)
    log_one_circle = log_fight + gate["scissors"] * (log_floor - log_fight)
    log_turning = log_one_circle + two_circle * (log_two_circle - log_one_circle)
    log_target = (
        (weights["disengaged"] + weights["merge"]) * log_merge
        + weights["offensive"] * (log_yo_yo + near * (log_corner - log_yo_yo))
        + weights["turning"] * (log_turning + near * (log_corner - log_turning))
        + weights["defensive"] * (log_fight + break_turn * (log_corner - log_fight))
    )
    log_target = log_target.clamp_min(log_floor)
    cap = (
        weights["disengaged"] * gate["level_cap"]
        + weights["offensive"] * high_yo_yo * gate["yo_yo_cap"]
        + weights["turning"] * two_circle * (1.0 - cash)
        + weights["defensive"] * (1.0 - break_turn)
    )
    # Falling behind a target that is not turning: no banking (the engine closes).
    no_bank = weights["offensive"] * low_yo_yo * (1.0 - turning_target)

    # The pursuit point: the gun's lead point moved tau along his turn (second order: tau > 0
    # lead, < 0 lag; 0 the gun's solution), mixed with the lead point itself by the gun
    # weight (inside gun range the gun decides).
    time_squared_half = 0.5 * pursuit_time * pursuit_time
    turn_rate = fight.target_nose_rate
    pursuit_x = (
        aim_x
        + fight.target_velocity_x * pursuit_time
        - turn_rate * fight.target_velocity_y * time_squared_half
    )
    pursuit_y = (
        aim_y
        + fight.target_velocity_y * pursuit_time
        + turn_rate * fight.target_velocity_x * time_squared_half
    )
    point_x = pursuit_x + gun_weight * (aim_x - pursuit_x)
    point_y = pursuit_y + gun_weight * (aim_y - pursuit_y)

    # Boundary: within a margin of turn radii of the edge and flying outward, blend in a
    # steer toward the centre. Without a target the wanted direction is the centre.
    distance_from_centre = torch.sqrt(state.x * state.x + state.y * state.y).clamp_min(TINY)
    in_corner_speeds = state.speed / c["corner_speed"]
    turn_radius = c["sustained_radius"] * (in_corner_speeds * in_corner_speeds).clamp_min(1.0)
    margin = read("boundary_margin") * turn_radius
    depth = (distance_from_centre - (c["arena_radius"] - margin)) / margin
    ramp_low = read("boundary_ramp_low")
    ramp = (depth - ramp_low) / (read("boundary_ramp_high") - ramp_low)
    outward = state.x * fight.velocity_x + state.y * fight.velocity_y > 0.0
    centre_weight = torch.where(outward & has_target, ramp.clamp(0.0, 1.0), 0.0)
    centre_weight = torch.where(has_target, centre_weight, 1.0)
    point_length = torch.sqrt(point_x * point_x + point_y * point_y).clamp_min(TINY)
    point_scale = (1.0 - centre_weight) / point_length
    centre_scale = centre_weight / distance_from_centre
    wanted_x = point_scale * point_x - centre_scale * state.x
    wanted_y = point_scale * point_y - centre_scale * state.y
    stick_error = signed_angle(state.heading_cos, state.heading_sin, wanted_x, wanted_y)

    # The committed turn sense (design B3, decided: hysteresis by concrete rules):
    # - release when he comes forward of my 3/9 line (by the release band), or the target
    #   changes;
    # - overshoot (Shaw pp. 67, 75-76): he is aft, inside the overshoot range, crossing my six
    #   from the side I turn to at more than the overshoot rate, with a small track-crossing
    #   angle, and I am no faster than him: reverse (the sense becomes the short way), level;
    # - commit when he moves aft with no sense held: the plan's sense. The plan is two-circle
    #   when my energy advantage is above the plan threshold, else one-circle; at a later pass
    #   it is the last pass's plan, switched when it gained less than the switch gain in
    #   angles off (Shaw pp. 112-113: the stalemate breaker inside the fight). Two-circle
    #   turns his way (toward his side when he is straight), one-circle the other way; the
    #   drawn side breaks an exact tie.
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
    overshoot = (
        aft
        & (held != 0.0)
        & (bearing_rate * held > 0.0)
        & (bearing_rate.abs() > read("overshoot_rate") * c["sustained_rate"])
        & (fight.distance < read("overshoot_range") * c["sustained_radius"])
        & (crossing_cos > _cos(read("overshoot_crossing")))
        & (state.speed <= fight.target_speed)
    )
    short_way = torch.where(situation.bearing > 0.0, 1.0, -1.0)
    held = torch.where(overshoot, short_way, held)
    commit = aft & (held == 0.0)
    his_turn = torch.where(
        situation.target_sense.abs() > 0.5, torch.sign(situation.target_sense), 0.0
    )
    dead_astern = math.pi - bearing_size < read("tie_band")
    his_side = torch.where(dead_astern, memory["side"], short_way)
    two_circle_sense = torch.where(his_turn != 0.0, his_turn, his_side)
    sense = torch.where(commit, plan * two_circle_sense, held)
    last_pass = torch.where(commit, plan * situation.angles_off, last_pass)

    # Stick: P(turn), then the level by two further gates (the cap folding full into
    # sustained), the side by the committed sense while he is aft (unless the boundary steer
    # dominates), else by the error's sign.
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

    # Power: track the speed target. Climb above v* + band, dive below v* - band, coast
    # between; the dive only with a purpose (offensive, turning, defensive: not disengaged
    # nor at the merge, where the engine brings the speed). Each fades over the
    # store's edge band (the bottom for the dive, the top for the climb), as its effect does.
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
    power_floor = read("power_floor")
    power = power_floor + (1.0 - 3.0 * power_floor) * torch.stack(
        [1.0 - dive - climb, dive, climb], -1
    )

    # Gun: aligned times in range at the lead point, zero without a target or while the gun
    # cannot fire. A shot costs no store; never shoot through a team-mate (author 2026-11-23).
    ready = (state.cooldown <= 0) & ~fight.ally_in_line
    fire = torch.where(has_target & ready, gate["aligned"] * gate["in_range"], 0.0)
    return Explanation(
        marginals=Marginals(stick, power, fire, fight.controlled, fight.target),
        situation=situation,
        weights=weights,
        yo_yo=yo_yo,
        pursuit_time=pursuit_time,
        gun_weight=gun_weight,
        speed_target=torch.exp(log_target),
        cap=cap,
        stick_error=stick_error,
        gun_error=gun_error,
        sense=sense,
        last_pass=last_pass,
        overshoot=overshoot,
    )


def _log(x: Tensor | float) -> Tensor | float:
    """The natural log of a number or a tensor."""
    return torch.log(x) if isinstance(x, Tensor) else math.log(x)


def _cos(x: Tensor | float) -> Tensor | float:
    """The cosine of a number or a tensor."""
    return torch.cos(x) if isinstance(x, Tensor) else math.cos(x)


def _last(x: Tensor | float) -> Tensor | float:
    """A per-ship value (E, N) laid along a last axis (a number as it is)."""
    return x[..., None] if isinstance(x, Tensor) else x


def _gates(
    config: ScriptedConfig,
    persona: dict[str, Tensor] | None,
    gates: dict[str, Gate],
    quantities: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Every gate of `gates` in one logistic, by name: floor + (1 - 2 floor) sigmoid(sharpness
    ln(quantity / centre)), the quantities stacked in the gates' order."""
    quantity = torch.stack([quantities[name] for name in gates], -1)
    centre, sharpness, floor = _gate_tensors(config, persona, gates, quantity.device)
    logit = sharpness * torch.log(quantity.clamp_min(TINY) / centre)
    probability = floor + (1.0 - 2.0 * floor) * torch.sigmoid(
        logit.clamp(-LOGIT_LIMIT, LOGIT_LIMIT)
    )
    return dict(zip(gates, probability.unbind(-1), strict=True))


def _gate_tensors(
    config: ScriptedConfig,
    persona: dict[str, Tensor] | None,
    gates: dict[str, Gate],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """The gates' centres, signed sharpness and floors in the order of `gates`: (G,) each for
    the yardstick, built once; for a persona its centres and sharpness (E, N, G), its
    temperature dividing the tempered gates' sharpness."""
    names = tuple(gates)
    centres = tuple(config.values[gates[name].centre] for name in names)
    table = _yardstick_gates(names, centres, config.values["temperature"], device)
    if persona is None:
        return table[0], table[1], table[2]
    temperature = value(config, persona, "temperature")
    reference = persona[next(iter(persona))]
    centre = torch.stack(
        [
            torch.as_tensor(value(config, persona, gates[name].centre), device=device).expand_as(
                reference
            )
            for name in names
        ],
        -1,
    )
    if isinstance(temperature, Tensor):
        sharpness = torch.where(table[4] > 0.5, table[3] / temperature[..., None], table[3])
    else:
        sharpness = table[1].expand_as(centre)
    return centre, sharpness, table[2]


@built_once
def _yardstick_gates(
    names: tuple[str, ...], centres: tuple[float, ...], temperature: float, device: torch.device
) -> Tensor:
    """(5, G) float32 for the gates `names` (of GATES or STICK_GATES): the centres, the signed
    sharpness (1 / (scale x temperature), positive rising; the temperature only where the gate
    is tempered), the floors, the signed sharpness without the temperature, and 1 where the
    gate is tempered."""
    gates = [{**GATES, **STICK_GATES}[name] for name in names]
    rows = [
        list(centres),
        [
            (1.0 if gate.rising else -1.0) / (gate.scale * (temperature if gate.tempered else 1.0))
            for gate in gates
        ],
        [gate.floor for gate in gates],
        [(1.0 if gate.rising else -1.0) / gate.scale for gate in gates],
        [1.0 if gate.tempered else 0.0 for gate in gates],
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def sample(parts: Marginals, uniforms: Tensor) -> Tensor:
    """Draw (E, N, 3) long [stick, power, fire] from the marginals by inverse transform,
    one uniform (E, N, 3) per head; rows that are not controlled are 0."""
    stick = (uniforms[..., 0:1] >= parts.stick.cumsum(-1)[..., :-1]).sum(-1)
    power = (uniforms[..., 1:2] >= parts.power.cumsum(-1)[..., :-1]).sum(-1)
    fire = (uniforms[..., 2] < parts.fire).long()
    return torch.stack([stick, power, fire], -1) * parts.controlled[..., None]


def act(
    config: ScriptedConfig,
    state,
    team: int | None,
    persona: dict[str, Tensor] | None,
    uniforms: Tensor,
    memory: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """The opponent's actions for the rows of `team`, or of every ship of both teams when
    `team` is None (one call; the same actions as the two single-team calls added).

    state: `lean.contract.State`; persona: from `personas` (None: the yardstick); uniforms:
    (E, N, 3) float32 on [0, 1) from `draw_uniforms`; memory: from `fresh_memory` (renewed
    for a new match by `renew_memory`), as the previous call returned it.

    Returns ((E, N, 3) long [stick, power, fire], the new memory). Action rows of the other
    team (with a `team`) and rows that are dead or do not exist are 0; a dead or absent row's
    kept target is cleared (-1), the other team's rows keep theirs.
    """
    action, memory, _ = decide(config, state, team, persona, uniforms, memory)
    return action, memory


def decide(
    config: ScriptedConfig,
    state,
    team: int | None,
    persona: dict[str, Tensor] | None,
    uniforms: Tensor,
    memory: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor], Marginals]:
    """`act`, also returning the marginals the actions were drawn from (the behaviour-cloning
    teacher's full distribution, lean.learn)."""
    reasons = explain(config, state, team, persona, memory)
    parts = reasons.marginals
    action = sample(parts, uniforms)
    controlled = parts.controlled
    new = {
        "target": torch.where(controlled, parts.target, -1),
        "side": memory["side"],
        "sense": torch.where(controlled, reasons.sense, 0.0),
        "last_pass": torch.where(controlled, reasons.last_pass, 0.0),
    }
    if team is not None:  # the other team's rows keep their memory
        acting = state.team == team
        new = {name: torch.where(acting, value, memory[name]) for name, value in new.items()}
    return action, new, parts

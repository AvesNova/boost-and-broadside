"""The scripted opponent: Imugi's combat core without doctrines, memory layers or Frontline.

One decision per living ship of one team per call, read from the true game state:
nearest living enemy, a drag-law lead, a logistic stick choice toward the lead point,
logistic power and gun gates, a steer toward the centre near the arena edge, and one
ordinary uniform draw per action head. Everything is (E, N): E environments, N ship slots.

Action indices follow `lean.contract.State`: stick 0 straight, then (1, 2) fine, (3, 4)
sustained, (5, 6) full; power 0 coast, 1 boost, 2 reverse; fire 0 or 1.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# The eight logistic gates, in the order of the persona's "gate_centre" and
# "gate_sharpness" columns. Each maps a positive, dimensionless quantity to a probability.
GATES = (
    "turn",  # angle error in stall angles: turn at all
    "level_two",  # same error: at least the sustained stick level, given a turn
    "level_three",  # same error: the full stick level, given at least sustained
    "boost_slow",  # speed in corner speeds: boost because slow
    "boost_far",  # store fraction times distance in gun ranges: boost because far
    "reverse",  # distance in gun ranges: reverse because very close
    "aligned",  # aim error in target angular sizes: the gun bears
    "in_range",  # distance in gun ranges: the target is within reach
)
TURN, LEVEL_TWO, LEVEL_THREE, BOOST_SLOW, BOOST_FAR, REVERSE, ALIGNED, IN_RANGE = range(8)
COAST_POWER, BOOST_POWER, REVERSE_POWER = 0, 1, 2
TINY = 1e-6  # keeps lengths and ratios away from zero
# The game's name in `state.constants` where it differs from the characteristic's name here.
GAME_CONSTANT_NAMES = {"stall_angle": "stall", "bullet_energy_cost": "bullet_cost"}
LOGIT_LIMIT = 30.0


@dataclass
class ScriptedConfig:
    """The yardstick's parameters and the ship characteristics it is stated in.

    Gate centres are in the unit named in `GATES`; gate scales are in natural-log units
    of the gate's quantity (the 10 to 90 percent transition spans a factor
    exp(4.39 * scale)). A rising gate's probability grows with its quantity. The centres
    and scales are v2 Imugi's `real_scale_elimination` set.

    A characteristic is read from `state.constants[name]` when the game provides that
    name ((E, N) float32) and from the field here otherwise, so the fields are the
    fallback for one ship type.
    """

    # Gate centres, scales and directions, in `GATES` order.
    gate_centre: tuple[float, ...] = (0.0106, 0.0775, 0.386, 2.61, 7.22, 0.21, 1.01, 1.04)
    gate_scale: tuple[float, ...] = (0.172, 0.328, 0.267, 0.0343, 0.29, 0.16, 0.243, 0.127)
    gate_rising: tuple[bool, ...] = (True, True, True, False, True, False, False, False)
    # Each gate's probability is squeezed into [floor, 1 - floor]; the gun gates are not.
    gate_floor: tuple[float, ...] = (0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.0, 0.0)
    side_floor: float = 0.02  # probability of turning the long way round

    # Target choice.
    target_keep_factor: float = 0.8  # switch only to an enemy closer than this times the kept one
    weakest_weight: float = 0.0  # 0 nearest enemy; larger prefers enemies with less store

    # Lead: first-order intercept with the drag law's time of flight.
    lead_cap_in_gun_range: float = 4.78  # the distance the time of flight is computed for is capped
    lead_gain: float = 1.0  # fraction of the computed lead applied

    # Stand-off: the reverse gate's centre is at least this many gun ranges.
    stand_off_in_gun_range: float = 0.0

    # Boundary steer: inside `margin` turn radii of the edge and flying outward, blend a
    # steer to the centre with a weight that ramps from 0 to 1 between the two depths
    # (depth 0 at the margin's inner edge, 1 at the arena edge).
    boundary_margin_in_turn_radius: float = 2.28
    boundary_ramp_low: float = 0.0554
    boundary_ramp_high: float = 0.52
    arena_radius: float = 4200.0  # metres; overridden by state.constants["arena_radius"]

    # Target angular size clamp, radians.
    target_size_minimum: float = math.radians(1.0)
    target_size_maximum: float = math.radians(45.0)

    # Stick layout: True when even stick indices turn left (counter-clockwise).
    left_is_even: bool = True

    # Fallback ship characteristics (v2's Vizia, the middle type).
    gun_range: float = 634.8  # metres
    bullet_launch_speed: float = 2037.4  # metres per second, cruise plus muzzle speed
    bullet_drag: float = 5e-4  # 1 / metre, quadratic drag law
    stall_angle: float = 0.2705  # radians
    corner_speed: float = 268.2  # metres per second
    # A team-mate within this many of its own hit radii of the line of fire blocks the shot.
    ally_clearance_in_hit_radii: float = 2.0
    max_acceleration: float = 353.16  # metres per second squared
    max_energy: float = 497563.0  # joules per kilogram
    hit_radius: float = 12.0  # metres
    bullet_energy_cost: float = 575.06  # joules per kilogram per shot

    # Persona population: standard deviation at spread 1.
    centre_log_spread: float = 0.35  # of ln(gate centre)
    temperature_log_spread: float = 0.3  # of ln(temperature)
    stand_off_spread: float = 0.5  # gun ranges, half-normal
    weakest_weight_spread: float = 1.0  # half-normal
    lead_gain_spread: float = 0.2
    hold_ticks_maximum: int = 6  # at spread 1 the action hold is uniform on 1..maximum


def personas(
    config: ScriptedConfig,
    envs: int,
    slots: int,
    generator: torch.Generator | None,
    device: torch.device | str,
    spread: float,
) -> dict[str, Tensor]:
    """Per-ship parameters, drawn once at reset.

    spread = 0 gives every ship the yardstick setting exactly (no draw is used);
    spread > 0 gives a population around it: gate centres log-normal about the tuned
    values, a temperature per ship, a stand-off distance, a preference for weak targets,
    a lead gain and an action hold.

    Returns, all on `device`:
      gate_centre     (E, N, 8) float32, in each gate's unit (`GATES` order)
      gate_sharpness  (E, N, 8) float32, signed 1 / (scale * temperature); positive rises
      stand_off       (E, N) float32, gun ranges
      weakest_weight  (E, N) float32
      lead_gain       (E, N) float32
      boundary_margin (E, N) float32, turn radii
      hold_ticks      (E, N) long, a decision is redrawn every this many ticks (needs memory)
      hold_phase      (E, N) long, offset of the redraw tick
    """
    gates = len(GATES)
    centre = torch.tensor(config.gate_centre, dtype=torch.float32, device=device)
    scale = torch.tensor(config.gate_scale, dtype=torch.float32, device=device)
    sign = torch.tensor([1.0 if rising else -1.0 for rising in config.gate_rising], device=device)
    shape = (envs, slots)
    if spread > 0.0:
        normal = torch.randn((*shape, gates + 5), generator=generator, device=device)
        uniform = torch.rand((*shape, 2), generator=generator, device=device)
    else:
        normal = torch.zeros((*shape, gates + 5), device=device)
        uniform = torch.zeros((*shape, 2), device=device)
    temperature = torch.exp(spread * config.temperature_log_spread * normal[..., gates])
    hold_span = 1.0 + min(spread, 1.0) * (config.hold_ticks_maximum - 1)
    hold_ticks = (1.0 + uniform[..., 0] * hold_span).floor().clamp(1, config.hold_ticks_maximum)
    hold_ticks = (
        hold_ticks.long() if spread > 0.0 else torch.ones(shape, dtype=torch.long, device=device)
    )
    return {
        "gate_centre": centre * torch.exp(spread * config.centre_log_spread * normal[..., :gates]),
        "gate_sharpness": (sign / scale) / temperature[..., None],
        "stand_off": config.stand_off_in_gun_range
        + spread * config.stand_off_spread * normal[..., gates + 1].abs(),
        "weakest_weight": config.weakest_weight
        + spread * config.weakest_weight_spread * normal[..., gates + 2].abs(),
        "lead_gain": (
            config.lead_gain + spread * config.lead_gain_spread * normal[..., gates + 3]
        ).clamp(0.0, 1.5),
        "boundary_margin": config.boundary_margin_in_turn_radius
        * torch.exp(spread * config.centre_log_spread * normal[..., gates + 4]),
        "hold_ticks": hold_ticks,
        "hold_phase": (uniform[..., 1] * hold_ticks).floor().long(),
    }


def draw_uniforms(
    envs: int, slots: int, generator: torch.Generator | None, device: torch.device | str
) -> Tensor:
    """The random draw one `act` call needs: (E, N, 3) float32 uniform on [0, 1), one
    per action head [stick, power, fire]."""
    return torch.rand((envs, slots, 3), generator=generator, device=device)


def fresh_memory(envs: int, slots: int, device: torch.device | str) -> dict[str, Tensor]:
    """The optional memory `act` reads and overwrites in place: the kept target
    ("target", (E, N) long, -1 for none) and the last action ("action", (E, N, 3) long).
    Without it `act` has no target hysteresis and no action hold."""
    return {
        "target": torch.full((envs, slots), -1, dtype=torch.long, device=device),
        "action": torch.zeros((envs, slots, 3), dtype=torch.long, device=device),
    }


def characteristic(config: ScriptedConfig, state, name: str) -> Tensor | float:
    """A ship characteristic: the game's per-ship constant ((E, N)) when it provides
    `name`, the config's fallback float otherwise."""
    constants = state.constants
    game_name = GAME_CONSTANT_NAMES.get(name, name)
    return constants[game_name] if game_name in constants else getattr(config, name)


def world_velocity(state) -> tuple[Tensor, Tensor]:
    """Each ship's velocity (E, N) in metres per second: the proper speed along the
    flight path, which is the nose turned back by the angle of attack. Assumes a
    refractive index of 1 (no fields)."""
    direction_x = state.heading_cos * state.angle_cos + state.heading_sin * state.angle_sin
    direction_y = state.heading_sin * state.angle_cos - state.heading_cos * state.angle_sin
    return state.speed * direction_x, state.speed * direction_y


def signed_angle(cos: Tensor, sin: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """The angle from the nose (cos, sin) to the direction (x, y), positive to the left."""
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


@dataclass
class Engagement:
    """Each ship's target geometry, (E, N) each; values are arbitrary where
    `has_target` is False."""

    controlled: Tensor  # bool: rows of the acting team that exist and are alive
    has_target: Tensor  # bool: controlled and a living enemy exists
    target: Tensor  # long: the target's slot
    offset_x: Tensor  # metres, target minus own position
    offset_y: Tensor
    distance: Tensor  # metres
    target_velocity_x: Tensor  # metres per second
    target_velocity_y: Tensor
    velocity_x: Tensor  # own velocity
    velocity_y: Tensor
    ally_in_line: Tensor  # bool: a living team-mate is in the line of fire, nearer than the target


def _per_other(value: Tensor | float) -> Tensor | float:
    """A per-ship value (E, N) laid along the "other ship" axis of (E, own, other) pairs."""
    return value[:, None, :] if isinstance(value, Tensor) else value


def engage(
    state,
    team: int,
    weakest_weight: Tensor | float = 0.0,
    max_energy: Tensor | float = 1.0,
    kept_target: Tensor | None = None,
    keep_factor: float = 1.0,
    clearance: Tensor | float = 24.0,
) -> Engagement:
    """Choose each ship's target: the living enemy with the smallest score, where the
    score is the distance times (1 + weakest_weight * the enemy's store fraction). An
    enemy equal to `kept_target` has its score multiplied by `keep_factor` (below 1), so
    it is kept unless another scores better by that factor. A team-mate blocks the shot when
    it is ahead along the nose, nearer than the target, and within `clearance` metres of the
    line of fire."""
    present = state.exists & state.alive
    own_team = state.team == team
    controlled = present & own_team
    enemy = present & ~own_team
    velocity_x, velocity_y = world_velocity(state)
    offset_x = state.x[:, None, :] - state.x[:, :, None]  # (E, own, other)
    offset_y = state.y[:, None, :] - state.y[:, :, None]
    score = offset_x * offset_x + offset_y * offset_y
    if not (isinstance(weakest_weight, float) and weakest_weight == 0.0):
        store = (state.energy / max_energy)[:, None, :]
        preference = 1.0 + weakest_weight[..., None] * store
        score = score * preference * preference
    if kept_target is not None:
        slots = torch.arange(state.x.shape[1], device=state.x.device)
        score = torch.where(slots == kept_target[..., None], score * keep_factor**2, score)
    score = torch.where(enemy[:, None, :], score, math.inf)
    target = score.argmin(-1)
    features = torch.stack([state.x, state.y, velocity_x, velocity_y], -1)  # (E, N, 4)
    chosen = torch.gather(features, 1, target[..., None].expand(-1, -1, 4))
    target_x, target_y, target_velocity_x, target_velocity_y = chosen.unbind(-1)
    target_x = target_x - state.x
    target_y = target_y - state.y
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    along = offset_x * state.heading_cos[..., None] + offset_y * state.heading_sin[..., None]
    across = offset_x * state.heading_sin[..., None] - offset_y * state.heading_cos[..., None]
    slots = torch.arange(state.x.shape[1], device=state.x.device)
    ally = present[:, None, :] & own_team[:, None, :] & (slots[:, None] != slots[None, :])
    blocking = ally & (along > 0.0) & (along < distance[..., None]) & (across.abs() < clearance)
    return Engagement(
        controlled=controlled,
        has_target=controlled & enemy.any(-1, keepdim=True),
        target=target,
        offset_x=target_x,
        offset_y=target_y,
        distance=distance,
        target_velocity_x=target_velocity_x,
        target_velocity_y=target_velocity_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        ally_in_line=blocking.any(-1),
    )


@dataclass
class Marginals:
    """The three independent action distributions of one decision; they are also the
    soft labels for behaviour cloning. Rows that are not controlled are arbitrary."""

    stick: Tensor  # (E, N, 7), sums to 1
    power: Tensor  # (E, N, 3), sums to 1: coast, boost, reverse
    fire: Tensor  # (E, N), probability of firing
    controlled: Tensor  # (E, N) bool
    target: Tensor  # (E, N) long, the chosen target's slot


def marginals(
    config: ScriptedConfig,
    state,
    team: int,
    persona: dict[str, Tensor],
    kept_target: Tensor | None = None,
) -> Marginals:
    """The opponent's action distributions for the rows of `team`.

    state: `lean.contract.State`; persona: from `personas`; kept_target: (E, N) long
    slot of last tick's target (-1 for none) for the hysteresis, or None for none.
    """
    gun_range = characteristic(config, state, "gun_range")
    stall_angle = characteristic(config, state, "stall_angle")
    max_energy = characteristic(config, state, "max_energy")
    bullet_drag = characteristic(config, state, "bullet_drag")
    arena_radius = characteristic(config, state, "arena_radius")

    fight = engage(
        state,
        team,
        persona["weakest_weight"],
        max_energy,
        kept_target,
        config.target_keep_factor,
        clearance=_per_other(
            config.ally_clearance_in_hit_radii * characteristic(config, state, "hit_radius")
        ),
    )
    has_target = fight.has_target
    safe_distance = fight.distance.clamp_min(TINY)
    range_ratio = safe_distance / gun_range

    # Lead: where the target is, relative to the shooter, when a bullet fired now arrives.
    # The bullet inherits the shooter's velocity, so the relative velocity carries it.
    lead_distance = torch.minimum(range_ratio, range_ratio.new_tensor(config.lead_cap_in_gun_range))
    flight_time = torch.expm1(bullet_drag * gun_range * lead_distance) / (
        bullet_drag * characteristic(config, state, "bullet_launch_speed")
    )
    lead_time = flight_time * persona["lead_gain"]
    aim_x = fight.offset_x + (fight.target_velocity_x - fight.velocity_x) * lead_time
    aim_y = fight.offset_y + (fight.target_velocity_y - fight.velocity_y) * lead_time
    aim_error = signed_angle(state.heading_cos, state.heading_sin, aim_x, aim_y)

    # Boundary: within a margin of turn radii of the edge and flying outward, blend in a
    # steer toward the centre. Without a target the wanted direction is the centre.
    distance_from_centre = torch.sqrt(state.x * state.x + state.y * state.y).clamp_min(TINY)
    turn_radius = state.speed * state.speed / characteristic(config, state, "max_acceleration")
    margin = (persona["boundary_margin"] * turn_radius).clamp_min(1.0)
    depth = (distance_from_centre - (arena_radius - margin)) / margin
    ramp = (depth - config.boundary_ramp_low) / (
        config.boundary_ramp_high - config.boundary_ramp_low
    )
    outward = state.x * fight.velocity_x + state.y * fight.velocity_y > 0.0
    centre_weight = torch.where(outward & has_target, ramp.clamp(0.0, 1.0), 0.0)
    centre_weight = torch.where(has_target, centre_weight, 1.0)
    aim_scale = (1.0 - centre_weight) / torch.sqrt(aim_x * aim_x + aim_y * aim_y).clamp_min(TINY)
    centre_scale = centre_weight / distance_from_centre
    wanted_x = aim_scale * aim_x - centre_scale * state.x
    wanted_y = aim_scale * aim_y - centre_scale * state.y
    stick_error = signed_angle(state.heading_cos, state.heading_sin, wanted_x, wanted_y)

    # All eight gates in one logistic: sigmoid(sharpness * (ln quantity - ln centre)).
    turn_size = stick_error.abs() / stall_angle
    target_size = (
        2.0 * torch.atan(characteristic(config, state, "hit_radius") / safe_distance)
    ).clamp(config.target_size_minimum, config.target_size_maximum)
    quantity = torch.stack(
        [
            turn_size,
            turn_size,
            turn_size,
            state.speed / characteristic(config, state, "corner_speed"),
            state.energy / max_energy * range_ratio,
            range_ratio,
            aim_error.abs() / target_size,
            range_ratio,
        ],
        -1,
    )
    centre = persona["gate_centre"]
    reverse_centre = torch.maximum(centre[..., REVERSE], persona["stand_off"])
    centre = torch.cat(
        [centre[..., :REVERSE], reverse_centre[..., None], centre[..., REVERSE + 1 :]], -1
    )
    logit = persona["gate_sharpness"] * torch.log(quantity.clamp_min(TINY) / centre)
    floor = quantity.new_tensor(config.gate_floor)
    gate = floor + (1.0 - 2.0 * floor) * torch.sigmoid(logit.clamp(-LOGIT_LIMIT, LOGIT_LIMIT))

    # Stick: P(turn), then the level by two further gates, the side by the error's sign.
    turn, above_one, above_two = gate[..., TURN], gate[..., LEVEL_TWO], gate[..., LEVEL_THREE]
    turn_above_one = turn * above_one
    levels = torch.stack(
        [turn - turn_above_one, turn_above_one * (1.0 - above_two), turn_above_one * above_two], -1
    )
    left = torch.where(stick_error > 0.0, 1.0 - config.side_floor, config.side_floor)[..., None]
    left_levels = left * levels
    sides = (
        (left_levels, levels - left_levels)
        if config.left_is_even
        else (levels - left_levels, left_levels)
    )
    stick = torch.cat(
        [(1.0 - turn)[..., None], torch.stack([sides[1], sides[0]], -1).flatten(-2)], -1
    )

    # Power: reverse when very close; else boost when slow, or far with store; else coast.
    target_gate = has_target.to(gate.dtype)
    reverse = torch.where(has_target, gate[..., REVERSE], config.gate_floor[REVERSE])
    boost = 1.0 - (1.0 - gate[..., BOOST_SLOW]) * (1.0 - gate[..., BOOST_FAR] * target_gate)
    boost = (1.0 - reverse) * boost
    power = torch.stack([1.0 - boost - reverse, boost, reverse], -1)

    # Gun: aligned times in range, zero without a target or while the gun cannot fire.
    ready = (state.cooldown <= 0) & (
        state.energy >= characteristic(config, state, "bullet_energy_cost")
    )
    ready = ready & ~fight.ally_in_line  # never shoot through a team-mate (author 2026-11-23)
    fire = torch.where(has_target & ready, gate[..., ALIGNED] * gate[..., IN_RANGE], 0.0)
    return Marginals(stick, power, fire, fight.controlled, fight.target)


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
    team: int,
    persona: dict[str, Tensor],
    uniforms: Tensor,
    memory: dict[str, Tensor] | None = None,
) -> Tensor:
    """The opponent's actions for the rows of `team`.

    state: `lean.contract.State`; persona: from `personas`; uniforms: (E, N, 3) float32
    on [0, 1) from `draw_uniforms`; memory: from `fresh_memory`, overwritten in place, or
    None for a memoryless opponent (no target hysteresis, no action hold).

    Returns (E, N, 3) long [stick, power, fire]; rows of the other team and rows that
    are dead or do not exist are 0.
    """
    if memory is None:
        return sample(marginals(config, state, team, persona), uniforms)
    parts = marginals(config, state, team, persona, memory["target"])
    action = sample(parts, uniforms)
    redraw = (state.tick[:, None] + persona["hold_phase"]) % persona["hold_ticks"] == 0
    action = torch.where(redraw[..., None], action, memory["action"]) * parts.controlled[..., None]
    memory["target"] = torch.where(parts.controlled, parts.target, -1)
    memory["action"] = action
    return action

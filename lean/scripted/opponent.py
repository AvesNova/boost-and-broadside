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

from .parameters import GATES, ScriptedConfig, value

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
    call (every caller: collect, evaluation, the sandbox): "target" (E, N) long, the kept
    target's slot (-1 for none); "side" (E, N) float32, +1 left or -1 right, the turn side
    for the symmetric head-on pass, drawn once per ship per match."""
    left = torch.rand((envs, slots), **random_keywords(generator, device)) < 0.5
    return {
        "target": torch.full((envs, slots), -1, dtype=torch.long, device=device),
        "side": torch.where(left, 1.0, -1.0),
    }


def renew_memory(
    memory: dict[str, Tensor], done: Tensor, generator: torch.Generator | None
) -> dict[str, Tensor]:
    """The memory with the environments in `done` (E,) bool started afresh (a new match):
    no kept target and a new head-on side. Draws for every environment (no host read)."""
    envs, slots = memory["target"].shape
    fresh = fresh_memory(envs, slots, generator, memory["target"].device)
    return {name: torch.where(done[:, None], fresh[name], memory[name]) for name in memory}


def signed_angle(cos: Tensor, sin: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """The angle from the nose (cos, sin) to the direction (x, y), positive to the left."""
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


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
    features = torch.stack(
        [enemies.x, enemies.y, enemy_velocity_x, enemy_velocity_y], -1
    )  # (E, N, 4)
    chosen = torch.gather(features, 1, target[..., None].expand(-1, -1, 4))
    target_x, target_y, target_velocity_x, target_velocity_y = chosen.unbind(-1)
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
    c = state.constants
    enemies = enemy_view(state)
    gun_range = c["gun_range"]

    def read(name: str) -> Tensor | float:
        return value(config, persona, name)

    centre, sharpness, floor = _gate_tensors(config, persona, state.x.device)
    lead_gain, margin_factor = read("lead_gain"), read("boundary_margin")
    weakest_weight, regain_share = read("weakest_weight"), read("regain_share")

    fight = engage(
        state,
        team,
        weakest_weight,
        memory["target"],
        read("target_keep_factor"),
        clearance=_per_other(read("ally_clearance") * c["hit_radius"]),
        enemies=enemies,
    )
    has_target = fight.has_target
    safe_distance = fight.distance.clamp_min(TINY)
    range_ratio = safe_distance / gun_range

    # Lead: where the target is, relative to the shooter, when a bullet fired now arrives.
    # The bullet inherits the shooter's velocity, so the relative velocity carries it. In a
    # medium of index n a bullet's map speed is its proper speed over n, so its time of flight
    # is n times longer (the shooter's index taken for the whole path).
    lead_distance = range_ratio.clamp_max(read("lead_cap"))
    bullet_drag = c["bullet_drag"]
    flight_time = torch.expm1(bullet_drag * gun_range * lead_distance) * (
        lead_gain / (bullet_drag * c["bullet_launch_speed"])
    )
    if fight.index is not None:
        flight_time = flight_time * fight.index
    aim_x = fight.offset_x + (fight.target_velocity_x - fight.velocity_x) * flight_time
    aim_y = fight.offset_y + (fight.target_velocity_y - fight.velocity_y) * flight_time
    aim_error = signed_angle(state.heading_cos, state.heading_sin, aim_x, aim_y)

    # Boundary: within a margin of turn radii of the edge and flying outward, blend in a
    # steer toward the centre. Without a target the wanted direction is the centre.
    distance_from_centre = torch.sqrt(state.x * state.x + state.y * state.y).clamp_min(TINY)
    in_corner_speeds = state.speed / c["corner_speed"]
    turn_radius = c["sustained_radius"] * (in_corner_speeds * in_corner_speeds).clamp_min(1.0)
    margin = margin_factor * turn_radius
    depth = (distance_from_centre - (c["arena_radius"] - margin)) / margin
    ramp_low = read("boundary_ramp_low")
    ramp = (depth - ramp_low) / (read("boundary_ramp_high") - ramp_low)
    outward = state.x * fight.velocity_x + state.y * fight.velocity_y > 0.0
    centre_weight = torch.where(outward & has_target, ramp.clamp(0.0, 1.0), 0.0)
    centre_weight = torch.where(has_target, centre_weight, 1.0)
    aim_scale = (1.0 - centre_weight) / torch.sqrt(aim_x * aim_x + aim_y * aim_y).clamp_min(TINY)
    centre_scale = centre_weight / distance_from_centre
    wanted_x = aim_scale * aim_x - centre_scale * state.x
    wanted_y = aim_scale * aim_y - centre_scale * state.y
    stick_error = signed_angle(state.heading_cos, state.heading_sin, wanted_x, wanted_y)

    # Every gate in one logistic: sigmoid(sharpness * (ln quantity - ln centre)).
    turn_size = stick_error.abs()  # radians
    target_size = (2.0 * torch.atan(c["hit_radius"] / safe_distance)).clamp(
        read("target_size_minimum"), read("target_size_maximum")
    )
    quantities = {
        "turn": turn_size,
        "level_two": turn_size,
        "level_three": turn_size,
        "dive_slow": state.speed / c["stall_speed"],
        "dive_in_turn": in_corner_speeds,
        "climb_fast": in_corner_speeds,
        "climb_close": range_ratio,
        "regain_far": range_ratio,
        "regain_low": state.energy * c["inverse_max_energy"],
        "aligned": aim_error.abs() / target_size,
        "in_range": range_ratio,
    }
    quantity = torch.stack([quantities[name] for name in GATES], -1)
    logit = sharpness * torch.log(quantity.clamp_min(TINY) / centre)
    gate = floor + (1.0 - 2.0 * floor) * torch.sigmoid(logit.clamp(-LOGIT_LIMIT, LOGIT_LIMIT))
    gate = dict(zip(GATES, gate.unbind(-1), strict=True))

    # Stick: P(turn), then the level by two further gates, the side by the error's sign; a
    # wanted direction within the head-on band of straight behind turns to the kept side.
    turn, above_one, above_two = gate["turn"], gate["level_two"], gate["level_three"]
    turn_above_one = turn * above_one
    levels = torch.stack(
        [turn - turn_above_one, turn_above_one * (1.0 - above_two), turn_above_one * above_two], -1
    )
    behind = stick_error.abs() > math.pi - read("head_on_band")
    turn_left = torch.where(behind, memory["side"] > 0.0, stick_error > 0.0)
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

    # Power, for the core flight model (dive: store into speed and the 1 g held freed for the
    # turn; climb: speed into store). Dive when slow, or below corner speed in a hard turn.
    # Otherwise climb when well above corner speed, or very close to the target, or (a share
    # of the ticks) when far from every known enemy with a low store. Otherwise coast. A slow
    # ship never climbs; it dives only with store to spend (the dive fades over the game's
    # edge band of the store, as the dive itself does), else it coasts.
    slow = 1.0 - (1.0 - gate["dive_slow"]) * (1.0 - gate["dive_in_turn"] * turn_above_one)
    dive = slow * (state.energy * c["inverse_fade"]).clamp(0.0, 1.0)
    far = torch.where(has_target, gate["regain_far"], 1.0)
    close = torch.where(has_target, gate["climb_close"], 0.0)
    regain = regain_share * far * gate["regain_low"]
    climb = (1.0 - slow) * (1.0 - (1.0 - gate["climb_fast"]) * (1.0 - close) * (1.0 - regain))
    power_floor = read("power_floor")
    power = power_floor + (1.0 - 3.0 * power_floor) * torch.stack(
        [1.0 - dive - climb, dive, climb], -1
    )

    # Gun: aligned times in range, zero without a target or while the gun cannot fire.
    # A shot costs no store; never shoot through a team-mate (author 2026-11-23).
    ready = (state.cooldown <= 0) & ~fight.ally_in_line
    fire = torch.where(has_target & ready, gate["aligned"] * gate["in_range"], 0.0)
    return Marginals(stick, power, fire, fight.controlled, fight.target)


def _gate_tensors(
    config: ScriptedConfig, persona: dict[str, Tensor] | None, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """The gates' centres, signed sharpness and floors in `GATES` order: (G,) each for the
    yardstick, built once; for a persona its centres and sharpness (E, N, G), its temperature
    dividing the tempered gates' sharpness."""
    centres = tuple(config.values[gate.centre] for gate in GATES.values())
    table = _yardstick_gates(centres, config.values["temperature"], device)
    if persona is None:
        return table[0], table[1], table[2]
    temperature = value(config, persona, "temperature")
    reference = persona[next(iter(persona))]
    centre = torch.stack(
        [
            torch.as_tensor(value(config, persona, gate.centre), device=device).expand_as(reference)
            for gate in GATES.values()
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
    centres: tuple[float, ...], temperature: float, device: torch.device
) -> Tensor:
    """(5, G) float32 in `GATES` order: the centres, the signed sharpness (1 / (scale x
    temperature), positive rising; the temperature only where the gate is tempered), the
    floors, the signed sharpness without the temperature, and 1 where the gate is tempered."""
    gates = GATES.values()
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
    parts = marginals(config, state, team, persona, memory)
    action = sample(parts, uniforms)
    target = torch.where(parts.controlled, parts.target, -1)
    if team is not None:  # the other team's rows keep their memory
        target = torch.where(state.team == team, target, memory["target"])
    return action, {"target": target, "side": memory["side"]}, parts

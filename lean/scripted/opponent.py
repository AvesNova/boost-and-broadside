"""The scripted opponent: a look-ahead scoring agent (author 2026-11-26, design in
docs/rewrite/sections/scripted-lookahead/progress.md and core-decisions.md "Scripted look-ahead").

One decision per living ship per call, from what its team's observation carries (its own team
as it is, enemies as last seen: `lean.game.core.enemy_view`), every ship characteristic read
from the game's constants. Per ship:

1. Candidate paths (lean.scripted.paths): the 21 flight actions (7 stick indices x 3 powers)
   flown for the look-ahead points, for every ship (own ships as they are, enemies as seen).
2. The nearest `ENEMIES` known enemies (a hidden one's distance grows with its roam; past the
   roam cutoff it is dropped, and its weight falls linearly toward the cutoff).
3. Pair tables: for each of them, every (my candidate a, his candidate b) at every point: my
   score S_me[a, b] and his S_him[a, b] from one geometry pass (`pair_tables`), in score
   seconds (lean.scripted.parameters): gun, threat, angles, energy, collision; his own-only
   terms (speed, boundary, slow medium) join S_him.
4. His reply collapsed per row (`collapse`): w(a, b) = softmax_b S_him[a, b] / tau_him; score(a)
   = log(sum_b w e^(beta S_me[a, b])) / beta (beta -> 0: the weighted mean). Summed over the
   enemies by their weights, plus my own-only terms (speed, boundary, fields: slow medium and
   cover, the search for the centre when no enemy is known, collisions with team-mates who keep
   their pending action) and the stickiness bonus for the pending flight action.
5. Flight = softmax over the 21 scores at tau_me; fire = the trigger's probability now (aim and
   range gates at the lead point, the hit worth firing, never through a team-mate, the gun
   ready). The joint over the 42 actions is flight x fire; the only randomness is the sampling
   (two uniforms per ship) and the persona drawn at the deal.

Action indices follow `lean.contract`: stick 0 straight, (1, 2) Sustain, (3, 4) Max, (5, 6)
Snap, the side by the game's stick table; power 0 coast, 1 dive, 2 climb; fire 0 or 1; joint =
(stick x 3 + power) x 2 + fire.
"""

import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from lean.contract import FIRES, JOINT, POWERS, device_constant, random_keywords
from lean.game import card
from lean.game.core import (
    SHOOTER_HIT_CONSTANTS,
    TARGET_HIT_CONSTANTS,
    enemy_view,
    field_blocks,
    flight_tick,
    hit_damage,
    hit_width,
    maximum,
    medium,
    minimum,
    path_velocity,
    slot_indices,
    smootherstep,
)

from . import kernels
from . import paths as paths_module
from .parameters import CROWDED, ScriptedConfig, value
from .paths import CANDIDATES, FLIGHTS, POWER_LOOK, Paths, power_outlook, predict

ENEMIES = 4  # the nearest known enemies each ship scores against
MATES = 3  # the nearest team-mates whose pending path each ship keeps clear of
# The crowd's reach for `store_crowd` (gun ranges; round 4, an experiment).
CROWD_RANGE = 2.0
TINY = 1e-6


def intervals(points) -> tuple[float, ...]:
    """Each look-ahead point stands for the time since the one before (seconds)."""
    return tuple(b - a for a, b in zip((0.0,) + tuple(points[:-1]), points, strict=True))


def draw_uniforms(
    envs: int, slots: int, generator: torch.Generator | None, device: torch.device | str
) -> Tensor:
    """The random draw one call needs: (E, N, 2) float32 uniform on [0, 1), [flight, fire]."""
    return torch.rand((envs, slots, 2), **random_keywords(generator, device))


@dataclass
class Choice:
    """One decision, (E, N, ...) each; rows that are not controlled are arbitrary."""

    joint: Tensor  # (E, N, JOINT): the distribution over the 42 actions (the BC teacher's)
    flight: Tensor  # (E, N, 21): over flight = stick x 3 + power (stick x power)
    fire: Tensor  # (E, N): the probability of firing
    scores: Tensor  # (E, N, 7): the stick scores, score seconds (at the table power)
    power_scores: Tensor  # (E, N, 3): the power scores, score seconds (the power look)
    controlled: Tensor  # (E, N) bool
    enemies: Tensor  # (E, N, ENEMIES) long: the enemies scored against (slots)
    enemy_weight: Tensor  # (E, N, ENEMIES): their weights (0: none there)


def _lay(value: Tensor | float, dims: int) -> Tensor | float:
    """A per-ship value (E, N) with `dims` trailing axes added; a number as it is."""
    if isinstance(value, Tensor):
        return value.reshape(value.shape + (1,) * dims)
    return value


def _per_ship(value, state) -> Tensor:
    """A constant as an (E, N) tensor."""
    return torch.as_tensor(value, dtype=state.x.dtype, device=state.x.device).expand_as(state.x)


def _exp(value: Tensor | float) -> Tensor | float:
    """e to a tensor or a number."""
    return torch.exp(value) if isinstance(value, Tensor) else math.exp(value)


def _at(value: Tensor | float, index: Tensor) -> Tensor | float:
    """A per-ship value (E, N) read at slots `index` (E, N, K) -> (E, N, K); a number as it is."""
    if not isinstance(value, Tensor):
        return value
    envs, slots, count = index.shape
    return value.gather(1, index.reshape(envs, slots * count)).view(envs, slots, count)


# The path fields the pair stage reads per candidate and point (the kernel's order).
PATH_FIELDS = kernels.FIELDS


def _gather_paths(paths: Paths, index: Tensor) -> dict[str, Tensor]:
    """The candidate paths of slots `index` (E, N, K): each field (E, N, K, C, P), gathered
    field by field (an indexed read the consumers fuse; no stacked copy)."""
    envs, slots, count = index.shape
    shape = paths.x.shape[2:]
    rows = index.reshape(envs, slots * count, 1, 1).expand(-1, -1, *shape)

    def gathered(field: Tensor) -> Tensor:
        return field.gather(1, rows).view(envs, slots, count, *shape)

    return {name: gathered(getattr(paths, name)) for name in PATH_FIELDS}


def _own(paths: Paths) -> dict[str, Tensor]:
    """A ship's own candidate paths laid as (E, N, 1, C, P)."""
    names = PATH_FIELDS
    return {name: getattr(paths, name)[:, :, None] for name in names}


def time_weights(horizon: Tensor | float, device, points=None) -> Tensor:
    """(P,) or (E, N, P): each point's weight, its interval discounted by e^(-t / horizon);
    the look-ahead's points unless `points` are given."""
    points = paths_module.LOOK_AHEAD if points is None else points
    times = device_constant(points, device)
    return device_constant(intervals(points), device) * torch.exp(-times / _lay(horizon, 1))


def own_components(c, paths: Paths, margin) -> Paths:
    """The paths with their own-only rates per point (unweighted) attached: boundary (in the
    edge band, more flying outward; growing past the edge) and slow (ln n in half-octaves where
    the medium is slow). The speed term reads the paths' speed. `c` the ships' constants
    ((E, N) rows of the paths' slots); `margin` the boundary margin (turn radii), a number or
    the slots' own rows (E, N)."""
    distance = torch.sqrt(paths.x * paths.x + paths.y * paths.y).clamp_min(TINY)
    over_corner = paths.speed * _lay(c["inverse_corner_speed"], 2)
    turn_radius = _lay(c["sustained_radius"], 2) * (over_corner * over_corner).clamp_min(1.0)
    margin = _lay(margin, 2) * turn_radius
    depth = (distance - (c["arena_radius"] - margin)) / margin
    outward = (paths.x * paths.path_cos + paths.y * paths.path_sin) / distance
    boundary = smootherstep(depth.clamp(0.0, 1.0)) * 0.5 * (1.0 + outward)
    boundary = boundary + (depth - 1.0).clamp_min(0.0)
    slow = paths.log_index.clamp_min(0.0) / card.LN_SQRT2
    return dataclasses.replace(paths, boundary=boundary, slow=slow)


def own_score(speed, boundary, slow, log_preferred, weights, point_weight) -> Tensor:
    """The own-only score per candidate (the points summed): - speed weight x |ln(speed /
    preferred)| - boundary weight x boundary - field weight x slow, each rate weighted by the
    points' time weights. Fields (..., P); `weights` (speed, boundary, field) and
    `log_preferred` laid to broadcast against them."""
    speed_weight, boundary_weight, field_weight = weights
    rate = (
        speed_weight * (torch.log(speed.clamp_min(TINY)) - log_preferred).abs()
        + boundary_weight * boundary
        + field_weight * slow
    )
    return -(rate * point_weight).sum(-1)


def _laid_constants(constants, index: Tensor | None, dims: int) -> dict:
    """The constants the scores read, per ship (index None: (E, N) + `dims` axes) or at slots
    `index` (E, N, K) + `dims` - 1 axes."""
    names = (
        "hit_radius",
        "core_radius",
        "nose_inner_cos",
        "inverse_nose_width",
        "inverse_max_energy",
        "inverse_corner_speed",
        "corner_speed",
        "sustained_radius",
        "sustained_speed",
        "gun_range",
        "bullet_drag",
        "bullet_launch_speed",
        "muzzle_speed",
        "inverse_reference_bullet_speed",
        "explosive_share",
        "inverse_collision_speed",
        "hit_damage",
        "arena_radius",
        *(name for name in TARGET_HIT_CONSTANTS if name not in ("hit_radius", "core_radius")),
    )
    laid = {}
    for name in names:
        item = constants[name]
        if index is not None:
            item = _at(item, index)
            laid[name] = _lay(item, dims - 1)
        else:
            laid[name] = _lay(item, dims)
    widths = _widths_at(constants, index)
    extra = dims - 1 if index is not None else dims
    laid["hit_widths"] = widths.reshape(widths.shape[:-1] + (1,) * extra + widths.shape[-1:])
    return laid


def _widths_at(constants, index: Tensor | None) -> Tensor:
    """The game's hit widths by aspect (core.hit_widths, a constants column) per ship, (E, N,
    POINTS), or at slots `index` (E, N, K): (E, N, K, POINTS)."""
    widths = constants["hit_widths"].movedim(1, -1)
    if index is None:
        return widths
    envs, slots, count = index.shape
    flat = index.reshape(envs, slots * count, 1).expand(-1, -1, widths.shape[-1])
    return widths.gather(1, flat).view(envs, slots, count, -1)


def _flight_time(distance, c, cap, index):
    """The bullet's time of flight to `distance` by the drag law, in a medium of index n (its
    map speed is its proper speed over n), the distance capped."""
    drag = c["bullet_drag"]
    reach = minimum(distance, cap)
    return torch.expm1(drag * reach) * index / (drag * c["bullet_launch_speed"])


def gun_solution(rel_x, rel_y, nose_x, nose_y, velocity_x, velocity_y, target, shooter, time, read):
    """The shooter's gun solution on the target, (0 to 1): its nose against the lead point
    (the drag law's time of flight in the shooter's medium), the tolerance the target's size
    plus the look-ahead's blur at `time`; inside the gun range (a smoothstep from 0.8 to 1.2
    gun ranges); the hit's worth by the game's hit model (`hit_worth`: kinetic at that
    distance, the target's nose cone at the bullet's path). rel: target minus shooter;
    velocity: target minus shooter over the map. `target`, `shooter`: dicts of constants and
    the paths' nose and log index. The target's size is half the game's hit width at the
    bullet's aspect (core.hit_width on the hit_widths column: the deflector's turn)."""
    distance_squared = rel_x * rel_x + rel_y * rel_y
    distance = torch.sqrt(distance_squared + TINY)
    gun_range = shooter["gun_range"]
    dims = distance.dim() - 2
    flight = _flight_time(
        distance, shooter, read("lead_cap", dims) * gun_range, torch.exp(shooter["log_index"])
    )
    lead_x = rel_x + velocity_x * flight
    lead_y = rel_y + velocity_y * flight
    lead_length = torch.sqrt(lead_x * lead_x + lead_y * lead_y + TINY)
    off = 1.0 - (nose_x * lead_x + nose_y * lead_y) / lead_length  # 1 - cos(aim error)
    hit_cos = -(shooter["bullet_x"] * target["nose_cos"] + shooter["bullet_y"] * target["nose_sin"])
    # The target's size: half the game's hit width at the bullet's aspect (the deflector).
    half_width = 0.5 * hit_width(target["hit_widths"], hit_cos)
    tolerance = half_width / distance + read("aim_blur", dims) * time
    tolerance = maximum(tolerance, 0.5 * read("target_size_minimum", dims))
    aligned = torch.exp(-off / (tolerance * tolerance))
    band = ((distance / gun_range - 0.8) / 0.4).clamp(0.0, 1.0)
    in_range = 1.0 - band * band * (3.0 - 2.0 * band)
    return aligned * in_range * hit_worth(shooter, target, distance, hit_cos)


def hit_worth(shooter, target, distance, hit_cos) -> Tensor:
    """A dead-centre hit's damage as a share of the shooter's hit damage, by the game's hit model
    (`core.hit_damage`, the one call): the bullet launched at the shooter's
    `bullet_launch_speed` slowed by the drag law over `distance` (speed x e^(-drag d)), the
    target's nose cone at `hit_cos`. `shooter`, `target`: constants by name, laid to broadcast."""
    speed = shooter["bullet_launch_speed"] * torch.exp(-shooter["bullet_drag"] * distance)
    centre = torch.zeros_like(speed)
    return hit_damage(shooter, target, speed, centre, hit_cos) / shooter["hit_damage"]


def _bullet_direction(paths: dict[str, Tensor], c) -> tuple[Tensor, Tensor]:
    """The launch direction (the ship's map velocity plus the muzzle speed along the nose)."""
    launch_x = paths["world_speed"] * paths["path_cos"] + c["muzzle_speed"] * paths["nose_cos"]
    launch_y = paths["world_speed"] * paths["path_sin"] + c["muzzle_speed"] * paths["nose_sin"]
    inverse = torch.rsqrt(launch_x * launch_x + launch_y * launch_y + TINY)
    return launch_x * inverse, launch_y * inverse


def _contact(previous_x, previous_y, rel_x, rel_y, duration, reach, inverse_collision_speed):
    """A predicted contact on one segment: the relative positions at its two ends, straight
    between; the closest distance's closeness exp(-(d / reach)^2) times the closing speed in
    collision-damage units (a full shield at 1, `lean.game.core.collisions`)."""
    move_x, move_y = rel_x - previous_x, rel_y - previous_y
    move_squared = move_x * move_x + move_y * move_y + TINY
    along = (-(previous_x * move_x + previous_y * move_y) / move_squared).clamp(0.0, 1.0)
    closest_x, closest_y = previous_x + along * move_x, previous_y + along * move_y
    closest_squared = closest_x * closest_x + closest_y * closest_y
    closing = torch.sqrt(move_squared) / duration
    return torch.exp(-closest_squared / (reach * reach)) * closing * inverse_collision_speed


def pair_tables(mine, his, my_c, his_c, now_rel, weights, horizon, read):
    """S_me and S_him (E, N, K, C, C), score seconds, summed over the look-ahead points: my
    candidates on axis -2, his on -1. `mine` (E, N, 1, C, 1, P) and `his` (E, N, K, 1, C, P)
    path fields; constants laid to broadcast; `now_rel` (E, N, K, 1, 1) x and y of him minus me
    now (the first segment's start); `weights` the term weights; `horizon` per ship."""
    my_bullet = _bullet_direction(mine, my_c)
    his_bullet = _bullet_direction(his, his_c)
    me_shooting = {**my_c, "bullet_x": my_bullet[0], "bullet_y": my_bullet[1]}
    him_shooting = {**his_c, "bullet_x": his_bullet[0], "bullet_y": his_bullet[1]}
    reach = read("collision_margin", 3) * (my_c["hit_radius"] + his_c["hit_radius"])
    closeness_range = read("closeness_range", 3) * my_c["gun_range"]
    s_me = s_him = 0.0
    previous_x, previous_y = now_rel
    started = 0.0
    for k, time in enumerate(paths_module.LOOK_AHEAD):

        def at(fields, name):
            return fields[name][..., k]

        rel_x = at(his, "x") - at(mine, "x")
        rel_y = at(his, "y") - at(mine, "y")
        my_velocity_x = at(mine, "world_speed") * at(mine, "path_cos")
        my_velocity_y = at(mine, "world_speed") * at(mine, "path_sin")
        his_velocity_x = at(his, "world_speed") * at(his, "path_cos")
        his_velocity_y = at(his, "world_speed") * at(his, "path_sin")
        velocity_x, velocity_y = his_velocity_x - my_velocity_x, his_velocity_y - my_velocity_y
        my_nose = {"nose_cos": at(mine, "nose_cos"), "nose_sin": at(mine, "nose_sin")}
        his_nose = {"nose_cos": at(his, "nose_cos"), "nose_sin": at(his, "nose_sin")}
        me_shooting.update(
            log_index=at(mine, "log_index"),
            bullet_x=my_bullet[0][..., k],
            bullet_y=my_bullet[1][..., k],
        )
        him_shooting.update(
            log_index=at(his, "log_index"),
            bullet_x=his_bullet[0][..., k],
            bullet_y=his_bullet[1][..., k],
        )
        gun = gun_solution(
            rel_x,
            rel_y,
            my_nose["nose_cos"],
            my_nose["nose_sin"],
            velocity_x,
            velocity_y,
            {**his_c, **his_nose},
            me_shooting,
            time,
            read,
        )
        threat = gun_solution(
            -rel_x,
            -rel_y,
            his_nose["nose_cos"],
            his_nose["nose_sin"],
            -velocity_x,
            -velocity_y,
            {**my_c, **my_nose},
            him_shooting,
            time,
            read,
        )
        distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y + TINY)
        my_off = (my_nose["nose_cos"] * rel_x + my_nose["nose_sin"] * rel_y) / distance
        his_off = -(his_nose["nose_cos"] * rel_x + his_nose["nose_sin"] * rel_y) / distance
        angles = 0.5 * (my_off - his_off)  # +1 nose on and on his tail, -1 he on mine
        my_speed, his_speed = at(mine, "speed"), at(his, "speed")
        energy = (
            0.5 * (my_speed * my_speed - his_speed * his_speed)
            + at(mine, "energy")
            - at(his, "energy")
        ) * my_c["inverse_max_energy"]
        near = closeness_range / (distance + closeness_range)
        near = near * near / (near * near + (1.0 - near) * (1.0 - near))  # 1 / (1 + (d / r)^2)
        position = near * (weights["angles"] * angles + weights["energy"] * energy)
        # The engagement range (McGrew): each side's angular score times his range peak.
        my_peak = torch.exp(
            -(distance - read("desired_range", 3) * my_c["gun_range"]).abs()
            / (read("range_width", 3) * my_c["gun_range"])
        )
        his_peak = torch.exp(
            -(distance - read("desired_range", 3) * his_c["gun_range"]).abs()
            / (read("range_width", 3) * his_c["gun_range"])
        )
        my_range = weights["aggression"] * 0.5 * (1.0 + angles) * my_peak
        his_range = weights["aggression"] * 0.5 * (1.0 - angles) * his_peak
        contact = _contact(
            previous_x,
            previous_y,
            rel_x,
            rel_y,
            time - started,
            reach,
            my_c["inverse_collision_speed"],
        )
        weight = _lay(horizon[..., k], 3)
        discount = _exp(-time / read("horizon", 3))
        collision = weights["collision"] * contact * discount
        s_me = s_me + weight * (gun - weights["threat"] * threat + position + my_range) - collision
        s_him = (
            s_him + weight * (threat - weights["threat"] * gun - position + his_range) - collision
        )
        previous_x, previous_y = rel_x, rel_y
        started = time
    return s_me, s_him


def collapse(s_me: Tensor, s_him: Tensor, reply_temperature, caution) -> Tensor:
    """His columns collapsed per row: w(a, b) = softmax over b of S_him[a, b] / tau_him; score(a)
    = (1 / beta) log(sum_b w(a, b) e^(beta S_me[a, b])), stably: the weighted mean m plus (1 /
    beta) log sum_b w e^(beta (S_me - m)), and m + beta / 2 x the weighted variance where |beta|
    is below 1e-3 (its limit). Shapes (..., C, C) -> (..., C); the parameters (E, N) or numbers
    for (E, N, K, C, C)."""
    log_w = torch.log_softmax(s_him / _lay(reply_temperature, 3), -1)
    w = log_w.exp()
    mean = (w * s_me).sum(-1, keepdim=True)
    centred = s_me - mean
    beta = _lay(caution, 2) if isinstance(caution, Tensor) else caution
    if isinstance(beta, Tensor):
        beta = beta.expand(mean.shape[:-2] + (1,))
        small = beta.abs() < 1e-3
        safe = torch.where(small, 1.0, beta)
        exact = torch.logsumexp(log_w + safe[..., None] * centred, -1) / safe
        variance = (w * centred * centred).sum(-1)
        spread = torch.where(small, 0.5 * beta * variance, exact)
    elif abs(beta) < 1e-3:
        spread = 0.5 * beta * (w * centred * centred).sum(-1)
    else:
        spread = torch.logsumexp(log_w + beta * centred, -1) / beta
    return mean[..., 0] + spread


RECKON_TURN = 2.0  # seconds: a hidden enemy keeps his last nose rate this long, then flies straight


def dead_reckon(enemies, seconds: Tensor):
    """Every ship as its enemy team knows it, carried forward from its last sighting by
    `seconds` (E, N) since seen (author 2026-11-26): its last speed and nose rate for up to
    RECKON_TURN seconds (the arc, as the candidate paths fly one), then straight on; the medium at
    the new place. A visible ship (0 s) is unchanged. No worst case for the hidden stretch: the
    caution inside the look-ahead covers that."""
    turning = seconds.clamp_max(RECKON_TURN)
    straight = seconds - turning
    has_fields = enemies.fields.shape[-1] > 0
    inverse_index = torch.exp(-enemies.log_index) if has_fields else 1.0
    speed = enemies.speed * inverse_index
    rate = enemies.nose_rate
    path = torch.atan2(enemies.heading_sin, enemies.heading_cos) - enemies.angle
    turned = rate * turning
    half = 0.5 * turned
    chord = torch.where(half.abs() < 1e-3, 1.0 - half * half / 6.0, torch.sin(half) / half)
    arc = speed * turning * chord
    end = path + turned
    x = enemies.x + arc * torch.cos(path + half) + speed * straight * torch.cos(end)
    y = enemies.y + arc * torch.sin(path + half) + speed * straight * torch.sin(end)
    heading = end + enemies.angle
    heading_cos, heading_sin = torch.cos(heading), torch.sin(heading)
    nose_rate = torch.where(seconds > RECKON_TURN, 0.0, rate)
    changes = dict(x=x, y=y, heading_cos=heading_cos, heading_sin=heading_sin, nose_rate=nose_rate)
    if has_fields:
        log_index, gradient_x, gradient_y = medium(enemies.fields, x, y)
        changes.update(log_index=log_index, gradient_x=gradient_x, gradient_y=gradient_y)
    velocity_x, velocity_y = path_velocity(
        enemies.speed, heading_cos, heading_sin, enemies.angle_cos, enemies.angle_sin
    )
    return enemies.replace(**changes, velocity_x=velocity_x, velocity_y=velocity_y)


def _seek(state, enemies, own_paths: Paths, point_weight: Tensor, read) -> Tensor:
    """Per candidate (E, N, C): the time-weighted heading toward the freshest dropped enemy's
    last-seen position (0 to 1 per second: 1 flying straight at it), 0 when none is dropped."""
    c = state.constants
    if state.ticks_since_seen is None:
        return torch.zeros_like(own_paths.x[..., 0])
    same_team = state.team[:, :, None] == state.team[:, None, :]
    seen_alive = (enemies.exists & enemies.alive)[:, None, :] & ~same_team
    seconds = state.ticks_since_seen.to(state.x.dtype) * c["tick_length"]
    roam = (seconds * c["sustained_speed"])[:, None, :]
    dropped = seen_alive & (roam > read("roam_cutoff", 1) * c["arena_radius"])
    freshest = torch.where(dropped, seconds[:, None, :], math.inf).min(-1)
    has = torch.isfinite(freshest.values)
    sighted = enemy_view(state)  # the last-seen rows, not reckoned
    last_x = sighted.x.gather(1, freshest.indices)
    last_y = sighted.y.gather(1, freshest.indices)
    to_x = last_x[..., None, None] - own_paths.x
    to_y = last_y[..., None, None] - own_paths.y
    length = torch.sqrt(to_x * to_x + to_y * to_y + TINY)
    toward = (own_paths.path_cos * to_x + own_paths.path_sin * to_y) / length
    return (0.5 * (1.0 + toward) * point_weight).sum(-1) * has[..., None]


def known_enemies(state, enemies, roam_cutoff) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per ship and slot (E, N, N): whether the slot is a known living enemy (seen, or hidden
    with a roam within the cutoff), its last-seen distance and its roam (metres), and the
    cutoff (metres; a number or (E, N, 1))."""
    c = state.constants
    same_team = state.team[:, :, None] == state.team[:, None, :]
    known = (enemies.exists & enemies.alive)[:, None, :] & ~same_team
    offset_x = enemies.x[:, None, :] - state.x[:, :, None]
    offset_y = enemies.y[:, None, :] - state.y[:, :, None]
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y)
    limit = _lay(roam_cutoff, 1) * c["arena_radius"]
    if state.ticks_since_seen is not None:
        seconds = state.ticks_since_seen.to(distance.dtype) * c["tick_length"]
        roam = (seconds * c["sustained_speed"])[:, None, :].expand_as(distance)
    else:
        roam = torch.zeros_like(distance)
    return known & (roam <= limit), distance, roam, limit


def crowd_weight(state, enemies, near, read) -> Tensor | float:
    """The crowd weight w (E, N) in [0, 1) (parameters.CROWDED): (m - 1) / (m - 1 +
    crowd_half), m by `crowd_measure` (0 the scored enemies within CROWD_RANGE gun ranges,
    `near`; 1 the known enemies; 2 my team's living ships), at least 1."""
    measure = read("crowd_measure")
    if isinstance(measure, Tensor) or measure > 0.5:
        known = known_enemies(state, enemies, read("roam_cutoff"))[0].sum(-1).to(near.dtype)
        alive = state.exists & state.alive
        same = (state.team[:, :, None] == state.team[:, None, :]) & alive[:, None, :]
        own = same.sum(-1).to(near.dtype)
        if isinstance(measure, Tensor):
            m = torch.where(measure < 0.5, near, torch.where(measure < 1.5, known, own))
        else:
            m = known if measure < 1.5 else own
    else:
        m = near
    excess = (m - 1.0).clamp_min(0.0)
    return excess / (excess + read("crowd_half"))


def nearest_enemies(state, enemies, roam_cutoff) -> tuple[Tensor, Tensor, Tensor]:
    """Per ship (E, N, ENEMIES): the nearest known living enemies' slots, their weights, and
    whether there is one. A hidden enemy's distance is its last-seen distance plus its roam
    (seconds since seen x its sustained speed); it is a candidate while the roam is at most the
    cutoff (arena radii) and weighs 1 - roam / (cutoff x arena radius). Fewer than ENEMIES:
    the rest are absent (weight 0, slot 0). Without fog no roam."""
    envs, slots = state.x.shape
    known, distance, roam, limit = known_enemies(state, enemies, roam_cutoff)
    score = torch.where(known, distance + roam, math.inf)
    padding = torch.full((envs, slots, ENEMIES), math.inf, device=score.device)
    nearest = torch.cat([score, padding], -1).topk(ENEMIES, -1, largest=False)
    present = torch.isfinite(nearest.values)
    slot = torch.where(present, nearest.indices, 0).clamp_max(slots - 1)
    roam_there = roam.gather(-1, slot)
    weight = torch.where(present, (1.0 - roam_there / limit).clamp(0.0, 1.0), 0.0)
    return slot, weight, present


def nearest_mates(state) -> tuple[Tensor, Tensor]:
    """Per ship (E, N, MATES): the nearest living team-mates' slots and whether there is one."""
    envs, slots = state.x.shape
    alive = state.exists & state.alive
    index = slot_indices(slots, state.x.device).ship
    mate = (
        (state.team[:, :, None] == state.team[:, None, :])
        & alive[:, None, :]
        & (index[:, None] != index[None, :])
    )
    offset_x = state.x[:, None, :] - state.x[:, :, None]
    offset_y = state.y[:, None, :] - state.y[:, :, None]
    score = torch.where(mate, offset_x * offset_x + offset_y * offset_y, math.inf)
    padding = torch.full((envs, slots, MATES), math.inf, device=score.device)
    nearest = torch.cat([score, padding], -1).topk(MATES, -1, largest=False)
    present = torch.isfinite(nearest.values)
    return torch.where(present, nearest.indices, 0).clamp_max(slots - 1), present


def after_pending(state):
    """Every ship one tick on, under its pending flight action (the action chosen on the last
    tick, which the game's next tick applies before the one chosen now: the one-tick delay):
    the game's own `flight_tick` (the firing tick's thrust cut left out), with the velocity
    rows recomputed. What an action chosen now starts from: its bullet leaves from this pose,
    its stick turns the nose from here. Own ships only: an enemy's pending action is not
    observed."""
    c = state.constants
    flown = flight_tick(
        c,
        state.fields,
        c["tick_length"],
        SimpleNamespace(**{name: getattr(state, name) for name in FLOWN}),
        state.pending[..., 0],
        state.pending[..., 1],
    )
    velocity_x, velocity_y = path_velocity(
        flown.speed, flown.heading_cos, flown.heading_sin, flown.angle_cos, flown.angle_sin
    )
    changes = {name: getattr(flown, name) for name in FLOWN}
    return state.replace(**changes, velocity_x=velocity_x, velocity_y=velocity_y)


# The rows `flight_tick` reads and writes (the game's _fly passes the same).
FLOWN = (
    "x",
    "y",
    "heading_cos",
    "heading_sin",
    "speed",
    "angle",
    "angle_cos",
    "angle_sin",
    "nose_rate",
    "energy",
    "log_index",
    "gradient_x",
    "gradient_y",
)


def explain(config: ScriptedConfig, state, team: int | None, persona) -> Choice:
    """The look-ahead agent's distributions for the rows of `team`, or of every ship of both
    teams when `team` is None (each ship decides exactly as in its own team's call)."""
    c = state.constants
    enemies = enemy_view(state)
    envs, slots = state.x.shape
    device = state.x.device

    def read(name: str, dims: int = 0):
        """A parameter: a number, or the persona's (E, N) with `dims` trailing axes."""
        return _lay(value(config, persona, name), dims)

    has_fields = state.fields.shape[-1] > 0
    present = state.exists & state.alive
    controlled = present if team is None else present & (state.team == team)
    if enemies is not state:  # fog: hidden enemies carried forward from their last sighting
        enemies = dead_reckon(enemies, state.ticks_since_seen.to(state.x.dtype) * c["tick_length"])
    enemy_slot, enemy_weight, _ = nearest_enemies(state, enemies, read("roam_cutoff"))
    engaged = enemy_weight.sum(-1).clamp(0.0, 1.0)
    # The crowd (round 4, an experiment): the scored enemies within CROWD_RANGE gun ranges.
    crowd_x = _at(enemies.x, enemy_slot) - state.x[..., None]
    crowd_y = _at(enemies.y, enemy_slot) - state.y[..., None]
    near = (crowd_x * crowd_x + crowd_y * crowd_y) < (CROWD_RANGE * _lay(c["gun_range"], 1)) ** 2
    crowd = (enemy_weight * near).sum(-1)
    # The crowd's rows (parameters.CROWDED): from here on each reads X x X_crowd^w. A ratio
    # of 1 as a number (the yardstick's default) changes nothing and computes nothing.
    duel_read = read
    if any(
        isinstance(value(config, persona, f"{name}_crowd"), Tensor)
        or value(config, persona, f"{name}_crowd") != 1.0
        for name in CROWDED
    ):
        weight = crowd_weight(state, enemies, crowd, duel_read)

        def read(name: str, dims: int = 0):
            if name not in CROWDED:
                return duel_read(name, dims)
            ratio = value(config, persona, f"{name}_crowd")
            base = value(config, persona, name)
            if not isinstance(ratio, Tensor) and ratio == 1.0:
                return _lay(base, dims)
            log_ratio = torch.log(ratio) if isinstance(ratio, Tensor) else math.log(ratio)
            return _lay(base * torch.exp(weight * log_ratio), dims)

    # The one-tick delay: own ships' candidates and shots start after the pending tick.
    ahead = after_pending(state)
    own_paths, seen_paths, power_scores = _predict_views(
        c, ahead, enemies, config, read, engaged, crowd
    )
    horizon = time_weights(read("horizon"), device)
    if horizon.dim() == 1:
        horizon = horizon.expand(envs, slots, -1)
    own_weights = tuple(read(name) for name in ("speed_weight", "boundary_weight", "field_weight"))
    now_x = _at(enemies.x, enemy_slot) - state.x[..., None]  # (E, N, K): him minus me now
    now_y = _at(enemies.y, enemy_slot) - state.y[..., None]
    if kernels.USE_KERNEL and device.type == "cuda":
        collapsed, cover = _pair_kernel(
            c, state.fields, own_paths, seen_paths, enemy_slot, now_x, now_y, horizon, read
        )
    else:  # the plain twin: the tables (E, N, ENEMIES, C, C), then the collapse
        his_paths = _gather_paths(seen_paths, enemy_slot)  # (E, N, K, C, P)
        # His own-only terms, with my weights and my preferred speed for his type.
        his_corner = _at(_per_ship(enemies.constants["corner_speed"], state), enemy_slot)
        log_preferred = torch.log(read("preferred_speed", 1) * his_corner)
        his_own = own_score(
            his_paths["speed"],
            his_paths["boundary"],
            his_paths["slow"],
            _lay(log_preferred, 2),
            tuple(_lay(weight, 3) for weight in own_weights),
            horizon[:, :, None, None, :],
        )  # (E, N, K, C)
        mine = {name: item[:, :, :, :, None] for name, item in _own(own_paths).items()}
        his = {name: item[:, :, :, None] for name, item in his_paths.items()}
        my_c = _laid_constants(c, None, 3)
        his_c = _laid_constants(enemies.constants, enemy_slot, 3)
        now_rel = (now_x[..., None, None], now_y[..., None, None])
        weights = {
            "threat": read("threat_weight", 3),
            "angles": read("angles_weight", 3),
            "energy": read("energy_weight", 3),
            "collision": read("collision_weight", 3),
            "aggression": read("aggression", 3),
        }
        s_me, s_him = pair_tables(mine, his, my_c, his_c, now_rel, weights, horizon, read)
        s_him = s_him + his_own[..., None, :]
        collapsed = collapse(s_me, s_him, read("reply_temperature"), read("caution"))
        cover = covered(state.fields, own_paths, his_paths, horizon) if has_fields else None
    scores = (enemy_weight[..., None] * collapsed).sum(-2)

    # My own-only terms.
    point_weight = horizon[:, :, None, :]  # (E, N, 1, P)
    log_preferred = _lay(
        torch.log(read("preferred_speed") * _per_ship(c["corner_speed"], state)), 2
    )
    scores = scores + own_score(
        own_paths.speed,
        own_paths.boundary,
        own_paths.slow,
        log_preferred,
        tuple(_lay(weight, 2) for weight in own_weights),
        point_weight,
    )
    # No enemy known (author 2026-11-26): no goal point. A preference for energy (speed^2 / 2 +
    # store, in full stores) and a gentle seek toward the freshest dropped last-seen position.
    disengaged = _lay(1.0 - enemy_weight.sum(-1).clamp(0.0, 1.0), 1)
    stored = (0.5 * own_paths.speed * own_paths.speed + own_paths.energy) * _lay(
        c["inverse_max_energy"], 2
    )
    scores = scores + disengaged * _lay(read("disengaged_energy_weight"), 1) * (
        stored * point_weight
    ).sum(-1)
    scores = scores + disengaged * _lay(read("seek_weight"), 1) * _seek(
        state, enemies, own_paths, point_weight, read
    )
    # Cover: behind a field core from an enemy who has the angles on me now (the enemy flying
    # straight on), weighted like the enemy.
    if has_fields:
        pressure = cover_pressure(state, enemies, enemy_slot, enemy_weight, read)
        scores = scores + _lay(read("field_weight"), 1) * (pressure[..., None] * cover).sum(-2)
    # Team-mates keep their pending action: keep clear of their paths.
    scores = scores - _lay(read("collision_weight"), 1) * _mate_contacts(state, own_paths, read)
    # Stickiness: the pending stick (the one chosen on the last tick).
    candidates = torch.arange(CANDIDATES, device=device)
    pending_stick, pending_power = state.pending[..., 0], state.pending[..., 1]
    scores = scores + _lay(read("stickiness"), 1) * (candidates == pending_stick[..., None]).float()
    stick = torch.softmax(scores / _lay(read("temperature"), 1), -1)
    # The power from the own-ship terms over the power look, with its own stickiness.
    powers = torch.arange(POWERS, device=device)
    held = (powers == pending_power[..., None]).float()
    power = torch.softmax(
        (power_scores + _lay(read("power_stickiness"), 1) * held)
        / _lay(read("power_temperature"), 1),
        -1,
    )
    flight = (stick[..., :, None] * power[..., None, :]).flatten(-2)
    fire = _trigger(state, ahead, enemies, enemy_slot, enemy_weight, read) * controlled
    joint = (flight[..., :, None] * torch.stack([1.0 - fire, fire], -1)[..., None, :]).flatten(-2)
    return Choice(
        joint=joint,
        flight=flight,
        fire=fire,
        scores=scores,
        power_scores=power_scores,
        controlled=controlled,
        enemies=enemy_slot,
        enemy_weight=enemy_weight,
    )


PREDICTED = (
    "x",
    "y",
    "heading_cos",
    "heading_sin",
    "speed",
    "angle",
    "nose_rate",
    "energy",
    "log_index",
    "gradient_x",
    "gradient_y",
    "type",  # the ship card row, for the game's steady-flight table (core.steady_lookup)
)


def power_scores(c, speed, energy, read_or_values, engaged, has_corner, crowd=1.0) -> Tensor:
    """The power scores (E, N, POWERS): the own-ship terms over the power look (`POWER_LOOK`):
    minus the speed term (|ln(speed / preferred)|) plus energy (speed^2 / 2 + store, in full
    stores) at the energy weight while engaged and the disengaged energy weight while not, plus
    the store itself (in full stores) at the store weight (round 3: the store held, not only
    its share of the total energy),
    each point's interval discounted by e^(-t / power_horizon). `speed`, `energy` (E, N,
    POWERS, P); `read_or_values(name)` a row (a number or (E, N)); `engaged` (E, N) 0 to 1.
    The store weight is scaled by (1 + k) / (1 + k x max(crowd, 1)), k the `store_crowd` row
    (0: off), `crowd` the scored enemies near (round 4, an experiment: more store alone, less
    in a crowd)."""
    read = read_or_values
    weights = time_weights(read("power_horizon"), speed.device, POWER_LOOK)
    weights = weights[..., None, :] if weights.dim() == 3 else weights
    log_preferred = torch.log(read("preferred_speed") * has_corner)
    energy_weight = read("energy_weight") * engaged + read("disengaged_energy_weight") * (
        1.0 - engaged
    )
    inverse_max_energy = _lay(c["inverse_max_energy"], 2)
    k = read("store_crowd")
    crowd = crowd.clamp_min(1.0) if isinstance(crowd, Tensor) else max(crowd, 1.0)
    crowding = (1.0 + k) / (1.0 + k * crowd)
    stored = (0.5 * speed * speed + energy) * inverse_max_energy
    rate = (
        -_lay(read("speed_weight"), 2)
        * (torch.log(speed.clamp_min(TINY)) - _lay(log_preferred, 2)).abs()
        + _lay(energy_weight, 2) * stored
        + _lay(read("store_weight") * crowding, 2) * energy * inverse_max_energy
    )
    return (rate * weights).sum(-1)


def _predict_views(
    c, state, enemies, config, read, engaged, crowd=1.0
) -> tuple[Paths, Paths, Tensor]:
    """The candidate paths of every ship as it is (own) and as its enemies see it (seen),
    with the own-only components attached, computed in one pass (the two views side by side on
    the slot axis) and materialised once: every field a view of one stacked tensor (P, E, 2 N,
    len(PATH_FIELDS), C), so every write is a run of C (the kernel's layout) and no consumer
    recomputes the paths. `state`: the own ships one tick on under their pending action
    (`after_pending`; the enemies as seen now), so the own view starts its candidates one tick
    later (`predict`'s start) with its points at the same times. Each view's table power: the
    own ship's pending power (in its observation); an enemy's is not observed, so it is the
    power his outlook scores best with the yardstick's rows (`config`; no persona of his is
    read), engaged, from straight flight.
    Also returns the own ships' power scores (E, N, POWERS) from their outlook holding the
    pending stick, with their own rows (`read`)."""
    slots = state.x.shape[1]
    ships = SimpleNamespace(
        **{name: torch.cat([getattr(state, name), getattr(enemies, name)], 1) for name in PREDICTED}
    )
    constants = dataclasses.replace(c, ship_type=torch.cat([c.ship_type, c.ship_type], -1))
    pending_stick, pending_power = state.pending[..., 0], state.pending[..., 1]
    outlook_stick = torch.cat([pending_stick, torch.zeros_like(pending_stick)], 1)
    speed, energy = power_outlook(constants, state.fields, ships, outlook_stick)
    corner = _per_ship(c["corner_speed"], state)
    own = power_scores(
        c, speed[:, :slots], energy[:, :slots], lambda name: read(name), engaged, corner, crowd
    )
    seen = power_scores(
        c, speed[:, slots:], energy[:, slots:], lambda name: config.values[name], 1.0, corner
    )
    table_power = torch.cat([pending_power, seen.argmax(-1)], 1)
    margin = read("boundary_margin")
    if isinstance(margin, Tensor):
        margin = torch.cat([margin, torch.full_like(margin, config.values["boundary_margin"])], 1)
    tick = c["tick_length"]
    start = torch.cat([torch.full_like(state.x, tick), torch.zeros_like(state.x)], 1)
    paths = own_components(
        constants, predict(constants, state.fields, ships, table_power, start), margin
    )
    # Materialised once, (P, E, 2 N, PATH_FIELDS, C): every write a run of C.
    stacked = torch.stack([getattr(paths, name).permute(3, 0, 1, 2) for name in PATH_FIELDS], -2)

    def view(start: int) -> Paths:
        part = stacked.narrow(2, start, slots).permute(3, 1, 2, 4, 0)  # (fields, E, N, C, P)
        fields_ = dict(zip(PATH_FIELDS, part.unbind(0), strict=True))
        now = SimpleNamespace(
            **{
                name: item.narrow(1, start, slots) if isinstance(item, Tensor) else item
                for name, item in vars(paths.now).items()
            }
        )
        return Paths(**fields_, now=now, stacked=stacked)

    return view(0), view(slots), own


def _pair_kernel(c, fields_, own_paths, seen_paths, enemy_slot, now_x, now_y, horizon, read):
    """The collapsed pair scores (E, N, ENEMIES, C) from the fused kernel
    (lean.scripted.kernels.pair_scores), its inputs laid out as it reads them."""
    envs, slots, count = enemy_slot.shape
    ships = envs * slots
    views = 2

    def per_ship(item) -> Tensor:
        return torch.as_tensor(item, dtype=torch.float32, device=now_x.device).expand(envs, slots)

    constants = torch.stack([per_ship(c[name]) for name in kernels.CONSTANTS], -1)
    constants = torch.cat([constants, _widths_at(c, None).to(constants.dtype)], -1)
    times = device_constant(paths_module.LOOK_AHEAD, now_x.device)
    discounts = torch.exp(-times / read("horizon", 1))
    parameters = torch.cat(
        [
            torch.stack([per_ship(read(name)) for name in kernels.PARAMETERS], -1),
            horizon,
            discounts.expand(envs, slots, -1),
        ],
        -1,
    )
    his_ship = enemy_slot + slots * torch.arange(envs, device=now_x.device)[:, None, None]
    scores, cover = kernels.pair_scores(
        own_paths.stacked,
        constants.reshape(ships, -1).contiguous(),
        parameters.reshape(ships, -1).contiguous(),
        times,
        his_ship.reshape(-1).to(torch.int32),
        torch.stack([now_x, now_y], -1).reshape(-1, 2).contiguous(),
        fields_.contiguous(),
        count,
        slots,
        views,
    )
    return scores.view(envs, slots, count, -1), cover.view(envs, slots, count, -1)


def cover_pressure(state, enemies, enemy_slot, enemy_weight, read) -> Tensor:
    """Per scored enemy (E, N, ENEMIES): what his cover is worth, his angular advantage on me
    now (0 to 1) times his closeness and his weight."""
    rel_x = _at(enemies.x, enemy_slot) - state.x[..., None]
    rel_y = _at(enemies.y, enemy_slot) - state.y[..., None]
    distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y + TINY)
    my_off = (
        state.heading_cos[..., None] * rel_x + state.heading_sin[..., None] * rel_y
    ) / distance
    his_cos, his_sin = _at(enemies.heading_cos, enemy_slot), _at(enemies.heading_sin, enemy_slot)
    his_off = -(his_cos * rel_x + his_sin * rel_y) / distance
    advantage = (0.5 * (his_off - my_off)).clamp_min(0.0)
    reach = read("closeness_range") * state.constants["gun_range"]
    near = 1.0 / (1.0 + (distance / _lay(reach, 1)) ** 2)
    return enemy_weight * advantage * near


def covered(fields: Tensor, own_paths: Paths, his_paths: dict, horizon: Tensor) -> Tensor:
    """Per scored enemy and my candidate (E, N, ENEMIES, C): the time-weighted share of the
    look-ahead points behind a field core (`field_blocks`) from him flying straight on
    (candidate 0). The plain twin of the kernel's cover."""
    straight_x = his_paths["x"][..., 0, :][..., None, :]  # (E, N, K, 1, P)
    straight_y = his_paths["y"][..., 0, :][..., None, :]
    my_x, my_y = own_paths.x[:, :, None], own_paths.y[:, :, None]  # (E, N, 1, C, P)
    blocked = field_blocks(fields, my_x, my_y, straight_x, straight_y).float()
    return (blocked * horizon[:, :, None, None, :]).sum(-1)


def _mate_contacts(state, own_paths: Paths, read) -> Tensor:
    """Per candidate (E, N, C): predicted contacts with the nearest team-mates, each flying his
    pending flight action, summed over the segments and the mates (collision-damage units,
    discounted by the horizon)."""
    c = state.constants
    mate, present = nearest_mates(state)
    envs, slots, _ = mate.shape
    mate_flight = _at(state.pending[..., 0], mate)  # (E, N, MATES): his pending stick
    points = len(paths_module.LOOK_AHEAD)

    def mate_path(field: Tensor) -> Tensor:  # (E, N, C, P) -> (E, N, MATES, P)
        flat = field.flatten(2)  # (E, N, C P)
        rows = flat.gather(1, mate.reshape(envs, slots * MATES, 1).expand(-1, -1, flat.shape[-1]))
        rows = rows.view(envs, slots * MATES, CANDIDATES, points)
        chosen = rows.gather(
            2, mate_flight.reshape(envs, slots * MATES, 1, 1).expand(-1, -1, 1, points)
        )
        return chosen.view(envs, slots, MATES, points)

    mate_x, mate_y = mate_path(own_paths.x)[..., None, :], mate_path(own_paths.y)[..., None, :]
    my_x, my_y = own_paths.x[:, :, None], own_paths.y[:, :, None]  # (E, N, 1, C, P)
    previous_x = (_at(state.x, mate) - state.x[..., None])[..., None]
    previous_y = (_at(state.y, mate) - state.y[..., None])[..., None]
    radius = c["hit_radius"]
    summed = (
        _lay(radius, 2) + _at(radius, mate)[..., None]
        if isinstance(radius, Tensor)
        else 2.0 * radius
    )
    reach = read("collision_margin", 2) * summed
    total = 0.0
    started = 0.0
    for k, time in enumerate(paths_module.LOOK_AHEAD):
        rel_x, rel_y = mate_x[..., k] - my_x[..., k], mate_y[..., k] - my_y[..., k]
        contact = _contact(
            previous_x,
            previous_y,
            rel_x,
            rel_y,
            time - started,
            reach,
            _lay(c["inverse_collision_speed"], 2),
        )
        total = total + contact * _exp(-time / read("horizon", 2))
        previous_x, previous_y = rel_x, rel_y
        started = time
    return (total * present[..., None]).sum(-2)


def aim(shooter, enemies, enemy_slot, read, seconds: float = 0.0):
    """The trigger's aim per scored enemy (E, N, ENEMIES): the distance, the bullet's flight
    time and the nose's error (radians, signed) against the lead point, from the shooter's pose
    (`shooter`: anything with the State's rows) and the enemies `seconds` on from their
    view, straight at their map velocity (the shot's tick: their own pending action is not
    observed)."""
    c = shooter.constants
    his_velocity_x, his_velocity_y = _map_velocity(enemies)
    his_velocity_x = _at(his_velocity_x, enemy_slot)
    his_velocity_y = _at(his_velocity_y, enemy_slot)
    rel_x = _at(enemies.x, enemy_slot) + his_velocity_x * seconds - shooter.x[..., None]
    rel_y = _at(enemies.y, enemy_slot) + his_velocity_y * seconds - shooter.y[..., None]
    distance = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp_min(TINY)
    index = torch.exp(shooter.log_index)[..., None] if shooter.fields.shape[-1] > 0 else 1.0
    gun_range = _lay(c["gun_range"], 1)
    laid = {name: _lay(c[name], 1) for name in ("bullet_drag", "bullet_launch_speed")}
    flight = _flight_time(distance, laid, read("lead_cap", 1) * gun_range, index)
    my_velocity_x, my_velocity_y = _map_velocity(shooter)
    velocity_x = his_velocity_x - my_velocity_x[..., None]
    velocity_y = his_velocity_y - my_velocity_y[..., None]
    aim_x, aim_y = rel_x + velocity_x * flight, rel_y + velocity_y * flight
    nose_cos, nose_sin = shooter.heading_cos[..., None], shooter.heading_sin[..., None]
    error = torch.atan2(nose_cos * aim_y - nose_sin * aim_x, nose_cos * aim_x + nose_sin * aim_y)
    return distance, flight, error


def _trigger(state, ahead, enemies, enemy_slot, enemy_weight, read) -> Tensor:
    """The probability of firing now (E, N): the best of the scored enemies' (aligned at the
    lead point within the trigger's tolerance of the target's size, in range, the hit worth it,
    no team-mate in the line), with the gun ready. The one-tick delay: the shot leaves on the
    next tick, so the aim, the range, the line and the bullet's launch are those of `ahead`
    (the own ships one tick on under their pending action, `after_pending`) against the
    enemies one tick on, and the gun must be ready then (the pending shot, if it fires,
    restarts the cooldown)."""
    c = state.constants
    tick = c["tick_length"]
    distance, flight, error = aim(ahead, enemies, enemy_slot, read, tick)
    gun_range = _lay(c["gun_range"], 1)
    nose_cos, nose_sin = ahead.heading_cos[..., None], ahead.heading_sin[..., None]
    in_range = torch.sigmoid(
        -torch.log(distance / (read("fire_range", 1) * gun_range)) / read("fire_range_scale", 1)
    )
    # Worth it (author 2026-11-25): the expected hit (its damage x its chance, below) at the
    # predicted aspect (his nose turned at its rate over the flight time) takes at least
    # fire_worth ticks of my shield recharge, or his whole shield (and more than 0: with the
    # nose cone's rule on, inside his nose cone it does nothing). The hit by the game's hit
    # model (`hit_worth`: core.hit_damage, one call; round 3), dead centre, my bullet's launch
    # speed slowed over the distance.
    launch_x = ahead.velocity_x[..., None] + _lay(c["muzzle_speed"], 1) * nose_cos
    launch_y = ahead.velocity_y[..., None] + _lay(c["muzzle_speed"], 1) * nose_sin
    turned = _at(enemies.nose_rate, enemy_slot) * (flight + tick)
    his_cos, his_sin = _at(enemies.heading_cos, enemy_slot), _at(enemies.heading_sin, enemy_slot)
    turned_cos, turned_sin = torch.cos(turned), torch.sin(turned)
    nose_x = his_cos * turned_cos - his_sin * turned_sin
    nose_y = his_sin * turned_cos + his_cos * turned_sin
    hit_cos = -(launch_x * nose_x + launch_y * nose_y) * torch.rsqrt(
        launch_x * launch_x + launch_y * launch_y + TINY
    )
    mine = {name: _lay(c[name], 1) for name in SHOOTER_HIT_CONSTANTS + ("bullet_drag",)}
    mine["bullet_launch_speed"] = torch.sqrt(launch_x * launch_x + launch_y * launch_y + TINY)
    his = {name: _at(enemies.constants[name], enemy_slot) for name in TARGET_HIT_CONSTANTS}
    # The target's size: the game's hit width at that aspect (the deflector's turn narrows it);
    # the chance a centred shot lands within it: the bullet spread and the lead error (the
    # target's unpredicted turn, lead_error x his sustained turn acceleration, over the flight)
    # spread the bullet sideways by a normal of that deviation (round 4, 2026-11-27).
    half_width = 0.5 * hit_width(_widths_at(enemies.constants, enemy_slot), hit_cos)
    size = maximum(2.0 * torch.atan(half_width / distance), read("target_size_minimum", 1))
    size = size.clamp_max(math.radians(45.0))
    aligned = torch.sigmoid(
        -torch.log(error.abs().clamp_min(TINY) / (size * read("fire_tolerance", 1)))
        / read("fire_aim_scale", 1)
    )
    his_speed = _at(enemies.constants["sustained_speed"], enemy_slot)
    his_turn = his_speed * his_speed / _at(enemies.constants["sustained_radius"], enemy_slot)
    lead = read("lead_error", 1) * 0.5 * his_turn * flight
    sideways = flight * torch.sqrt(_lay(c["spread_speed"], 1) ** 2 + lead * lead + TINY)
    chance = torch.erf(half_width / (math.sqrt(2.0) * sideways))
    hit = _lay(c["hit_damage"], 1) * hit_worth(mine, his, distance, hit_cos) * chance
    worth = read("fire_worth", 1) * _lay(c["shield_rate"] * c["tick_length"], 1)
    shield = _at(enemies.shield, enemy_slot)
    worthwhile = (hit >= worth) | ((hit >= shield) & (hit > 0.0))
    clear = ~_mate_in_line(ahead, distance, read)
    chance = aligned * in_range * (worthwhile & clear & (enemy_weight > 0.0)).float()
    # The gun at the shot's tick: the game's count down after the pending tick (the store's
    # block on the pending shot left out).
    pending_shot = (state.pending[..., 2] == 1) & (state.cooldown == 0)
    cooldown = torch.where(pending_shot, c["fire_interval"], state.cooldown)
    ready = cooldown <= 1
    return torch.where(ready, chance.amax(-1), 0.0)


def _mate_in_line(state, distance: Tensor, read) -> Tensor:
    """(E, N, ENEMIES) bool: a living team-mate ahead along my nose, nearer than each enemy and
    within the clearance of the line of fire."""
    c = state.constants
    slots = state.x.shape[1]
    index = slot_indices(slots, state.x.device).ship
    alive = state.exists & state.alive
    mate = (
        (state.team[:, :, None] == state.team[:, None, :])
        & alive[:, None, :]
        & (index[:, None] != index[None, :])
    )
    offset_x = state.x[:, None, :] - state.x[:, :, None]
    offset_y = state.y[:, None, :] - state.y[:, :, None]
    along = offset_x * state.heading_cos[..., None] + offset_y * state.heading_sin[..., None]
    across = offset_x * state.heading_sin[..., None] - offset_y * state.heading_cos[..., None]
    clearance = read("ally_clearance", 1) * _lay(c["hit_radius"], 1)
    near_line = mate & (along > 0.0) & (across.abs() < clearance)  # (E, N, other)
    nearer = along[:, :, None, :] < distance[..., None]  # (E, N, ENEMIES, other)
    return (near_line[:, :, None, :] & nearer).any(-1)


def _map_velocity(state) -> tuple[Tensor, Tensor]:
    """Each ship's velocity over the map (E, N): the stored proper velocity over n."""
    if state.fields.shape[-1] == 0:
        return state.velocity_x, state.velocity_y
    index = torch.exp(-state.log_index)
    return state.velocity_x * index, state.velocity_y * index


def sample(choice: Choice, uniforms: Tensor) -> Tensor:
    """Draw (E, N, 3) long [stick, power, fire] from the choice by inverse transform, one uniform
    (E, N, 2) per head [flight, fire]; rows that are not controlled are 0."""
    flight = (uniforms[..., 0:1] >= choice.flight.cumsum(-1)[..., :-1]).sum(-1)
    fire = (uniforms[..., 1] < choice.fire).long()
    action = torch.stack([flight // POWERS, flight % POWERS, fire], -1)
    return action * choice.controlled[..., None]


def act(config: ScriptedConfig, state, team: int | None, persona, uniforms: Tensor) -> Tensor:
    """The opponent's actions for the rows of `team`, or of every ship of both teams when
    `team` is None (one call; the same actions as the two single-team calls added): (E, N, 3)
    long [stick, power, fire]. state: `lean.contract.State`; persona: from `personas` (None:
    the yardstick); uniforms: (E, N, 2) from `draw_uniforms`. No memory: the stickiness reads
    the state's pending action."""
    return decide(config, state, team, persona, uniforms)[0]


def decide(config: ScriptedConfig, state, team: int | None, persona, uniforms: Tensor):
    """`act`, also returning the Choice the actions were drawn from (its `joint` is the
    behaviour-cloning teacher's full distribution, lean.learn)."""
    choice = explain(config, state, team, persona)
    return sample(choice, uniforms), choice


assert JOINT == FLIGHTS * FIRES

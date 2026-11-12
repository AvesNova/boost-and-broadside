"""Hits (game spec 7.1) and damage, death and shields (game spec 7.2).

Friendly fire is on: every bullet is paired with every ship but its own shooter, as
(B, N K, N) pair tensors that the compiled stage reduces without storing.
"""

import functools
import math
import operator
from typing import NamedTuple

import torch

from bnb.contracts import ASPECT_BOUNDS, ASPECTS, Action

from ..rules import COAST, QUIET_STICKS, Rules
from ..ship_types import ShipTypes, per_ship
from ..state import Bullets, Ships, Sweep, replace, ship_team

LN_2 = math.log(2.0)
NO_HIT = torch.iinfo(torch.int64).max


class Hits(NamedTuple):
    """One tick's swept hits, before shields and spawn protection."""

    damage: torch.Tensor  # (B, N, N) hit damage from shooter to target
    bullet_hit: torch.Tensor  # (B, N, K) the bullet hit something this tick
    first_shooter: torch.Tensor  # (B, N) owner of the earliest hit on each ship, -1 if none
    # (B, N, len(ASPECTS)) hit damage from the enemy team by aspect class (statistics only)
    enemy_damage_by_aspect: torch.Tensor
    first_aspect: torch.Tensor  # (B, N) aspect class of the earliest hit on each ship, -1 if none


class Contacts(NamedTuple):
    """Every (bullet, ship) pair of one tick's swept test, (B, N K, N): tau, the fraction of
    the move at the closest approach; miss_squared, that distance squared (m^2); contact,
    the pair is a hit candidate (an active bullet, a ship alive at tick start that is not
    its shooter, within the ship's hit radius)."""

    tau: torch.Tensor
    miss_squared: torch.Tensor
    contact: torch.Tensor


def contacts(
    types: ShipTypes,
    sweep: Sweep,
    ship_type: torch.Tensor,
    active: torch.Tensor,
    alive_at_start: torch.Tensor,
) -> Contacts:
    """The relative swept test between this tick's bullet and ship motions.

    sweep: the motions from `move`; ship_type, alive_at_start: (B, N); active: (B, N, K),
    the bullets with moves left. Bullet o K + k belongs to ship o.
    """
    B, N, K = active.shape
    per_bullet = (B, N * K, 1)
    start_x = sweep.bullet_x.reshape(per_bullet) - sweep.ship_x[:, None, :]  # (B, N K, N): D0
    start_y = sweep.bullet_y.reshape(per_bullet) - sweep.ship_y[:, None, :]
    relative_x = (
        sweep.bullet_displacement_x.reshape(per_bullet) - sweep.ship_displacement_x[:, None, :]
    )  # Delta
    relative_y = (
        sweep.bullet_displacement_y.reshape(per_bullet) - sweep.ship_displacement_y[:, None, :]
    )
    # abs() turns a -0.0 from the clamp into +0.0, so ties at tau = 0 order by bullet index
    # in the integer key below, eagerly and compiled alike.
    tau = (
        (
            -(start_x * relative_x + start_y * relative_y)
            / (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
        )
        .clamp(0.0, 1.0)
        .abs()
    )
    miss_x, miss_y = start_x + tau * relative_x, start_y + tau * relative_y
    miss_squared = miss_x * miss_x + miss_y * miss_y
    # The target's hit radius squared; -1 (no contact possible) for a ship dead at tick start.
    reach = torch.where(alive_at_start, per_ship(types.hit_radius, ship_type) ** 2, -1.0)
    ship = torch.arange(N, device=alive_at_start.device)
    not_shooter = ship.repeat_interleave(K)[:, None] != ship  # (N K, N)
    contact = (miss_squared < reach[:, None, :]) & not_shooter & active.reshape(per_bullet)
    return Contacts(tau=tau, miss_squared=miss_squared, contact=contact)


def kinetic(
    speed: torch.Tensor, reference_speed: torch.Tensor, explosive_share: torch.Tensor
) -> torch.Tensor:
    """A hit's damage over its type's (at the reference speed): the explosive share holds,
    the rest goes as the bullet's kinetic energy, (speed / reference_speed)^2. speed: the
    bullet's own proper speed through the medium, not its speed relative to the target
    (which would favour the head-on pass)."""
    ratio = speed / reference_speed
    return explosive_share + (1.0 - explosive_share) * ratio * ratio


def closeness(impact_squared: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """A hit's damage over a centre hit's, (1 - (b / R)^2)^2 for the closest approach b of
    the bullet's relative path to the target's centre and the target's hit radius R: 1 at
    the centre, 0 with zero slope at the rim (a graze does little, and miss, graze, hit and
    centre are continuous). Its mean over a uniform b in [0, R] is 8/15 (`MEAN_CLOSENESS`)."""
    left = (1.0 - impact_squared / (radius * radius)).clamp_min(0.0)
    return left * left


MEAN_CLOSENESS = 8.0 / 15.0  # closeness averaged over a uniform impact parameter


def impact_squared(sweep: Sweep, target: torch.Tensor, K: int) -> torch.Tensor:
    """(B, N K) the squared distance b^2 from each bullet's target's centre to the bullet's
    relative path this tick, the line through its swept move (the target is any ship index
    for a bullet that hit nothing; its value is then unused). b^2 = (D0 x Delta)^2 / |Delta|^2,
    at most |D0|^2, the start distance, should the relative move vanish."""
    B, N = sweep.ship_x.shape
    start_x = sweep.bullet_x.reshape(B, N * K) - sweep.ship_x.gather(1, target)
    start_y = sweep.bullet_y.reshape(B, N * K) - sweep.ship_y.gather(1, target)
    relative_x = sweep.bullet_displacement_x.reshape(B, N * K) - sweep.ship_displacement_x.gather(
        1, target
    )
    relative_y = sweep.bullet_displacement_y.reshape(B, N * K) - sweep.ship_displacement_y.gather(
        1, target
    )
    cross = start_x * relative_y - start_y * relative_x
    line = cross * cross / (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
    return torch.minimum(line, start_x * start_x + start_y * start_y)


def contact_order(tau: torch.Tensor, miss_squared: torch.Tensor) -> torch.Tensor:
    """The int64 key that orders contacts in time: smallest tau first, ties (the clamp puts
    many contacts at exactly tau = 0 or 1) to the closer pass, the smaller miss distance.
    Both are +0.0 or more, so their float bits order like the floats: tau's 30 bits above
    the miss's 31. An order by sim index here would favour a team: team-major storage puts
    team 0's ships and bullets first."""
    return (tau.view(torch.int32).long() << 31) | miss_squared.view(torch.int32).long()


def first_contact(pairs: Contacts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each bullet hits only its first contact by `contact_order` (an exact tie in both to
    the lowest ship index: min returns the first minimal index). Returns (order, target,
    hit), each (B, N K): the first contact's order key, NO_HIT for a bullet that hit
    nothing."""
    order, target = torch.where(
        pairs.contact, contact_order(pairs.tau, pairs.miss_squared), NO_HIT
    ).min(-1)
    return order, target, order < NO_HIT


def hits(
    rules: Rules,
    types: ShipTypes,
    sweep: Sweep,
    ships: Ships,
    bullets: Bullets,
    alive_at_start: torch.Tensor,
) -> Hits:
    """Hits of this tick's swept test (`contacts`), each bullet on its first contact.

    sweep: the motions from `move`; ships, bullets: after the moves; alive_at_start:
    (B, N). Active bullets are those with moves_left > 0.
    """
    B, N, K = bullets.moves_left.shape
    order, target, hit = first_contact(
        contacts(types, sweep, ships.ship_type, bullets.moves_left > 0, alive_at_start)
    )
    ship = torch.arange(N, device=alive_at_start.device)

    # Damage by aspect: theta between the bullet's reversed direction and its target's nose.
    direction_x = bullets.direction_x.reshape(B, N * K)
    direction_y = bullets.direction_y.reshape(B, N * K)
    nose_cos, nose_sin = ships.heading_cos.gather(1, target), ships.heading_sin.gather(1, target)
    theta = torch.atan2(
        (direction_x * nose_sin - direction_y * nose_cos).abs(),
        -(direction_x * nose_cos + direction_y * nose_sin),
    )

    # The shooter's damage: a centre hit's at its bullet's reference speed (`kinetic`), times
    # the closeness of the pass to the target's centre (`closeness`).
    def shooters(table):
        return per_ship(table, ships.ship_type).repeat_interleave(K, 1)

    full = shooters(types.hit_damage) * kinetic(
        bullets.proper_speed.reshape(B, N * K),
        shooters(types.reference_bullet_speed),
        shooters(types.explosive_share),
    )
    radius = per_ship(types.hit_radius, ships.ship_type).gather(1, target)  # the target's
    full = full * closeness(impact_squared(sweep, target, K), radius)
    # The shooter's arming distance X: the damage ramps as a smoothstep of the distance the
    # bullet has travelled over [0, X] (a minimum effective range; X = 0 is full damage at
    # once). The distance is its moves so far, this one included, at its present speed:
    # exact without drag, and the drag is small this early in a bullet's life.
    arming = shooters(types.arming_distance)
    life = torch.round(shooters(types.bullet_life) / rules.tick_length)
    moves = life - bullets.moves_left.reshape(B, N * K).to(life.dtype) + 1.0
    travelled = moves * rules.tick_length * bullets.proper_speed.reshape(B, N * K)
    armed = (travelled / arming.clamp_min(1e-6)).clamp(0.0, 1.0)
    full = full * torch.where(arming > 0.0, armed * armed * (3.0 - 2.0 * armed), 1.0)
    head_on = rules.head_on_fraction * per_ship(types.head_on_fraction_scale, ships.ship_type)
    head_on = head_on.gather(1, target)  # the target's armour
    # Its half-strength angle: the aspect is halfway from the head-on fraction to 1 there.
    half = rules.head_on_angle * per_ship(types.head_on_angle_scale, ships.ship_type)
    ratio = theta / half.gather(1, target)
    # The cone's sharpness p: exp(-ln2 ratio^p), 2 a Gaussian cone; higher, a flat-topped
    # armoured nose that falls away quickly past its half-strength angle.
    sharpness = per_ship(types.head_on_sharpness, ships.ship_type).gather(1, target)
    aspect = 1.0 - (1.0 - head_on) * torch.exp(-LN_2 * ratio.pow(sharpness))
    bullet_damage = torch.where(hit, full * aspect, 0.0)

    # Scatter each bullet's hit to its (shooter, target): O(bullets), not O(bullets x ships).
    # A bullet that hit nothing adds 0. The sum's order can matter only when three or more
    # of one shooter's bullets hit one target in the same tick.
    owner = ship.repeat_interleave(K)
    damage = (
        torch.zeros((B, N * N), dtype=bullet_damage.dtype, device=bullet_damage.device)
        .scatter_add(1, owner * N + target, bullet_damage)
        .reshape(B, N, N)
    )
    # The earliest hit on each target: the smallest contact order (tau, then the miss
    # distance), an exact tie in both to the lowest bullet index o K + k. Two exact int64
    # mins: the order, then the bullet among those that reach it.
    bullet = torch.arange(N * K, device=bullet_damage.device)
    blank = torch.full((B, N), NO_HIT, dtype=torch.int64, device=bullet_damage.device)
    earliest_order = blank.scatter_reduce(1, target, order, "amin")  # a miss is NO_HIT
    first = hit & (order == earliest_order.gather(1, target))
    earliest = blank.scatter_reduce(1, target, torch.where(first, bullet, NO_HIT), "amin")
    shooter = torch.where(earliest < NO_HIT, earliest // K, -1)

    # The statistics' aspect classes (`bnb.contracts.events.ASPECTS`): each bullet's, the
    # enemy damage each target took by class, and the class of each target's earliest hit.
    aspect_class = sum((theta > bound).long() for bound in ASPECT_BOUNDS)
    team = ship_team(rules.ships_per_team, bullet_damage.device)
    from_enemy = team.repeat_interleave(K) != team[target]  # (B, N K)
    classes = len(ASPECTS)
    enemy_damage_by_aspect = (
        torch.zeros((B, N * classes), dtype=bullet_damage.dtype, device=bullet_damage.device)
        .scatter_add(
            1, target * classes + aspect_class, torch.where(from_enemy, bullet_damage, 0.0)
        )
        .reshape(B, N, classes)
    )
    first_aspect = torch.where(
        earliest < NO_HIT, aspect_class.gather(1, earliest.clamp_max(N * K - 1)), -1
    )
    return Hits(
        damage=damage,
        bullet_hit=hit.reshape(B, N, K),
        first_shooter=shooter,
        enemy_damage_by_aspect=enemy_damage_by_aspect,
        first_aspect=first_aspect,
    )


def boundary_damage(
    rules: Rules, arena_radius: torch.Tensor, ships: Ships, alive_at_start: torch.Tensor
) -> torch.Tensor:
    """Game spec 5: (boundary_damage_base + boundary_damage_per_metre depth) tick_length for
    a living ship outside the arena (radius (B,)) after its move."""
    depth = torch.sqrt(ships.x * ships.x + ships.y * ships.y) - arena_radius[:, None]
    rate = rules.boundary_damage_base + rules.boundary_damage_per_metre * depth
    return torch.where(alive_at_start & (depth > 0.0), rate * rules.tick_length, 0.0)


def quiet(action: Action) -> torch.Tensor:
    """(B, N) bool: the ship's action is quiet (game spec 7.2), coasting with the stick
    straight or at a slight turn and not shooting, so its shield's recharge runs. Any other
    input (boost, reverse, a normal or sharp turn, a shot) resets its wait to the delay."""
    steady = functools.reduce(operator.or_, (action.stick == index for index in QUIET_STICKS))
    return (action.power == COAST) & steady & (action.shoot == 0)


class Damage(NamedTuple):
    """What apply_damage returns: the ships after the damage and the recharge, and per ship
    (B, N) died, shield_lost and shield_restored."""

    ships: Ships
    died: torch.Tensor  # (B, N)
    shield_lost: torch.Tensor  # (B, N) what the shield actually lost, overkill ignored
    shield_restored: torch.Tensor  # (B, N) the recharge applied


def apply_damage(
    rules: Rules,
    types: ShipTypes,
    ships: Ships,
    alive_at_start: torch.Tensor,
    total: torch.Tensor,
    quiet_input: torch.Tensor,
) -> Damage:
    """Phase 5 for ships alive at tick start: shields, recharge, and who dies.

    Damage beyond the shield breaks it (the overkill is ignored); any damage to a shield
    already broken at tick start kills. A hit resets the recharge wait to the delay, and so
    does any input that is not quiet (quiet_input (B, N), see `quiet`): an unhurt ship on a
    quiet tick counts its wait down and recharges once the wait is over, so the recharge
    needs the delay's length of unbroken quiet flight. Ships dead at tick start lose and
    restore nothing.
    """
    tick_length = rules.tick_length
    shield0, wait0 = ships.shield, ships.shield_recharge_wait
    hurt = total > 0.0
    died = alive_at_start & hurt & (shield0 <= 0.0)
    shield = (shield0 - total).clamp_min(0.0)
    rate = rules.shield_recharge_rate * per_ship(types.shield_recharge_rate_scale, ships.ship_type)
    delay = rules.shield_recharge_delay * per_ship(
        types.shield_recharge_delay_scale, ships.ship_type
    )
    recharge = torch.minimum(
        per_ship(types.max_shield, ships.ship_type) - shield,
        rate * (tick_length - wait0).clamp(0.0, tick_length),
    )
    # an unhurt ship cannot have died
    restored = torch.where(alive_at_start & ~hurt & quiet_input, recharge, 0.0)
    counted = (wait0 - tick_length).clamp_min(0.0)
    wait = torch.where(hurt | ~quiet_input, delay, counted)
    ships = replace(
        ships,
        shield=torch.where(alive_at_start, shield + restored, shield0),
        shield_recharge_wait=torch.where(alive_at_start, wait, wait0),
    )
    return Damage(
        ships=ships,
        died=died,
        shield_lost=torch.where(alive_at_start, shield0 - shield, 0.0),
        shield_restored=restored,
    )

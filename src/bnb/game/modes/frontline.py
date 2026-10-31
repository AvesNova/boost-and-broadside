"""Frontline (modes spec 5): v1's zone rules, the v2 mirror-symmetric map, spawns and
spawn protection.

Five zones sit on a ring. At front 0 their roles by zone are neutral, team-0 spawn, team-0
defense, team-1 defense, team-1 spawn, and every capture rotates them one place. Only the
two defense zones have meters; a meter moves toward its attacker on a strict alive
majority inside the zone. Roles are in team 0's frame (0 team-0 spawn ... 4 team-1 spawn);
a positive front favours team 0.

The rule functions take any leading axes and work in the dtype they are given (the
fixtures run them in float64).
"""

import math
from typing import NamedTuple

import torch

from ..physics.numerics import lookup
from ..state import Fields

ZONES = 5
TEAM0_SPAWN, TEAM0_DEFENSE, NEUTRAL, TEAM1_DEFENSE, TEAM1_SPAWN = range(ZONES)
BASE_ROLES = (NEUTRAL, TEAM0_SPAWN, TEAM0_DEFENSE, TEAM1_DEFENSE, TEAM1_SPAWN)  # by zone, front 0
EULER_GAMMA = 0.5772156649015329


def roles_from_front(front: torch.Tensor) -> torch.Tensor:
    """(..., 5) int32 roles: role(z) = base[(z - front) mod 5]."""
    zone = torch.arange(ZONES, device=front.device)
    return lookup(BASE_ROLES, (zone - front[..., None]) % ZONES)


def _captures_left(remaining: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """v1's terminal distance of each zone (spec 7): with `remaining` captures to the end
    and the zone falling at capture j = order (mod 5), j taken in the window of five
    captures that starts at clip(remaining - 4, -4, 1)."""
    first = (remaining - 4).clamp(-4, 1)
    return remaining - first - (order - first) % ZONES


def terminal_distances(front: torch.Tensor, threshold) -> tuple[torch.Tensor, torch.Tensor]:
    """(offensive, defensive), each (..., 5): captures to team 0's win and to its loss, per
    zone. Team 1's are team 0's swapped (the map is a mirror image)."""
    zone = torch.arange(ZONES, device=front.device)
    front = front[..., None]
    threshold = threshold[..., None] if isinstance(threshold, torch.Tensor) else threshold
    return (_captures_left(threshold - front, zone - front - 2),
            _captures_left(threshold + front, front + 3 - zone))


def harmonic(counts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """H_n as v1 computed it, digamma(n + 1) + gamma in `dtype` (in float32 H_0 rounds to
    -2.4e-7; it is never used, since a meter moves only on a strict majority)."""
    return torch.digamma(counts.to(dtype) + 1.0) + EULER_GAMMA


def zone_membership(ship_position: torch.Tensor, zone_position: torch.Tensor,
                    zone_radius: torch.Tensor) -> torch.Tensor:
    """(..., N, Z): the ship's distance to the zone's centre is at most its radius.
    ship_position (..., N, 2), zone_position (..., Z, 2), zone_radius (..., Z)."""
    offset = ship_position[..., :, None, :] - zone_position[..., None, :, :]
    return (offset * offset).sum(-1) <= zone_radius[..., None, :] ** 2


class Capture(NamedTuple):
    """One tick of the meters. progress, direction (..., Z): the majority's push on a defense
    zone in team 0's frame (+1 a team-0 majority, -1 a team-1 one, 0 none or a completion;
    clipped meters included); the attempt ledgers
    (..., Z, N); gain and loss (..., N), each capturing side's shares; team0_captured,
    team1_captured (...,); front (...,) and roles (..., Z) after the captures."""

    progress: torch.Tensor
    direction: torch.Tensor
    attack_ledger: torch.Tensor
    defense_ledger: torch.Tensor
    gain: torch.Tensor
    loss: torch.Tensor
    team0_captured: torch.Tensor
    team1_captured: torch.Tensor
    front: torch.Tensor
    roles: torch.Tensor


def _sides(team: torch.Tensor, roles: torch.Tensor):
    """(attacker, defender) ship masks per zone (..., Z, N), and whether team 0 attacks
    (..., Z). Only the defense zones have sides."""
    team0_attacks = roles == TEAM1_DEFENSE
    contested = team0_attacks | (roles == TEAM0_DEFENSE)
    on_team0 = (team == 0)[..., None, :]
    attacker = contested[..., None] & (on_team0 == team0_attacks[..., None])
    defender = contested[..., None] & ~attacker
    return attacker, defender, team0_attacks, contested


def _split(amount: torch.Tensor, members: torch.Tensor) -> torch.Tensor:
    """amount (..., Z) split evenly over members (..., Z, N)."""
    count = members.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(members, amount[..., None] / count, 0.0)


def _normalised(values: torch.Tensor) -> torch.Tensor:
    total = values.sum(-1, keepdim=True)
    return torch.where(total > 0.0, values / torch.where(total > 0.0, total, 1.0), 0.0)


def capture_step(team: torch.Tensor, alive: torch.Tensor, inside: torch.Tensor,
                 roles: torch.Tensor, progress: torch.Tensor, attack_ledger: torch.Tensor,
                 defense_ledger: torch.Tensor, front: torch.Tensor, tick_seconds: float,
                 capture_seconds: float) -> Capture:
    """v1's meters, attempt ledgers and completions (spec 5).

    team, alive (..., N); inside (..., N, Z), already masked to living ships; roles,
    progress (..., Z); the ledgers (..., Z, N); front (...,).
    """
    present = inside.transpose(-1, -2)  # (..., Z, N)
    attacker, defender, team0_attacks, contested = _sides(team, roles)
    on_team0 = (team == 0)[..., None, :]
    net = (present & on_team0).sum(-1) - (present & ~on_team0).sum(-1)  # team 0 minus team 1
    direction = torch.where(contested, torch.sign(net), 0)  # the majority, team 0's frame
    toward_attacker = torch.where(team0_attacks, direction, -direction)
    rate = harmonic(net.abs(), progress.dtype) * tick_seconds / capture_seconds
    moved = (progress + toward_attacker * rate).clamp(0.0, 1.0)
    new = torch.where(contested, moved, 0.0)
    delta = new - progress

    # A rise credits the attackers present and debits the defenders absent (dead count as
    # absent; with none absent, every defender); a fall scales both ledgers down.
    present_attackers = attacker & present
    absent_defenders = defender & ~present
    charged = torch.where(absent_defenders.any(-1, keepdim=True), absent_defenders, defender)
    rise = delta.clamp_min(0.0)
    scale = torch.where(delta < 0.0, new / torch.where(progress > 0.0, progress, 1.0), 1.0)
    attack_ledger = attack_ledger * scale[..., None] + _split(rise, present_attackers)
    defense_ledger = defense_ledger * scale[..., None] + _split(rise, charged)

    # A completion: half by presence on this tick, half by the attempt's ledger.
    captured = new >= 1.0
    gain = 0.5 * _split(torch.ones_like(new), present_attackers) + 0.5 * _normalised(attack_ledger)
    loss = 0.5 * _split(torch.ones_like(new), charged) + 0.5 * _normalised(defense_ledger)
    gain = torch.where(captured[..., None], gain, 0.0).sum(-2)
    loss = torch.where(captured[..., None], loss, 0.0).sum(-2)
    team0_captured = (captured & team0_attacks).any(-1)
    team1_captured = (captured & ~team0_attacks).any(-1)

    # A completion zeroes every meter and push (the roles rotate); an empty meter clears its
    # ledgers.
    completed = (team0_captured | team1_captured)[..., None]
    new = torch.where(completed, 0.0, new)
    direction = torch.where(completed, 0, direction)
    clear = (new <= 0.0)[..., None]
    front = front + team0_captured.to(front.dtype) - team1_captured.to(front.dtype)
    return Capture(progress=new, direction=direction,
                   attack_ledger=torch.where(clear, 0.0, attack_ledger),
                   defense_ledger=torch.where(clear, 0.0, defense_ledger), gain=gain, loss=loss,
                   team0_captured=team0_captured, team1_captured=team1_captured, front=front,
                   roles=roles_from_front(front))


def zone_progress_events(team: torch.Tensor, inside: torch.Tensor, roles: torch.Tensor,
                         previous_progress: torch.Tensor, progress: torch.Tensor,
                         team0_captured: torch.Tensor, team1_captured: torch.Tensor
                         ) -> torch.Tensor:
    """(..., N, 2) held and absent: v1's meter-movement attribution (objective progress).

    The side a movement favours splits |delta| among its ships inside the zone; the other
    side's ships outside it are charged |delta| (all of them, if none is outside). A
    completing tick pays nothing.
    """
    present = inside.transpose(-1, -2)
    attacker, defender, _, _ = _sides(team, roles)
    completing = (team0_captured | team1_captured)[..., None]
    delta = torch.where(completing, 0.0, progress - previous_progress)
    favoured = torch.where((delta > 0.0)[..., None], attacker, defender)
    other = torch.where((delta > 0.0)[..., None], defender, attacker)
    outside = other & ~present
    charged = torch.where(outside.any(-1, keepdim=True), outside, other)
    amount = delta.abs()
    held = _split(amount, favoured & present).sum(-2)
    absent = -_split(amount, charged).sum(-2)
    return torch.stack([held, absent], -1)


def timeout_result(front: torch.Tensor) -> torch.Tensor:
    """+1 team 0 wins, -1 team 1 wins, 0 a draw, from the front's sign at the clock."""
    return torch.sign(front)


# ------------------------------------------------------------------ the map (v2)

def zone_ring(arena_radius: float, zone_radius: float, ring_fraction: float,
              rotation: torch.Tensor, handedness: torch.Tensor):
    """Zone centres (B, 5) x, y at angles rotation + handedness 2 pi k / 5 on the ring, and
    their radii (B, 5)."""
    angle = rotation[:, None] + handedness[:, None] * (2.0 * math.pi / ZONES) \
        * torch.arange(ZONES, device=rotation.device)
    distance = ring_fraction * arena_radius
    return (distance * torch.cos(angle), distance * torch.sin(angle),
            torch.full_like(angle, zone_radius))


def mirror_fields(fields: Fields, map_uniforms: torch.Tensor, arena_radius: float,
                  rotation: torch.Tensor) -> Fields:
    """The point-symmetric map's fields made mirror-symmetric about the axis at angle
    `rotation` through the centre and zone 0: field P + i is field i reflected about it,
    and an odd last field lies on the axis at signed distance 0.85 R_a (2u - 1) (u the
    field's first uniform). Radii, bands and levels are as drawn."""
    B, M = fields.x.shape
    P = M // 2
    cos2, sin2 = torch.cos(2.0 * rotation)[:, None], torch.sin(2.0 * rotation)[:, None]
    x, y = fields.x[:, :P], fields.y[:, :P]
    reflected_x, reflected_y = cos2 * x + sin2 * y, sin2 * x - cos2 * y
    along = 0.85 * arena_radius * (2.0 * map_uniforms[:, 4 * (M - 1)] - 1.0)
    on_axis = (along * torch.cos(rotation))[:, None], (along * torch.sin(rotation))[:, None]
    keep = M - 2 * P  # 1 for an odd M
    return Fields(x=torch.cat([x, reflected_x, on_axis[0][:, :keep]], 1),
                  y=torch.cat([y, reflected_y, on_axis[1][:, :keep]], 1),
                  radius=fields.radius, band_width=fields.band_width, level=fields.level)

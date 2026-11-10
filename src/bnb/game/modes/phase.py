"""The mode phase (modes spec 2): after damage, death and shields, before the clocks. It
updates the mode state, emits the objective events and decides whether the match is won.
Also the mode's other hooks into the tick: spawn protection, the result at the clock, and
a new match's mode state and map.

Each hook takes one mode (static, `ModeRules`) and runs on that mode's slice of the batch
(`layout.per_mode`): it dispatches on the mode's name, so a slice holds one mode's rules. A
mode without zones leaves its zone slots (Z = ZONE_SLOTS in every mode) absent: zero, and untouched.
"""

import math
from typing import NamedTuple

import torch

from ..rules import Rules
from ..state import Fields, ModeState, Ships, apply, enemy, replace, ship_team
from . import frontline
from .rules import FRONTLINE, TEAM_DEATHMATCH, ModeRules


class Phase(NamedTuple):
    """mode: the state after this tick; objective_progress, objective (B, N): the mode's
    signed per-ship amounts, each summing to zero per environment; won (B,): the win rule
    ended the match; result (B,) int32: the result for team 0 if it ended now (+1, -1, 0)."""

    mode: ModeState
    objective_progress: torch.Tensor
    objective: torch.Tensor
    won: torch.Tensor
    result: torch.Tensor


def _in_zones(mode: ModeState, ships: Ships) -> torch.Tensor:
    """(B, N, Z): the ship's position is inside the zone."""
    return frontline.zone_membership(
        torch.stack([ships.x, ships.y], -1),
        torch.stack([mode.zone_x, mode.zone_y], -1),
        mode.zone_radius,
    )


def mode_phase(
    rules: Rules,
    mode_rules: ModeRules,
    mode: ModeState,
    ships: Ships,
    alive: torch.Tensor,
    died: torch.Tensor,
    active: torch.Tensor | None = None,
) -> Phase:
    """ships: after the moves; alive (B, N): alive after the damage phase; died (B, N);
    active (B, 2) int32 or None: each team's active ships (None: every ship), which sets
    team deathmatch's winning scores (`deathmatch_targets`)."""
    zero = torch.zeros_like(ships.x)
    no = torch.zeros_like(mode.front, dtype=torch.bool)
    if mode_rules.name == TEAM_DEATHMATCH:
        # Every death scores for the other team, whatever its cause: (B, 2) by the scoring
        # team, its enemy's deaths.
        score = mode.score + enemy(died.to(torch.int32)).sum(-1, dtype=torch.int32)
        reached = score >= deathmatch_targets(rules, mode_rules, active)
        result = reached[:, 0].to(torch.int32) - reached[:, 1].to(torch.int32)  # both: a draw
        return Phase(replace(mode, score=score), zero, zero, reached.any(-1), result)
    if mode_rules.name == FRONTLINE:
        team = ship_team(rules.ships_per_team, ships.x.device)
        inside = _in_zones(mode, ships) & alive[..., None]
        capture = frontline.capture_step(
            team,
            inside,
            mode.role,
            mode.progress,
            mode.attack_ledger,
            mode.defense_ledger,
            mode.front,
            rules.tick_length,
            mode_rules.capture_seconds,
        )
        held = frontline.zone_progress_events(
            team,
            inside,
            capture.roles,
            mode.progress,
            capture.progress,
            capture.team0_captured,
            capture.team1_captured,
        )
        # The observation shows each meter's actual movement, which a clipped meter does not
        # make whatever the majority.
        completed = (capture.team0_captured | capture.team1_captured)[:, None]
        movement = torch.where(completed, 0, torch.sign(capture.progress - mode.progress))
        state = replace(
            mode,
            front=capture.front,
            role=capture.roles,
            progress=capture.progress,
            movement=movement.to(torch.int32),
            attack_ledger=capture.attack_ledger,
            defense_ledger=capture.defense_ledger,
        )
        won = capture.front.abs() >= mode_rules.front_threshold
        return Phase(
            state,
            held.sum(-1),
            capture.gain - capture.loss,
            won,
            torch.sign(capture.front).to(torch.int32),
        )
    return Phase(mode, zero, zero, no, torch.zeros_like(mode.front))


def deathmatch_targets(
    rules: Rules, mode_rules: ModeRules, active: torch.Tensor | None
) -> torch.Tensor | int:
    """Team deathmatch's winning score per team: kills_per_ship times the enemy team's
    active ships, (B, 2) int32 by the scoring team from active (B, 2), or the one int
    kills_per_ship * ships_per_team when every ship is active (active None)."""
    if active is None:
        return mode_rules.kills_per_ship * rules.ships_per_team
    return mode_rules.kills_per_ship * active.flip(-1).to(torch.int32)


def clock_result(mode_rules: ModeRules, mode: ModeState) -> torch.Tensor:
    """(B,) int32: the result for team 0 when the clock ends the match."""
    if mode_rules.name == TEAM_DEATHMATCH:
        return torch.sign(mode.score[:, 0] - mode.score[:, 1]).to(torch.int32)
    if mode_rules.name == FRONTLINE:
        return frontline.timeout_result(mode.front).to(torch.int32)
    return torch.zeros_like(mode.front)


def protected(rules: Rules, mode_rules: ModeRules, mode: ModeState, ships: Ships) -> torch.Tensor:
    """(B, N): hits on these ships deal no damage (Frontline: inside the own team's spawn
    zone, by the roles at tick start and the post-move position; other modes: none)."""
    if mode_rules.name != FRONTLINE:
        return torch.zeros_like(ships.alive)
    own = torch.stack(
        [mode.role == frontline.TEAM0_SPAWN, mode.role == frontline.TEAM1_SPAWN], 1
    ).repeat_interleave(rules.ships_per_team, 1)  # (B, N, Z)
    return (_in_zones(mode, ships) & own).any(-1)


def new_match(
    mode_rules: ModeRules,
    arena_radius: torch.Tensor,
    map_scale: torch.Tensor,
    uniforms: torch.Tensor,
    map_uniforms: torch.Tensor,
    fields: Fields,
    previous: ModeState,
) -> tuple[ModeState, Fields]:
    """A new match's mode state and map from the mode's tick uniforms (B, U_mode) and the
    environments' arena radius and map scale (B,): the drawn point-symmetric fields as they
    are, or, in Frontline, a random ring (rotation, then handedness) with fields mirrored
    about its neutral axis. previous: the mode state as it stands. A mode without zones
    zeroes only the score; its other leaves are zero already and pass through unchanged
    (no copy of the absent slots)."""
    if mode_rules.name != FRONTLINE:
        return replace(previous, score=torch.zeros_like(previous.score)), fields
    blank = apply(torch.zeros_like, previous)
    rotation = 2.0 * math.pi * uniforms[:, 0]
    handedness = torch.where(uniforms[:, 1] < 0.5, 1.0, -1.0)
    zone_x, zone_y, radius = frontline.zone_ring(
        arena_radius,
        mode_rules.zone_radius * map_scale,
        mode_rules.zone_ring_fraction,
        rotation,
        handedness,
    )
    roles = frontline.roles_from_front(blank.front)
    state = replace(
        blank,
        zone_x=zone_x,
        zone_y=zone_y,
        zone_radius=radius,
        role=roles,
        handedness=handedness,
        ring_rotation=rotation,
    )
    return state, frontline.mirror_fields(fields, map_uniforms, arena_radius, rotation)

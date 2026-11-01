"""The mode phase (modes spec 2): after damage, death and shields, before the clocks. It
updates the mode state, emits the objective events and decides whether the match is won.
Also the mode's other hooks into the tick: spawn protection, the result at the clock, and
a new match's mode state and map.

Each hook takes one mode (static, `ModeRules`) and runs on that mode's slice of the batch
(`layout.per_mode`): it dispatches on the mode's name, so a slice holds one mode's rules. A
mode without zones leaves its zone slots (Z = 5 in every mode) absent: zero, and untouched.
"""

import math
from typing import NamedTuple

import torch

from ..rules import Rules
from ..state import Fields, ModeState, Ships, apply, by_team, replace
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


def _team(rules: Rules, device) -> torch.Tensor:
    return torch.arange(rules.num_ships, device=device) // rules.ships_per_team


def _inside(mode: ModeState, ships: Ships) -> torch.Tensor:
    """(B, N, Z): the ship's position is inside the zone."""
    return frontline.zone_membership(torch.stack([ships.x, ships.y], -1),
                                     torch.stack([mode.zone_x, mode.zone_y], -1),
                                     mode.zone_radius)


def mode_phase(rules: Rules, mode_rules: ModeRules, mode: ModeState, ships: Ships,
               alive: torch.Tensor, died: torch.Tensor) -> Phase:
    """ships: after the moves; alive (B, N): alive after the damage phase; died (B, N)."""
    zero = torch.zeros_like(ships.x)
    no = torch.zeros_like(mode.front, dtype=torch.bool)
    if mode_rules.name == TEAM_DEATHMATCH:
        # Every death scores for the other team, whatever its cause.
        deaths = by_team(died.to(torch.int32)).sum(-1)  # (B, 2) by the dead ship's team
        score = mode.score + deaths.flip(-1)
        reached = score >= mode_rules.kills_per_ship * rules.ships_per_team
        result = reached[:, 0].to(torch.int32) - reached[:, 1].to(torch.int32)  # both: a draw
        return Phase(replace(mode, score=score), zero, zero, reached.any(-1), result)
    if mode_rules.name == FRONTLINE:
        team = _team(rules, ships.x.device)
        inside = _inside(mode, ships) & alive[..., None]
        capture = frontline.capture_step(team, alive, inside, mode.role, mode.progress,
                                         mode.attack_ledger, mode.defense_ledger, mode.front,
                                         rules.tick_length, mode_rules.capture_seconds)
        held = frontline.zone_progress_events(team, inside, capture.roles, mode.progress,
                                              capture.progress, capture.team0_captured,
                                              capture.team1_captured)
        # The observation shows each meter's actual movement (the fixture's `direction` is
        # the majority's push, which a clipped meter does not follow).
        completed = (capture.team0_captured | capture.team1_captured)[:, None]
        movement = torch.where(completed, 0, torch.sign(capture.progress - mode.progress))
        state = replace(mode, front=capture.front, role=capture.roles,
                        progress=capture.progress, movement=movement.to(torch.int32),
                        attack_ledger=capture.attack_ledger,
                        defense_ledger=capture.defense_ledger)
        won = capture.front.abs() >= mode_rules.front_threshold
        return Phase(state, held.sum(-1), capture.gain - capture.loss, won,
                     torch.sign(capture.front).to(torch.int32))
    return Phase(mode, zero, zero, no, torch.zeros_like(mode.front))


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
    own = torch.stack([mode.role == frontline.TEAM0_SPAWN, mode.role == frontline.TEAM1_SPAWN],
                      1).repeat_interleave(rules.ships_per_team, 1)  # (B, N, Z)
    return (_inside(mode, ships) & own).any(-1)


def new_match(mode_rules: ModeRules, arena_radius: torch.Tensor, map_scale: torch.Tensor,
              uniforms: torch.Tensor, map_uniforms: torch.Tensor, fields: Fields,
              previous: ModeState) -> tuple[ModeState, Fields]:
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
        arena_radius, mode_rules.zone_radius * map_scale, mode_rules.zone_ring_fraction, rotation,
        handedness)
    roles = frontline.roles_from_front(blank.front)
    state = replace(blank, zone_x=zone_x, zone_y=zone_y, zone_radius=radius, role=roles,
                    handedness=handedness, ring_rotation=rotation)
    return state, frontline.mirror_fields(fields, map_uniforms, arena_radius, rotation)

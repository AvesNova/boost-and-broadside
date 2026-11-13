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

from ..physics.numerics import lookup
from ..rules import Rules
from ..state import Fields, ModeState, Ships, apply, by_team, enemy, replace, ship_team
from . import frontline
from .rules import ELIMINATION, FRONTLINE, TEAM_DEATHMATCH, ModeRules

KILL_MODES = (TEAM_DEATHMATCH, ELIMINATION)  # every death scores for the other team


class Phase(NamedTuple):
    """mode: the state after this tick; objective_progress, objective (B, N): the mode's
    signed per-ship amounts, each summing to zero per environment; won (B,): the win rule
    ended the match; result (B,) int32: the result for team 0 if it ended now (+1, -1, 0);
    respawn_open (B, 2) bool, by team: a dead ship of the team whose respawn wait is over
    respawns on this tick (Frontline: the team's respawn wave lands; every tick in the
    other modes)."""

    mode: ModeState
    objective_progress: torch.Tensor
    objective: torch.Tensor
    won: torch.Tensor
    result: torch.Tensor
    respawn_open: torch.Tensor


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
    team deathmatch's winning scores (`deathmatch_targets`). Elimination scores as team
    deathmatch with one kill per enemy ship to win: a team with no ship left loses."""
    zero = torch.zeros_like(ships.x)
    no = torch.zeros_like(mode.front, dtype=torch.bool)
    every_tick = torch.ones_like(mode.score, dtype=torch.bool)
    if mode_rules.name in KILL_MODES:
        # Every death scores for the other team, whatever its cause: (B, 2) by the scoring
        # team, its enemy's deaths.
        score = mode.score + enemy(died.to(torch.int32)).sum(-1, dtype=torch.int32)
        reached = score >= deathmatch_targets(rules, mode_rules, active)
        result = reached[:, 0].to(torch.int32) - reached[:, 1].to(torch.int32)  # both: a draw
        return Phase(replace(mode, score=score), zero, zero, reached.any(-1), result, every_tick)
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
            frontline.attacker_locked(mode.lock, mode.role),
        )
        # The capture lock: a zone that fell holds the team that lost it for the lock's
        # length, scaled by the ring's size (`frontline.lock_ticks`).
        length = frontline.lock_ticks(
            mode_rules.lock_seconds_per_root_km, mode.zone_x, mode.zone_y, rules.tick_length
        )
        lock = frontline.locks_after(mode.lock, capture.captured, mode.role, length)
        # The respawn waves: each team's clock counts down, and a wave lands at 0.
        interval = frontline.whole_ticks(mode_rules.wave_seconds, rules.tick_length, mode.front)
        wave_wait, landed = frontline.waves_after(mode.wave_wait, interval[:, None])
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
            lock=lock,
            wave_wait=wave_wait,
        )
        won = capture.front.abs() >= mode_rules.front_threshold
        return Phase(
            state,
            held.sum(-1),
            capture.gain - capture.loss,
            won,
            torch.sign(capture.front).to(torch.int32),
            landed,
        )
    return Phase(mode, zero, zero, no, torch.zeros_like(mode.front), every_tick)


def deathmatch_targets(
    rules: Rules, mode_rules: ModeRules, active: torch.Tensor | None
) -> torch.Tensor | int:
    """Team deathmatch's (and elimination's) winning score per team: kills_per_ship (1 in
    elimination) times the enemy team's
    active ships, (B, 2) int32 by the scoring team from active (B, 2), or the one int
    kills_per_ship * ships_per_team when every ship is active (active None)."""
    if active is None:
        return mode_rules.kills_per_ship * rules.ships_per_team
    return mode_rules.kills_per_ship * active.flip(-1).to(torch.int32)


def clock_result(
    mode_rules: ModeRules, mode: ModeState, alive: torch.Tensor, health: torch.Tensor
) -> torch.Tensor:
    """(B,) int32: the result for team 0 when the clock ends the match. alive (B, N): the
    survivors after the damage phase; health (B, N): each ship's shield over its type's full
    shield. Elimination (game v12): the team with more health left over its living ships
    wins (a ship counts its shield fraction, 0 to 1: a shieldless survivor counts 0), equal
    health a draw. Zero-sum, so a side that runs from a fight it is losing cannot run out the
    clock into a draw."""
    if mode_rules.name == TEAM_DEATHMATCH:
        return torch.sign(mode.score[:, 0] - mode.score[:, 1]).to(torch.int32)
    if mode_rules.name == ELIMINATION:
        left = by_team(torch.where(alive, health, 0.0)).sum(-1)  # (B, 2)
        return torch.sign(left[:, 0] - left[:, 1]).to(torch.int32)
    if mode_rules.name == FRONTLINE:
        return frontline.timeout_result(mode.front).to(torch.int32)
    return torch.zeros_like(mode.front)


def result_win_loss(result: torch.Tensor) -> torch.Tensor:
    """(..., 2) float: a win/draw/loss result (...) as its weights of a win and of a loss,
    one-hot (a draw is neither)."""
    return torch.stack([result > 0, result < 0], -1).to(torch.float32)


def kill_share(score: torch.Tensor) -> torch.Tensor:
    """(B,) float: team 0's share of the kills, its score over both teams' (B, 2); 0.5 when
    neither team scored."""
    score = score.to(torch.float32)
    total = score.sum(-1)
    return torch.where(total > 0, score[:, 0] / total.clamp_min(1.0), 0.5)


def win_loss(mode_rules: ModeRules, mode: ModeState, result: torch.Tensor) -> torch.Tensor:
    """(B, 2) float: team 0's weights of a win and of a loss for a match that ended now with
    `result` (B,) (team 1's: flipped); a draw is the rest. Team deathmatch scores a game by
    its kill share, as its rating does: (share, 1 - share) (`kill_share`, of the final
    score), never a draw. Elimination and Frontline: the result one-hot."""
    if mode_rules.name == TEAM_DEATHMATCH:
        share = kill_share(mode.score)
        return torch.stack([share, 1.0 - share], -1)
    return result_win_loss(result)


def protected(rules: Rules, mode_rules: ModeRules, mode: ModeState, ships: Ships) -> torch.Tensor:
    """(B, N): hits on these ships deal no damage (Frontline: inside the own team's spawn
    zone, by the roles at tick start and the post-move position; other modes: none)."""
    if mode_rules.name != FRONTLINE:
        return torch.zeros_like(ships.alive)
    own = torch.stack(
        [mode.role == frontline.TEAM0_SPAWN, mode.role == frontline.TEAM1_SPAWN], 1
    ).repeat_interleave(rules.ships_per_team, 1)  # (B, N, Z)
    return (_in_zones(mode, ships) & own).any(-1)


def refit_zones(
    mode_rules: ModeRules, arena_radius: torch.Tensor, map_scale: torch.Tensor, mode: ModeState
) -> ModeState:
    """The mode state with Frontline's zone ring rebuilt for these rules (live tuning): the
    match's own ring rotation and handedness, the rules' zone radius and ring fraction, and
    the environments' arena radius and map scale (B,). The zones' roles, meters and the
    front are kept. A mode without zones passes through unchanged."""
    if mode_rules.name != FRONTLINE:
        return mode
    zone_x, zone_y, radius = _ring(
        mode_rules, arena_radius, map_scale, mode.ring_rotation, mode.handedness
    )
    return replace(mode, zone_x=zone_x, zone_y=zone_y, zone_radius=radius)


def _ring(mode_rules: ModeRules, arena_radius, map_scale, rotation, handedness):
    """Frontline's zone ring for these rules (`frontline.zone_ring`): the zone radius at
    the environments' map scale and the ring fraction of their arena, each held at its
    floor (`ModeRules.zone_radius_floor`, `zone_spacing_floor`: m at map scale 1, times the
    ship set's map scale, so a smaller arena does not shrink them)."""
    return frontline.zone_ring(
        arena_radius,
        mode_rules.zone_radius * map_scale,
        mode_rules.zone_ring_fraction,
        rotation,
        handedness,
        mode_rules.zone_radius_floor * mode_rules.set_map_scale,
        mode_rules.zone_spacing_floor * mode_rules.set_map_scale,
    )


def new_match(
    mode_rules: ModeRules,
    arena_radius: torch.Tensor,
    map_scale: torch.Tensor,
    uniforms: torch.Tensor,
    map_uniforms: torch.Tensor,
    fields: Fields,
    previous: ModeState,
    disc_fraction,
    tick_length: float,
) -> tuple[ModeState, Fields]:
    """A new match's mode state and map from the mode's tick uniforms (B, U_mode) and the
    environments' arena radius and map scale (B,): the drawn point-symmetric fields as they
    are, a fixed layout's fields (`layout_fields`) where the mode's rules have one, or, in
    Frontline, a random ring (rotation, then handedness) with fields mirrored about its
    neutral axis (the odd field within disc_fraction of the arena radius), no capture
    locks, and each team's first respawn wave drawn uniformly on 1..interval ticks (the
    mode's uniforms 2 and 3; tick_length, s). previous: the mode
    state as it stands. A mode without zones zeroes only
    the score; its other leaves are zero already and pass through unchanged (no copy of
    the absent slots)."""
    if mode_rules.name != FRONTLINE:
        if mode_rules.layout is not None and mode_rules.layout.fields:
            fields = layout_fields(mode_rules, map_uniforms, fields)
        return replace(previous, score=torch.zeros_like(previous.score)), fields
    blank = apply(torch.zeros_like, previous)
    rotation = 2.0 * math.pi * uniforms[:, 0]
    handedness = torch.where(uniforms[:, 1] < 0.5, 1.0, -1.0)
    zone_x, zone_y, radius = _ring(mode_rules, arena_radius, map_scale, rotation, handedness)
    roles = frontline.roles_from_front(blank.front)
    state = replace(
        blank,
        zone_x=zone_x,
        zone_y=zone_y,
        zone_radius=radius,
        role=roles,
        handedness=handedness,
        ring_rotation=rotation,
        wave_wait=frontline.first_waves(
            uniforms[:, 2:4],
            frontline.whole_ticks(mode_rules.wave_seconds, tick_length, rotation)[:, None],
        ),
    )
    return state, frontline.mirror_fields(
        fields, map_uniforms, arena_radius, rotation, disc_fraction
    )


def layout_rotation(mode_rules: ModeRules, map_uniforms: torch.Tensor):
    """(cos, sin) (B,) of a fixed layout's turn this match: the spawn axis's draw (the map's
    last uniform) when the layout is rotated, else none."""
    if mode_rules.layout.rotated:
        angle = 2.0 * math.pi * map_uniforms[:, -1]
        return torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(map_uniforms[:, -1])
    return zero + 1.0, zero


def layout_fields(mode_rules: ModeRules, map_uniforms: torch.Tensor, fields: Fields) -> Fields:
    """The fixed layout's fields (B, M) for a new match: its fields, then their images under
    the 180-degree rotation, the whole turned by `layout_rotation`. Field p and its image
    draw one level from the field's choices with the uniform the random map reads for
    field p's level (map uniform 4 p + 3). The band width is the drawn map's (the
    rules'). Built from the layout's numbers as constants (no host-to-device copy, so it
    is safe inside a CUDA graph's capture)."""
    cos, sin = layout_rotation(mode_rules, map_uniforms)
    xs, ys, radii, levels = [], [], [], []
    for index, field in enumerate(mode_rules.layout.fields):
        xs.append(field.x * cos - field.y * sin)
        ys.append(field.x * sin + field.y * cos)
        radii.append(torch.zeros_like(cos) + field.radius)
        uniform = map_uniforms[:, 4 * index + 3]
        choice = (len(field.levels) * uniform).to(torch.int32).clamp_max(len(field.levels) - 1)
        levels.append(lookup(field.levels, choice))
    x, y, radius = torch.stack(xs, -1), torch.stack(ys, -1), torch.stack(radii, -1)
    level = torch.stack(levels, -1).to(fields.level.dtype)
    return replace(
        fields,
        x=torch.cat([x, -x], 1),
        y=torch.cat([y, -y], 1),
        radius=torch.cat([radius, radius], 1),
        level=torch.cat([level, level], 1),
    )

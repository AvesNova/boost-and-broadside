"""The game state: typed dataclasses of tensors, registered as pytrees.

Opaque outside `bnb.game`: the runtime copies, clones and moves it as a pytree and never
reads a field. Ship order is team-major (ship `n = team * S + slot`), so a (B, N, ...)
tensor viewed as (B, 2, S, ...) is indexed by team for free.
"""

import dataclasses

import torch

from bnb.contracts.tree import contract

from .rules import TEAMS, Rules


@contract
class Ships:
    """(B, N), one entry per ship.

    Floats: position x, y (m); the nose direction (heading_cos, heading_sin); proper_speed
    (m/s); angle_of_attack (rad) with its cos and sin; nose_rate (rad/s); energy (J); the
    medium at the ship, cached from its last evaluation (log_refractive_index and its
    gradient, 1/m); shield; shield_recharge_wait (s).
    Flags: alive; fired and spawned this tick.
    Ints: respawn_wait and gun_cooldown (ticks); next_slot, the bullet slot the next shot
    uses; ship_type.
    """

    x: torch.Tensor
    y: torch.Tensor
    heading_cos: torch.Tensor
    heading_sin: torch.Tensor
    proper_speed: torch.Tensor
    angle_of_attack: torch.Tensor
    angle_of_attack_cos: torch.Tensor
    angle_of_attack_sin: torch.Tensor
    nose_rate: torch.Tensor
    energy: torch.Tensor
    log_refractive_index: torch.Tensor
    log_index_gradient_x: torch.Tensor
    log_index_gradient_y: torch.Tensor
    shield: torch.Tensor
    shield_recharge_wait: torch.Tensor
    alive: torch.Tensor
    fired: torch.Tensor
    spawned: torch.Tensor
    respawn_wait: torch.Tensor
    gun_cooldown: torch.Tensor
    next_slot: torch.Tensor
    ship_type: torch.Tensor


@contract
class Bullets:
    """(B, N, K), by owner and slot: position, unit direction, proper speed, the cached
    medium, and moves_left (0 = inactive)."""

    x: torch.Tensor
    y: torch.Tensor
    direction_x: torch.Tensor
    direction_y: torch.Tensor
    proper_speed: torch.Tensor
    log_refractive_index: torch.Tensor
    log_index_gradient_x: torch.Tensor
    log_index_gradient_y: torch.Tensor
    moves_left: torch.Tensor


@contract
class Fields:
    """(B, M): centre, radius and band width (m), level in {-2, -1, 1, 2}."""

    x: torch.Tensor
    y: torch.Tensor
    radius: torch.Tensor
    band_width: torch.Tensor
    level: torch.Tensor


@contract
class Match:
    """episode_tick (B,) ticks into the episode; the spawn axis (B,); ticks_since_seen
    (B, 2, N), ticks since team T last saw ship n (0 while visible); fogged (B,), the
    environment hides what its teams cannot see (fixed at reset by the caller)."""

    episode_tick: torch.Tensor
    spawn_axis_x: torch.Tensor
    spawn_axis_y: torch.Tensor
    ticks_since_seen: torch.Tensor
    fogged: torch.Tensor


@contract
class ModeState:
    """The mode's part of the match (modes spec 6). score (B, 2) int32, deaths scored per
    team (team deathmatch). Frontline, with Z zones (Z = 0 in other modes): front (B,)
    int32, captures ahead for team 0; the zones' centres and radii (B, Z); role (B, Z)
    int32 in team 0's frame; progress (B, Z), the meters; direction (B, Z) int32, the
    meters' last push in team 0's frame (+1 a team-0 majority); attack_ledger and
    defense_ledger (B, Z, N), the attempt ledgers; handedness (B,) +-1 and ring_rotation
    (B,) rad, the ring's draw."""

    score: torch.Tensor
    front: torch.Tensor
    zone_x: torch.Tensor
    zone_y: torch.Tensor
    zone_radius: torch.Tensor
    role: torch.Tensor
    progress: torch.Tensor
    direction: torch.Tensor
    attack_ledger: torch.Tensor
    defense_ledger: torch.Tensor
    handedness: torch.Tensor
    ring_rotation: torch.Tensor


@contract
class State:
    ships: Ships
    bullets: Bullets
    fields: Fields
    match: Match
    mode: ModeState


@contract
class Sweep:
    """This tick's straight-line motions, which the hit test sweeps: tick-start positions
    and displacements of the ships (B, N) and of the bullets after firing (B, N, K).
    Passed from `move` to `resolve`."""

    ship_x: torch.Tensor
    ship_y: torch.Tensor
    ship_displacement_x: torch.Tensor
    ship_displacement_y: torch.Tensor
    bullet_x: torch.Tensor
    bullet_y: torch.Tensor
    bullet_displacement_x: torch.Tensor
    bullet_displacement_y: torch.Tensor


replace = dataclasses.replace

FLOAT, BOOL, INT = torch.float32, torch.bool, torch.int32
SHIP_DTYPES = {field.name: FLOAT for field in dataclasses.fields(Ships)} | dict(
    alive=BOOL, fired=BOOL, spawned=BOOL, respawn_wait=INT, gun_cooldown=INT, next_slot=INT,
    ship_type=INT)
BULLET_DTYPES = {field.name: FLOAT for field in dataclasses.fields(Bullets)} | dict(moves_left=INT)
FIELD_DTYPES = {field.name: FLOAT for field in dataclasses.fields(Fields)} | dict(level=INT)
MODE_INTEGERS = ("score", "front", "role", "direction")


def blank(rules: Rules, batch: int, device=None) -> State:
    """Every value zero and every flag false (fogged true), in the canonical shapes."""
    B, N, K, M = batch, rules.num_ships, rules.bullet_slots, rules.num_fields

    def zeros(dtypes: dict, shape: tuple) -> dict:
        return {k: torch.zeros(shape, dtype=d, device=device) for k, d in dtypes.items()}

    Z = rules.mode.num_zones
    shapes = dict(score=(B, TEAMS), front=(B,), attack_ledger=(B, Z, N),
                  defense_ledger=(B, Z, N), handedness=(B,), ring_rotation=(B,))
    mode = {field.name: torch.zeros(shapes.get(field.name, (B, Z)), device=device,
                                    dtype=INT if field.name in MODE_INTEGERS else FLOAT)
            for field in dataclasses.fields(ModeState)}
    return State(ships=Ships(**zeros(SHIP_DTYPES, (B, N))),
                 bullets=Bullets(**zeros(BULLET_DTYPES, (B, N, K))),
                 fields=Fields(**zeros(FIELD_DTYPES, (B, M))),
                 match=Match(episode_tick=torch.zeros(B, dtype=INT, device=device),
                             spawn_axis_x=torch.zeros(B, device=device),
                             spawn_axis_y=torch.zeros(B, device=device),
                             ticks_since_seen=torch.zeros((B, TEAMS, N), dtype=INT,
                                                          device=device),
                             fogged=torch.ones(B, dtype=BOOL, device=device)),
                 mode=ModeState(**mode))


def sweep_like(state: State) -> Sweep:
    """A zero Sweep shaped for this state (a buffer for runtimes that cut between stages)."""
    ship, bullet = state.ships.x, state.bullets.x
    return Sweep(*(torch.zeros_like(ship) for _ in range(4)),
                 *(torch.zeros_like(bullet) for _ in range(4)))


def where(mask: torch.Tensor, new, old):
    """`new` where mask, `old` elsewhere, field by field for two dataclasses of one type.
    mask broadcasts from the left over each field's trailing axes."""
    out = {}
    for field in dataclasses.fields(old):
        chosen, kept = getattr(new, field.name), getattr(old, field.name)
        expanded = mask.reshape(mask.shape + (1,) * (kept.dim() - mask.dim()))
        out[field.name] = torch.where(expanded, chosen, kept)
    return type(old)(**out)


def apply(function, value):
    """A dataclass of the same type with `function` applied to every field."""
    return type(value)(**{field.name: function(getattr(value, field.name))
                          for field in dataclasses.fields(value)})


def by_team(values: torch.Tensor) -> torch.Tensor:
    """(B, N, ...) per-ship values as (B, 2, S, ...)."""
    return values.reshape(values.shape[0], TEAMS, values.shape[1] // TEAMS, *values.shape[2:])

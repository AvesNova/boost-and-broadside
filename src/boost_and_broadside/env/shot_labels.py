"""Counterfactual shot labels: what every shot did, and what a shot would have done.

Every living ship that does not fire a real bullet launches a *ghost* whenever
its ghost cooldown allows. A ghost has the real bullet's launch state, spread
and physics but deals no damage, costs no power, never reveals its shooter and
is never rendered or observed. A real shot resets both cooldowns, so launches of
either kind are at least one firing cooldown apart, and a real shot fired while
the ghost cooldown is still running *replaces* that ghost in its slot: the
ghost's example is discarded. Any other launch reclaims a slot rather than
following the ring cursor: a free one, else the oldest spent flight (a ghost, or
a real bullet that already hit), else the oldest live bullet. With launches a
firing cooldown apart and a lifetime of ``bullet_lifetime`` the existing
``max_bullets`` slots always hold a free one; when a same-tick respawn
compresses launches, a ghost gives way before any real bullet does, so gameplay
never depends on the ghosts.

Each launched bullet flies its whole lifetime, through any hit (a real bullet
that hits stops being ``bullet_active`` but keeps ``bullet_flying``). Along the
way its slot records:

- the state at two launch-drawn ages: ``tau1`` and a fallback ``tau2``,
- its first hit, from the same swept distances that decide gameplay hits, with
  the hit ship's side relative to the shooter, or else its closest pass to any
  other living ship, with that ship's side,
- the state at that encounter.

A state *snapshot* is the displacement from the launch point and the velocity,
both rotated into the launch attitude's frame, and the local refractive index.
Age ``a`` is the bullet after ``a`` transport steps; age 0 is the launch point
after the launch tick's ship movement.

When a bullet's lifetime runs out its record is emitted as the ship's example
for that tick. That is always :func:`shot_label_delay` ticks after its launch,
so a consumer can file it under the launch decision with a fixed offset.
Bullets cleared by an environment reset never expire and so never emit, which
is how examples cut short by episode termination are discarded.

Everything is dense over all slots, fixed-shape, and free of host syncs.
"""

from __future__ import annotations

import functools

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.state import TensorState

# Encounter sides, relative to the shooter.
NO_HIT = 0
ENEMY = 1
ALLY = 2

# A closest distance that means "nothing was ever in range": large and finite,
# so masked arithmetic never meets an inf.
NO_ENCOUNTER_DSQ = 1.0e12

# Above any bullet age, so a slot's rank dominates its age when the launch code
# reclaims the oldest slot of a rank.
MAX_AGE_KEY = 1 << 16

# Snapshot channels.
SNAP_DX, SNAP_DY, SNAP_VX, SNAP_VY, SNAP_INDEX = range(5)
SNAPSHOT_DIM = 5

# Example layout: what a slot emits on expiry.
EX_TAU1 = 0
EX_TAU_FALLBACK = 1
EX_EVENT_TAU = 2
EX_HIT = 3
EX_CLOSEST_DSQ = 4
EX_CLOSEST_SIDE = 5
EX_SNAP1 = slice(6, 6 + SNAPSHOT_DIM)
EX_SNAP_FALLBACK = slice(EX_SNAP1.stop, EX_SNAP1.stop + SNAPSHOT_DIM)
EX_SNAP_EVENT = slice(EX_SNAP_FALLBACK.stop, EX_SNAP_FALLBACK.stop + SNAPSHOT_DIM)
EXAMPLE_DIM = EX_SNAP_EVENT.stop

# Record layout: the launch pose, then the example.
_LAUNCH_POS = slice(0, 2)
_LAUNCH_ATT = slice(2, 4)
_EXAMPLE = 4
RECORD_DIM = _EXAMPLE + EXAMPLE_DIM


def _rec(index: int | slice) -> int | slice:
    """An example index shifted to its place in the record."""
    if isinstance(index, slice):
        return slice(index.start + _EXAMPLE, index.stop + _EXAMPLE)
    return index + _EXAMPLE


def shot_label_delay(config: ShipConfig) -> int:
    """Ticks from a bullet's launch to its expiry, which is also its last age.

    Counted with the same float32 countdown ``advance_bullets`` runs, so the
    offset can never disagree with the simulation by a rounding.
    """
    return _countdown_moves(config.bullet_lifetime, config.dt)


@functools.cache
def _countdown_moves(lifetime: float, dt: float) -> int:
    remaining = torch.tensor(lifetime, dtype=torch.float32)
    moves = 0
    while True:
        remaining = remaining - dt
        if not bool(remaining > 0):
            return moves
        moves += 1


def _complex(pair: torch.Tensor) -> torch.Tensor:
    return torch.complex(pair[..., 0], pair[..., 1])


def _snapshot(state: TensorState, config: ShipConfig) -> torch.Tensor:
    """Every slot's current state in its own launch frame, ``(B, N, K, 5)``."""
    from boost_and_broadside.env.frontline import toroidal_displacement

    record = state.bullet_shot_record
    frame = torch.conj(_complex(record[..., _LAUNCH_ATT]))
    displacement = (
        toroidal_displacement(
            state.bullet_pos - _complex(record[..., _LAUNCH_POS]), config.world_size
        )
        * frame
    )
    velocity = state.bullet_vel * frame
    return torch.stack(
        [
            displacement.real,
            displacement.imag,
            velocity.real,
            velocity.imag,
            state.bullet_local_index,
        ],
        dim=-1,
    )


def _write(record: torch.Tensor, index: int | slice, mask: torch.Tensor, value) -> None:
    """``record[..., index] = value`` where ``mask``, in place."""
    if isinstance(index, slice):
        record[..., index] = torch.where(mask.unsqueeze(-1), value, record[..., index])
    else:
        record[..., index] = torch.where(mask, value, record[..., index])


def record_launch(
    state: TensorState,
    launched: torch.Tensor,
    config: ShipConfig,
    generator: torch.Generator,
) -> None:
    """Start a fresh record in every slot written this tick.

    Called after the launch has written position, velocity and index, so the
    age-0 snapshot is the launch point. Slots not written keep their record.

    Args:
        state: State whose launched slots already hold the new bullets.
        launched: ``(B, N, K)`` bool, the slots written this tick.
        config: Physics configuration.
        generator: The label stream; gameplay randomness is never drawn here.
    """
    delay = shot_label_delay(config)
    record = state.bullet_shot_record
    shape = launched.shape
    taus = torch.randint(
        0,
        delay + 1,
        (*shape, 2),
        generator=generator,
        device=launched.device,
    ).to(record.dtype)
    attitude = state.ship_attitude.unsqueeze(-1).expand(shape)
    pos = state.bullet_pos
    _write(record, _LAUNCH_POS, launched, torch.stack([pos.real, pos.imag], dim=-1))
    _write(record, _LAUNCH_ATT, launched, torch.stack([attitude.real, attitude.imag], dim=-1))
    _write(record, _rec(EX_TAU1), launched, taus[..., 0])
    _write(record, _rec(EX_TAU_FALLBACK), launched, taus[..., 1])
    _write(record, _rec(EX_EVENT_TAU), launched, 0.0)
    _write(record, _rec(EX_HIT), launched, float(NO_HIT))
    _write(record, _rec(EX_CLOSEST_DSQ), launched, NO_ENCOUNTER_DSQ)
    _write(record, _rec(EX_CLOSEST_SIDE), launched, float(NO_HIT))
    _write(record, _rec(EX_SNAP_EVENT), launched, 0.0)
    state.bullet_flying = state.bullet_flying | launched
    state.bullet_age = torch.where(launched, 0, state.bullet_age)
    _record_scheduled_snapshots(state, launched, config)


def _record_scheduled_snapshots(
    state: TensorState, candidates: torch.Tensor, config: ShipConfig
) -> None:
    """Snapshot ``candidates`` whose age is one of their two drawn ages."""
    record = state.bullet_shot_record
    age = state.bullet_age.to(record.dtype)
    at_tau1 = candidates & (age == record[..., _rec(EX_TAU1)])
    at_fallback = candidates & (age == record[..., _rec(EX_TAU_FALLBACK)])
    snapshot = _snapshot(state, config)
    _write(record, _rec(EX_SNAP1), at_tau1, snapshot)
    _write(record, _rec(EX_SNAP_FALLBACK), at_fallback, snapshot)


def advance(state: TensorState, config: ShipConfig) -> None:
    """Age every flight by one transport step, emitting the ones that expired.

    Called right after ``advance_bullets``: a slot whose lifetime ran out this
    tick did not move, so its record is complete and becomes its ship's
    example. At most one slot per ship expires per tick, because launches are
    at least a firing cooldown apart.
    """
    expired = state.bullet_flying & (state.bullet_time <= 0)
    example = state.bullet_shot_record[..., _EXAMPLE:]
    state.shot_example = torch.where(expired.unsqueeze(-1), example, 0.0).sum(dim=2)
    state.shot_example_valid = expired.any(dim=2)

    state.bullet_flying = state.bullet_flying & (state.bullet_time > 0)
    state.bullet_age = state.bullet_age + state.bullet_flying.int()
    _record_scheduled_snapshots(state, state.bullet_flying, config)


def record_encounters(
    state: TensorState,
    enemy_dsq: torch.Tensor,
    ally_dsq: torch.Tensor,
    config: ShipConfig,
) -> None:
    """Fold this tick's swept distances into every undecided flight.

    The first tick within the collision radius of a living ship decides the
    shot: that ship's side, the distance and the snapshot are latched and
    nothing later changes them. Until then every new closest pass replaces the
    last. A tie between sides goes to the enemy.

    Args:
        state: State after this tick's transport.
        enemy_dsq: ``(B, N, K)`` this tick's least squared swept distance from
            each slot to a living enemy of its shooter.
        ally_dsq: ``(B, N, K)`` the same for the shooter's living allies.
        config: Physics configuration.
    """
    record = state.bullet_shot_record
    undecided = state.bullet_flying & (record[..., _rec(EX_HIT)] == NO_HIT)
    nearest = torch.minimum(enemy_dsq, ally_dsq)
    side = torch.where(ally_dsq < enemy_dsq, float(ALLY), float(ENEMY))
    hit = undecided & (nearest < config.collision_radius**2)
    closer = undecided & ~hit & (nearest < record[..., _rec(EX_CLOSEST_DSQ)])
    event = hit | closer
    _write(record, _rec(EX_HIT), hit, side)
    _write(record, _rec(EX_CLOSEST_DSQ), event, nearest)
    _write(record, _rec(EX_CLOSEST_SIDE), event, side)
    _write(record, _rec(EX_EVENT_TAU), event, state.bullet_age.to(record.dtype))
    _write(record, _rec(EX_SNAP_EVENT), event, _snapshot(state, config))

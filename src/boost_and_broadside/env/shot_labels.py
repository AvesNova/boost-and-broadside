"""Counterfactual shot labels, re-simulated by the learner from stored truth.

Every decision of every living ship gets one label: what a bullet fired on that
decision would have done. Nothing is launched in the environment. Each tick,
the environment copies every ship's launch state (:func:`capture`) -- read
after the tick's kinematics and before its collisions, which is both where a
real launch reads from and what that tick's swept hit test checks bullets
against -- and the trainer stores it per decision row with the field layout.
After the rollout, :func:`resimulate` fires the same bullet the environment
would (``physics.bullet_launch_velocity``), flies it with the environment's
own transport (``physics.fly_bullets``) and tests it with its own swept
distance (``physics._swept_dist_sq``) against the stored ships of the rows that
follow. A shot launched on row ``t`` moves :func:`shot_label_delay` times,
tested against rows ``t`` to ``t + delay - 1``.

Each shot records:

- the state at two drawn ages: ``tau1`` and a fallback ``tau2``,
- its first hit, with the hit ship's side relative to the shooter, or else
  its closest pass to any other living ship, with that ship's side,
- the state at that encounter.

A state *snapshot* is the displacement from the launch point and the velocity,
both rotated into the launch attitude's frame, and the local refractive index.
Age ``a`` is the bullet after ``a`` transport steps; age 0 is the launch point.

A shot is invalid when its ship is dead on the launch row, or its flight runs
past the end of the rollout or across an episode end. Gameplay never sees any
of this: the environment only copies state it already has.

Everything is dense, fixed-shape, and free of host syncs.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

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

# Snapshot channels.
SNAP_DX, SNAP_DY, SNAP_VX, SNAP_VY, SNAP_INDEX = range(5)
SNAPSHOT_DIM = 5

# Example layout: one shot's label.
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


def shot_label_delay(config: ShipConfig) -> int:
    """Transport steps in a bullet's lifetime, which is also its last age.

    Counted with the same float32 countdown ``advance_bullets`` runs, so the
    flight can never disagree with the simulation's by a rounding.
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


class ShotLaunchState(NamedTuple):
    """Every ship as a bullet fired this tick would leave it, ``(..., N)`` each.

    Read after a tick's kinematics and before its collisions: the state a
    launch reads, and the one that tick's swept hit test checks bullets
    against.
    """

    position: torch.Tensor  # complex64
    velocity: torch.Tensor  # complex64
    attitude: torch.Tensor  # complex64
    local_index: torch.Tensor  # float32
    field_gradient: torch.Tensor  # complex64
    alive: torch.Tensor  # bool
    team_id: torch.Tensor  # int


class FieldLayout(NamedTuple):
    """An environment's fields, ``(..., M)`` each. Fixed between resets."""

    field_pos: torch.Tensor  # complex64
    field_radius: torch.Tensor
    field_transition_width: torch.Tensor
    field_index: torch.Tensor


def capture(state: TensorState) -> tuple[ShotLaunchState, FieldLayout]:
    """Copy the launch state and field layout the re-simulation needs."""
    launch = ShotLaunchState(
        state.ship_pos.clone(),
        state.ship_vel.clone(),
        state.ship_attitude.clone(),
        state.ship_local_index.clone(),
        state.ship_field_gradient.clone(),
        state.ship_alive.clone(),
        state.ship_team_id.clone(),
    )
    fields = FieldLayout(
        state.field_pos.clone(),
        state.field_radius.clone(),
        state.field_transition_width.clone(),
        state.field_index.clone(),
    )
    return launch, fields


def _flat_snapshot(
    position: torch.Tensor,
    velocity: torch.Tensor,
    local_index: torch.Tensor,
    launch_pos: torch.Tensor,
    frame: torch.Tensor,
    config: ShipConfig,
) -> torch.Tensor:
    """``(..., 5)`` snapshot in the launch attitude's frame."""
    from boost_and_broadside.env.frontline import toroidal_displacement

    displacement = toroidal_displacement(position - launch_pos, config.world_size) * frame
    velocity = velocity * frame
    return torch.stack(
        [displacement.real, displacement.imag, velocity.real, velocity.imag, local_index],
        dim=-1,
    )


def launchable_rows(episode_end: torch.Tensor, delay: int) -> torch.Tensor:
    """``(T, B)`` bool: a shot launched on this row has its whole flight here.

    A shot launched on row ``t`` is tested against the ships of rows ``t`` to
    ``t + delay - 1``, so that span must lie inside the rollout and inside one
    episode. An episode ending on the flight's last row is fine: that row's
    ships were read before the reset.
    """
    T = episode_end.shape[0]
    ends = torch.zeros((T + 1, *episode_end.shape[1:]), device=episode_end.device)
    ends[1:] = episode_end.float().cumsum(0)
    rows = torch.arange(T, device=episode_end.device)
    last = rows + delay - 1  # the flight's last row
    in_rollout = last <= T - 1
    last = last.clamp(max=T - 1)
    # Episode ends on rows t .. last - 1.
    crossed = ends[last] - ends[rows] > 0
    return in_rollout.view(T, *([1] * (episode_end.dim() - 1))) & ~crossed


def resimulate(
    launch: ShotLaunchState,
    fields: FieldLayout,
    episode_end: torch.Tensor,
    config: ShipConfig,
    generator: torch.Generator | None = None,
    taus: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    max_bullets_per_chunk: int = 1 << 21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fly one counterfactual shot from every ship on every row of a rollout.

    Each shot is the bullet the ship would have fired on that row's decision:
    the same launch, flight and swept hit test the environment runs
    (``physics.bullet_launch_velocity``, ``physics.fly_bullets``,
    ``physics._swept_dist_sq``), against the stored ships of the following
    rows. A shot whose flight runs past the rollout or across an episode end,
    or whose ship was dead on its launch row, is invalid.

    Args:
        launch: ``(T, B, N)`` launch states, one per decision row.
        fields: ``(T, B, M)`` field layouts, one per row.
        episode_end: ``(T, B)`` bool, the episode ended after this row.
        config: Physics configuration.
        generator: Stream for the spread and the queried ages.
        taus: Optional ``(T, B, N, 2)`` queried ages in place of a draw.
        noise: Optional ``(T, B, N)`` complex spread in place of a draw.
        max_bullets_per_chunk: Bound on bullets flown at once.

    Returns:
        ``(T, B, N, EXAMPLE_DIM)`` examples and ``(T, B, N)`` validity.
    """
    from boost_and_broadside.env import physics

    T, B, N = launch.position.shape
    device = launch.position.device
    delay = shot_label_delay(config)
    if taus is None:
        taus = torch.randint(0, delay + 1, (T, B, N, 2), generator=generator, device=device).float()
    if noise is None:
        spread = torch.randn((T, B, N, 2), generator=generator, device=device)
        noise = torch.complex(spread[..., 0], spread[..., 1]) * config.bullet_spread
    valid = launchable_rows(episode_end, delay).unsqueeze(-1) & launch.alive
    examples = torch.zeros((T, B, N, EXAMPLE_DIM), device=device)
    L = T - delay + 1
    if L <= 0:
        return examples, torch.zeros_like(valid)

    chunk = max(1, max_bullets_per_chunk // (L * N))
    radius_sq = config.collision_radius**2
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)

        def rows(x: torch.Tensor, start: int) -> torch.Tensor:
            return x[start : start + L, b0:b1].flatten(0, 1)

        layout = FieldLayout(*(rows(f, 0) for f in fields))
        launch_pos = rows(launch.position, 0)
        frame = torch.conj(rows(launch.attitude, 0))
        local_index = rows(launch.local_index, 0)
        field_gradient = rows(launch.field_gradient, 0)
        velocity = physics.bullet_launch_velocity(
            rows(launch.velocity, 0), rows(launch.attitude, 0), local_index, rows(noise, 0), config
        )
        position = launch_pos
        tau = rows(taus, 0)
        tau1, tau2 = tau[..., 0], tau[..., 1]

        def snap() -> torch.Tensor:
            return _flat_snapshot(position, velocity, local_index, launch_pos, frame, config)

        first = snap()
        snap1 = torch.where((tau1 == 0).unsqueeze(-1), first, 0.0)
        snap2 = torch.where((tau2 == 0).unsqueeze(-1), first, 0.0)
        hit = torch.full_like(tau1, float(NO_HIT))
        closest = torch.full_like(tau1, NO_ENCOUNTER_DSQ)
        side_seen = torch.full_like(tau1, float(NO_HIT))
        event_tau = torch.zeros_like(tau1)
        snap_event = torch.zeros_like(first)
        for age in range(1, delay + 1):
            flight = physics.fly_bullets(
                position, velocity, local_index, field_gradient, layout, config
            )
            position, velocity = flight.position, flight.velocity
            local_index, field_gradient = flight.local_index, flight.field_gradient
            current = snap()
            snap1 = torch.where((tau1 == age).unsqueeze(-1), current, snap1)
            snap2 = torch.where((tau2 == age).unsqueeze(-1), current, snap2)

            ships = rows(launch.position, age - 1)
            dsq = physics._swept_dist_sq(
                ships,
                flight.start.unsqueeze(-1),
                flight.midpoint.unsqueeze(-1),
                position.unsqueeze(-1),
                config.world_size,
            )  # (rows, N shooters, N targets)
            team = rows(launch.team_id, age - 1)
            target = rows(launch.alive, age - 1).unsqueeze(1) & not_self
            same = team.unsqueeze(2) == team.unsqueeze(1)
            far = torch.full_like(dsq, NO_ENCOUNTER_DSQ)
            enemy = torch.where(target & ~same, dsq, far).amin(-1)
            ally = torch.where(target & same, dsq, far).amin(-1)

            undecided = hit == NO_HIT
            nearest = torch.minimum(enemy, ally)
            side = torch.where(ally < enemy, float(ALLY), float(ENEMY))
            hits = undecided & (nearest < radius_sq)
            event = hits | (undecided & (nearest < closest))
            hit = torch.where(hits, side, hit)
            closest = torch.where(event, nearest, closest)
            side_seen = torch.where(event, side, side_seen)
            event_tau = torch.where(event, float(age), event_tau)
            snap_event = torch.where(event.unsqueeze(-1), current, snap_event)

        example = torch.cat(
            [
                torch.stack([tau1, tau2, event_tau, hit, closest, side_seen], dim=-1),
                snap1,
                snap2,
                snap_event,
            ],
            dim=-1,
        )
        examples[:L, b0:b1] = example.view(L, b1 - b0, N, EXAMPLE_DIM)
    examples = torch.where(valid.unsqueeze(-1), examples, 0.0)
    return examples, valid

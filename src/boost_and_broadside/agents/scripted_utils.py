"""Shared geometry utilities for deterministic scripted agents."""

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.state import TensorState


def compute_obstacle_repulsion(
    state: TensorState,
    ship_config: ShipConfig,
    tti_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute potential-field repulsion vectors from all obstacles.

    For each ship, computes the TTI to every obstacle via the quadratic
    constant-velocity approximation, then sums weighted unit vectors pointing
    away from each obstacle.  Weight = (1 - TTI/tti_max)^2, so repulsion grows
    sharply as a collision becomes imminent.

    Returns:
        repulsion: (B, N) complex64 — summed repulsion vectors (unnormalized magnitude)
        min_tti:   (B, N) float32  — TTI to nearest obstacle (inf = no threat)
    """
    B, N = state.ship_pos.shape
    M = state.num_obstacles
    device = state.device
    W, H = ship_config.world_size

    if M == 0:
        return (
            torch.zeros(B, N, dtype=state.ship_pos.dtype, device=device),
            torch.full((B, N), float("inf"), device=device),
        )

    # Relative position ship − obstacle, toroidal wrapped  (B, N, M)
    rel_r = state.ship_pos.real.unsqueeze(2) - state.obstacle_pos.real.unsqueeze(1)
    rel_i = state.ship_pos.imag.unsqueeze(2) - state.obstacle_pos.imag.unsqueeze(1)
    rel_r = (rel_r + W / 2) % W - W / 2
    rel_i = (rel_i + H / 2) % H - H / 2

    # Relative velocity  (B, N, M)
    vel_r = state.ship_vel.real.unsqueeze(2) - state.obstacle_vel.real.unsqueeze(1)
    vel_i = state.ship_vel.imag.unsqueeze(2) - state.obstacle_vel.imag.unsqueeze(1)

    # Combined hitbox radius per obstacle  (B, 1, M)
    hit_r = (ship_config.obstacle_collision_radius + state.obstacle_radius).unsqueeze(1)

    # Quadratic TTI: |vel|²t² + 2·dot(rel,vel)·t + |rel|² − r² = 0
    a = vel_r * vel_r + vel_i * vel_i                       # (B, N, M)
    b = rel_r * vel_r + rel_i * vel_i
    c = rel_r * rel_r + rel_i * rel_i - hit_r * hit_r

    disc = b * b - a * c
    sqrt_disc = disc.clamp(min=0.0).sqrt()
    tti_raw = (-b - sqrt_disc) / (a + 1e-8)
    tti_raw = tti_raw.clamp(min=0.0)

    already_inside = c < 0
    no_future_collision = (disc < 0) | (a < 1e-8)

    tti = torch.where(already_inside, torch.zeros_like(tti_raw), tti_raw)
    tti = torch.where(
        no_future_collision & ~already_inside,
        torch.full_like(tti, float("inf")),
        tti,
    )

    # Repulsion weight: (1 − TTI/tti_max)²
    weight = (1.0 - tti / tti_max).clamp(0.0, 1.0) ** 2   # (B, N, M)

    # Repulsion direction: unit vector from obstacle toward ship
    dist = (rel_r * rel_r + rel_i * rel_i).sqrt() + 1e-8
    dir_r = rel_r / dist
    dir_i = rel_i / dist

    # Weighted sum across obstacles
    rep_r = (weight * dir_r).sum(dim=2)                     # (B, N)
    rep_i = (weight * dir_i).sum(dim=2)
    repulsion = torch.complex(rep_r, rep_i)

    min_tti = tti.min(dim=2).values                         # (B, N)
    return repulsion, min_tti


def predict_interception(
    state: TensorState,
    ship_config: ShipConfig,
    target_idx: torch.Tensor,
    closest_dist: torch.Tensor,
) -> torch.Tensor:
    """Return the unit direction vector to the first-order interception point.

    Accounts for target and shooter motion during bullet travel time.

    Args:
        target_idx:   (B, N) long  — index of the target ship for each shooter
        closest_dist: (B, N) float — current distance to that target

    Returns:
        (B, N) complex64 — unit vector toward predicted intercept
    """
    W, H = ship_config.world_size

    target_pos = torch.gather(state.ship_pos, 1, target_idx)
    target_vel = torch.gather(state.ship_vel, 1, target_idx)

    t_intercept = closest_dist / ship_config.bullet_speed

    pred_pos         = target_pos + target_vel * t_intercept
    shooter_future   = state.ship_pos + state.ship_vel * t_intercept

    diff = pred_pos - shooter_future
    diff.real = (diff.real + W / 2) % W - W / 2
    diff.imag = (diff.imag + H / 2) % H - H / 2

    return diff / (torch.abs(diff) + 1e-8)


def compute_team_target_bearings(
    state: TensorState, ship_config: ShipConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each ship, return geometry toward its team's shared target.

    The team target is the alive enemy closest to the team's toroidal center of mass.

    Returns:
        bearings:   (B, N) complex64 — unit vector from each ship toward its team's target
        distances:  (B, N) float     — toroidal distance from each ship to its team's target
        target_idx: (B, N) long      — ship index of the team target (same for all on a team)
        has_target: (B, N) bool
    """
    B, N = state.ship_pos.shape
    W, H = ship_config.world_size
    device = state.device

    bearings   = torch.zeros(B, N, dtype=state.ship_pos.dtype, device=device)
    distances  = torch.zeros(B, N, dtype=torch.float32, device=device)
    target_idx = torch.zeros(B, N, dtype=torch.long, device=device)
    has_target = torch.zeros(B, N, dtype=torch.bool, device=device)

    # Use ship 0 as a stable toroidal reference for computing CoM displacements.
    # Any fixed anchor works; all ships in an active match are well within half-world radius.
    anchor = state.ship_pos[:, 0]  # (B,)

    for team in (0, 1):
        friend_mask = (state.ship_team_id == team) & state.ship_alive  # (B, N)
        enemy_mask  = (state.ship_team_id != team) & state.ship_alive  # (B, N)

        # Toroidal CoM of alive friendlies
        disp = state.ship_pos - anchor.unsqueeze(1)
        disp.real = (disp.real + W / 2) % W - W / 2
        disp.imag = (disp.imag + H / 2) % H - H / 2

        f     = friend_mask.float()
        count = f.sum(dim=1, keepdim=True).clamp(min=1)
        com   = anchor + torch.complex(
            (disp.real * f).sum(dim=1) / count.squeeze(1),
            (disp.imag * f).sum(dim=1) / count.squeeze(1),
        )  # (B,)

        # Closest alive enemy to CoM
        diff_to_enemy = state.ship_pos - com.unsqueeze(1)
        diff_to_enemy.real = (diff_to_enemy.real + W / 2) % W - W / 2
        diff_to_enemy.imag = (diff_to_enemy.imag + H / 2) % H - H / 2
        dist_to_enemy = torch.abs(diff_to_enemy)

        dist_masked = torch.where(
            enemy_mask, dist_to_enemy, torch.tensor(float("inf"), device=device)
        )
        min_dists, team_tgt_idx = dist_masked.min(dim=1)  # (B,)
        team_has_target = min_dists < float("inf")         # (B,)

        # Broadcast team target index to all ships on this team: (B,) → (B, N)
        team_tgt_idx_bn = team_tgt_idx.unsqueeze(1).expand(B, N)

        target_pos = torch.gather(state.ship_pos, 1, team_tgt_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Bearing and distance from each ship to the shared team target
        diff = target_pos.unsqueeze(1) - state.ship_pos
        diff.real = (diff.real + W / 2) % W - W / 2
        diff.imag = (diff.imag + H / 2) % H - H / 2
        dist    = torch.abs(diff)        # (B, N)
        bearing = diff / (dist + 1e-8)  # (B, N)

        is_friend = (state.ship_team_id == team)
        bearings   = torch.where(is_friend, bearing, bearings)
        distances  = torch.where(is_friend, dist, distances)
        target_idx = torch.where(is_friend, team_tgt_idx_bn, target_idx)
        has_target = torch.where(
            is_friend,
            team_has_target.unsqueeze(1).expand(B, N),
            has_target,
        )

    return bearings, distances, target_idx, has_target


def select_targets(
    state: TensorState, ship_config: ShipConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the nearest alive enemy for each ship and return the bearing to it.

    Returns:
        closest_dist:  (B, N) float  — toroidal distance to nearest enemy
        target_idx:    (B, N) long   — ship index of nearest enemy
        has_target:    (B, N) bool   — False when no alive enemy exists
        bearing:       (B, N) complex64 — unit vector from each ship toward its target
    """
    world_width, world_height = ship_config.world_size
    device = state.device

    pos_targets = state.ship_pos.unsqueeze(1)   # (B, 1, N)
    pos_sources = state.ship_pos.unsqueeze(2)   # (B, N, 1)

    diff = pos_targets - pos_sources            # (B, N, N) — from source to target
    diff.real = (diff.real + world_width / 2) % world_width - world_width / 2
    diff.imag = (diff.imag + world_height / 2) % world_height - world_height / 2

    dist = torch.abs(diff)                      # (B, N, N)

    team_src = state.ship_team_id.unsqueeze(2)
    team_tgt = state.ship_team_id.unsqueeze(1)
    enemy_mask = team_src != team_tgt
    alive_tgt = state.ship_alive.unsqueeze(1)
    valid_tgt = enemy_mask & alive_tgt

    dist_masked = torch.where(valid_tgt, dist, torch.tensor(float("inf"), device=device))
    closest_dist, target_idx = torch.min(dist_masked, dim=2)
    has_target = closest_dist < float("inf")

    bearing_diff = torch.gather(diff, 2, target_idx.unsqueeze(2)).squeeze(2)  # (B, N)
    bearing = bearing_diff / (torch.abs(bearing_diff) + 1e-8)

    return closest_dist, target_idx, has_target, bearing

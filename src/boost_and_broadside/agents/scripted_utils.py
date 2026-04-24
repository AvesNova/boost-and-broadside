"""Shared geometry utilities for deterministic scripted agents."""

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.state import TensorState


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

"""TensorState: the complete GPU-resident state of all parallel environments."""

import torch
from dataclasses import dataclass


@dataclass
class TensorState:
    """Complete state of all parallel environments as GPU tensors.

    All tensors share the same device. Shape notation:
      B = num_envs, N = max_ships, K = max_bullets per ship.

    The dataclass is NOT frozen so that physics functions can write into fields
    in-place. Callers that need a snapshot must call .clone() explicitly.
    """

    step_count: torch.Tensor  # (B,) int32

    # Ship kinematic state
    ship_pos: torch.Tensor  # (B, N) complex64  — world position
    ship_vel: torch.Tensor  # (B, N) complex64  — velocity
    ship_attitude: torch.Tensor  # (B, N) complex64  — unit heading vector
    ship_ang_vel: torch.Tensor  # (B, N) float32    — angular velocity (rad/s)

    # Ship resource state
    ship_health: torch.Tensor  # (B, N) float32
    ship_power: torch.Tensor  # (B, N) float32
    ship_cooldown: torch.Tensor  # (B, N) float32    — seconds until next shot

    # Identity / status
    ship_team_id: torch.Tensor  # (B, N) int32
    ship_alive: torch.Tensor  # (B, N) bool
    ship_is_shooting: torch.Tensor  # (B, N) bool

    # Action taken at the previous step (for observation)
    prev_action: torch.Tensor  # (B, N, 3) float32  — [power, turn, shoot]

    # Bullet ring-buffer (K = max_bullets per ship)
    bullet_pos: torch.Tensor  # (B, N, K) complex64
    bullet_vel: torch.Tensor  # (B, N, K) complex64
    bullet_time: torch.Tensor  # (B, N, K) float32  — remaining lifetime (s)
    bullet_active: torch.Tensor  # (B, N, K) bool

    # Ring-buffer write cursor
    bullet_cursor: torch.Tensor  # (B, N) int64

    # Per-step and per-episode damage attribution (shooter × target)
    damage_matrix: (
        torch.Tensor
    )  # (B, N, N) float32  — damage dealt this step; zeroed each step
    cumulative_damage_matrix: (
        torch.Tensor
    )  # (B, N, N) float32  — accumulated this episode; zeroed on reset

    # Obstacle state (M = num_obstacles per env)
    obstacle_pos: torch.Tensor  # (B, M) complex64  — world position
    obstacle_vel: torch.Tensor  # (B, M) complex64  — velocity
    obstacle_radius: torch.Tensor  # (B, M) float32    — circle radius
    obstacle_gcenter: torch.Tensor  # (B, M) complex64  — per-obstacle harmonic gravity center

    # Per-step flag: True for ships that were killed by an obstacle this step
    ship_hit_obstacle: torch.Tensor  # (B, N) bool

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.ship_pos.shape[0]

    @property
    def max_ships(self) -> int:
        return self.ship_pos.shape[1]

    @property
    def max_bullets(self) -> int:
        return self.bullet_pos.shape[2]

    @property
    def num_obstacles(self) -> int:
        return self.obstacle_pos.shape[1]

    @property
    def device(self) -> torch.device:
        return self.ship_pos.device

    def clone(self) -> "TensorState":
        """Deep copy — all tensors are cloned onto the same device."""
        return TensorState(
            step_count=self.step_count.clone(),
            ship_pos=self.ship_pos.clone(),
            ship_vel=self.ship_vel.clone(),
            ship_attitude=self.ship_attitude.clone(),
            ship_ang_vel=self.ship_ang_vel.clone(),
            ship_health=self.ship_health.clone(),
            ship_power=self.ship_power.clone(),
            ship_cooldown=self.ship_cooldown.clone(),
            ship_team_id=self.ship_team_id.clone(),
            ship_alive=self.ship_alive.clone(),
            ship_is_shooting=self.ship_is_shooting.clone(),
            prev_action=self.prev_action.clone(),
            bullet_pos=self.bullet_pos.clone(),
            bullet_vel=self.bullet_vel.clone(),
            bullet_time=self.bullet_time.clone(),
            bullet_active=self.bullet_active.clone(),
            bullet_cursor=self.bullet_cursor.clone(),
            damage_matrix=self.damage_matrix.clone(),
            cumulative_damage_matrix=self.cumulative_damage_matrix.clone(),
            obstacle_pos=self.obstacle_pos.clone(),
            obstacle_vel=self.obstacle_vel.clone(),
            obstacle_radius=self.obstacle_radius.clone(),
            obstacle_gcenter=self.obstacle_gcenter.clone(),
            ship_hit_obstacle=self.ship_hit_obstacle.clone(),
        )

"""TensorState: the complete GPU-resident state of all parallel environments."""

from dataclasses import dataclass, fields

import torch


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
    damage_matrix: torch.Tensor  # (B, N, N) float32  — damage dealt this step; zeroed each step
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
            **{field.name: getattr(self, field.name).clone() for field in fields(self)}
        )

    def slice_envs(self, env_slice: slice) -> "TensorState":
        """Return a view-backed state containing only the selected environments."""
        return TensorState(
            **{field.name: getattr(self, field.name)[env_slice] for field in fields(self)}
        )

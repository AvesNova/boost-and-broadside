"""TensorState: the complete GPU-resident state of all parallel environments."""

from dataclasses import dataclass, fields

import torch


@dataclass
class TensorState:
    """Complete state of all parallel environments as GPU tensors.

    All tensors share the same device. Shape notation:
      B = num_envs, N = max_ships, K = max_bullets per ship,
      M = num_fields.

    The dataclass is NOT frozen so physics functions can advance state between
    steps (the refractive-field map itself remains static). The load-bearing
    convention is that they do so by *reassignment*
    (`state.x = f(state.x)`), not in-place mutation of the existing tensor. This
    is what lets a snapshot alias the pre-step tensors cheaply: a later
    reassignment on the live state rebinds the attribute and leaves the aliased
    tensor untouched. Two snapshot mechanisms rely on this:

      - `.clone()` — deep copy, safe to read from for every field.
      - `_make_prev_state_proxy` (env/wrapper.py) — aliases all tensors and swaps
        in only pre-damage `ship_health`/`ship_alive`. Reward components must read
        *only* those two fields from the proxy; any other field aliases the live,
        already-advanced state.

    Converting a physics field's advance to in-place mutation (e.g. for the
    allocation win STYLE_GUIDE §6.5 favors) would silently break that proxy — so
    it is deliberately reassignment. The few in-place writes on state fields
    (`damage_matrix`) are per-step scratch/outputs that
    nothing snapshots, which is why they are exempt.
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
    bullet_remaining_damage: torch.Tensor  # (B, N, K) float32
    bullet_field_alpha: torch.Tensor  # (B, N, K, M) float32
    bullet_local_index: torch.Tensor  # (B, N, K) float32
    bullet_field_gradient: torch.Tensor  # (B, N, K) complex64 — grad(n)

    # Ring-buffer write cursor
    bullet_cursor: torch.Tensor  # (B, N) int64

    # Per-step and per-episode damage attribution (shooter × target)
    damage_matrix: torch.Tensor  # (B, N, N) float32  — damage dealt this step; zeroed each step
    cumulative_damage_matrix: (
        torch.Tensor
    )  # (B, N, N) float32  — accumulated this episode; zeroed on reset

    # Static refractive-field map. Parent and delta-index values are computed
    # during map construction and never discovered in the per-step hot path.
    field_pos: torch.Tensor  # (B, M) complex64
    field_radius: torch.Tensor  # (B, M) float32 — nominal interface radius
    field_transition_width: torch.Tensor  # (B, M) float32 — complete interface band
    field_index_level: torch.Tensor  # (B, M) int8 — {-2, -1, +1, +2}
    field_index: torch.Tensor  # (B, M) float32 — absolute interior n
    field_damage_level: torch.Tensor  # (B, M) int8 — {0, 1, 2}
    field_damage: torch.Tensor  # (B, M) float32 — damage per complete crossing
    field_parent: torch.Tensor  # (B, M) int64 — direct parent, -1 for roots
    field_delta_index: torch.Tensor  # (B, M) float32 — n_inside - n_parent

    # Cached ship-field evaluation. The alpha cache is also the previous alpha
    # used by smooth total-variation interface damage.
    ship_field_alpha: torch.Tensor  # (B, N, M) float32
    ship_local_index: torch.Tensor  # (B, N) float32
    ship_field_gradient: torch.Tensor  # (B, N) complex64 — grad(n)
    ship_field_damage: torch.Tensor  # (B, N) float32 — interface damage this step

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
    def num_fields(self) -> int:
        return self.field_pos.shape[1]

    @property
    def num_obstacles(self) -> int:
        """Deprecated read-only alias for pre-field integrations."""

        return self.num_fields

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

"""TensorEnv: pure physics engine for vectorized parallel game environments.

This class owns the physics simulation only. Observation construction, reward
computation, and episode tracking are handled by MVPEnvWrapper (env/wrapper.py).
"""

import math
import torch
from typing import Any

from boost_and_broadside.config import ShipConfig, EnvConfig
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.env.physics import (
    _get_lookup_tables, update_ships, update_bullets, resolve_collisions
)


class TensorEnv:
    """Vectorized GPU physics engine for B parallel game instances.

    Attributes:
        num_envs: Number of parallel environments (B).
        ship_config: Physics constants.
        env_config: Environment sizing (ships, bullets, episode length).
        device: Torch device for all tensors.
        state: Live TensorState; updated each call to step().
    """

    def __init__(
        self,
        num_envs: int,
        ship_config: ShipConfig,
        env_config: EnvConfig,
        device: str | torch.device,
    ) -> None:
        self.num_envs    = num_envs
        self.ship_config = ship_config
        self.env_config  = env_config
        self.device      = torch.device(device)
        self.state: TensorState | None = None

        # Pre-cached per-step constants — built once, reused every step
        self._physics_tables: tuple | None = None
        self._self_mask: torch.Tensor | None = None
        # Pre-built reset templates — avoid per-episode allocation
        self._reset_alive: torch.Tensor | None = None
        self._reset_base_teams: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Allocate state tensors and reset all environments.

        Args:
            options: Optional dict. Supported keys:
                - "team_sizes": (n_team0, n_team1) tuple.
            seed: Optional RNG seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self._allocate_state()
        self._build_cached_tensors()
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_envs(mask, options)

    def _allocate_state(self) -> None:
        """Pre-allocate all state tensors on device."""
        B = self.num_envs
        N = self.env_config.num_ships
        K = self.env_config.max_bullets
        dev = self.device

        self.state = TensorState(
            step_count     = torch.zeros((B,),       dtype=torch.int32,    device=dev),
            ship_pos       = torch.zeros((B, N),     dtype=torch.complex64, device=dev),
            ship_vel       = torch.zeros((B, N),     dtype=torch.complex64, device=dev),
            ship_attitude  = torch.zeros((B, N),     dtype=torch.complex64, device=dev),
            ship_ang_vel   = torch.zeros((B, N),     dtype=torch.float32,   device=dev),
            ship_health    = torch.zeros((B, N),     dtype=torch.float32,   device=dev),
            ship_power     = torch.zeros((B, N),     dtype=torch.float32,   device=dev),
            ship_cooldown  = torch.zeros((B, N),     dtype=torch.float32,   device=dev),
            ship_team_id   = torch.zeros((B, N),     dtype=torch.int32,     device=dev),
            ship_alive     = torch.zeros((B, N),     dtype=torch.bool,      device=dev),
            ship_is_shooting = torch.zeros((B, N),   dtype=torch.bool,      device=dev),
            prev_action    = torch.zeros((B, N, 3),  dtype=torch.float32,   device=dev),
            bullet_pos     = torch.zeros((B, N, K),  dtype=torch.complex64, device=dev),
            bullet_vel     = torch.zeros((B, N, K),  dtype=torch.complex64, device=dev),
            bullet_time    = torch.zeros((B, N, K),  dtype=torch.float32,   device=dev),
            bullet_active  = torch.zeros((B, N, K),  dtype=torch.bool,      device=dev),
            bullet_cursor  = torch.zeros((B, N),     dtype=torch.long,      device=dev),
        )

    def _build_cached_tensors(self) -> None:
        """Build per-step constants and reset templates once after state allocation."""
        N = self.env_config.num_ships
        n_team0 = N // 2
        n_team1 = N - n_team0

        # Physics lookup tables — 5 small 1-D tensors, reused every step
        self._physics_tables = _get_lookup_tables(self.ship_config, self.device)

        # Self-gravity exclusion mask — (1, N, N) eye, reused every step
        self._self_mask = torch.eye(N, device=self.device, dtype=torch.bool).unsqueeze(0)

        # Reset alive template — always the same pattern (all ships alive)
        self._reset_alive = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._reset_alive[:n_team0 + n_team1] = True

        # Reset team-ID template — first n_team0 slots = team 0, next n_team1 = team 1
        self._reset_base_teams = torch.zeros(N, dtype=torch.int32, device=self.device)
        self._reset_base_teams[n_team0:n_team0 + n_team1] = 1

    def reset_envs(
        self,
        mask: torch.Tensor,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments selected by the boolean mask.

        Sync-free: no .item(), no nonzero(), no Python branching on tensor values.
        Random tensors are always allocated at full (B, N) size; wasted values for
        non-reset envs are trivially cheap compared to the cost of a GPU→CPU sync.

        Args:
            mask: (B,) bool — True for envs that need resetting.
            options: Same as reset() options.
        """
        B = self.num_envs
        N = self.env_config.num_ships
        K = self.env_config.max_bullets
        world_w, world_h = self.ship_config.world_size

        n_team0 = N // 2
        n_team1 = N - n_team0
        if options and "team_sizes" in options:
            n_team0, n_team1 = options["team_sizes"]
            # Custom team sizes — build fresh templates (rare path, not hot path)
            reset_alive = torch.zeros(N, dtype=torch.bool, device=self.device)
            reset_alive[:n_team0 + n_team1] = True
            reset_base_teams = torch.zeros(N, dtype=torch.int32, device=self.device)
            reset_base_teams[n_team0:n_team0 + n_team1] = 1
        else:
            # Hot path — reuse pre-allocated templates
            reset_alive      = self._reset_alive
            reset_base_teams = self._reset_base_teams

        mask_n  = mask.unsqueeze(1).expand(B, N)          # (B, N)
        mask_nk = mask.view(B, 1, 1).expand(B, N, K)      # (B, N, K)
        mask_n3 = mask.view(B, 1, 1).expand(B, N, 3)      # (B, N, 3)

        # step_count
        self.state.step_count.masked_fill_(mask, 0)

        # Positions — uniformly random in world
        rand_x   = torch.rand(B, N, device=self.device) * world_w   # (B, N)
        rand_y   = torch.rand(B, N, device=self.device) * world_h   # (B, N)
        new_pos  = torch.complex(rand_x, rand_y)
        self.state.ship_pos = torch.where(mask_n, new_pos, self.state.ship_pos)

        # Attitude — random unit vectors
        rand_angle = torch.rand(B, N, device=self.device) * (2 * math.pi)  # (B, N)
        new_att    = torch.polar(torch.ones(B, N, device=self.device), rand_angle)
        self.state.ship_attitude = torch.where(mask_n, new_att, self.state.ship_attitude)

        # Velocity — along attitude at configured speed
        if self.ship_config.random_speed:
            speed = (
                self.ship_config.min_speed
                + torch.rand(B, N, device=self.device)
                * (self.ship_config.max_speed - self.ship_config.min_speed)
            )                                                        # (B, N)
        else:
            speed = torch.full(
                (B, N), self.ship_config.default_speed, device=self.device
            )
        self.state.ship_vel = torch.where(mask_n, speed * new_att, self.state.ship_vel)

        # Resources — masked_fill_ for scalar constants (in-place CUDA kernel, no sync)
        self.state.ship_health  .masked_fill_(mask_n, self.ship_config.max_health)
        self.state.ship_power   .masked_fill_(mask_n, self.ship_config.max_power)
        self.state.ship_cooldown.masked_fill_(mask_n, 0.0)
        self.state.ship_ang_vel .masked_fill_(mask_n, 0.0)

        # Team assignment: randomly shuffle which N slots belong to team 0 vs team 1.
        # A full shuffle (C(N, n_team0) possible assignments) prevents the policy
        # from exploiting any fixed slot-to-team mapping, unlike a simple whole-team flip.
        #
        # Expand templates to (B, N) — expand creates a view, no copy
        base_team_ids = reset_base_teams.unsqueeze(0).expand(B, N)        # (B, N) view
        new_alive     = reset_alive.unsqueeze(0).expand(B, N)             # (B, N) view

        perm         = torch.rand(B, N, device=self.device).argsort(dim=1)
        new_team_ids = base_team_ids.gather(1, perm)                      # (B, N) — new tensor

        self.state.ship_team_id = torch.where(mask_n, new_team_ids, self.state.ship_team_id)
        self.state.ship_alive   = torch.where(mask_n, new_alive,    self.state.ship_alive)

        # Clear bullets
        self.state.bullet_active.masked_fill_(mask_nk, False)
        self.state.bullet_time  .masked_fill_(mask_nk, 0.0)
        self.state.bullet_cursor.masked_fill_(mask_n,  0)

        # Clear previous action
        self.state.prev_action.masked_fill_(mask_n3, 0.0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance all environments by one physics tick.

        The caller (wrapper) is responsible for:
        - Snapshotting health/alive BEFORE calling step() if needed for rewards.
        - Calling reset_envs() on done environments AFTER computing rewards.

        Args:
            actions: (B, N, 3) int tensor — [power, turn, shoot].

        Returns:
            (dones, truncated) — each is a (B,) bool tensor.
        """
        self.state.prev_action = actions.float()

        self.state = update_ships(
            self.state, actions, self.ship_config,
            self._physics_tables, self._self_mask,
        )
        self.state = update_bullets(self.state, self.ship_config)
        self.state, dones = resolve_collisions(self.state, self.ship_config)

        self.state.step_count += 1
        truncated = self.state.step_count >= self.env_config.max_episode_steps

        return dones, truncated

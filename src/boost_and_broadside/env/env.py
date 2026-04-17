"""TensorEnv: pure physics engine for vectorized parallel game environments.

This class owns the physics simulation only. Observation construction, reward
computation, and episode tracking are handled by MVPEnvWrapper (env/wrapper.py).
"""

import numpy as np
import torch
from typing import Any

from boost_and_broadside.config import ShipConfig, EnvConfig
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.env.physics import (
    update_ships,
    update_bullets,
    update_obstacles,
    resolve_collisions,
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
        self.num_envs = num_envs
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)
        self.state: TensorState | None = None

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
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_envs(mask, options)

    def _allocate_state(self) -> None:
        """Pre-allocate all state tensors on device."""
        B = self.num_envs
        N = self.env_config.num_ships
        K = self.env_config.max_bullets
        M = self.env_config.num_obstacles
        dev = self.device

        self.state = TensorState(
            step_count=torch.zeros((B,), dtype=torch.int32, device=dev),
            ship_pos=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_vel=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_attitude=torch.zeros((B, N), dtype=torch.complex64, device=dev),
            ship_ang_vel=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_health=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_power=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_cooldown=torch.zeros((B, N), dtype=torch.float32, device=dev),
            ship_team_id=torch.zeros((B, N), dtype=torch.int32, device=dev),
            ship_alive=torch.zeros((B, N), dtype=torch.bool, device=dev),
            ship_is_shooting=torch.zeros((B, N), dtype=torch.bool, device=dev),
            prev_action=torch.zeros((B, N, 3), dtype=torch.float32, device=dev),
            bullet_pos=torch.zeros((B, N, K), dtype=torch.complex64, device=dev),
            bullet_vel=torch.zeros((B, N, K), dtype=torch.complex64, device=dev),
            bullet_time=torch.zeros((B, N, K), dtype=torch.float32, device=dev),
            bullet_active=torch.zeros((B, N, K), dtype=torch.bool, device=dev),
            bullet_cursor=torch.zeros((B, N), dtype=torch.long, device=dev),
            damage_matrix=torch.zeros((B, N, N), dtype=torch.float32, device=dev),
            cumulative_damage_matrix=torch.zeros(
                (B, N, N), dtype=torch.float32, device=dev
            ),
            obs_pos=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            obs_vel=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            obs_radius=torch.zeros((B, M), dtype=torch.float32, device=dev),
            obs_gravity_center=torch.zeros((B, M), dtype=torch.complex64, device=dev),
        )

    def reset_envs(
        self,
        mask: torch.Tensor,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments selected by the boolean mask.

        Args:
            mask: (B,) bool — True for envs that need resetting.
            options: Same as reset() options.
        """
        num_reset = int(mask.sum().item())
        if num_reset == 0:
            return

        world_w, world_h = self.ship_config.world_size
        N = self.env_config.num_ships

        n_team0 = N // 2
        n_team1 = N - n_team0
        if options and "team_sizes" in options:
            n_team0, n_team1 = options["team_sizes"]

        idx = torch.nonzero(mask, as_tuple=True)[0]  # indices of envs to reset

        self.state.step_count[mask] = 0

        # Positions — uniformly random in world
        rand_x = torch.rand((num_reset, N), device=self.device) * world_w
        rand_y = torch.rand((num_reset, N), device=self.device) * world_h
        self.state.ship_pos[idx] = torch.complex(rand_x, rand_y)

        # Attitude — random unit vectors
        rand_angle = torch.rand((num_reset, N), device=self.device) * 2 * np.pi
        att = torch.polar(torch.ones_like(rand_angle), rand_angle)
        self.state.ship_attitude[idx] = att

        # Velocity — along attitude at configured speed
        if self.ship_config.random_speed:
            speed = self.ship_config.min_speed + torch.rand(
                (num_reset, N), device=self.device
            ) * (self.ship_config.max_speed - self.ship_config.min_speed)
        else:
            speed = torch.full(
                (num_reset, N), self.ship_config.default_speed, device=self.device
            )
        self.state.ship_vel[idx] = speed * att

        # Resources
        self.state.ship_health[idx] = self.ship_config.max_health
        self.state.ship_power[idx] = self.ship_config.max_power
        self.state.ship_cooldown[idx] = 0.0
        self.state.ship_ang_vel[idx] = 0.0

        # Team assignment: randomly shuffle which N slots belong to team 0 vs team 1.
        # A full shuffle (C(N, n_team0) possible assignments) prevents the policy
        # from exploiting any fixed slot-to-team mapping, unlike a simple whole-team flip.
        new_alive = torch.zeros((num_reset, N), dtype=torch.bool, device=self.device)
        new_alive[:, : n_team0 + n_team1] = True

        base_team_ids = torch.zeros(
            (num_reset, N), dtype=torch.int32, device=self.device
        )
        base_team_ids[:, n_team0 : n_team0 + n_team1] = 1  # last n_team1 slots = team 1

        # Independent random permutation per env → any slot can be any team.
        perm = torch.rand((num_reset, N), device=self.device).argsort(dim=1)
        new_team_ids = base_team_ids.gather(1, perm)

        self.state.ship_team_id[idx] = new_team_ids
        self.state.ship_alive[idx] = new_alive

        # Clear bullets
        self.state.bullet_active[idx] = False
        self.state.bullet_time[idx] = 0.0
        self.state.bullet_cursor[idx] = 0

        # Clear damage attribution
        self.state.cumulative_damage_matrix[mask] = 0.0

        # Clear previous action
        self.state.prev_action[mask] = 0.0

        # Obstacles — energy-based initialization for bounded orbits.
        # Each obstacle is given a total mechanical energy E sampled from
        # [0, E_max] where E_max = ½G·R_max² (PE at the orbit boundary).
        # Position radius r and speed are derived so that KE + PE = E exactly,
        # guaranteeing the orbit never exceeds R_max from initialization alone.
        M = self.env_config.num_obstacles
        if M > 0:
            R_max = min(world_w, world_h) / 2.0
            G = self.ship_config.obs_gravity
            E_max = 0.5 * G * R_max**2

            energy = torch.rand((num_reset, M), device=self.device) * E_max  # (R, M)

            # Max orbital radius for this energy: r_max = sqrt(2E/G)
            r_max = torch.sqrt(2.0 * energy / G)  # (R, M)
            r = torch.rand((num_reset, M), device=self.device) * r_max  # (R, M)

            # Position: world center + r in a random direction, then wrap
            theta_pos = torch.rand((num_reset, M), device=self.device) * (2.0 * np.pi)
            obs_x = (world_w / 2 + r * torch.cos(theta_pos)) % world_w
            obs_y = (world_h / 2 + r * torch.sin(theta_pos)) % world_h
            self.state.obs_pos[idx] = torch.complex(obs_x, obs_y)

            # Velocity: remaining KE = E − PE, speed = sqrt(2·KE), random direction
            KE = energy - 0.5 * G * r**2  # (R, M) — always ≥ 0 since r ≤ r_max
            speed = torch.sqrt(KE.clamp(min=0.0) * 2.0)  # (R, M)
            theta_vel = torch.rand((num_reset, M), device=self.device) * (2.0 * np.pi)
            self.state.obs_vel[idx] = torch.polar(speed, theta_vel)

            radii = self.ship_config.obs_radius_min + torch.rand(
                (num_reset, M), device=self.device
            ) * (self.ship_config.obs_radius_max - self.ship_config.obs_radius_min)
            self.state.obs_radius[idx] = radii

            self.state.obs_gravity_center[idx] = complex(world_w / 2, world_h / 2)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        self.state = update_ships(self.state, actions, self.ship_config)
        self.state = update_bullets(self.state, self.ship_config)
        self.state = update_obstacles(self.state, self.ship_config)
        self.state, dones = resolve_collisions(self.state, self.ship_config)

        self.state.step_count += 1
        truncated = self.state.step_count >= self.env_config.max_episode_steps

        return dones, truncated

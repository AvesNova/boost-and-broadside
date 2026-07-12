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
    resolve_collisions,
    resolve_obstacle_collisions,
)
from boost_and_broadside.env.obstacle_physics import step_obstacles_harmonic
from boost_and_broadside.env.obstacle_cache import ObstacleCache


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
        obstacle_cache: ObstacleCache | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.ship_config = ship_config
        self.env_config = env_config
        self.device = torch.device(device)
        self.obstacle_cache = obstacle_cache
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
            obstacle_pos=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            obstacle_vel=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            obstacle_radius=torch.zeros((B, M), dtype=torch.float32, device=dev),
            obstacle_gcenter=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            ship_hit_obstacle=torch.zeros((B, N), dtype=torch.bool, device=dev),
        )

    def reset_envs(
        self,
        mask: torch.Tensor,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments selected by the boolean mask.

        Fully branchless: new values are generated for every env and applied
        only where mask is True. Generating full-size randoms is trivially
        cheap compared to the host-device sync that counting/indexing the mask
        would force — this runs every step of the training hot path.

        Args:
            mask: (B,) bool — True for envs that need resetting.
            options: Same as reset() options.
        """
        B = self.num_envs
        world_w, world_h = self.ship_config.world_size
        N = self.env_config.num_ships

        n_team0 = N // 2
        n_team1 = N - n_team0
        if options and "team_sizes" in options:
            n_team0, n_team1 = options["team_sizes"]

        s = self.state
        m = mask.unsqueeze(1)  # (B, 1) — broadcasts over ships/obstacles

        s.step_count = torch.where(mask, 0, s.step_count)

        # Positions — uniformly random in world
        rand_x = torch.rand((B, N), device=self.device) * world_w
        rand_y = torch.rand((B, N), device=self.device) * world_h
        s.ship_pos = torch.where(m, torch.complex(rand_x, rand_y), s.ship_pos)

        # Attitude — random unit vectors
        rand_angle = torch.rand((B, N), device=self.device) * 2 * np.pi
        att = torch.polar(torch.ones_like(rand_angle), rand_angle)
        s.ship_attitude = torch.where(m, att, s.ship_attitude)

        # Velocity — along attitude at configured speed
        if self.ship_config.random_speed:
            speed = self.ship_config.min_speed + torch.rand(
                (B, N), device=self.device
            ) * (self.ship_config.max_speed - self.ship_config.min_speed)
        else:
            speed = torch.full(
                (B, N), self.ship_config.default_speed, device=self.device
            )
        s.ship_vel = torch.where(m, speed * att, s.ship_vel)

        # Resources
        s.ship_health = torch.where(m, self.ship_config.max_health, s.ship_health)
        s.ship_power = torch.where(m, self.ship_config.max_power, s.ship_power)
        s.ship_cooldown = torch.where(m, 0.0, s.ship_cooldown)
        s.ship_ang_vel = torch.where(m, 0.0, s.ship_ang_vel)

        if self.env_config.single_team:
            # All ships share one randomly chosen team id (0 or 1) per env.
            # Random team prevents the policy overfitting to always seeing itself as team 0.
            team_id = torch.randint(0, 2, (B,), device=self.device, dtype=torch.int32)
            s.ship_team_id = torch.where(m, team_id.unsqueeze(1), s.ship_team_id)
            s.ship_alive = s.ship_alive | m
        else:
            # Two-team setup: randomly shuffle ship slots across both teams.
            new_alive = torch.zeros((B, N), dtype=torch.bool, device=self.device)
            new_alive[:, : n_team0 + n_team1] = True

            base_team_ids = torch.zeros(
                (B, N), dtype=torch.int32, device=self.device
            )
            base_team_ids[:, n_team0 : n_team0 + n_team1] = 1  # last n_team1 slots = team 1

            # Independent random permutation per env → any slot can be any team.
            perm = torch.rand((B, N), device=self.device).argsort(dim=1)
            new_team_ids = base_team_ids.gather(1, perm)

            s.ship_team_id = torch.where(m, new_team_ids, s.ship_team_id)
            s.ship_alive = torch.where(m, new_alive, s.ship_alive)

        # Clear bullets
        m3 = mask.view(B, 1, 1)
        s.bullet_active = s.bullet_active & ~m3
        s.bullet_time = torch.where(m3, 0.0, s.bullet_time)
        s.bullet_cursor = torch.where(m, 0, s.bullet_cursor)

        # Clear damage attribution
        s.cumulative_damage_matrix = torch.where(m3, 0.0, s.cumulative_damage_matrix)

        # Clear previous action
        s.prev_action = torch.where(m3, 0.0, s.prev_action)

        # Inject obstacle snapshot from cache (rotation + translation applied inside)
        if self.obstacle_cache is not None:
            obs_pos, obs_vel, obs_radius, obs_gcenter = self.obstacle_cache.sample(
                B, self.ship_config.world_size, self.device
            )
            s.obstacle_pos = torch.where(m, obs_pos, s.obstacle_pos)
            s.obstacle_vel = torch.where(m, obs_vel, s.obstacle_vel)
            s.obstacle_radius = torch.where(m, obs_radius, s.obstacle_radius)
            s.obstacle_gcenter = torch.where(m, obs_gcenter, s.obstacle_gcenter)

        # Clear obstacle death flag
        s.ship_hit_obstacle = s.ship_hit_obstacle & ~m

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
        self.state.ship_hit_obstacle.zero_()

        self.state = update_ships(self.state, actions, self.ship_config)
        if self.env_config.max_bullets > 0:
            self.state = update_bullets(self.state, self.ship_config)
        self.state = step_obstacles_harmonic(self.state, self.ship_config, enable_pbd=False)
        self.state = resolve_obstacle_collisions(self.state, self.ship_config)
        self.state, dones = resolve_collisions(self.state, self.ship_config)

        self.state.step_count += 1
        truncated = self.state.step_count >= self.env_config.max_episode_steps

        return dones, truncated

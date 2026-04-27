"""Pre-computed converged obstacle map cache.

Generates a bank of stable obstacle configurations before training begins.
Each snapshot stores obstacle positions, velocities, radii, and the gravity
center for one environment. During training, episode resets sample a random
snapshot and apply a random rotation + toroidal translation so no two
episodes see the exact same map layout.
"""

import math  # needed for math.ceil in generate()

import torch

from boost_and_broadside.config import EnvConfig, ObstacleCacheConfig, ShipConfig
from boost_and_broadside.env.obstacle_physics import (
    check_convergence,
    convergence_period_steps,
    init_obstacles_orbital,
    step_obstacles_harmonic,
)
from boost_and_broadside.env.state import TensorState


class ObstacleCache:
    """GPU-resident bank of pre-computed converged obstacle maps.

    Shape notation: C = cache_size, M = num_obstacles per env.

    Args:
        pos:     (C, M) complex64 — converged obstacle positions.
        vel:     (C, M) complex64 — converged obstacle velocities.
        radius:  (C, M) float32  — obstacle radii.
        gcenter: (C,) complex64  — per-snapshot gravity center.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        radius: torch.Tensor,
        gcenter: torch.Tensor,
    ) -> None:
        self._pos = pos
        self._vel = vel
        self._radius = radius
        self._gcenter = gcenter

    def __len__(self) -> int:
        return self._pos.shape[0]

    @staticmethod
    def generate(
        ship_config: ShipConfig,
        env_config: EnvConfig,
        cache_config: ObstacleCacheConfig,
        device: torch.device,
    ) -> "ObstacleCache":
        """Simulate B_cache parallel envs until cache_size converge.

        Runs harmonic gravity + PBD. Converged envs are harvested into the
        cache and immediately reinitialized so the simulation is continuous.
        If max_steps is reached before cache_size maps are collected, the
        available maps are tiled to fill the requested cache_size with a warning.

        Args:
            ship_config:   Physics constants.
            env_config:    Env sizing (num_obstacles).
            cache_config:  Cache generation parameters.
            device:        GPU device.

        Returns:
            Populated ObstacleCache.
        """
        B = cache_config.num_cache_envs
        M = env_config.num_obstacles
        target = cache_config.cache_size
        period_steps = convergence_period_steps(ship_config)

        # Simulation state
        pos, vel, radius, gcenter = init_obstacles_orbital(B, M, ship_config, device)
        collision_free = torch.zeros(B, dtype=torch.int32, device=device)

        # Cache storage accumulators (pre-allocated to target size for speed)
        cache_pos = torch.empty(target, M, dtype=torch.complex64, device=device)
        cache_vel = torch.empty(target, M, dtype=torch.complex64, device=device)
        cache_radius = torch.empty(target, M, dtype=torch.float32, device=device)
        cache_gcenter = torch.empty(target, dtype=torch.complex64, device=device)
        collected = 0

        # Minimal TensorState proxy for obstacle physics functions
        _dummy_state = _make_obstacle_state(pos, vel, radius, gcenter, ship_config, device)

        for step in range(cache_config.max_steps):
            _dummy_state.obstacle_pos = pos
            _dummy_state.obstacle_vel = vel
            _dummy_state.obstacle_radius = radius
            _dummy_state.obstacle_gcenter = gcenter

            _dummy_state = step_obstacles_harmonic(_dummy_state, ship_config, enable_pbd=True)
            pos = _dummy_state.obstacle_pos
            vel = _dummy_state.obstacle_vel

            converged, collision_free = check_convergence(
                pos, radius, collision_free, period_steps, ship_config
            )

            if converged.any():
                conv_idx = torch.nonzero(converged, as_tuple=True)[0]
                n_new = min(conv_idx.shape[0], target - collected)
                take = conv_idx[:n_new]

                cache_pos[collected : collected + n_new] = pos[take]
                cache_vel[collected : collected + n_new] = vel[take]
                cache_radius[collected : collected + n_new] = radius[take]
                cache_gcenter[collected : collected + n_new] = gcenter[take]
                collected += n_new

                if collected >= target:
                    break

                # Reinitialize harvested envs so they contribute new maps
                new_pos, new_vel, new_radius, new_gcenter = init_obstacles_orbital(
                    conv_idx.shape[0], M, ship_config, device
                )
                pos[conv_idx] = new_pos
                vel[conv_idx] = new_vel
                radius[conv_idx] = new_radius
                gcenter[conv_idx] = new_gcenter
                collision_free[conv_idx] = 0

        if collected == 0:
            raise RuntimeError(
                f"Obstacle cache: zero converged maps in {cache_config.max_steps} steps. "
                "Increase max_steps or reduce num_obstacles / obstacle_radius_max."
            )

        if collected < target:
            print(
                f"[ObstacleCache] Warning: only {collected}/{target} maps converged "
                f"in {cache_config.max_steps} steps — tiling to fill cache."
            )
            reps = math.ceil(target / collected)
            cache_pos = cache_pos[:collected].repeat(reps, 1)[:target]
            cache_vel = cache_vel[:collected].repeat(reps, 1)[:target]
            cache_radius = cache_radius[:collected].repeat(reps, 1)[:target]
            cache_gcenter = cache_gcenter[:collected].repeat(reps)[:target]

        return ObstacleCache(cache_pos, cache_vel, cache_radius, cache_gcenter)

    def sample(
        self,
        B: int,
        world_size: tuple[float, float],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw B random snapshots and apply random rotation + translation.

        The rotation preserves orbital relative positions; the toroidal
        translation prevents the policy from learning absolute-position cues.

        Args:
            B:          Number of environments to populate.
            world_size: (world_w, world_h).
            device:     Target device.

        Returns:
            pos:     (B, M) complex64
            vel:     (B, M) complex64
            radius:  (B, M) float32
            gcenter: (B,) complex64
        """
        world_w, world_h = world_size
        C = len(self)

        idx = torch.randint(0, C, (B,), device=device)
        pos = self._pos[idx].to(device)        # (B, M)
        vel = self._vel[idx].to(device)        # (B, M)
        radius = self._radius[idx].to(device)  # (B, M)
        gcenter = self._gcenter[idx].to(device)  # (B,)

        # Toroidal translation only — shifts the entire configuration by a random
        # offset mod (world_w, world_h).  Every pairwise toroidal distance is
        # preserved exactly, so a collision-free snapshot stays collision-free.
        off_x = torch.rand(B, device=device) * world_w
        off_y = torch.rand(B, device=device) * world_h

        pos = torch.complex(
            (pos.real + off_x.unsqueeze(1)) % world_w,
            (pos.imag + off_y.unsqueeze(1)) % world_h,
        )
        gcenter = torch.complex(
            (gcenter.real + off_x) % world_w,
            (gcenter.imag + off_y) % world_h,
        )

        return pos, vel, radius, gcenter


def _make_obstacle_state(
    pos: torch.Tensor,
    vel: torch.Tensor,
    radius: torch.Tensor,
    gcenter: torch.Tensor,
    config: ShipConfig,
    device: torch.device,
) -> TensorState:
    """Minimal TensorState for use during cache generation (ships not needed)."""
    B, M = pos.shape
    zero_bn = torch.zeros((B, 1), device=device)
    zero_bn_c = torch.zeros((B, 1), dtype=torch.complex64, device=device)
    return TensorState(
        step_count=torch.zeros((B,), dtype=torch.int32, device=device),
        ship_pos=zero_bn_c,
        ship_vel=zero_bn_c,
        ship_attitude=zero_bn_c,
        ship_ang_vel=zero_bn.squeeze(-1),
        ship_health=zero_bn.squeeze(-1),
        ship_power=zero_bn.squeeze(-1),
        ship_cooldown=zero_bn.squeeze(-1),
        ship_team_id=torch.zeros((B, 1), dtype=torch.int32, device=device),
        ship_alive=torch.zeros((B, 1), dtype=torch.bool, device=device),
        ship_is_shooting=torch.zeros((B, 1), dtype=torch.bool, device=device),
        prev_action=torch.zeros((B, 1, 3), dtype=torch.float32, device=device),
        bullet_pos=torch.zeros((B, 1, 1), dtype=torch.complex64, device=device),
        bullet_vel=torch.zeros((B, 1, 1), dtype=torch.complex64, device=device),
        bullet_time=torch.zeros((B, 1, 1), dtype=torch.float32, device=device),
        bullet_active=torch.zeros((B, 1, 1), dtype=torch.bool, device=device),
        bullet_cursor=torch.zeros((B, 1), dtype=torch.long, device=device),
        damage_matrix=torch.zeros((B, 1, 1), dtype=torch.float32, device=device),
        cumulative_damage_matrix=torch.zeros((B, 1, 1), dtype=torch.float32, device=device),
        obstacle_pos=pos,
        obstacle_vel=vel,
        obstacle_radius=radius,
        obstacle_gcenter=gcenter,
        ship_hit_obstacle=torch.zeros((B, 1), dtype=torch.bool, device=device),
    )

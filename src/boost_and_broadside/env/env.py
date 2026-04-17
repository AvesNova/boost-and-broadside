"""TensorEnv: pure physics engine for vectorized parallel game environments.

This class owns the physics simulation only. Observation construction, reward
computation, and episode tracking are handled by MVPEnvWrapper (env/wrapper.py).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Any

from boost_and_broadside.config import ShipConfig, EnvConfig
from boost_and_broadside.env.state import TensorState

if TYPE_CHECKING:
    from boost_and_broadside.env.obstacle_cache import ObstacleCache
from boost_and_broadside.env.physics import (
    update_ships,
    update_bullets,
    update_obstacles,
    step_obstacle_physics,
    resolve_collisions,
    _gravity_harmonic,
    _gravity_keplerian,
    _delta_pe_harmonic,
    _delta_pe_keplerian,
)


# ------------------------------------------------------------------
# Obstacle init strategies — four combinations of (init, physics).
# All share the same signature and return (obstacle_pos, obstacle_vel) complex64 (R, M).
# Selected once at TensorEnv.__init__ time via _OBSTACLE_INIT_FNS.
# ------------------------------------------------------------------


def _init_harmonic_energy(
    G: float,
    R_max: float,
    num_reset: int,
    M: int,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    world_w: float,
    world_h: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy-based init for harmonic physics. Samples E ∈ [0, ½G·R_max²]."""
    E_max = 0.5 * G * R_max**2
    energy = torch.rand((num_reset, M), device=device) * E_max
    r_max = torch.sqrt(2.0 * energy / G)
    r = torch.rand((num_reset, M), device=device) * r_max
    KE = energy - 0.5 * G * r**2
    speed = torch.sqrt(KE.clamp(min=0.0) * 2.0)
    theta_pos = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    theta_vel = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    obstacle_x = (center_x + r * torch.cos(theta_pos)) % world_w
    obstacle_y = (center_y + r * torch.sin(theta_pos)) % world_h
    return torch.complex(obstacle_x, obstacle_y), torch.polar(speed, theta_vel)


def _init_keplerian_energy(
    G: float,
    R_max: float,
    num_reset: int,
    M: int,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    world_w: float,
    world_h: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy-based init for Keplerian physics.

    Places each obstacle AT its apoapsis r_a with tangential speed f·v_circ(r_a).
    Starting at the apoapsis guarantees the orbit never exceeds r_a ≤ R_max.
    """
    r_a = torch.rand((num_reset, M), device=device) * R_max
    f = torch.rand((num_reset, M), device=device)
    v_circ = torch.sqrt(G / r_a.clamp(min=1.0))
    v = f * v_circ
    alpha = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    # Tangential direction: i·e^(iα) = (−sin α, cos α); CW or CCW randomly.
    sign = (torch.rand((num_reset, M), device=device) > 0.5).float() * 2.0 - 1.0
    obstacle_x = (center_x + r_a * torch.cos(alpha)) % world_w
    obstacle_y = (center_y + r_a * torch.sin(alpha)) % world_h
    v_x = sign * (-torch.sin(alpha)) * v
    v_y = sign * torch.cos(alpha) * v
    return torch.complex(obstacle_x, obstacle_y), torch.complex(v_x, v_y)


def _init_harmonic_orbital(
    G: float,
    R_max: float,
    num_reset: int,
    M: int,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    world_w: float,
    world_h: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbital init for harmonic physics.

    Centered ellipse with semi-axes a=r_a, b=β·r_a and angular frequency ω=√G.
    θ=0 is the apoapsis, so the orbit orientation φ = α directly.
    """
    r_a = torch.rand((num_reset, M), device=device) * R_max
    beta = torch.rand((num_reset, M), device=device)
    theta = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    alpha = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    a = r_a
    b = beta * r_a
    omega = float(np.sqrt(G))
    r_lx = a * torch.cos(theta)
    r_ly = b * torch.sin(theta)
    v_lx = -a * omega * torch.sin(theta)
    v_ly = b * omega * torch.cos(theta)
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    obstacle_x = (center_x + cos_a * r_lx - sin_a * r_ly) % world_w
    obstacle_y = (center_y + sin_a * r_lx + cos_a * r_ly) % world_h
    v_x = cos_a * v_lx - sin_a * v_ly
    v_y = sin_a * v_lx + cos_a * v_ly
    sign = (torch.rand((num_reset, M), device=device) > 0.5).float() * 2.0 - 1.0
    return torch.complex(obstacle_x, obstacle_y), torch.complex(v_x * sign, v_y * sign)


def _init_keplerian_orbital(
    G: float,
    R_max: float,
    num_reset: int,
    M: int,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    world_w: float,
    world_h: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbital init for Keplerian physics using the perifocal frame formulas.

    r_a is the apoapsis. β=0 → radial plunge, β=1 → circular orbit.
    Rotation φ = α−π places the apoapsis at angle α in the global frame
    (perifocal convention: θ=0 is periapsis, apoapsis at θ=π).
    """
    r_a = torch.rand((num_reset, M), device=device) * R_max
    beta = torch.rand((num_reset, M), device=device)
    theta = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    alpha = torch.rand((num_reset, M), device=device) * (2.0 * np.pi)
    e = 1.0 - beta**2
    p = (r_a * beta**2).clamp(min=1e-6)
    V0 = torch.sqrt(G / p)
    r = p / (1.0 + e * torch.cos(theta))
    r_lx = r * torch.cos(theta)
    r_ly = r * torch.sin(theta)
    v_lx = -V0 * torch.sin(theta)
    v_ly = V0 * (torch.cos(theta) + e)
    phi = alpha - np.pi  # apoapsis at θ=π → rotate by α−π to align with α
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    obstacle_x = (center_x + cos_p * r_lx - sin_p * r_ly) % world_w
    obstacle_y = (center_y + sin_p * r_lx + cos_p * r_ly) % world_h
    v_x = cos_p * v_lx - sin_p * v_ly
    v_y = sin_p * v_lx + cos_p * v_ly
    sign = (torch.rand((num_reset, M), device=device) > 0.5).float() * 2.0 - 1.0
    return torch.complex(obstacle_x, obstacle_y), torch.complex(v_x * sign, v_y * sign)


def _spawn_ships_rejection(
    num_reset: int,
    N: int,
    M: int,
    obstacle_pos: torch.Tensor,
    obstacle_radius: torch.Tensor,
    config: ShipConfig,
    device: torch.device,
) -> torch.Tensor:
    """Place ships using rejection sampling to keep them clear of all obstacles.

    Each candidate position is accepted only if it is at least safe_dist away
    from every obstacle edge. safe_dist accounts for the ship's collision hitbox
    plus a velocity lookahead buffer so ships don't immediately fly into obstacles.

    Args:
        num_reset:      R — number of environments being reset.
        N:              Ships per environment.
        M:              Obstacles per environment.
        obstacle_pos:   (R, M) complex64 — obstacle centres.
        obstacle_radius:(R, M) float32   — per-obstacle radii.
        config:         ShipConfig providing world_size, collision params, speed, dt.
        device:         Torch device.

    Returns:
        (R, N) complex64 ship positions.
    """
    world_w, world_h = config.world_size
    # Safety margin: obstacle radius + ship hitbox + velocity buffer
    buffer = (
        config.obstacle_collision_radius
        + config.default_speed * config.dt * config.ship_spawn_safety_k
    )

    placed_pos = torch.zeros(num_reset, N, dtype=torch.complex64, device=device)
    placed = torch.zeros(num_reset, N, dtype=torch.bool, device=device)

    while not placed.all():
        cand_x = torch.rand(num_reset, N, device=device) * world_w
        cand_y = torch.rand(num_reset, N, device=device) * world_h
        cand = torch.complex(cand_x, cand_y)  # (R, N)

        if M > 0:
            # diff: (R, N, M) — toroidal vector from each obstacle to each candidate
            diff = cand.unsqueeze(2) - obstacle_pos.unsqueeze(1)
            diff = torch.complex(
                (diff.real + world_w / 2) % world_w - world_w / 2,
                (diff.imag + world_h / 2) % world_h - world_h / 2,
            )
            dist = torch.sqrt(diff.real**2 + diff.imag**2)  # (R, N, M)
            # safe_dist per (env, obstacle): obstacle_radius + buffer  →  (R, 1, M)
            safe_dist = obstacle_radius.unsqueeze(1) + buffer
            safe = (dist >= safe_dist).all(dim=2)  # (R, N)
        else:
            safe = torch.ones(num_reset, N, dtype=torch.bool, device=device)

        newly = safe & ~placed
        placed_pos[newly] = cand[newly]
        placed |= newly

    return placed_pos


_OBSTACLE_GRAVITY_FNS = {
    "harmonic": _gravity_harmonic,
    "keplerian": _gravity_keplerian,
}
_OBSTACLE_DELTA_PE_FNS = {
    "harmonic": _delta_pe_harmonic,
    "keplerian": _delta_pe_keplerian,
}
_OBSTACLE_INIT_FNS = {
    ("energy", "harmonic"): _init_harmonic_energy,
    ("energy", "keplerian"): _init_keplerian_energy,
    ("orbital", "harmonic"): _init_harmonic_orbital,
    ("orbital", "keplerian"): _init_keplerian_orbital,
}


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
        self.state: TensorState | None = None
        self._obstacle_cache = obstacle_cache

        # Resolve obstacle strategy callables once — no branching in the hot loop.
        self._obstacle_gravity_fn = _OBSTACLE_GRAVITY_FNS[ship_config.obstacle_physics]
        self._obstacle_delta_pe_fn = _OBSTACLE_DELTA_PE_FNS[ship_config.obstacle_physics]
        self._obstacle_init_fn = _OBSTACLE_INIT_FNS[(ship_config.obstacle_init, ship_config.obstacle_physics)]

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
            obstacle_gravity_center=torch.zeros((B, M), dtype=torch.complex64, device=dev),
            obstacle_hit=torch.zeros((B, M), dtype=torch.bool, device=dev),
            ship_obstacle_damage=torch.zeros((B, N), dtype=torch.float32, device=dev),
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

        # Ship positions are set after obstacles (rejection sampling below).

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

        # Clear previous action and per-step damage
        self.state.prev_action[mask] = 0.0
        self.state.ship_obstacle_damage[mask] = 0.0

        # Obstacles — init strategy and physics type resolved at __init__ time.
        M = self.env_config.num_obstacles
        self.state.obstacle_hit[mask] = False
        if M > 0:
            R_max = min(world_w, world_h) * 0.45
            G = self.ship_config.obstacle_gravity

            if self._obstacle_cache is not None:
                # Sample from pre-converged library with random rotation+translation.
                obs_pos, obs_vel, obs_radius, obs_gcenter = self._obstacle_cache.sample(
                    num_reset, world_w, world_h, self.device
                )
                self.state.obstacle_pos[idx] = obs_pos
                self.state.obstacle_vel[idx] = obs_vel
                self.state.obstacle_radius[idx] = obs_radius
                self.state.obstacle_gravity_center[idx] = obs_gcenter
            else:
                # Fresh random init via selected strategy.
                if self.ship_config.obstacle_random_gravity_centers:
                    center_x = torch.rand((num_reset, M), device=self.device) * world_w
                    center_y = torch.rand((num_reset, M), device=self.device) * world_h
                else:
                    center_x = torch.full((num_reset, M), world_w / 2, device=self.device)
                    center_y = torch.full((num_reset, M), world_h / 2, device=self.device)
                self.state.obstacle_gravity_center[idx] = torch.complex(center_x, center_y)

                obstacle_pos, obstacle_vel = self._obstacle_init_fn(
                    G, R_max, num_reset, M, center_x, center_y, world_w, world_h, self.device
                )
                self.state.obstacle_pos[idx] = obstacle_pos
                self.state.obstacle_vel[idx] = obstacle_vel

                radii = self.ship_config.obstacle_radius_min + torch.rand(
                    (num_reset, M), device=self.device
                ) * (self.ship_config.obstacle_radius_max - self.ship_config.obstacle_radius_min)
                self.state.obstacle_radius[idx] = radii

                # Warmup: run obstacle-only physics before ships spawn.
                warmup_steps = self.ship_config.obstacle_warmup_steps
                if warmup_steps > 0:
                    w_pos = self.state.obstacle_pos[idx]
                    w_vel = self.state.obstacle_vel[idx]
                    w_rad = self.state.obstacle_radius[idx]
                    w_gc = self.state.obstacle_gravity_center[idx]
                    for _ in range(warmup_steps):
                        w_pos, w_vel, _ = step_obstacle_physics(
                            w_pos, w_vel, w_rad, w_gc,
                            self.ship_config, self._obstacle_gravity_fn, self._obstacle_delta_pe_fn,
                        )
                    self.state.obstacle_pos[idx] = w_pos
                    self.state.obstacle_vel[idx] = w_vel

        # Ship placement — rejection sampling keeps ships clear of all obstacles.
        # Works correctly for M=0 (accepts every candidate immediately).
        ship_pos = _spawn_ships_rejection(
            num_reset, N, M,
            self.state.obstacle_pos[idx],
            self.state.obstacle_radius[idx],
            self.ship_config,
            self.device,
        )
        self.state.ship_pos[idx] = ship_pos

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
        self.state = update_obstacles(
            self.state, self.ship_config, self._obstacle_gravity_fn, self._obstacle_delta_pe_fn
        )
        self.state, dones = resolve_collisions(self.state, self.ship_config)

        self.state.step_count += 1
        truncated = self.state.step_count >= self.env_config.max_episode_steps

        return dones, truncated

"""ObstacleCache: pre-converged obstacle configuration library.

Builds a library of fully-converged obstacle states by running a ship-free
simulation and harvesting environments where every obstacle has completed
period_steps consecutive steps without a single collision.

At episode reset time, TensorEnv samples a random entry, applies a random
rotation about the world centre and a random toroidal offset, and uses the
result as the initial obstacle state. This ensures obstacles start in stable
orbits rather than the noisy transient phase.

Usage (once, before training):
    cache = ObstacleCache()
    cache.build(ship_config, num_obstacles=8, num_snapshots=2000,
                sim_envs=4096, max_steps=10000, period_steps=337, device="cuda")
    cache.save("obstacle_cache_8obs.pt")

Usage at training time:
    cache = ObstacleCache.load("obstacle_cache_8obs.pt", device="cuda")
    # TensorEnv receives the cache and calls cache.sample() in reset_envs.
"""

import math
import time

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.env import (
    _OBSTACLE_GRAVITY_FNS,
    _OBSTACLE_DELTA_PE_FNS,
    _OBSTACLE_INIT_FNS,
)
from boost_and_broadside.env.physics import step_obstacle_physics


class ObstacleCache:
    """Library of pre-converged obstacle configurations.

    Stored tensors (all on CPU after build/load):
        _pos     (S, M) complex64 — obstacle positions
        _vel     (S, M) complex64 — obstacle velocities
        _radius  (S, M) float32   — obstacle radii
        _gcenter (S, M) complex64 — gravity centres
    """

    def __init__(self) -> None:
        self._pos: torch.Tensor | None = None
        self._vel: torch.Tensor | None = None
        self._radius: torch.Tensor | None = None
        self._gcenter: torch.Tensor | None = None

    def __len__(self) -> int:
        if self._pos is None:
            return 0
        return self._pos.shape[0]

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        ship_config: ShipConfig,
        num_obstacles: int,
        num_snapshots: int,
        sim_envs: int,
        max_steps: int,
        period_steps: int,
        device: str,
    ) -> None:
        """Run obstacle simulation and collect num_snapshots converged configs.

        An environment is considered converged when ALL M obstacles have had
        a collision-free streak >= period_steps steps. On convergence the
        snapshot is saved and that environment is re-initialised.

        Args:
            ship_config:   Physics configuration.
            num_obstacles: M — obstacles per environment.
            num_snapshots: S — target number of snapshots to collect.
            sim_envs:      Number of parallel environments for simulation.
            max_steps:     Hard stop if not enough snapshots collected.
            period_steps:  Required collision-free streak length per obstacle.
            device:        Torch device string.
        """
        if num_obstacles == 0:
            raise ValueError("ObstacleCache.build: num_obstacles must be > 0")

        B = sim_envs
        M = num_obstacles
        dev = torch.device(device)

        gravity_fn = _OBSTACLE_GRAVITY_FNS[ship_config.obstacle_physics]
        delta_pe_fn = _OBSTACLE_DELTA_PE_FNS[ship_config.obstacle_physics]
        init_fn = _OBSTACLE_INIT_FNS[
            (ship_config.obstacle_init, ship_config.obstacle_physics)
        ]

        world_w, world_h = ship_config.world_size
        G = ship_config.obstacle_gravity
        R_max = min(world_w, world_h) * 0.45

        def _make_centers(R: int) -> tuple[torch.Tensor, torch.Tensor]:
            if ship_config.obstacle_random_gravity_centers:
                cx = torch.rand((R, M), device=dev) * world_w
                cy = torch.rand((R, M), device=dev) * world_h
            else:
                cx = torch.full((R, M), world_w / 2.0, device=dev)
                cy = torch.full((R, M), world_h / 2.0, device=dev)
            return cx, cy

        def _make_radii(R: int) -> torch.Tensor:
            return (
                torch.rand((R, M), device=dev)
                * (ship_config.obstacle_radius_max - ship_config.obstacle_radius_min)
                + ship_config.obstacle_radius_min
            )

        # Initial allocation
        cx, cy = _make_centers(B)
        gcenter = torch.complex(cx, cy)
        radius = _make_radii(B)
        pos, vel = init_fn(G, R_max, B, M, cx, cy, world_w, world_h, dev)

        streak = torch.zeros((B, M), dtype=torch.int32, device=dev)

        snap_pos: list[torch.Tensor] = []
        snap_vel: list[torch.Tensor] = []
        snap_radius: list[torch.Tensor] = []
        snap_gcenter: list[torch.Tensor] = []

        t0 = time.perf_counter()
        print(
            f"ObstacleCache.build: target={num_snapshots} snapshots, "
            f"B={B}, M={M}, period={period_steps}, max_steps={max_steps}"
        )

        for step in range(1, max_steps + 1):
            pos, vel, hit = step_obstacle_physics(
                pos, vel, radius, gcenter, ship_config, gravity_fn, delta_pe_fn
            )
            streak = torch.where(hit, torch.zeros_like(streak), streak + 1)

            # Env converged when ALL M obstacles have streak >= period_steps
            env_converged = (streak >= period_steps).all(dim=1)  # (B,)

            if env_converged.any():
                idx = env_converged.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    snap_pos.append(pos[i].cpu())
                    snap_vel.append(vel[i].cpu())
                    snap_radius.append(radius[i].cpu())
                    snap_gcenter.append(gcenter[i].cpu())

                # Re-initialise converged envs
                R = int(idx.shape[0])
                cx_r, cy_r = _make_centers(R)
                gcenter[idx] = torch.complex(cx_r, cy_r)
                radius[idx] = _make_radii(R)
                pos[idx], vel[idx] = init_fn(
                    G, R_max, R, M, cx_r, cy_r, world_w, world_h, dev
                )
                streak[idx] = 0

                collected = len(snap_pos)
                if collected >= num_snapshots:
                    print(
                        f"  Collected {collected} snapshots at step {step} "
                        f"({time.perf_counter() - t0:.1f}s)"
                    )
                    break

        collected = len(snap_pos)
        if collected == 0:
            raise RuntimeError(
                f"ObstacleCache.build: no converged snapshots in {max_steps} steps. "
                "Increase max_steps or decrease period_steps."
            )
        if collected < num_snapshots:
            print(
                f"  Warning: only {collected}/{num_snapshots} snapshots collected "
                f"in {max_steps} steps."
            )

        # Stack and store (cap to num_snapshots)
        S = min(collected, num_snapshots)
        self._pos = torch.stack(snap_pos[:S])       # (S, M) complex64
        self._vel = torch.stack(snap_vel[:S])
        self._radius = torch.stack(snap_radius[:S])
        self._gcenter = torch.stack(snap_gcenter[:S])
        print(f"  Cache built: {S} snapshots stored.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save cache to a .pt file."""
        if self._pos is None:
            raise RuntimeError("Cache is empty — call build() first.")
        torch.save(
            {
                "pos": self._pos,
                "vel": self._vel,
                "radius": self._radius,
                "gcenter": self._gcenter,
            },
            path,
        )
        print(f"ObstacleCache saved: {path}  ({len(self)} snapshots)")

    @classmethod
    def load(cls, path: str) -> "ObstacleCache":
        """Load cache from a .pt file (tensors kept on CPU; moved to device at sample time)."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        cache = cls()
        cache._pos = data["pos"]
        cache._vel = data["vel"]
        cache._radius = data["radius"]
        cache._gcenter = data["gcenter"]
        print(f"ObstacleCache loaded: {path}  ({len(cache)} snapshots)")
        return cache

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        num_envs: int,
        world_w: float,
        world_h: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample num_envs snapshots with independent random rotation + translation.

        Rotation is about the world centre — positions and velocities are rotated
        by the same angle θ, so the orbital geometry is preserved exactly. The
        gravity centres rotate with the configuration. A random toroidal offset is
        then applied (both positions and gravity centres wrap modulo world size).

        Args:
            num_envs: E — number of environments to fill.
            world_w:  World width in pixels.
            world_h:  World height in pixels.
            device:   Target device for returned tensors.

        Returns:
            (obstacle_pos, obstacle_vel, obstacle_radius, obstacle_gravity_center)
            all shape (E, M); pos and gcenter are complex64, vel complex64, radius float32.
        """
        if self._pos is None:
            raise RuntimeError("Cache is empty — call build() or load() first.")

        S = len(self)
        idx = torch.randint(0, S, (num_envs,))
        pos = self._pos[idx].to(device)         # (E, M) complex64
        vel = self._vel[idx].to(device)
        radius = self._radius[idx].to(device)
        gcenter = self._gcenter[idx].to(device)

        # Random rotation per env about world centre
        theta = torch.rand(num_envs, device=device) * (2.0 * math.pi)
        rot = torch.polar(
            torch.ones(num_envs, device=device), theta
        ).unsqueeze(1)  # (E, 1) complex64

        wc = complex(world_w / 2.0, world_h / 2.0)
        pos = (pos - wc) * rot + wc
        gcenter = (gcenter - wc) * rot + wc
        vel = vel * rot  # velocity is a vector — direction rotates, magnitude preserved

        # Random toroidal translation
        tx = torch.rand(num_envs, device=device) * world_w
        ty = torch.rand(num_envs, device=device) * world_h
        offset = torch.complex(tx, ty).unsqueeze(1)  # (E, 1)

        pos_x = (pos.real + offset.real) % world_w
        pos_y = (pos.imag + offset.imag) % world_h
        pos = torch.complex(pos_x, pos_y)

        gcx = (gcenter.real + offset.real) % world_w
        gcy = (gcenter.imag + offset.imag) % world_h
        gcenter = torch.complex(gcx, gcy)

        return pos, vel, radius, gcenter

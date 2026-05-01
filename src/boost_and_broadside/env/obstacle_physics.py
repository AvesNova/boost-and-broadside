"""GPU-vectorized obstacle physics: harmonic (spring) gravity with orbital initialization.

Obstacles orbit a per-environment gravity center under a harmonic spring
(F = G * distance). Cache generation uses PBD overlap separation + energy
reprojection to converge to stable collision-free orbits. Training runs
harmonic gravity only (no PBD).

Convergence criterion: all M obstacles complete two full harmonic periods
with zero inter-obstacle overlaps.
    T_period = 2π / sqrt(G)
    period_steps = max(1, round(2 * T_period / dt))
"""

import math

import torch

from boost_and_broadside.config import ShipConfig

_EPS = 1e-6  # division safety guard for direction normalization
from boost_and_broadside.env.state import TensorState


def _wrap_diff(diff_real: torch.Tensor, diff_imag: torch.Tensor, world_w: float, world_h: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap component-wise differences to [-half_world, +half_world]."""
    diff_real = (diff_real + world_w / 2) % world_w - world_w / 2
    diff_imag = (diff_imag + world_h / 2) % world_h - world_h / 2
    return diff_real, diff_imag


def convergence_period_steps(config: ShipConfig) -> int:
    """Steps required for two full harmonic periods."""
    G = config.obstacle_gravity_harmonic
    T_period = 2.0 * math.pi / math.sqrt(G)
    return max(1, round(2.0 * T_period / config.dt))


def init_obstacles_orbital(
    B: int,
    M: int,
    config: ShipConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize M obstacles per env with random orbital ellipses.

    Places each obstacle on a parametric ellipse centred at a random
    gravity center. The ellipse semi-axes and initial phase are randomised;
    the orbital velocity is chosen so the obstacle satisfies the harmonic
    orbit equations at t=0.

    Args:
        B: Number of environments.
        M: Number of obstacles per environment.
        config: Physics config (world_size, obstacle_radius_min/max, obstacle_gravity_harmonic).
        device: Torch device.

    Returns:
        pos:     (B, M) complex64 — initial positions (toroidally wrapped).
        vel:     (B, M) complex64 — initial velocities.
        radius:  (B, M) float32  — obstacle radii in [radius_min, radius_max].
        gcenter: (B, M) complex64 — per-obstacle gravity center.
    """
    world_w, world_h = config.world_size
    G = config.obstacle_gravity_harmonic
    omega = math.sqrt(G)

    # Independent gravity center per obstacle
    gcx = torch.rand(B, M, device=device) * world_w
    gcy = torch.rand(B, M, device=device) * world_h
    gcenter = torch.complex(gcx, gcy)  # (B, M)

    # Radii
    radius = (
        config.obstacle_radius_min
        + torch.rand(B, M, device=device) * (config.obstacle_radius_max - config.obstacle_radius_min)
    )  # (B, M)

    # Orbital parameters — broadcasted over (B, M)
    R_max = min(world_w, world_h) * 0.35  # keep orbits within world
    r_a = torch.sqrt(torch.rand(B, M, device=device)) * R_max  # semi-major axis a — triangle dist (PDF ∝ x)
    beta = torch.rand(B, M, device=device)                   # axis ratio b/a
    r_b = beta * r_a                                         # semi-minor axis b
    theta = torch.rand(B, M, device=device) * 2.0 * math.pi  # initial phase
    alpha = torch.rand(B, M, device=device) * 2.0 * math.pi  # orbit rotation angle
    sign = torch.randint(0, 2, (B, M), device=device).float() * 2.0 - 1.0  # ±1

    # Local ellipse coordinates
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rx_local = r_a * cos_theta
    ry_local = r_b * sin_theta
    vx_local = -r_a * omega * sin_theta
    vy_local = r_b * omega * cos_theta

    # Rotate by alpha into global frame
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    rx = rx_local * cos_alpha - ry_local * sin_alpha
    ry = rx_local * sin_alpha + ry_local * cos_alpha
    vx = (vx_local * cos_alpha - vy_local * sin_alpha) * sign
    vy = (vx_local * sin_alpha + vy_local * cos_alpha) * sign

    # Absolute position = local + gcenter
    pos_x = rx + gcenter.real  # (B, M)
    pos_y = ry + gcenter.imag
    pos_x = pos_x % world_w
    pos_y = pos_y % world_h

    pos = torch.complex(pos_x, pos_y)      # (B, M)
    vel = torch.complex(vx, vy)            # (B, M)

    return pos, vel, radius, gcenter


def step_obstacles_harmonic(
    state: TensorState,
    config: ShipConfig,
    enable_pbd: bool,
) -> TensorState:
    """Advance obstacle positions by one timestep under harmonic gravity.

    Semi-implicit Euler integration: velocity updated first, then position.
    When enable_pbd=True, overlapping obstacles are separated via PBD and
    the kinetic energy is adjusted to conserve total mechanical energy.

    Args:
        state:      Current TensorState (obstacle fields mutated in-place).
        config:     Physics config.
        enable_pbd: Whether to run PBD overlap correction (cache gen only).

    Returns:
        Mutated state.
    """
    world_w, world_h = config.world_size
    G = config.obstacle_gravity_harmonic
    dt = config.dt

    pos = state.obstacle_pos    # (B, M) complex64
    vel = state.obstacle_vel    # (B, M) complex64
    gc = state.obstacle_gcenter  # (B, M) complex64

    # Toroidal wrapped displacement from obstacle to its gravity center
    diff_r, diff_i = _wrap_diff(
        gc.real - pos.real,
        gc.imag - pos.imag,
        world_w, world_h,
    )  # (B, M) each

    # Semi-implicit Euler: v += F*dt, x += v*dt
    vel = torch.complex(
        vel.real + diff_r * G * dt,
        vel.imag + diff_i * G * dt,
    )
    pos = torch.complex(
        (pos.real + vel.real * dt) % world_w,
        (pos.imag + vel.imag * dt) % world_h,
    )

    if enable_pbd:
        pos, vel = _pbd_separation(pos, vel, state.obstacle_radius, gc, config)

    state.obstacle_pos = pos
    state.obstacle_vel = vel
    return state


def _pbd_separation(
    pos: torch.Tensor,   # (B, M) complex64
    vel: torch.Tensor,   # (B, M) complex64
    radius: torch.Tensor,  # (B, M) float32
    gcenter: torch.Tensor,  # (B, M) complex64
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PBD positional correction + per-obstacle energy-conserving speed rescale.

    1. Push overlapping pairs apart by half the overlap along their connecting axis.
    2. Keep velocity direction unchanged; rescale speed so that total mechanical
       energy (KE + PE) per obstacle is conserved after the position move.
       PE = 0.5 * G * |wrapped_dist_to_gcenter|^2  (harmonic spring).

    Returns:
        (pos, vel) updated tensors.
    """
    world_w, world_h = config.world_size
    G = config.obstacle_gravity_harmonic
    B, M = pos.shape

    # diff[b,i,j] = wrap(pos[b,i] - pos[b,j]) — vector FROM j TO i
    diff_r, diff_i = _wrap_diff(
        pos.real.unsqueeze(2) - pos.real.unsqueeze(1),
        pos.imag.unsqueeze(2) - pos.imag.unsqueeze(1),
        world_w, world_h,
    )
    dist = (diff_r**2 + diff_i**2).sqrt().clamp(min=_EPS)  # (B, M, M)

    r_sum = radius.unsqueeze(2) + radius.unsqueeze(1)       # (B, M, M)
    overlap = (r_sum - dist).clamp(min=0.0)                 # (B, M, M)

    diag = torch.eye(M, device=pos.device, dtype=torch.bool).unsqueeze(0)
    inv_dist = torch.where(diag, torch.zeros_like(dist), 1.0 / dist)

    # n[b,i,j]: unit normal FROM j TO i (the direction that pushes i away from j)
    n_r = diff_r * inv_dist  # (B, M, M)
    n_i = diff_i * inv_dist

    # --- Positional correction ---
    # Move i away from each j by half the overlap: disp_i = sum_j(0.5 * overlap * n[i,j])
    # Record wrapped dist-to-gcenter before moving (for energy conservation).
    gc_r = gcenter.real  # (B, M)
    gc_i = gcenter.imag
    to_gc_r_pre, to_gc_i_pre = _wrap_diff(pos.real - gc_r, pos.imag - gc_i, world_w, world_h)
    r_sq_pre = to_gc_r_pre**2 + to_gc_i_pre**2  # (B, M)

    disp_r = (0.5 * overlap * n_r).sum(dim=2)  # (B, M)
    disp_i = (0.5 * overlap * n_i).sum(dim=2)
    pos = torch.complex(
        (pos.real + disp_r) % world_w,
        (pos.imag + disp_i) % world_h,
    )

    # --- Speed rescale (keep direction, conserve E = 0.5*v^2 + 0.5*G*r^2) ---
    to_gc_r_post, to_gc_i_post = _wrap_diff(pos.real - gc_r, pos.imag - gc_i, world_w, world_h)
    r_sq_post = to_gc_r_post**2 + to_gc_i_post**2  # (B, M)

    speed_sq = vel.real**2 + vel.imag**2  # (B, M)
    # new_speed^2 = speed^2 + G*(r_pre^2 - r_post^2)  [PE released → KE gained, or vice-versa]
    new_speed_sq = (speed_sq + G * (r_sq_pre - r_sq_post)).clamp(min=0.0)
    scale = torch.where(speed_sq > 1e-12, (new_speed_sq / speed_sq.clamp(min=1e-12)).sqrt(), torch.ones_like(speed_sq))
    vel = torch.complex(vel.real * scale, vel.imag * scale)

    return pos, vel


def check_convergence(
    pos: torch.Tensor,    # (B, M) complex64
    radius: torch.Tensor,  # (B, M) float32
    collision_free_steps: torch.Tensor,  # (B,) int32 — mutable counter
    period_steps: int,
    config: ShipConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check convergence: no overlaps for two full harmonic periods.

    Updates the per-env collision-free step counter (reset on overlap, increment otherwise).

    Args:
        pos:                  (B, M) current obstacle positions.
        radius:               (B, M) obstacle radii.
        collision_free_steps: (B,) running counter of steps without collision.
        period_steps:         Target steps for two periods (from convergence_period_steps).
        config:               Physics config.

    Returns:
        converged:            (B,) bool — True when counter >= period_steps.
        collision_free_steps: (B,) updated counter.
    """
    world_w, world_h = config.world_size
    B, M = pos.shape

    # Pairwise distances
    diff_r, diff_i = _wrap_diff(
        pos.real.unsqueeze(2) - pos.real.unsqueeze(1),
        pos.imag.unsqueeze(2) - pos.imag.unsqueeze(1),
        world_w, world_h,
    )
    dist = torch.sqrt(diff_r**2 + diff_i**2)  # (B, M, M)

    r_sum = radius.unsqueeze(2) + radius.unsqueeze(1)  # (B, M, M)
    diag = torch.eye(M, device=pos.device, dtype=torch.bool).unsqueeze(0)
    any_overlap = ((dist < r_sum) & ~diag).any(dim=(1, 2))  # (B,)

    # Reset counter where there's overlap, increment where there isn't
    collision_free_steps = torch.where(any_overlap, torch.zeros_like(collision_free_steps), collision_free_steps + 1)
    converged = collision_free_steps >= period_steps

    return converged, collision_free_steps

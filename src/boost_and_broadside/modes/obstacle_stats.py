"""obstacle_stats mode: measure obstacle convergence time (time until collision-free).

Runs a ship-free obstacle simulation across B parallel environments. For each
obstacle slot, we track the current collision-free streak. Convergence is
declared when the streak reaches `period_steps` consecutive clean steps —
i.e. the obstacle has completed one full harmonic period without any collision.

The convergence step reported is the step at which that streak was first
achieved. Slots that never converge within max_steps are reported separately.

Outputs: period_steps used as threshold, convergence counts, and quartiles
(Q1/Q2/Q3), mean, 95th, 99th percentile of convergence steps over all
converged (B x M) obstacle slots.
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


def run_obstacle_stats_mode(
    num_envs: int,
    num_obstacles: int,
    max_steps: int,
    ship_config: ShipConfig,
    device: str,
) -> None:
    """Run obstacle convergence measurement and print statistics.

    Convergence criterion: a slot achieves `period_steps` consecutive steps
    with no collision. period_steps = one harmonic period = 2pi / (sqrt(G) * dt),
    applied as the threshold for both harmonic and Keplerian modes.

    Args:
        num_envs:      Number of parallel environments (B).
        num_obstacles: Obstacles per environment (M).
        max_steps:     Maximum simulation steps.
        ship_config:   Physics configuration (determines physics mode, gravity, etc.)
        device:        Torch device string.
    """
    if num_obstacles == 0:
        print("obstacle_stats: num_obstacles=0, nothing to simulate.")
        return

    B = num_envs
    M = num_obstacles
    dev = torch.device(device)

    gravity_fn = _OBSTACLE_GRAVITY_FNS[ship_config.obstacle_physics]
    delta_pe_fn = _OBSTACLE_DELTA_PE_FNS[ship_config.obstacle_physics]
    init_fn = _OBSTACLE_INIT_FNS[(ship_config.obstacle_init, ship_config.obstacle_physics)]

    world_w, world_h = ship_config.world_size
    G = ship_config.obstacle_gravity
    R_max = min(world_w, world_h) * 0.45

    # Period threshold: one harmonic orbital period T = 2pi / sqrt(G_harmonic).
    # Used as the required collision-free streak length for both physics modes.
    period_steps = max(1, round(2.0 * 2.0 * math.pi / math.sqrt(ship_config.obstacle_gravity_harmonic) / ship_config.dt))

    # --- Gravity centers ---
    if ship_config.obstacle_random_gravity_centers:
        center_x = torch.rand((B, M), device=dev) * world_w
        center_y = torch.rand((B, M), device=dev) * world_h
    else:
        center_x = torch.full((B, M), world_w / 2.0, device=dev)
        center_y = torch.full((B, M), world_h / 2.0, device=dev)
    obstacle_gravity_center = torch.complex(center_x, center_y)

    # --- Obstacle radii ---
    obstacle_radius = (
        torch.rand((B, M), device=dev)
        * (ship_config.obstacle_radius_max - ship_config.obstacle_radius_min)
        + ship_config.obstacle_radius_min
    )

    # --- Initial positions and velocities ---
    obstacle_pos, obstacle_vel = init_fn(
        G, R_max, B, M, center_x, center_y, world_w, world_h, dev
    )

    # streak[b, m]           = current consecutive collision-free steps
    # convergence_step[b, m] = step at which streak first reached period_steps (0 = not yet)
    # converged[b, m]        = bool, True once convergence_step is set
    streak = torch.zeros((B, M), dtype=torch.int32, device=dev)
    convergence_step = torch.zeros((B, M), dtype=torch.int32, device=dev)
    converged = torch.zeros((B, M), dtype=torch.bool, device=dev)

    sim_fps = 1.0 / ship_config.dt
    print(
        f"\nobstacle_stats: B={B} envs, M={M} obstacles, max_steps={max_steps}, "
        f"physics={ship_config.obstacle_physics}, init={ship_config.obstacle_init}, "
        f"device={device}"
    )
    print(
        f"  period_steps={period_steps}  ({period_steps / sim_fps:.2f}s sim)  "
        f"[2 x harmonic T = 2 x 2pi/sqrt({ship_config.obstacle_gravity_harmonic})]"
    )
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        obstacle_pos, obstacle_vel, obstacle_hit = step_obstacle_physics(
            obstacle_pos,
            obstacle_vel,
            obstacle_radius,
            obstacle_gravity_center,
            ship_config,
            gravity_fn,
            delta_pe_fn,
        )
        # Reset streak on collision; increment otherwise
        streak = torch.where(obstacle_hit, torch.zeros_like(streak), streak + 1)

        # Mark newly converged slots (streak just reached the threshold, not yet converged)
        newly_converged = (streak >= period_steps) & ~converged
        convergence_step = torch.where(
            newly_converged,
            torch.full_like(convergence_step, step),
            convergence_step,
        )
        converged = converged | newly_converged

    elapsed = time.perf_counter() - t0

    total_slots = B * M
    converged_n = int(converged.sum().item())
    not_converged_n = total_slots - converged_n

    w = 60
    print(f"{'-' * w}")

    if converged_n == 0:
        print(f"  No slots converged within {max_steps} steps.")
        print(f"{'-' * w}")
        print(f"  Total slots     : {total_slots:,}  (B={B} x M={M})")
        print(f"  Converged       : 0  (0.0%)")
        print(f"  Not converged   : {not_converged_n:,}  (100.0%)")
    else:
        flat = convergence_step[converged].float().cpu()
        q1, q2, q3 = torch.quantile(flat, torch.tensor([0.25, 0.50, 0.75])).tolist()
        p95, p99 = torch.quantile(flat, torch.tensor([0.95, 0.99])).tolist()
        mean = float(flat.mean().item())

        print(f"  Total slots     : {total_slots:,}  (B={B} x M={M})")
        print(f"  Converged       : {converged_n:,}  ({100 * converged_n / total_slots:.1f}%)")
        print(f"  Not converged   : {not_converged_n:,}  ({100 * not_converged_n / total_slots:.1f}%)  [within {max_steps} steps]")
        print(f"{'-' * w}")
        print(f"  Stats over converged slots only:")
        print(f"  Q1  (25th pct)  : {q1:8.1f} steps  ({q1 / sim_fps:.2f}s sim)")
        print(f"  Q2  (median)    : {q2:8.1f} steps  ({q2 / sim_fps:.2f}s sim)")
        print(f"  Q3  (75th pct)  : {q3:8.1f} steps  ({q3 / sim_fps:.2f}s sim)")
        print(f"  Mean            : {mean:8.1f} steps  ({mean / sim_fps:.2f}s sim)")
        print(f"  95th percentile : {p95:8.1f} steps  ({p95 / sim_fps:.2f}s sim)")
        print(f"  99th percentile : {p99:8.1f} steps  ({p99 / sim_fps:.2f}s sim)")

    print(f"{'-' * w}")
    print(f"  Wall time       : {elapsed:.2f}s  ({max_steps * B / elapsed:,.0f} env-steps/s)")
    print(f"{'-' * w}\n")

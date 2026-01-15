#!/usr/bin/env python3
"""
Ship Parameter Derivation

This script runs physics simulations to derive key ship parameters like max speed,
equilibrium speed, turn radius, and turn speeds. Results are saved to YAML.
"""

import numpy as np
import torch
import yaml
from typing import Any
import math
import os

from .ship import Ship, default_ship_config
from .constants import PowerActions, TurnActions
from .bullets import Bullets


def create_test_ship() -> Ship:
    """Create a ship for testing with standard parameters."""
    return Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=512.0,  # Center of default 1024x1024 world
        initial_y=512.0,
        initial_vx=100.0,  # Start with some forward velocity
        initial_vy=0.0,
    )


def measure_max_speed() -> float:
    """
    Measure maximum speed by running with full boost until energy runs out.
    Returns the maximum speed achieved.
    """
    print("Measuring maximum speed...")

    ship = create_test_ship()
    dt = 0.02
    max_speed = 0.0

    # Create action with max forward boost
    # [Power, Turn, Shoot]
    action = torch.zeros(3, dtype=torch.float32)
    action[0] = float(PowerActions.BOOST)
    action[1] = float(TurnActions.GO_STRAIGHT)
    action[2] = 0.0  # No Shoot

    # Run until energy is depleted or speed stops increasing
    steps = 0
    speed_history = []

    # Create a bullets container for the ship
    bullets = Bullets(max_bullets=100)

    while ship.power > 0.01 and steps < 10000:
        ship.forward(action, bullets, steps * dt, dt)
        speed = abs(ship.velocity)
        max_speed = max(max_speed, speed)
        speed_history.append(speed)
        steps += 1

        # Stop if speed hasn't increased significantly in last 60 steps
        if len(speed_history) > 60:
            recent_max = max(speed_history[-60:])
            if recent_max - max_speed < 0.1:
                break

    print(f"  Max speed: {max_speed:.2f} units/s (after {steps} steps)")
    return max_speed


def measure_equilibrium_speed() -> float:
    """
    Measure equilibrium speed by running with no boost until speed stabilizes.
    Returns the equilibrium speed.
    """
    print("Measuring equilibrium speed...")

    ship = create_test_ship()
    dt = 0.02

    # No thrust action (COAST)
    action = torch.zeros(3, dtype=torch.float32)
    action[0] = float(PowerActions.COAST)
    action[1] = float(TurnActions.GO_STRAIGHT)
    # Shoot is 0 (NO_SHOOT)

    # Run until speed stabilizes
    steps = 0
    speed_history = []
    stabilization_threshold = 0.01

    # Create a bullets container for the ship
    bullets = Bullets(max_bullets=100)

    while steps < 10000:
        ship.forward(action, bullets, steps * dt, dt)
        speed = abs(ship.velocity)
        speed_history.append(speed)
        steps += 1

        # Check if speed has stabilized (last 120 steps within threshold)
        if len(speed_history) > 120:
            recent_speeds = speed_history[-120:]
            speed_variance = np.var(recent_speeds)
            if speed_variance < stabilization_threshold:
                break

    equilibrium_speed = np.mean(speed_history[-60:])
    print(f"  Equilibrium speed: {equilibrium_speed:.2f} units/s (after {steps} steps)")
    return equilibrium_speed


def measure_turn_characteristics(sharp_turn: bool = False) -> dict[str, float]:
    """
    Measure turn characteristics by holding a turn until speed stabilizes,
    then measuring turn radius and time for complete circle.

    Args:
        sharp_turn: If True, use sharp turn (left+right), else normal turn (right only)

    Returns:
        Dict with turn_speed (rad/s), turn_radius, and circle_time
    """
    turn_type = "sharp" if sharp_turn else "normal"
    print(f"Measuring {turn_type} turn characteristics...")

    ship = create_test_ship()
    dt = 0.02

    # Set up turn action
    action = torch.zeros(3, dtype=torch.float32)
    action[0] = float(
        PowerActions.COAST
    )  # Maintain some speed? Or boost? Original code didn't specify boost, so COAST.
    # Actually original code for 'sharp_turn' was sharp_turn=1, right=1.
    # For normal turn it was right=1.

    if sharp_turn:
        # Sharp turn + Right (or Left)
        action[1] = float(TurnActions.SHARP_RIGHT)
    else:
        # Normal turn Right
        action[1] = float(TurnActions.TURN_RIGHT)

    # Phase 1: Let speed stabilize during turn
    steps = 0
    speed_history = []
    position_history = []

    # Create a bullets container for the ship
    bullets = Bullets(max_bullets=100)

    while steps < 3000:  # Allow more time for turn stabilization
        ship.forward(action, bullets, steps * dt, dt)
        speed = abs(ship.velocity)
        speed_history.append(speed)
        position_history.append(complex(ship.position))
        steps += 1

        # Check if speed has stabilized
        if len(speed_history) > 180:  # 3 seconds of data
            recent_speeds = speed_history[-180:]
            speed_variance = np.var(recent_speeds)
            if speed_variance < 0.1:
                break

    stable_speed = np.mean(speed_history[-60:])
    print(f"  Stable turning speed: {stable_speed:.2f} units/s")

    # Phase 2: Measure one complete circle
    start_position = complex(ship.position)
    start_heading = math.atan2(ship.velocity.imag, ship.velocity.real)

    circle_steps = 0
    min_circle_steps = 120  # At least 2 seconds before checking for completion
    max_circle_steps = 6000  # 100 seconds max

    while circle_steps < max_circle_steps:
        ship.forward(action, bullets, (steps + circle_steps) * dt, dt)
        circle_steps += 1

        if circle_steps > min_circle_steps:
            # Check if we've completed a circle (back near start position)
            distance_from_start = abs(ship.position - start_position)
            current_heading = math.atan2(ship.velocity.imag, ship.velocity.real)
            heading_diff = abs(current_heading - start_heading)
            # Handle angle wrapping
            if heading_diff > math.pi:
                heading_diff = 2 * math.pi - heading_diff

            if distance_from_start < 20 and heading_diff < 0.2:
                break

    # Calculate turn characteristics
    circle_time = circle_steps * dt
    angular_speed = (2 * math.pi) / circle_time if circle_time > 0 else 0

    # Estimate turn radius from the circular path
    # Use positions from the stable turning phase
    if len(position_history) > 120:
        stable_positions = position_history[-120:]
        # Find center of circular motion
        center_x = np.mean([pos.real for pos in stable_positions])
        center_y = np.mean([pos.imag for pos in stable_positions])
        center = center_x + 1j * center_y

        # Calculate average radius
        radii = [abs(pos - center) for pos in stable_positions]
        turn_radius = np.mean(radii)
    else:
        turn_radius = 0.0

    print(f"  Turn radius: {turn_radius:.2f} units")
    print(f"  Angular speed: {angular_speed:.3f} rad/s")
    print(f"  Circle time: {circle_time:.2f} s")

    return {
        "turn_speed_rad_per_s": angular_speed,
        "turn_radius": turn_radius,
        "circle_time": circle_time,
        "stable_speed": stable_speed,
    }


def derive_ship_parameters() -> dict[str, Any]:
    """
    Derive all ship parameters and return as a dictionary.
    This function can be called from tests or other modules.
    """
    # Measure all parameters
    max_speed = measure_max_speed()
    equilibrium_speed = measure_equilibrium_speed()
    normal_turn = measure_turn_characteristics(sharp_turn=False)
    sharp_turn = measure_turn_characteristics(sharp_turn=True)

    # Compile results - convert all numpy types to native Python types
    results = {
        "ship_parameters": {
            "max_speed": float(round(max_speed, 2)),
            "equilibrium_speed": float(round(equilibrium_speed, 2)),
            "normal_turn": {
                "angular_speed_rad_per_s": float(
                    round(normal_turn["turn_speed_rad_per_s"], 4)
                ),
                "turn_radius": float(round(normal_turn["turn_radius"], 2)),
                "circle_time_s": float(round(normal_turn["circle_time"], 2)),
                "stable_speed": float(round(normal_turn["stable_speed"], 2)),
            },
            "sharp_turn": {
                "angular_speed_rad_per_s": float(
                    round(sharp_turn["turn_speed_rad_per_s"], 4)
                ),
                "turn_radius": float(round(sharp_turn["turn_radius"], 2)),
                "circle_time_s": float(round(sharp_turn["circle_time"], 2)),
                "stable_speed": float(round(sharp_turn["stable_speed"], 2)),
            },
        },
        "measurement_info": {
            "timestep": 0.02,
            "world_bounds": [1024.0, 1024.0],
            "test_starting_velocity": 100.0,
            "description": "Ship parameters derived by running physics simulations",
        },
    }

    # Save to YAML file in the project
    # __file__ is src/env/ship_derived_parameters.py
    # dirname -> src/env
    # dirname -> src
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(src_dir, "derived_ship_parameters.yaml")

    with open(output_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"Results saved to {output_file}")
    print("\n=== Summary ===")
    params = results["ship_parameters"]
    print(f"Max Speed: {params['max_speed']:.2f} units/s")
    print(f"Equilibrium Speed: {params['equilibrium_speed']:.2f} units/s")
    print(
        f"Normal Turn: {params['normal_turn']['angular_speed_rad_per_s']:.4f} rad/s, radius {params['normal_turn']['turn_radius']:.2f}"
    )
    print(
        f"Sharp Turn: {params['sharp_turn']['angular_speed_rad_per_s']:.4f} rad/s, radius {params['sharp_turn']['turn_radius']:.2f}"
    )

    return results


if __name__ == "__main__":
    derive_ship_parameters()

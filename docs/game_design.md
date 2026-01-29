# Game Design: Boost and Broadside

**Boost and Broadside** is a high-speed, competitive 2D space dogfighting environment. Teams of ships compete to eliminate each other in a frictionless (but drag-limited) physics environment.

## 1. Overview

- **View**: Top-down 2D.
- **Map**: Continuous toroidal space (wraparound boundaries). Default size: `1024.0 x 1024.0`.
- **Goal**: Eliminate all ships on the opposing team.
- **Teams**: Two opposing teams (Team 0 vs Team 1). Supports asymmetric team sizes (NvM).

## 2. Physics Model

The environment uses a **Semi-Implicit Euler** integration scheme for stability.

### Movement
Ships behave like aircraft/spacecraft hybrid vehicles.
- **Thrust**: Applied in the direction of the ship's heading (attitude).
    - **Coast**: Low base thrust (maintains speed).
    - **Boost**: High thrust (consumes extra power).
    - **Reverse**: Negative thrust (braking).
- **Drag**: Air resistance proportional to velocity.
    - Higher drag during turns (induced drag).
- **Lift**: Lateral force applied during turns to drift the velocity vector towards the heading.

### Stats
| Attribute | Value | Description |
| :--- | :--- | :--- |
| **Max Health** | `100.0` | Ship structure points. |
| **Max Power** | `100.0` | Energy for boosting and shooting. |
| **Collision Radius** | `10.0` | Radius for collision detection (ship-ship and ship-bullet). |
| **Simulation DT** | `1/60` | Physics time step (60 Hz). |

## 3. Combat Mechanics

### Shooting
Ships fire projectile bullets.
- **Damage**: `10.0` per hit.
- **Speed**: `500.0` units/sec (relative to world, not shooter).
- **Lifetime**: `1.0` second.
- **Cost**: `3.0` Energy per shot.
- **Cooldown**: `0.1` seconds between shots.

### Power Management
Power regenerates over time but is consumed by actions.
- **Regeneration**: `+10.0` units/sec (Base).
- **Boost Cost**: Net loss of `-40.0` units/sec (Regen - Consumption).
- **Shooting Cost**: Immediate `-3.0` per shot.
- **Penalty**: If power hits 0, boosting and shooting are disabled until it regenerates.

### Collisions
- **Ship-Bullet**: Bullet is destroyed, ship takes `10.0` damage. Friendly fire is **enabled**.
- **Ship-Ship**: Minimal elastic collision (currently simplified or disabled in core loop depending on config).

## 4. Action Space

Agents control ships using a discrete action space with three independent components:

### 1. Power (3 actions)
| Index | Name | Description |
| :--- | :--- | :--- |
| 0 | **COAST** | Maintain movement, slowly regenerate power. |
| 1 | **BOOST** | High acceleration, consumes power. |
| 2 | **REVERSE** | Decelerate, regenerates power. |

### 2. Turn (7 actions)
| Index | Name | Description |
| :--- | :--- | :--- |
| 0 | **STRAIGHT** | No rotation. |
| 1 | **LEFT** | Turn left (approx 5 deg/step). |
| 2 | **RIGHT** | Turn right (approx 5 deg/step). |
| 3 | **SHARP LEFT** | Tight turn left (approx 15 deg/step), high drag. |
| 4 | **SHARP RIGHT** | Tight turn right (approx 15 deg/step), high drag. |
| 5 | **AIR BRAKE** | No turn, high drag (for slowing down). |
| 6 | **SHARP BRAKE** | No turn, very high drag. |

### 3. Shoot (2 actions)
| Index | Name | Description |
| :--- | :--- | :--- |
| 0 | **HOLD** | Do not fire. |
| 1 | **FIRE** | Fire principal weapon (if cooldown/power allows). |

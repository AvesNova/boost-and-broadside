# Game Design: Boost and Broadside

**Boost and Broadside** is a high-speed, competitive 2D space dogfighting environment. Teams of ships compete to eliminate each other in a frictionless (but drag-limited) physics environment.

> **Note — training overrides.** This document describes the `ShipConfig` defaults. The
> training profiles override some of them in `runs/shared.py`: bullet energy cost is `2.0`
> (not `3.0`), and `bullet_min_damage_frac=1.0` disables the head-on damage-reduction
> mechanic (by default, bullets striking within a narrow head-on cone deal a reduced
> fraction of their damage).

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

## 5. Training League and Ratings

Self-play training uses one schedule-controlled opponent region. At each rollout, that
region is divided among up to `league_k` distinct opponents sampled from the league by
prioritized fictitious self-play (PFSP). Random, scripted, the running-average policy,
and retained frozen checkpoints are ordinary league members. The default hard-PFSP
weight is `(1 - p_win)^2`, so opponents the live policy already dominates naturally
receive little curriculum weight.

Ratings are estimated separately from curriculum selection. Every completed training
or evaluation game adds a win, loss, or draw to a persistent pairwise count matrix. A
global Bradley-Terry maximum-likelihood fit converts those counts to the ELO scale, with
random as the sole fixed anchor at ELO 0. Scripted and all learned policies float. Since
the fit uses outcome ratios for every pair rather than incremental zero-sum updates,
oversampling a curriculum opponent does not drag its rating up or down.

Dedicated evaluation environments schedule informative directed pairs across the full
league, including checkpoint-vs-checkpoint games. Outcomes accumulate on the device and
are transferred once per PPO update; the CPU fit runs in the asynchronous logging worker.
Counts involving the changing live and average policies decay at different rates, while
frozen checkpoints and built-in agents retain their evidence.

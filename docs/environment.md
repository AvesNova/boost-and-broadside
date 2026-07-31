# Environment and physics

Boost and Broadside is a two-team, continuous 2D combat environment. Ships maneuver and
fire projectiles on a toroidal map until one team is eliminated or the episode horizon is
reached. Team sizes can be symmetric or asymmetric.

Engine defaults and the configuration of the reference training run differ in a few
places; both are tabulated [below](#engine-defaults-and-reference-run-overrides). For the
policy's view of the world, see [architecture](architecture.md); for experiment settings,
see [training](training.md).

## Tensorized simulation

[`TensorState`](../src/boost_and_broadside/env/state.py) stores batched ship, projectile,
and obstacle state in tensors. [`TensorEnv`](../src/boost_and_broadside/env/env.py) advances
the batch using tensor operations rather than a Python loop over environments or ships.
The dense projectile collision path is designed for GPU parallelism; its memory footprint
grows with batch size and the square of the ship count.

The environment layer has a deliberately narrow responsibility:

- [`physics.py`](../src/boost_and_broadside/env/physics.py) advances kinematics, power,
  firing, and projectile collisions;
- [`obstacle_physics.py`](../src/boost_and_broadside/env/obstacle_physics.py) advances
  optional orbital obstacles and resolves ship/obstacle contacts;
- [`env.py`](../src/boost_and_broadside/env/env.py) initializes teams, steps the state,
  and decides termination/truncation;
- [`wrapper.py`](../src/boost_and_broadside/env/wrapper.py) adds observations, decomposed
  rewards, episode statistics, and automatic reset for training.

## World and episode

The [`ShipConfig`](../src/boost_and_broadside/config/core.py) defaults define a
1024×1024 continuous world at a 60 Hz simulation timestep. Positions wrap at both map
boundaries, so there are no walls or corners. Velocity is updated before position — a
semi-implicit Euler step.

On reset, `EnvConfig.num_ships` is the **total** number of ships across both teams. A
normal reset divides them as evenly as possible; an explicit `(team0, team1)` size tuple
enables asymmetric evaluation. The reference run's `num_ships=8` therefore means 4-vs-4,
not 8-vs-8.

A combat episode ends when either team that existed at reset has no surviving ships. If
both teams survive to `max_episode_steps`, the episode is truncated and evaluation can
record a draw.

## Flight model

Each ship picks one option per step from three independent action factors, defined in
[`constants.py`](../src/boost_and_broadside/constants.py):

| Factor | Choices |
|---|---|
| Power (3) | coast, boost, reverse |
| Turn (7) | straight, left, right, sharp left, sharp right, air brake, sharp air brake |
| Shoot (2) | hold, fire |

The policy therefore produces 12 logits per ship — 3 + 7 + 2 — and samples three
sub-actions. It does not materialize a 42-way Cartesian-product action head.

Ships fly like simplified aircraft. A turn action does not rotate the ship by a fixed
rate; it holds the attitude at a fixed sideslip angle to the velocity — 5° for a normal
turn, 15° for a sharp turn. The sideslip induces lift and drag forces proportional to
speed squared, with per-action coefficients in
[`ShipConfig`](../src/boost_and_broadside/config/core.py), and it is the lift force that
curves the flight path — so turn rate rises with speed. Air-brake actions apply the
turn-level drag without the sideslip or its lift, and below a minimum speed a ship
stalls, losing turning authority.

Power is potential energy. Ignoring drag, the engine conserves the total
`½·speed² + c·power`: forward thrust converts power into speed the way an aircraft
trades altitude for airspeed by pitching down, and reverse thrust converts speed back
into power. A passive regeneration term adds power each step, and firing spends power
directly; boosting and firing are suppressed when the ship cannot pay the cost.

## Projectiles and damage

When a ship fires, the projectile velocity is the **sum of the firing ship's velocity and
the configured muzzle velocity**, plus configured spread. Projectiles wrap around the
world, expire after their configured lifetime, and use a per-ship ring buffer.

Projectile-to-ship collision is continuous over the step segment rather than a point-only
test. Friendly fire is enabled: a projectile excludes its owner but can damage a teammate.
The hit projectile is consumed and damage is recorded in a source/target matrix used by
both rewards and kill attribution.

The engine supports an incidence-angle damage multiplier that can reduce a head-on hit
to a fraction of nominal damage. The reference run sets `bullet_min_damage_frac=1.0`,
disabling the reduction, so every hit deals nominal damage there.

Ship-to-ship collision is **not** implemented in the core physics loop; ships may pass
through one another. Ship-to-obstacle and projectile collisions are implemented and
tested.

## Engine defaults and reference-run overrides

| Setting | Engine default | Reference run |
|---|---:|---:|
| World | 1024 × 1024 | 1024 × 1024 |
| Timestep | 1/60 s | 1/60 s |
| Health / power | 100 / 100 | 100 / 100 |
| Collision radius | 10 | 10 |
| Projectile muzzle speed | 500 | 500 |
| Projectile damage | 10 | 10 |
| Projectile lifetime | 1 s | 1 s |
| Firing cooldown | 0.1 s | 0.1 s |
| Firing energy cost | 3 | **2** |
| Minimum angular damage fraction | 0.1 | **1.0 (disabled)** |
| Obstacles | optional | **0** |

Engine values come from [`ShipConfig`](../src/boost_and_broadside/config/core.py); run
values from the preserved [W&B configuration
export](../checkpoints/resilient-resonance-682/wandb_export/config.json).

## Optional obstacles

Obstacle-enabled profiles add circles orbiting a per-environment harmonic gravity center.
Position-based dynamics resolves obstacle overlap during cache generation; converged maps
can be cached and sampled so training does not spend its hot loop settling new fields.

The headline combat results used no obstacles; obstacle support is an engine capability
with its own training profile. See [`runs/rl_obstacles.py`](../runs/rl_obstacles.py) and
[`obstacle_cache.py`](../src/boost_and_broadside/env/obstacle_cache.py).

## Validation

Physical behavior is covered by tensor-level tests:

- [`test_physics.py`](../tests/env/test_physics.py) covers motion, power, firing, damage,
  wraparound, and collision invariants;
- [`test_env.py`](../tests/env/test_env.py) covers reset, team layout, stepping, termination,
  and truncation;
- [`test_obstacle_physics.py`](../tests/env/test_obstacle_physics.py) covers orbital and
  collision behavior;
- [`test_rewards.py`](../tests/env/test_rewards.py) covers event accounting consumed by
  the training wrapper.

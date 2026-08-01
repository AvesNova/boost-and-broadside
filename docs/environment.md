# Environment and physics

Boost and Broadside is a two-team, continuous 2D combat environment. Ships maneuver and
fire projectiles on a 1024×1024 toroidal map at 60 Hz until one team is eliminated or the
episode horizon is reached. Optional static circular refractive fields change ship motion
without becoming solid walls.

## Tensorized simulation

[`TensorState`](../src/boost_and_broadside/env/state.py) stores batched ships, projectiles,
and fields as fixed-shape tensors. [`TensorEnv`](../src/boost_and_broadside/env/env.py)
advances thousands of environments without Python loops over environments or ships.
Field evaluation is an `O(B*N*M)` reduction over `(environment, ship, field)`; hierarchy
and material deltas are precomputed by
[`FieldMapCache`](../src/boost_and_broadside/env/field_cache.py), so stepping performs no
sorting, region selection, rejection sampling, or host synchronization.

The main layers are:

- [`physics.py`](../src/boost_and_broadside/env/physics.py): ship control, effective-mass
  transport, power, firing, and projectile collisions;
- [`field_physics.py`](../src/boost_and_broadside/env/field_physics.py): toroidal field
  profiles, hierarchy validation, and telescoping index composition;
- [`env.py`](../src/boost_and_broadside/env/env.py): reset, teams, stepping, and episode
  termination;
- [`wrapper.py`](../src/boost_and_broadside/env/wrapper.py): observations, decomposed
  rewards, statistics, and automatic reset.

`EnvConfig.num_ships` is the total across both teams. The primary `runs/rl.py` profile
uses eight ships (4-vs-4) and zero fields, preserving the original ambient-only hot path
exactly. `runs/rl_fields.py` is a four-field combat profile for smoke tests and future
training.

## Flight, proper speed, and power

Actions factor into power (coast/base thrust, boost, reverse), turn (straight, normal or
sharp sideslip, and air brakes), and shoot. Sideslip induces lift and drag; below the
configured minimum proper speed the ship stalls and loses turning authority.

The ambient medium has refractive index `n=1`. Inside fields, a ship has physical
effective mass

```text
m(x) = n(x)^2
d(mv)/dt = 0.5 |v|^2 grad(m) + F_ship
```

or equivalently

```text
a = F_ship/m + 0.5 |v|^2 grad(log m) - (v·grad(log m))v.
```

The simulator uses proper or medium-relative speed `u=n*v_world` for configured spawn
speed, stall, and the existing lift/drag interpretation. Thus a ship initialized inside
index `n` starts at `default_speed/n`. Low index increases world speed and control rate;
high index decreases them, while the log-symmetric tiers remain approximately reciprocal.

Ignoring drag, regeneration, firing, and damage, passive field motion conserves
`H=0.5*n^2*|v|^2`. Powered motion exchanges actual generalized mechanical work with the
ship battery, conserving

```text
E = 0.5*n^2*|v|^2 + power_speed_constant*power.
```

Forward thrust cannot spend unavailable power. Reverse stops at the kinetic-energy
minimum and cannot become a free backwards boost; recovered work is capped by available
battery storage. Drag is integrated with its exact scalar speed solution and dissipates
energy. Passive regeneration remains an explicit external source.

## Refractive-field profile

A field has a center, nominal radius `r`, complete transition width `w`, absolute interior
index, and independent interface damage. The band extends from `r-w/2` to `r+w/2`. For
minimum-image toroidal distance `d=distance(x, center)-r`:

```text
z = clamp(0.5 - d/w, 0, 1)
alpha = 6z^5 - 15z^4 + 10z^3.
```

The analytic gradient is used. Both derivatives are flat at the band edges and the
gradient is explicitly finite at the center. The four non-ambient material levels use one
configurable log step `s` (default `sqrt(2)`):

| Level | Exponent | Index | Passive world-speed factor |
|---|---:|---:|---:|
| `VERY_LOW` | -2 | `s^-2` = 0.5 | `s^2` = 2 |
| `LOW` | -1 | `s^-1` ≈ 0.707 | `s` ≈ 1.414 |
| `HIGH` | +1 | `s` ≈ 1.414 | `s^-1` ≈ 0.707 |
| `VERY_HIGH` | +2 | `s^2` = 2 | `s^-2` = 0.5 |

The ambient `AMBIENT=0` level is not sampled as a field. Smooth refraction, including
total internal reflection when transmission is impossible, emerges from the same force;
there is no collision, random branch, breakthrough speed, or force clamp.

### Integration order

Each field step uses a symmetric split:

1. half control/thrust work, exact scalar drag, and work-free lift rotation;
2. fixed-count midpoint passive transport substeps;
3. passive projection to preserve `n*|v|` at the newly evaluated index;
4. the second control half-step, then explicit passive power regeneration.

The midpoint force determines direction, so energy projection cannot substitute for the
correct refractive curvature. Projection is confined to the passive split and cannot
erase powered work. Two substeps at the configured 60 Hz, speeds, and minimum 40-pixel
band keep each ordinary step far narrower than an interface. Bullets deliberately remain
on their existing straight toroidal trajectories: bullet refraction is a future extension
point in the projectile update, and fields never absorb bullets.

## Smooth interface damage

Index and interface damage are independent. The three levels are `NONE=0`,
`STANDARD=D`, and `SEVERE=2D`, with default `D=10`. For cached previous and newly
evaluated alpha values, each field contributes

```text
damage_i = D_i * abs(alpha_i_next - alpha_i_previous).
```

Midpoint total variation also accounts for an approach and reflection within one step.
A complete monotonic crossing therefore costs exactly `D_i`, independent of speed and
band width; remaining still costs nothing, partial reflection costs proportionally, and
oscillation accumulates its traveled alpha variation. Reset initializes cached alpha at
the spawn point, so spawning inside a field does no artificial damage. Field damage is a
health mechanic and flows through normal health-loss and death rewards, not mechanical
energy or projectile-source attribution.

## Nesting and map validity

Fields are either completely disjoint or strictly nested. Partial intersections and
overlapping transition bands are rejected. A child `c` is contained in parent `p` only if

```text
distance(c,p) + r_c + w_c/2 <= r_p - w_p/2.
```

Disjoint bands require `distance(i,j) >= r_i+r_j+w_i/2+w_j/2`. All distances use the
same minimum-image toroidal geometry as runtime evaluation and rendering. Outer extent
`r+w/2` must be strictly less than half the shorter world dimension, avoiding ambiguous
antipodal circle topology.

Construction selects the smallest direct enclosing parent. Each cached field stores
`delta_n = n_child - n_parent` (roots subtract ambient 1), and runtime composes

```text
n(x) = 1 + sum(delta_n_i * alpha_i(x))
grad(n) = sum(delta_n_i * grad(alpha_i(x))).
```

This telescopes through arbitrary depth: a child core has the child's absolute index,
regardless of whether it is higher or lower than its parent, without hard per-step
`argmin`/`max` priority or derivative ridges. Cached map generation has bounded attempts
and fails clearly when requested geometry cannot be packed.

## Projectiles, collisions, and rendering

Projectile velocity is ship velocity plus configured muzzle velocity and spread. Bullets
wrap, expire through a per-ship ring buffer, and use continuous segment collision against
ships. Friendly fire is enabled. Ship-to-ship collision is not implemented, and fields
are always traversable.

Fields render as unfilled outlines with toroidal edge copies. Cyan/blue means lower/faster
index; violet means higher/slower index, with stronger levels brighter and more saturated.
Dotted, dashed, and solid borders mean none, standard, and severe damage respectively.
Solid means severe interface damage—not an impermeable wall. Parents draw first so nested
children remain visible.

## Measured field cost

The pure-environment benchmark (no bullets or policy inference) on an NVIDIA GeForce RTX
4070 Laptop GPU, with 4,096 environments, eight ships, 50 warmup ticks, and 500 timed
ticks, measured:

| Fields | Environment steps/s | Relative | State memory | Peak allocation | Tokens | Attention-pair factor |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9,490,454 | 1.000× | 4.58 MiB | 8.89 MiB | 8 | 1.000× |
| 1 | 1,194,662 | 0.126× | 4.85 MiB | 12.29 MiB | 9 | 1.266× |
| 2 | 1,201,744 | 0.127× | 5.12 MiB | 15.22 MiB | 10 | 1.562× |
| 4 | 1,194,044 | 0.126× | 5.67 MiB | 21.08 MiB | 12 | 2.250× |

The field path is launch-bound at these modest `M` values, so one through four fields
have similar throughput. The zero-field branch bypasses every field evaluation and retains
the old kinematics implementation. Policy inference is intentionally separate: fields add
tokens, so attention pair count grows theoretically as `(N+M)^2/N^2`; the benchmark's
last column reports that factor rather than blending policy cost into physics cost. Results
depend on hardware and clocks; reproduce them with `benchmarks/field_throughput.py`.

## Validation

- [`test_physics.py`](../tests/env/test_physics.py): unchanged ambient motion, power,
  firing, wraparound, and bullet collision;
- [`test_field_physics.py`](../tests/env/test_field_physics.py): profile gradients,
  toroidal geometry, hierarchy, materials, generation, and reset;
- [`test_field_transport.py`](../tests/env/test_field_transport.py): long-run energy,
  refraction/TIR, power exchange, smooth damage, and bullet non-absorption;
- [`test_env.py`](../tests/env/test_env.py),
  [`test_rewards.py`](../tests/env/test_rewards.py), and
  [`test_renderer.py`](../tests/ui/test_renderer.py): integration, attribution, numeric
  observations, and outline rendering.

The zero/one/two/four-field environment benchmark is in
[`benchmarks/field_throughput.py`](../benchmarks/field_throughput.py).

# Environment and physics

Frontline is a 5v5 objective game at 30 Hz on a translated 16384×16384 torus,
with five rotating-role zones and a 2600 px playable radius. A match ends at a net
front lead of three or after 300 seconds; timeout uses the front's sign.
Legacy elimination combat remains available on its smaller 60 Hz map.

## Frontline shields and lives

Ships carry 100 shield capacity. Enemy hits remove shields; a ship that entered a
physics tick with zero shields dies on subsequent enemy damage. All projectile hits
are aggregated before applying that rule, so simultaneous shots cannot both break a
shield and kill its ship. Friendly fire removes shields and delays recovery but cannot
finish a depleted ship. The soft outer boundary also depletes shields, then kills on
a later tick. Field interfaces and defense/spawn zones inflict no passive damage.

Damage resets a five-second recharge delay. After that undamaged interval, shields
recover at 15 units/second anywhere, capped at 100. A hit and recharge never pay out
on the same tick. The timer is an observed, predicted ship channel.

Initial placement and instant respawns use 15 shields, 40 power and 30 px/s proper
speed, facing the enemy defense, with the full recharge delay. This avoids a special
full-resource opening and leaves enough power and steering speed to act immediately.
Ships inside their current friendly spawn are invulnerable. Spawn membership is
sampled after movement at collision time; a zone-role change takes effect for the next
collision tick. Spawn offers protection, not a special healing rate.

Respawns preserve slot identity, clear previous-life damage attribution on the next
tick, and mark the transition discontinuous for auxiliary prediction. Recurrent match
memory persists. Unseen enemies with zero predicted shields remain valid beliefs;
zero shields no longer implies death. Existing checkpoints are incompatible with the
new `frontline_shields_v9` observation/feature contract.

## Tensorized simulation

[`TensorState`](../src/boost_and_broadside/env/state.py) stores batched ships, projectiles,
and fields as fixed-shape tensors. [`TensorEnv`](../src/boost_and_broadside/env/env.py)
advances thousands of environments without Python loops over environments or ships.
Ship field evaluation is an `O(B*N*M)` reduction over `(environment, ship, field)`.
Projectile transport applies the same fixed-shape reduction to the `N*K` ring-buffer
slots; inactive slots remain masked rather than being dynamically compacted. Each reset
generates a fresh independent layout directly on device. Stepping performs no sorting,
region selection, rejection sampling, or host synchronization.

The main layers are:

- [`physics.py`](../src/boost_and_broadside/env/physics.py): shared ship/projectile
  effective-mass transport, control, power, firing, drag, and swept collisions;
- [`field_physics.py`](../src/boost_and_broadside/env/field_physics.py): toroidal field
  profiles and bounded overlapping log-index composition;
- [`field_generation.py`](../src/boost_and_broadside/env/field_generation.py): direct
  on-reset layouts, including Frontline's common map translation;
- [`env.py`](../src/boost_and_broadside/env/env.py): reset, teams, stepping, and episode
  termination;
- [`wrapper.py`](../src/boost_and_broadside/env/wrapper.py): observations, decomposed
  rewards, statistics, and automatic reset.

## Team perception and map tokens

Frontline uses finite, team-shared sight. Every living allied ship tests targets within
`EnvConfig.vision_range` (1024 px in play and in the `rl` profile) along the shortest
toroidal displacement. A target seen by any ally is visible to the whole team.

Sight is broken by any opaque core the line crosses. A field's core is its nominal radius
less half its transition band, so the graded part of the interface stays transparent. One
test covers the three ways a line can be lost—looking into a core, out of one, or past
one—because each crosses the boundary. The sole exemption is a line with both endpoints
inside the same core: a disk is convex, so that line never leaves it and two ships sharing
a field still see each other. This is natural map occlusion only—there are no synthetic fog
volumes.

Capture zones are opaque by default under the same convex-core rule: ships sharing
one zone can see one another, but lines crossing its boundary are blocked. The explicit
`zones_occlude=False` override remains useful for controlled visibility experiments.

A successful shot reveals its firing ship to both teams for that state sample, regardless
of range or intervening field cores. The reveal uses `ship_is_shooting`, so a requested shot
that fails because of cooldown, power, or death does not reveal anything. The reveal exposes
the ship's ordinary visible state but not its private pending action.

The environment constructs Team 0 and Team 1 observations independently. An unseen enemy
ship has an explicit false visibility mask and every state channel is replaced with zero as
defense in depth. This includes position, velocity, health, power, cooldown, alive state,
local refractive state, and bullets. Policy-side belief tracking may retain a previously seen
enemy as a valid token, recursively replacing only its predictable physical channels and
adding time since observation. A never-seen enemy remains absent. Enemy pending actions and
hidden local field gradients remain zero rather than being predicted. Enemy pending actions
are private even while the enemy itself is visible. Allies and static map geometry remain
known. `vision_range=None` is the explicit omniscient compatibility mode.

In Team 0/Team 1 rendering modes, unseen world pixels receive a mild neutral-gray overlay.
The visible mask is the union of allied sight circles, each with every opaque core in range
subtracted along with the umbra behind it, and clipped to the core the observer stands in
when it stands in one. It is the same rule the environment applies, and
`tests/ui/test_renderer.py` holds it to that by probing the drawn mask against the
environment's own predicate across the viewport. The overlay is applied after fields, zones,
and the boundary, so unseen empty space and static outlines desaturate together;
visible/revealed units are then drawn at full contrast. It is rebuilt every frame — the
shadow geometry costs far less than the full-viewport composite that follows it, so there is
nothing to gain from holding a stale mask. The full-information spectator mode has no
overlay.

The entity-token axis is typed rather than inferred from position:

| Token type | Frontline count | Globally visible information |
|---|---:|---|
| Ship | 10 | Team-relative dynamic state when visible |
| Field | configured (`10` in play) | Geometry, target index |
| Zone | 5 | Position, role/owner, capture progress and direction |
| Boundary/global | 1 | Playable radius, front, win threshold, time remaining, mode |

The boundary and global state intentionally share one token. Team canonicalization swaps
zone ownership and roles as well as ship/bullet labels, and negates front/capture direction.
Finite-vision training therefore requires `ego_pass`; the legacy `shared_pass` cannot safely
serve one masked team view to both sides and is rejected.

`EnvConfig.num_ships` is the total across both teams, and `EnvConfig.num_fields` the count
of static-for-one-episode fields. `profiles/rl.py` trains at ten ships (5-vs-5) and ten fields.
There is no separate field-free profile: `num_fields` sets the token count and no weight
shape depends on it, so zero fields is a configuration -- the one run 682 trained under, and
the ambient-only hot path it still exercises -- rather than a different model.

Ships also observe `grad(n)` at their own position. The physics had been computing and
consuming it long before it reached the observation, as the force term in
`a = F/m + 0.5|v|^2 grad(log m) - (v.grad(log m))v`. Until it was exposed, a ship could see
which medium it occupied but not which way that medium was changing.

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
index. The band extends from `r-w/2` to `r+w/2`. For
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

Ships and projectiles use the same passive field-transport driver and choose either the
`two_step` or `midpoint` integrator independently. The default ship path remains midpoint
with two substeps. Each ship field step uses a symmetric split:

1. half control/thrust work, exact scalar drag, and work-free lift rotation;
2. fixed-count midpoint passive transport substeps;
3. passive projection to preserve `n*|v|` at the newly evaluated index;
4. the second control half-step, then explicit passive power regeneration.

The midpoint force determines direction, so energy projection cannot substitute for the
correct refractive curvature. Projection is confined to the passive split and cannot
erase powered work. Two substeps at the configured 60 Hz, speeds, and minimum 40-pixel
band keep each ordinary step far narrower than an interface.

The provisional Frontline play contract instead uses one `two_step` ship step at 30 Hz.
Its maximum configured displacement is 6 px against the same 40 px interface, and swept
projectile collision plus two projectile field substeps remain enabled. This keeps every
per-second gameplay rate unchanged while making single-game interactive latency practical.

Projectile transport uses exact quadratic-drag half-steps around the passive field step.
Its default `two_step` integrator uses an optical acceleration kick, drift, endpoint field
evaluation, and the same projection that preserves `n*|v|`. The selectable `midpoint`
integrator uses the ship-quality midpoint force at additional cost. Both paths retain the
half-tick position needed for two-segment swept collision detection. Integrator selection
is static Python configuration; it does not read tensor values or synchronize the GPU.

## Arbitrary overlap and map generation

Fields may partially intersect, share transition bands, coincide, or nest in any order.
For target `L_i=log(n_i)` and memberships `alpha_i`, composition is

```text
A = 1 - product_i(1-alpha_i)
log(n) = A * sum_i(alpha_i*L_i) / sum_i(alpha_i)
```

with zero log-index when no field contributes. Identical overlaps reinforce partial
coverage without exceeding their shared target. Different materials blend in signed log
space, so equally covered reciprocal targets cancel to ambient. The union coverage keeps
optical strength bounded as field count grows.

The analytic gradient uses vectorized exclusive prefix/suffix products, without unstable
division by `1-alpha` or a Python loop over fields. All distances use minimum-image
toroidal geometry.
Outer extent
`r+w/2` must be strictly less than half the shorter world dimension, avoiding ambiguous
antipodal circle topology. For the default 1024×1024 world and 40-pixel transition width,
this requires `r < 492`; changing the maximum radius beyond that requires a larger world
or a different field-topology definition.

Centers, radii, widths, and target materials are sampled directly on every
episode reset. Randomized low-discrepancy R2 samples cover combat toroids; randomized
sunflower samples stratify equal-area Frontline disks. Both reduce clustering without a
pairwise rejection loop and still permit useful overlap. Frontline fields share the same
random translated map center as the zones and boundary, with each complete outer extent
inside that boundary. The zero-field branch allocates an empty field axis and bypasses
field evaluation.

## Projectiles, collisions, and rendering

The configured muzzle speed is a proper speed relative to the firing ship: it is divided
by the local index before adding ship velocity and spread. Projectiles continuously
refract, experience quadratic drag, wrap, and expire through a fixed per-ship ring buffer.
The production pool uses ten slots, sufficient for the default one-second lifetime and
0.1-second cooldown without overwriting a live projectile.

Projectiles are also observable by the policy. `observation_from_state(...,
include_bullets=True)` flattens the per-ship ring buffers into one `(B, N*K, ...)` axis
carrying position, velocity, remaining lifetime, local index and index
gradient, and the shooter's team. Every slot is emitted; inactive ones are masked out of
attention rather than compacted, so the shape stays static. See
[architecture](architecture.md#bullet-cross-attention) for how the policy reads them.

Each tick retains start, half-tick, and final positions ephemerally and tests both swept
segments against ships, preventing fast projectiles from tunneling between endpoints.
On impact, incidence scaling is applied to the configured fixed bullet damage.
Friendly fire is enabled. Ship-to-ship collision is not implemented, and fields remain
traversable rather than absorbing projectiles as solid obstacles.

Fields render as translucent transition annuli plus outlines with toroidal edge copies.
Cyan/blue means lower/faster
index; violet means higher/slower index, with stronger levels brighter and more saturated.
Fields use uniform thin outlines. Alpha-blended annuli make
partial and coincident overlaps visible while nominal contours stay individually legible.

## Historical field cost (before the shield overhaul)

The pure-environment benchmark (no bullets or policy inference) on an NVIDIA GeForce RTX
4070 Laptop GPU, with 4,096 environments, eight ships, 30 warmup ticks, and 300 timed
ticks, measured:

| Fields | Environment steps/s | Relative | State memory | Reset µs/env | Peak allocation | Tokens | Attention-pair factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,543,526 | 1.000× | 5.58 MiB | 0.687 | 9.90 MiB | 8 | 1.000× |
| 4 | 223,241 | 0.145× | 6.48 MiB | 1.520 | 22.65 MiB | 12 | 2.250× |
| 10 | 228,889 | 0.148× | 7.84 MiB | 1.538 | 39.95 MiB | 18 | 5.062× |
| 20 | 231,166 | 0.150× | 10.11 MiB | 1.365 | 68.78 MiB | 28 | 12.250× |

The zero-field branch bypasses every field evaluation and retains the old kinematics
implementation. GPU throughput is nearly flat once the field path is active because these
small reductions are launch-bound. Policy
inference is intentionally separate: fields add tokens, so attention pair count grows
theoretically as `(N+M)^2/N^2`; the benchmark's last column reports that factor rather than
blending policy cost into physics cost. Results depend on hardware and clocks; reproduce
them with `benchmarks/field_throughput.py`.

## Validation

- [`test_physics.py`](../tests/env/test_physics.py): unchanged ambient motion, power,
  firing, wraparound, and bullet collision;
- [`test_field_physics.py`](../tests/env/test_field_physics.py): profile/composition
  gradients, reciprocal cancellation, identical and toroidal overlap, materials,
  generation, and reset;
- [`test_field_transport.py`](../tests/env/test_field_transport.py): long-run energy,
  refraction/TIR and power exchange;
- [`test_bullet_fields.py`](../tests/env/test_bullet_fields.py): selectable projectile
  integrators, refraction/TIR, proper-speed conservation, and
  high-resolution trajectory comparisons;
- [`test_perception.py`](../tests/env/test_perception.py): the four sight rules —
  team sharing, opaque-core occlusion in both directions, the circular range, and the
  firing reveal — plus optional zone occlusion and declared projectile perception;
- [`test_env.py`](../tests/env/test_env.py),
  [`test_rewards.py`](../tests/env/test_rewards.py), and
  [`test_renderer.py`](../tests/ui/test_renderer.py): integration, attribution, numeric
  observations, outline rendering, and the fog mask held against the environment's own
  sight predicate across the viewport.

The zero/one/two/four/ten/twenty-field environment benchmark is in
[`benchmarks/field_throughput.py`](../benchmarks/field_throughput.py). Saturated projectile
storage, drag, integrator, compilation, and capacity comparisons are in
[`benchmarks/bullet_throughput.py`](../benchmarks/bullet_throughput.py).

The 256-map fog distribution and isolated GPU observation profile are reproducible with
[`benchmarks/frontline_fog_suite.py`](../benchmarks/frontline_fog_suite.py). At the earlier
1600 px range and the earlier core rule, the 60-second scripted sample saw enemies 74.2% of the
time; team sharing added 21.2 percentage points over individual sight, and fields blocked 11.6%
of otherwise in-range
exposure. A 256-environment visibility pass measured 4.12 ms, while visibility plus both
masked team observations measured 15.80 ms on an RTX 4070 Laptop GPU.

The default 5v5 Frontline play preset uses one CPU thread, a 30 Hz tick/decision rate, and
a state-only scripted loop that skips unused reward and policy-observation work. Larger
fleets retain the requested CUDA device and use the compiled interactive path. The
end-to-end headless benchmark,
including perception and fog-aware Team 0 rendering, measured 32.74 ms per decision
(1.02× realtime) at 900 px. The terrain stencil is quarter resolution and cached for eight
ticks; dynamic unit visibility remains 30 Hz, and camera/view changes invalidate the cache.
The earlier omniscient path measured 27.54 ms (1.21× realtime), versus 94.12 ms (0.35×)
with a 16-thread tiny-tensor workload. Reproduce it with
[`benchmarks/play_throughput.py`](../benchmarks/play_throughput.py).

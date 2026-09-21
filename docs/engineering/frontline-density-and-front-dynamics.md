# Frontline density scaling and front dynamics

Two questions, measured together on 2026-09-21: what map geometry holds gameplay
constant as the fleet grows, and why scripted Frontline matches so rarely reach a
decision. The second turned out to be a property of the baseline 5v5 rules, not
of any scaling.

Status: **shipped.** `frontline_scale` and `scaled_frontline_geometry` live in
[`env/frontline.py`](../../src/boost_and_broadside/env/frontline.py) and are
applied once per mode — `config/resolve.py` for `train`, and
`_frontline_interactive_config` for `play`/`watch`. `FRONTLINE_WORLD_SIZE` is
65536 px. Every existing checkpoint predates this and is incompatible.

## The density-preserving map

One linear factor drives every Frontline length:

```
s = sqrt(num_ships / 10)
```

applied to `playable_radius`, `zone_ring_radius`, `zone_radius`,
`field_radius_min/max` and `field_transition_width_min/max`. Areas go as `s**2`,
so ships per unit area, zone area per ship and field area per ship all hold at
their 5v5 values. Zone count (5) and field count (10) do not change, and the
ring keeps a constant fraction of the playable radius.

| | 5v5 | 50v50 |
|---|---:|---:|
| `s` | 1.0000 | 3.1623 |
| `playable_radius` | 2600.0 | 8221.9 |
| `zone_ring_radius` | 1200.0 | 3794.7 |
| `zone_radius` | 330.0 | 1043.6 |
| field radius range | 30 – 750 | 94.9 – 2371.7 |
| ring / playable | 0.4615 | 0.4615 |
| ships per megapixel | 0.471 | 0.471 |

Ship physics is deliberately *not* scaled. Hull size, speed and weapon range are
what make an engagement feel the way it does; holding areal density fixed while
they stay fixed is the point.

`vision_range` is not scaled either, for the same reason: constant areal density
times a constant sight area holds the number of ships inside a ship's sight at its
5v5 value, which is the invariant that makes a local engagement feel the same.
Scaling it would instead hold "fraction of the map visible" constant and multiply
ships-in-sight by the fleet ratio. The pacing results below support this choice —
the contest, not the distance, sets the clock.

The scripted agent's `frontline_zone_radius` scales with the map because it stands
in for objective geometry; `frontline_combat_radius` and
`frontline_separation_radius` do not, being weapon- and hull-scale.

Checkpoint compatibility splits `FrontlineConfig` into rules and geometry.
`frontline_rules_match` compares only the nine rule fields, so a policy trained at
5v5 can be evaluated at 50v50 — comparing the whole dataclass would reject exactly
the zero-shot transfer the scaling exists to enable. `world_size` must still match
exactly, since a checkpoint's position basis is derived from the period.

### Why the world has to grow

`2600 * sqrt(10) = 8222 px` exceeds the 16384 px toroid's 8192 px half-period,
where minimum-image displacement stops being defined. Exact density preservation
at 50v50 is **unreachable** on the contract world. At 65536 px the half-period is
32768 px and the constraint does not bind until ~1433 ships.

65536 is four times the contract and a power of two, which matters:
`position_fourier_frequencies` grows 8 -> 10 by adding two *coarse* harmonics
(65536 and 32768 px) while the finest period stays exactly 128 px. The basis
extends downward in frequency; fine resolution is untouched.

Measured cost of the 4x world (RTX 4070 Laptop, B=256, 5v5):

| | 16384 | 65536 |
|---|---:|---:|
| position harmonics per axis | 8 | 10 |
| encoding error (vs float64) | 7.5e-05 | 3.0e-04 |
| implied positional precision | 0.0015 px | 0.0060 px |
| toroidal seam error | 2.2e-05 | 9.0e-05 |
| float32 ulp at world edge | 0.00195 px | 0.00781 px |
| policy-only forward | 8.967 ms | 9.026 ms (+0.7%) |
| parameters | 1,907,877 | 1,909,925 |

Everything degrades by exactly 4x and all of it stays negligible against a 10 px
collision radius and 3.3 px of travel per tick. Throughput is unaffected: token
count, attention shapes and physics kernels are identical. The real cost is that
`world_size` is baked into `observation_contract.position_frequencies`, the RoPE
axes and the assert at `env.py:62`, so changing it retires every checkpoint.

RoPE is enabled, so the 4x world uses 48 of 64 head dimensions rather than 40,
leaving 16 passthrough dims for content instead of 24. The rotary periodicity test
moved from `atol=1e-4` to `1e-3`: the finest harmonic's phase argument reaches
`2*pi*512` here, four times what it reached before, and the float32 frequency
buffer's residual scales with it. Measured 1.265e-04 for a full-world translation,
which is 0.0026 px of implied position. Building the frequency buffer in float64
and casting after the multiply would restore the tighter bound if it ever matters.

## Match pacing

`bnb`-free harness; scripted vs scripted, `max_episode_steps` raised to 54000
(30 min at 30 Hz) so matches end on the front-3 threshold rather than the clock.

| | 5v5 | 50v50 | ratio |
|---|---:|---:|---:|
| median time to first capture | 65.6 s | 53.3 s | 0.81x |
| median seconds between captures | 16.2 s | 23.4 s | 1.44x |
| median game length | 727.5 s | 1032.4 s | 1.42x |
| median captures per match | 21 | 69 | 3.29x |
| decided within 30 min | 263/512 (51%) | 40/128 (31%) | |

Pacing scales far better than traversal: distances grow 3.16x but inter-capture
time grows only 1.44x, and first capture gets *faster* with a much tighter
spread (p10-p90 of 45-65 s at 50v50 against 27-226 s at 5v5). More ships means
more simultaneous contests.

**The shipped 300 s cap truncates most scripted matches.** Median 5v5 game length
is 727 s against a 9000-step `max_episode_steps`; only ~10% decide inside 300 s.
This is a baseline 5v5 property, independent of any scaling. All medians above
are conditioned on matches that finished, so with 49% and 69% censored they are
biased low.

## The front is anti-persistent, not a random walk

A symmetric +-1 walk absorbing at +-3 has mean first passage `3*3 = 9` captures
and median 7. Observed medians are 21 (5v5) and 69 (50v50); a random walk exceeds
21 only 7.5% of the time and 69 about once in 13000 matches.

Measuring the *sign* of `front_delta` shows why:

| | 10 ships, 10-ship map | 10 ships, 100-ship map |
|---|---:|---:|
| ships per megapixel | 0.471 | 0.047 |
| capture transitions | 4494 | 7093 |
| `P(reversal)` | 0.8405 | 0.7706 |
| gap before a reversal | 13.7 s | 21.6 s |
| gap before a continuation | 69.6 s | 48.1 s |

Both at 18000 steps, 512 envs. Against a fair coin the baseline reversal rate is
`z = 24.8` (n=1228 on the first 6000-step run).

### Mechanism: the captured zone becomes the opponent's target

`roles_from_front` rotates the five-zone role pattern rigidly around the ring, so
both spawns advance and the post-capture configuration is congruent to the
pre-capture one. There is no *positional* restoring force. What there is:

```
front=0: z0=NEUTRAL  z1=TEAM0_SPAWN  z2=TEAM0_DEFENSE  z3=TEAM1_DEFENSE  z4=TEAM1_SPAWN
front=1: z0=TEAM1_SPAWN  z1=NEUTRAL  z2=TEAM0_SPAWN  z3=TEAM0_DEFENSE  z4=TEAM1_DEFENSE

team0 captures z3 (was TEAM1_DEFENSE) -> z3 becomes TEAM0_DEFENSE
team1's next target (TEAM0_DEFENSE) is ... z3
```

The zone you just took is immediately your opponent's next objective, and their
defenders are already standing in it. The counter-attacker needs no travel; the
attacker must hold ground it just fought onto. That yields negative
autocorrelation with no positional asymmetry, which is what the passage times
require.

An anti-persistent walk with `P(reversal) = 0.80` gives median 21, matching 5v5;
the directly measured 0.84 predicts median 29. Treat 0.80-0.85 as the range and
the mechanism as established, not the point estimate.

### Pacing is set by the contest, not by distance

Travel time was the obvious explanation and it is **refuted**. On a 3.16x larger
map with the same fleet, continuations got 31% *faster* (69.6 -> 48.1 s), not
slower. The density reading fits every cell:

- reversals are slower when sparse (13.7 -> 21.6 s): one ship in a zone gives
  harmonic pressure `H_1 = 1` and the full `capture_seconds`, where two or three
  stacked ships give `H_2 = 1.5` or `H_3 = 1.83` and finish in 5-7 s;
- continuations are faster when sparse (69.6 -> 48.1 s): at a tenth the density
  nobody contests the next zone, so the 69.6 s at baseline is the cost of
  *fighting*, not of walking;
- the sparse map produced ~1.9x more captures per unit time, and 512/512 envs
  scored against 451/512 at baseline.

This is the argument that density is the right invariant to preserve: holding it
constant holds the contest constant, and the contest sets the clock.

## Open

- The fourth cell, 100 ships on the 10-ship map (4.71 ships/Mpx), would complete
  the 2x2. It should show the fastest reversals, slowest continuations and the
  highest reversal rate.
- `H2` (the global `zone_capture_progress` reset on any capture), respawn massing,
  the harmonic pressure term and scripted defend/rally behaviour are **not** ruled
  out; the test showed H1 is present and large, not that it is alone. Separating
  the rest needs interventions, not instrumentation.
- The scaled-`vision_range` 50v50 variant was never run.
- If matches should decide, the highest-leverage lever is the H1 mechanism: a
  capture-ownership cooldown, a post-capture grace period, or offsetting the two
  teams' target residues so they do not contest the same zone back to back.
  Raising `front_win_threshold` does the opposite, and a longer
  `max_episode_steps` only stops truncating a game that still will not resolve.

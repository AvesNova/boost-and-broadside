# State-derived Frontline teacher

Working branch: `frontline/state-derived-strategy`, based on `feat/frontline-overhaul`.

## Rules

The controller has no strategic memory, random roles, ship-index preferences, assigned
zones, rally readiness, or healing latch. Action sampling remains stochastic; the
probability labels are deterministic functions of the current team-visible state.

1. **Combat:** Gaussian distance-weighted normalized health gives allied strength A
   (including self) and visible enemy strength E. The score is
   `tanh(log((A + 1e-6)/(E + 1e-6)) + aggression)`. It multiplies the weighted mean
   enemy bearing and `1-exp(-E)`, so empty/distant enemy populations exert negligible
   influence. Negative scores oppose commitment; the close dogfighter remains protected.
2. **Objectives:** Each ship contributes `health * exp(-(distance/R_Z)^2)` to each zone.
   Matrix products sum allied and visible enemy contributions; subtracting self gives
   the allied strength that would remain without the observer. Defense and offense use
   margins `m*exp(-aggression)` and `m*exp(aggression)`. Need is
   `0.5*softplus(2*(margin + enemy_pressure - allied_pressure_without_self))`.
   Divide need by `1+(distance/R_Z)^2` and normalize across the two defense zones.
   The objective force is the resulting weighted sum of unit zone bearings.
3. **Separation:** Sum short-range allied repulsion with Gaussian weights and
   `displacement/R_S`, divided by `max(1, sum(weights))`. This regularizes coincident
   positions and bounds dense-fleet repulsion without diluting it with distant allies.
   Exactly coincident identical ships cannot deterministically break symmetry; their
   sampled actions can separate them.
4. **Recovery:** For normalized health h, recovery influence is
   `q=(1-h)^2 / ((1-h)^2 + (h/h_R)^2 + 1e-8)`. The final navigation force is
   `(1-q)*(objective + combat) + q*spawn_bearing + separation`.
   Recovery grows smoothly and vanishes at full health. No respawn flag is used.

Navigation uses the existing flight probability ramps against this bearing and a
recovery-weighted mean objective/spawn travel distance. It does not shoot at zones.
Zero resultant force preserves current heading.

## Dogfighter and information boundary

The original combat calculation is shared unchanged by combat mode and Frontline.
Every action head uses its exact old probability distribution within `r0`, including
low-health ships. Between `r0` and `R_C`, probabilities are converted to log probabilities,
linearly interpolated, and softmaxed. All heads use the same alpha. Exact endpoint
selection retains original structural zeros; a fixed `1e-8` interior floor bounds the
otherwise infinite logits. Thus numerical continuity is within the probability floor,
not a claim of mathematical continuity at exact zero support.

Frontline callers supply the same `(B,2,N)` team visibility mask used by observations.
Every enemy-dependent rule and the dogfighter use that mask. If omitted, the controller
assumes no enemies are visible. It still knows allies and the static map. The match
benchmark now supplies authoritative visibility too; its former omission made it
omniscient. Combat mode retains its existing optional-mask contract.

## Parameters

| Concept | Configuration | Default |
|---|---|---|
| Offensive/defensive bias a | `frontline_aggression` | 1.0 |
| Tactical radius R_C | `frontline_combat_radius` | 600 px |
| Zone support R_Z | `frontline_zone_radius` | 900 px |
| Neutral margin m | `frontline_zone_margin` | 1 full-health ship |
| Separation R_S | `frontline_separation_radius` | 120 px |
| Recovery health h_R | `frontline_recovery_health` | 0.5 |

`r0` is the lower shooting-distance ramp endpoint (200 px by default). `R_C` must
exceed it. Strategic parameters do not enter the existing combat tuning vector.
The softplus width, unit combat-score slope, saturation and numerical floors are
fixed curve shapes, not extra tuning knobs. Aggression is a shared configuration
constant, not an unobserved per-ship or per-episode draw. Randomizing it for BC would
require making it observable or accepting inconsistent labels.

## Differences from the reference equations

- Normalize zone utilities and the enemy bearing to keep force magnitudes bounded.
- Use rational travel cost instead of a Gaussian travel cost to retain nonzero
  attraction to distant objectives; contribution itself remains Gaussian.
- Use smooth, bounded separation instead of an unbounded sum of unit repulsions.
- Recovery blends the navigation forces continuously instead of overriding them.
- Use log probabilities because the existing controller exposes probabilities, not
  raw logits. This is equivalent up to per-head additive logit constants.

These changes keep the intended marginal-contribution semantics and avoid force
magnitude growing simply because a larger fleet is present.

## Execution and limitations

The strategy uses `(B,N,N)` ship geometry and `(B,N,Z)` zone geometry, with two batched
matrix products for zone pressure. There are no ship/environment Python loops, scalar
GPU reads, or episode-reset synchronizations in the strategy. Memory is O(B*N²+B*N*Z);
zone aggregation is O(B*N*Z), summed once per team over five zones. The existing target selection computes
its own pairwise geometry, and team targeting retains its existing fixed two-team loop.
Further geometry reuse is a performance opportunity; no GPU speedup is claimed.

Nearest-target ties retain the legacy dogfighter's first-index tie behavior. Strategy
itself is permutation equivariant up to floating-point reduction order; preserving
legacy dogfighting means its exact equidistant-target tie behavior remains unchanged.

Reproduce finite-vision statistical playtests with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/frontline_scripted_suite.py \
  --device cpu --games 8 --team-size 4 --max-ticks 9000 --output /tmp/frontline-4v4.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/frontline_scripted_suite.py \
  --device cpu --games 4 --team-size 16 --max-ticks 9000 --output /tmp/frontline-16v16.json
```

Use `--device cuda` on a CUDA host. Interactive play remains `.venv/bin/bnb play`.

## Validation results

- Focused final agent/perception suite: 76 passed, including marginal demand,
  aggression, hidden-state privacy, recovery, separation, dead ships, missing masks,
  exact endpoint distributions, interior logit interpolation, and permutations through
  64v64. Both marginal and joint action sampling are covered.
- Direct comparison with the base branch's original controller: byte-identical sampled
  actions and probabilities on 128 randomized 4v4 combat states, both sampling modes,
  with team targeting disabled and enabled.
- Full repository suite: 1,499 passed, 12 skipped, and one invocation-related failure
  in 816.23 seconds. `OMP_NUM_THREADS=1` conflicts with one smoke test's
  environment-inheritance assertion; all 20 smoke tests pass when that command override
  is removed. No controller-related failures occurred. The full run included the BC and
  Frontline PPO training integrations; the final focused run covers the subsequent
  once-per-team aggregation optimization and additional invariant tests.
- Ruff lint, formatting check, and `git diff --check` pass.

Small finite-vision, field-enabled self-play samples, seed 20260908, 300-second matches:

| Fleet | Games | Games with captures | Captures/game | Team 0 / Team 1 / draw | Wall time |
|---|---:|---:|---:|---|---:|
| 4v4 | 8 | 25% | 0.375 | 1 / 1 / 6 | 225.7 s |
| 16v16 | 4 | 75% | 1.0 | 2 / 1 / 1 | 351.5 s |

No spawn- or boundary-attributed deaths occurred in either batch. Larger fleets had
far more combat deaths (615.25/game versus 13.375/game), so the larger sample demonstrates
operability, not fleet-size-independent match dynamics or settled combat balance.
Neither sample reached the front victory threshold. These are small diagnostics,
not statistically strong balance estimates. They were run before the algebraically
equivalent optimization from per-observer sums to once-per-team sums; the final focused
suite and CPU profile were run after that optimization.

Raw results: [4v4](internal/frontline-strategy-4v4.json),
[16v16](internal/frontline-strategy-16v16.json).

The [rendered replay](internal/frontline-strategy-replay.gif) and
[contact sheet](internal/frontline-strategy-contact.png) cover the first 120 seconds of
one 4v4 game. Visual inspection shows spatially differentiated defense, forward fights,
and spawn recovery without roles. Combat persists around the defenses, and this sample
has no capture during the rendered interval. This is an offscreen spectator playtest;
interactive human steering and subjective feel remain untested.

Final CPU-only profile, one Torch thread, batch 32, 5 warmups and 50 measured calls,
excluding visibility, physics, and rendering:

| Fleet | Strategy ms/batch | Whole controller ms/batch |
|---|---:|---:|
| 4v4 | 2.83 | 12.77 |
| 16v16 | 10.61 | 26.62 |
| 64v64 | 108.91 | 163.12 |

[Raw profile](internal/frontline-strategy-cpu-profile.json). Other validation processes
were active, so these are indicative CPU timings, not isolated performance claims.
CUDA was unavailable; GPU throughput and peak allocation remain unmeasured.

## Remaining tuning questions

- The 2026-09-16 head-to-head sweep selected the current defaults and found
  the controller stronger than both the pinned legacy controller and the latest
  available checkpoint in small standard-horizon samples. See the
  [full experiment record](internal/frontline-agent-tuning.md).
- The selected positive aggression is a deterministic team-wide parameter, not a
  hidden identity. It needs more seeds and learned-policy checkpoints before it
  should be treated as a universally optimal strategic bias.
- Weighted objective bearings can balance between zones. Recovery and local combat
  usually perturb that balance, but long-lived stalling needs trajectory-level study.
- Separation applies to strategic navigation only. Close dogfights intentionally retain
  the old controller, including any close-range bunching.
- The rules do not route around damaging fields or explicitly hold outside damaging
  defenses. Recovery handles health loss reactively; sustained environmental attrition
  may warrant geometry-aware navigation in a separate iteration.
- More maps, unequal fleets, human play feedback, GPU profiling, and BC learning curves
  are needed before treating the measured advantage as a settled rating.

# Frontline overhaul progress and plan

Last updated: 2026-09-09

This is the living internal record for the Frontline Conquest Overhaul. Update it when
a gameplay contract changes, evidence changes a recommendation, a milestone advances,
or a new blocker appears. The original root-agent instructions and implementation spec
remain the source of truth for scope and human gates.

## Current position

- Branch: `frontline/02-fields`
- Integration base: `feat/frontline-overhaul`
- Human gate: Gate 2, overlapping field physics
- Status: ready for human Gate 2 review
- Gate 1 was approved by the user's instruction to continue and merged into the
  integration base on 2026-09-09.
- Boundary: stop after overlap physics, on-reset generation, renderer changes, examples,
  and performance evidence are ready for human review.
- Draft PR: not opened because GitHub CLI/credentials are unavailable in this workspace
- Compare URL: <https://github.com/AvesNova/boost-and-broadside/compare/feat/frontline-overhaul...frontline/02-fields?expand=1>

## Gate 1 implemented

- 16384 × 16384 toroidal world with a smaller circular playable region and random map
  translation.
- Centralized camera with fit, reset, pan, zoom, selection follow, and spectator mode.
- Five fixed physical zones with roles derived from an unwrapped integer front.
- Cyclic role order: neutral → Team 0 spawn → Team 0 defense → Team 1 defense → Team 1
  spawn → neutral. The damaging capturable defenses are adjacent.
- Defense-only capture, atomic simultaneous captures, ±5 immediate victory, and a
  five-minute timeout resolved by the sign of the front.
- Immediate same-slot low-health respawn, friendly spawn healing, hostile spawn damage,
  symmetric defense damage, soft-boundary damage, and separate source attribution.
- Frontline outcomes routed through match, evaluation, tournament, rating, reward, and
  artifact paths.
- Respawn preserves recurrent/GAE continuity while teleport-crossing auxiliary targets
  are masked.
- Zone/HUD rendering, selected-ship keyboard control, fully scripted spectator mode,
  and 1×/2×/4×/8× playback speeds.
- GPU-batched scripted-vs-scripted benchmark with first/second capture timing and capture
  duration overrides.

## Current provisional gameplay values

| Parameter | Value | Status |
|---|---:|---|
| Zone ring radius | 1200 px | Provisional |
| Zone radius | 330 px | Provisional; enlarged after playtest |
| Playable radius | 2600 px | Provisional |
| Capture duration | 8 s | Selected from 256-game sweep; still needs human playtest |
| Capture pressure | Sign of ship-count majority | Provisional, intentionally flat |
| Defense damage | 2 health/s | Provisional |
| Respawn health | 25 | Provisional |
| Friendly spawn healing | 12 health/s | Provisional |
| Hostile spawn damage | 8 health/s | Provisional |
| Front win threshold | ±5 | Provisional |
| Match duration | 300 s | Provisional |
| Frontline field count | 10 | Provisional; low-discrepancy placement |
| Frontline field radius | 30–750 px | Provisional; maximum raised from 490 px |
| Frontline physics/decision rate | 30 Hz | Per-second rules unchanged |

Capture and stabilization use the same fixed rate. A larger majority does not accelerate
the meter: 4v0, 4v2, 1v0, and 2v1 are equivalent; ties pause it.

## Current scripted baseline

The scripted controller remains a crude baseline and behavior-cloning warm start, not an
optimal hand-coded strategist.

- Episode identities: 50% offensive, 25% defensive, 25% timid.
- A single enemy occupying a defense makes it contested, even if no defender is present.
- Exactly one contested defense draws both available fleets there. With both or neither
  contested, ships follow their tendencies.
- Nearby combat takes priority over point navigation except for a timid healing retreat.
- Offensive ships gather one-third of the shortest spawn-to-enemy-defense route. They
  advance after every living offensive ship reaches or passes the rally threshold.
  Respawns naturally cause regrouping. Timid offensive followers join but do not count
  toward readiness.
- Uncontested defensive ships orbit 20 px outside the damaging point, alternating orbit
  direction by stable within-team rank. Nearby enemies interrupt patrol.
- Timid ships retreat below 30% health and heal fully; timid respawns always heal fully.
- Offensive/defensive respawns heal fully unless local combat or a contested point calls
  them away.
- Tendencies survive death and are redrawn only at episode reset.

Until team perception exists, the Gate 1 scripted controller reads authoritative state.
It must move to the same perception contract as learned agents during the fog-of-war
milestone.

## Capture-duration evidence

This Gate 1 sweep predates enabled fields and the 30 Hz Frontline runtime. Each setting
used 256 full 4v4 games, the same seed, the current 50/25/25 identities, attacker rally,
defender patrol, and flat-majority capture rule. Treat it as the rationale for selecting
eight seconds, not as current field-enabled match statistics.

| Seconds | Any capture | Two captures | Captures/game | Mean first | Mean second | T0/T1/draw |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.0% | 96.5% | 9.82 | 53.5 s | 74.4 s | 112/104/40 |
| 5 | 93.0% | 81.6% | 4.82 | 102.9 s | 133.0 s | 83/98/75 |
| 6 | 88.3% | 75.4% | 3.56 | 112.9 s | 151.5 s | 94/92/70 |
| 7 | 83.6% | 67.2% | 2.60 | 123.3 s | 165.8 s | 84/87/85 |
| 8 | 77.7% | 52.0% | 1.93 | 132.3 s | 177.8 s | 81/82/93 |
| 9 | 72.7% | 45.7% | 1.55 | 140.0 s | 191.9 s | 76/79/101 |
| 10 | 65.6% | 34.8% | 1.20 | 146.7 s | 200.4 s | 70/72/114 |
| 15 | 43.4% | 10.9% | 0.55 | 176.6 s | 234.1 s | 48/53/155 |
| 20 | 27.0% | 2.3% | 0.29 | 198.9 s | 249.2 s | 34/35/187 |
| 25 | 12.9% | 0.4% | 0.13 | 213.2 s | 277.7 s | 16/16/224 |

First/second times are conditional on a match reaching that capture. Eight seconds was
selected because it produces roughly two captures per match, captures in most matches,
balanced team outcomes, and no ±5 threshold endings in this population. Statistical
results guide tuning but do not replace human playtesting.

## Verification record

- Latest focused gameplay/controller/UI run: 52 passed.
- Defender patrol smoke: 1.89 circuits in 30 seconds, 0% of ticks inside the damage
  radius, full health retained.
- Capture sweep: 2,560 full games across ten capture durations on an RTX 4070 Laptop GPU.
- Gate 1 boundary full repository suite: 1,359 passed, 6 skipped, 0 failed in 397.99
  seconds after the final scripted-policy and 8-second capture changes.
- Gate 2 focused physics/transport/projectile/schema/UI suite: 162 passed, 2 skipped.
- Gate 2 pre-performance-revision full repository suite: 1,358 passed, 6 skipped, 0
  failed in 402.65 seconds. The current lightweight revision ran 129 focused field,
  transport, projectile, Frontline, interactive, evaluation, tournament, and UI tests,
  all passing; the full suite was not repeated during this tuning iteration.
- Gate 2 field benchmark: 4,096 environments on an RTX 4070 Laptop GPU. Zero/four/ten/
  twenty fields measured 1,543,526/223,241/228,889/231,166 environment steps/s. The
  ten-field state used 7.84 MiB, peaked at 39.95 MiB, and reset in 1.538 µs/environment.
  Policy inference was excluded.
- Three-update, 128-environment CUDA PPO spot check without compilation: final cumulative
  throughput was 447 SPS at zero fields, 365 SPS at the production four fields, and 380 SPS
  at ten fields. These deliberately short runs show a roughly 15–18% fielded penalty at
  this underfilled width, not a stable training forecast.
- End-to-end one-environment play benchmark, including scripted decisions, physics, and
  offscreen rendering: 27.54 ms/decision or 1.21× realtime with one CPU thread. The same
  tiny-tensor workload at 16 threads measured 94.12 ms/decision or 0.35× realtime.
- In 4,096 ten-field unit-disk samples, low-discrepancy placement increased mean nearest-
  neighbor spacing from 0.320 to 0.507 radius units and reduced its coefficient of variation
  from 0.566 to 0.111 versus IID area-uniform samples.
- A full 64-game scripted batch was deliberately stopped after roughly six minutes because
  it no longer met the requested lightweight iteration budget; no statistics are claimed
  from the interrupted run.

## Gate 2 implementation

- Fields compose with bounded union coverage and an alpha-weighted target average in
  signed log-index space, using vectorized exclusive products for the analytic gradient.
- Partial, coincident, toroidal, and nested overlaps are legal. Parent relationships,
  parent-relative deltas, laminar packing, cached map banks, retry settings, and their
  training/evaluation plumbing have been removed.
- Layouts generate directly on every masked episode reset. Randomized sunflower/R2
  low-discrepancy samples reduce clustering without rejection loops. Frontline centers use
  the same translated origin as zones and stay wholly inside the practical boundary.
- Field observations expose one absolute target log-index instead of three
  parent-relative channels. Checkpoint observation schema v5 rejects old weights.
- Renderer transition annuli alpha-blend at intersections while each field keeps its
  material color and independent dotted/dashed/solid damage outline.
- Tests cover identical reinforcement, reciprocal cancellation, arbitrary and toroidal
  overlap, finite-difference gradients, on-reset generation, and independent overlap damage.
- Frontline play is a state-only CPU path: one thread avoids tiny-tensor thread-pool
  overhead, one shared scripted controller avoids duplicate analysis, and 30 Hz physics
  avoids computing two 60 Hz ticks per displayed decision. Batched training remains CUDA.
- Deterministic visual fixture:
  [10-field Frontline map, seed 20260909](frontline-field-example-seed-20260909.png),
  reproducible with `benchmarks/render_field_example.py`.

## Gate 2 human review requested

- Play the field-enabled 8-second Frontline preset with `uv run bnb play` or
  `.venv/bin/bnb play`.
- Judge optical feel and whether bending/reflection remains understandable in mixed overlaps.
- Check projectile trajectories through overlaps and reciprocal-cancellation regions.
- Judge the provisional 10-field low-discrepancy density and 30–750 px radius range.
- Check whether overlap bands and independent damage patterns remain legible during combat.

## Known limitations and open questions

- Scripted identities and strategy are not policy observation features. This adds
  multimodal BC labels; keep heuristics simple until we decide whether roles should be
  observable or deterministic.
- Capture statistics are scripted-policy dependent and do not predict learned or human
  balance exactly.
- No capture-time sweep setting above 7 seconds reached ±5 in the sampled population;
  timeout sign currently decides most matches.
- The full vision-mode and minimap requirements are deferred until the perception work
  that can enforce non-leakage correctly.
- Actual draft PR creation remains blocked by unavailable GitHub tooling/credentials.

## Next plan

1. Wait for explicit Gate 2 review and approval before merging or starting perception work.
2. Preserve the later mandatory stops: fog distribution, recursive beliefs, map
   architecture comparison, and curriculum decision.

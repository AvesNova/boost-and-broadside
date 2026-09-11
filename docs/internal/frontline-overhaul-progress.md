# Frontline overhaul progress and plan

Last updated: 2026-09-10

This is the living internal record for the Frontline Conquest Overhaul. Update it when
a gameplay contract changes, evidence changes a recommendation, a milestone advances,
or a new blocker appears. The original root-agent instructions and implementation spec
remain the source of truth for scope and human gates.

## Current position

- Branch: `frontline/07-map-memory`
- Integration base: `feat/frontline-overhaul`
- Human gate: Gate 5, map-memory architecture comparison
- Status: in progress; implementation and preliminary performance evidence are ready, but
  meaningful learning curves and evaluation quality remain outstanding
- Gate 1 was approved by the user's instruction to continue and merged into the
  integration base on 2026-09-09.
- Gate 2 was approved explicitly and merged into the integration base on 2026-09-10.
- Gate 3 was approved by the user's instruction to proceed and merged into the integration
  base on 2026-09-10.
- Gate 4 was approved on 2026-09-10. The point-estimate + age + recurrence design is sufficient
  for now; uncertainty values, categorical targets, and other output representations are
  deferred until a proper training run provides evidence.
- Gate 4 was merged into the integration base and pushed on 2026-09-10 as `331d3d6`.
- Boundary: stop after the full-attention versus K/V-only comparison includes meaningful
  learning curves and final evaluation quality and is ready for human review.
- Draft PR: not opened because GitHub CLI is unavailable; use the compare URL after publication
- Compare URL: <https://github.com/AvesNova/boost-and-broadside/compare/feat/frontline-overhaul...frontline/07-map-memory?expand=1>

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
| Team vision range | 1600 px | Provisional; Gate 3 distribution measured |
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

The scripted controller now receives the same authoritative team-shared visibility mask
as the learned policy. Hidden enemies cannot trigger local combat, point-contested logic,
or targeting, so behavior-cloning labels do not leak privileged state.

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
- Gate 3 focused perception/observation/scripted/model/match/checkpoint/UI suite: 326 passed,
  2 skipped; the final visible-enemy-action privacy change then passed 137 directly affected
  tests. Static checks and whitespace checks pass.
- Shot-reveal/fog-overlay revision: 158 perception/environment/renderer/checkpoint tests
  passed, 2 skipped; the direct overlay suite is 19/19 and includes field-shadow pixels.
- Gate 3 fog suite: 256 independent 4v4 maps for 60 simulated seconds on an RTX 4070 Laptop
  GPU, completed in 109.24 seconds (140.61 aggregate simulated game-seconds/wall-second),
  peaking at 59.40 MiB while computing three visibility ranges every tick.
- Isolated 256-environment CUDA profile: production visibility computes in 4.12 ms/batch
  (62,171 envs/s); visibility plus both masked team observations computes in 15.80 ms/batch
  (16,203 envs/s), excluding bullets. This is far above the recent end-to-end PPO spot-check
  rate, so perception is vectorized and is not currently the training bottleneck.
- The fog-aware one-environment play path measures 32.74 ms/decision or 1.02× realtime at
  900 px with one CPU thread. The quarter-resolution terrain stencil refreshes every eight
  ticks, while ship/shot visibility stays at 30 Hz and camera/view changes refresh immediately.
- Latest Gate 4 compact belief/buffer/PPO/evaluation/checkpoint/config regression: 192 passed,
  2 skipped in 52.76 seconds after the never-seen auxiliary mask correction. After fixing
  Frontline map-token launch sizing, all 96 resolution/VRAM tests passed in 0.74 seconds.
  Earlier broader component runs also passed, but their coverage overlaps these sets. The full
  repository suite was deliberately not repeated in line with the requested lighter cycle.
- Two-update CUDA BC/PPO smoke at 256 environments completed under the Frontline v8 contract at
  811 environment steps/s and 6,488 ship tokens/s; loss decreased from 7.617 to 6.157.
- A 131,072-step, 16-update CUDA BC sample completed in roughly 2.4 minutes. Throughput rose
  from 861 SPS at update 4 to 927 SPS at update 16, while total loss fell from 5.941 to 5.524.
- Gate 4 natural-occlusion suite: 128 independent 4v4 games for 60 seconds on an RTX 4070
  Laptop GPU. The scripted baseline simulated 7,680 aggregate game-seconds in 73.03 seconds
  (105.16× aggregate realtime), peaking at 29.67 MiB. A two-team belief-cache batch measured
  12.61 ms at width 128, or 10,148 environments/s. Learned-checkpoint inference was measured
  separately and took 191.07 seconds.
- Gate 5 config/model/checkpoint/policy-I/O regression: 316 passed, 2 skipped. A matched
  129,024-step CUDA BC integration
  sample completed in 170.1 seconds for full attention and 160.4 seconds for K/V memory, a
  5.7% K/V wall-time reduction. Update-20 throughput was 758 versus 781 environment steps/s;
  losses were 5.383 versus 5.375. This is a pipeline/throughput sample, not a learning result.
- At the production 16 map objects, isolated CUDA sequence forward/backward peak allocation
  fell from 1,435 MiB to 841 MiB with K/V memory (41.4%). In the latest timing pass it improved
  sequence throughput by 8.3%, while inference was 2.7% slower. At 106 map objects, update peak
  fell from 5,606 MiB to 1,479 MiB and K/V update throughput was 2.65x full attention.

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

## Gate 3 implementation

- Finite range and natural field-core line-of-sight use shortest toroidal displacement.
  Sight is shared across living allies; no synthetic occlusion was added. A successful shot
  globally reveals its firing ship for the current state sample, while failed shoot commands
  do not.
- Each environment observation carries independently masked Team 0 and Team 1 views.
  Hidden enemy position, velocity, health, power, cooldown, alive state, local field state,
  bullets, and actions are zeroed behind explicit visibility masks. Enemy pending actions
  remain private even while the enemy ship is visible.
- Static fields remain globally known. Typed entity tokens now cover ships, fields, five
  zones, and one combined boundary/global token with zone roles, capture state, front,
  threshold, timer, and game-mode channels.
- Team canonicalization swaps ship and zone ownership, zone roles, front direction, capture
  direction, and bullet ownership. Finite vision rejects legacy `shared_pass` training;
  `ego_pass` selects the correct masked team view before canonicalization.
- Scripted, policy, match, tournament, Elo, behavior-cloning, and opponent paths all consume
  the same team perception. A regression test perturbs every hidden enemy channel and keeps
  the scripted ally action distribution byte-identical.
- Renderer modes `FULL`, `TEAM_0`, and `TEAM_1` apply the authoritative mask to ships, health
  bars, bullets, prediction ghosts, and the new minimap. Team modes also place a mild gray
  veil over unseen empty space, field-cast shadows, zones, fields, and boundary outlines.
  `V` cycles the perspective; team render modes refuse to draw without an authoritative
  visibility result.
- W&B reports enemy visible/range-only fractions, field occlusion, never-seen fraction,
  individual sight, team-sharing gain, hidden age, reacquisitions, and duration bins.
  Checkpoint observation schema `team_perception_v7` rejects older incompatible encoders.

## Gate 4 implementation

- Every acting policy perspective owns a fixed-shape GPU belief cache. Visible truth refreshes
  it, never-seen enemies stay absent, and a previously seen hidden enemy is advanced recursively
  by that policy's learned next-state head. Reacquisition immediately overwrites the estimate.
- Hidden belief tokens carry explicit validity and age. Predicted health may describe believed
  alive state, but token existence is separate, so a low-health forecast cannot accidentally
  erase memory. Hidden enemy actions and local field gradients remain zero.
- Team 0 and Team 1 caches are independent. Frozen league policies, the running-average policy,
  Elo ladder policies, and ordinary checkpoint match agents each advance their own prediction;
  none reuse the live policy's belief or hidden truth.
- Authoritative physical targets live in a separate rollout tensor used only for auxiliary
  supervision and diagnostics. Visible and previously seen hidden ships are supervised;
  never-seen slots, terminal transitions, and death-to-respawn teleports are masked.
- W&B now reports visible/hidden physical error for position, velocity, attitude, angular
  velocity, health, power, cooldown, and local refractive index, with hidden errors bucketed at
  0.1, 0.5, 1, 2, 5, 10, and 30 seconds.
- The registered RL and BC profiles now actually use the Frontline environment: 16384² world,
  30 Hz, 4v4, ten fields, five zones plus global token, 1600 px team sight, eight-second capture,
  and ego-only perception. `front_advance_weight=0.25` is provisional. This fixed a stale profile
  integration that would otherwise have trained the old small omniscient deathmatch.
- Checkpoint observation schema `recursive_belief_v8` rejects encoders trained before the new
  visibility/validity/age semantics.

## Gate 5 implementation (in progress)

- `ModelConfig.map_read_mode` selects `full_attention` or `kv_memory`; command-line overrides
  accept and validate both literal values, so matched training and sweeps need no code edits.
- Full attention preserves the existing path. K/V mode encodes map inputs once, projects them
  to a 64-wide latent, and lets ship queries cross-attend to layer-specific map keys/values.
  Map objects never query ships, enter the feed-forward or recurrent trunk, or receive heads.
- The K/V bank is recomputed from the current observation on every rollout or update forward;
  nothing derived from model weights is cached across optimizer steps.
- Normal registered training now sets projectile cross-attention layers to zero. Projectile
  physics and the reusable bullet encoder/K/V infrastructure remain available through model
  configuration and checkpoint metadata.
- The K/V model has 1.950M parameters versus 1.909M for full attention: its small map projection
  and per-layer K/V projections add weights, while unused field temporal adapters are omitted.
- Tests pin query/memory dimensions, map influence on ship logits, rollout/update agreement,
  literal override validation, and the existing full-attention behavior.

Raw preliminary evidence:

- [`frontline-map-memory-gate5-throughput.json`](frontline-map-memory-gate5-throughput.json)
- [`frontline-map-memory-gate5-training.json`](frontline-map-memory-gate5-training.json)

Reproduce the isolated scaling comparison with
[`benchmarks/map_memory_suite.py`](../../benchmarks/map_memory_suite.py). The short training
pair establishes that both paths optimize end to end and estimates operational throughput;
it is explicitly not the Gate 5 learning-curve or final-policy comparison.

## Belief-model evidence

The offline comparison used natural 4v4 occlusions and an early 131,072-step BC checkpoint.
For the learned-policy trajectory, its recursive head and a last-observation plus constant-
velocity baseline saw exactly the same frames. Errors include the operational ambiguity after
an enemy respawns unseen: the teleport label itself is excluded, but subsequent estimates remain
wrong until evidence or learned dynamics corrects them.

| Status/age | Samples | Learned position | Baseline position |
|---|---:|---:|---:|
| Visible | 688,439 | 3.4 px | 0.1 px |
| Hidden, all | 119,287 | 557 px | 629 px |
| Hidden ≤0.1 s | 2,470 | 209 px | 201 px |
| Hidden 0.1–0.5 s | 13,060 | 263 px | 238 px |
| Hidden 0.5–1 s | 12,171 | 332 px | 282 px |
| Hidden 1–2 s | 19,514 | 416 px | 349 px |
| Hidden 2–5 s | 37,239 | 607 px | 566 px |
| Hidden 5–10 s | 27,432 | 808 px | 1,023 px |
| Hidden 10–30 s | 7,372 | 759 px | 1,629 px |

Across all hidden samples, learned/baseline mean errors were 103.5/105.8 px/s velocity,
26.6/25.5 health, 19.8/19.0 power, and 0.0045/0.0053 seconds cooldown. The minimally trained
head is worse than the kinematic baseline through roughly five seconds, but substantially
better on long-hidden position and modestly better on aggregate position and cooldown. Its
3.4 px visible one-step position error versus the baseline's 0.1 px shows it is still early in
training, not a converged model comparison.

Interpretation: the simple recursive point estimate is implemented correctly and already learns
some useful long-horizon correction, but a 131k-step warm-up is not enough evidence to select a
more complex output distribution. The immediate large error is consistent with lifecycle
ambiguity: an unseen same-tick respawn can relocate a ship, while ordinary one-frame motion
cannot explain a roughly 200 px mean by itself. This is an inference from the game contract and
error scale; the benchmark does not yet split ordinary occlusion from post-respawn occlusion.
Age exposes staleness to the policy, while recurrence can learn when to discount it. Keep the
simple representation for the first serious BC curve; decide on concentration or explicit
uncertainty only if downstream policy quality or a trained head still degrades sharply with age.

Raw output: [`frontline-belief-gate4.json`](frontline-belief-gate4.json). Reproduce with
[`benchmarks/frontline_belief_suite.py`](../../benchmarks/frontline_belief_suite.py).

## Fog-distribution evidence

The suite used 256 independent field layouts and scripted 4v4 trajectories controlled with
the production 1600 px range. The 1200 and 2000 px columns are counterfactual geometry probes
over those same trajectories, so range comparisons do not confound map or combat randomness.

| Vision range | Enemy visible | Range-only visible | In-range blocked by fields | Individual visible | Team-sharing gain | Mean hidden age |
|---:|---:|---:|---:|---:|---:|---:|
| 1200 px | 57.7% | 61.6% | 6.4% | 36.5% | +21.2 pp | 5.16 s |
| **1600 px** | **74.2%** | **83.7%** | **11.6%** | **53.0%** | **+21.2 pp** | **3.66 s** |
| 2000 px | 81.3% | 95.7% | 15.1% | 60.9% | +20.4 pp | 2.85 s |

At 1600 px, per-map enemy visibility ranged from 59.5% at p10 to 87.1% at p90. All enemy
slots were seen at least once within 60 seconds, with 22.9 reacquisitions/game and 91.8% of
started hidden runs completing inside the horizon. Completed occlusions were broadly useful:
455 lasted at most 0.1 s, 924 fell in 0.1–0.5 s, 735 in 0.5–1 s, 813 in 1–2 s, 1,207 in
2–5 s, 1,176 in 5–10 s, and 544 in 10–30 s.

Interpretation: team sharing is substantial rather than cosmetic (+21.2 percentage points),
and fields materially interrupt otherwise valid sight without dominating it (11.6% of
in-range exposure). The 1600 px setting is permissive—roughly three quarters of enemy-slot
time is visible—but still produces multi-second uncertainty and strong map variance. Keep it
for the first belief-model experiments; 1200 px would create heavier fog but would mix the
recursive-belief comparison with a major observation-distribution change.

Raw outputs:

- [`frontline-fog-gate3.json`](frontline-fog-gate3.json)
- [`frontline-perception-throughput.json`](frontline-perception-throughput.json)

Reproduce with [`benchmarks/frontline_fog_suite.py`](../../benchmarks/frontline_fog_suite.py).

## Gate 2 human review (approved)

- Play the field-enabled 8-second Frontline preset with `uv run bnb play` or
  `.venv/bin/bnb play`.
- Judge optical feel and whether bending/reflection remains understandable in mixed overlaps.
- Check projectile trajectories through overlaps and reciprocal-cancellation regions.
- Judge the provisional 10-field low-discrepancy density and 30–750 px radius range.
- Check whether overlap bands and independent damage patterns remain legible during combat.

## Gate 3 human review (approved)

- Play or spectate with `V` cycling full/Team 0/Team 1 and confirm the information density
  feels right at the provisional 1600 px range.
- Check field-core occlusion at overlaps, map seams, and while a ship is inside a field.
- Confirm the minimap, health bars, bullets, and prediction ghosts never reveal hidden ships.
- The user approved proceeding with the 1600 px range for initial recursive-belief work.

## Gate 4 human review (approved)

- Keep point estimate + age + recurrence as the initial belief representation.
- Do not interpret the 131k-step smoke checkpoint as a final model-quality comparison.
- Revisit uncertainty values, categorical targets, multi-scale/concentration outputs,
  lifecycle handling, and latent-only alternatives after a proper training run.

## Known limitations and open questions

- Scripted identities and strategy are not policy observation features. This adds
  multimodal BC labels; keep heuristics simple until we decide whether roles should be
  observable or deterministic.
- Capture statistics are scripted-policy dependent and do not predict learned or human
  balance exactly.
- No capture-time sweep setting above 7 seconds reached ±5 in the sampled population;
  timeout sign currently decides most matches.
- Beliefs are deterministic point estimates with age, not calibrated distributions. An unseen
  respawn can make a fresh-looking estimate abruptly wrong; the acting team may not know that
  the lifecycle discontinuity occurred.
- The 131k-step learned checkpoint is an early pipeline/learning smoke, not a converged quality
  result. It should not decide the final uncertainty representation by itself.
- The combined boundary/global token is the first architecture, not the Gate 5 map-memory
  comparison. Static fields remain globally visible by design.
- GitHub CLI is unavailable, so draft PR creation must use the compare URL or GitHub UI.
- Gate 5 timing varies with GPU compilation/power state; the memory deltas were stable across
  repeats, while the matched end-to-end sample is the conservative throughput estimate.

## Next plan

1. Run a meaningfully long, matched BC learning comparison for full attention and K/V memory,
   then evaluate both checkpoints against the same scripted/reference populations.
2. Select the default architecture only after throughput, VRAM, learning curve, and final
   evaluation evidence can be considered together; then stop for Gate 5 human review.
3. Preserve the later mandatory curriculum stop.

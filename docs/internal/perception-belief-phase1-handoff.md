# Phase 1 handoff — one-time physical delta calibration

Phase 1 of [`perception-belief-overarching-plan.md`](perception-belief-overarching-plan.md).
September 27, 2026.

**Outcome: COMPLETE. The Phase 1 exit gate is fully satisfied.**

## Scope and history

| | |
|---|---|
| branch | `fix/seat-symmetry-pending-action` |
| starting commit | `1dfc54a` (`docs: finish Phase 0 evaluation handoff`) |
| implementation commits | `035b138` (`feat: define physical delta calibration`), `233e61b` (`feat: select fixed physical delta scales`) |
| handoff commit | the commit containing this document |
| recommended next block | Phase 2 only |

The phase measured authoritative truth-to-truth changes over one production
decision. It did not change the current next-state head or belief architecture;
those remain Phase 3 work. It did make the selected scales available as one
source of truth for that work and replaced one Phase-0 diagnostic's invalid
`max_speed` assumption with the measured position scale.

## What was implemented

- `train/rl/physical_deltas.py` defines the ordered 11-channel physical delta,
  fixed scales, exact-zero normalization, toroidal position difference, wrapped
  attitude difference, and natural-log refractive-index difference.
- `benchmarks/physical_delta_calibration.py` runs the production `rl` profile
  under `StochasticScriptedAgent` on both teams, collects pre/post-decision
  truth, excludes destinations marked `ship_respawned`, reports all-sample and
  conditional-nonzero distributions, and records runtime/configuration metadata.
- `perception-belief-phase1-calibration.json` is the complete reproducible CUDA
  result, including the full configs, distributions, selected constants, and
  runtime context.
- The belief position-outlier diagnostic now uses ten calibrated position
  scales (25 px), not `ShipConfig.max_speed`. `max_speed` is a spawn parameter,
  not a physical speed cap. Its comment now correctly says that an untrained
  head measures model error, not necessarily a lifecycle leak.
- Focused tests pin physical units and ordering, toroidal and angular wrapping,
  snapshot timing, exact-zero normalization, positive scales, and required x/y
  symmetry.

## Collection conditions and result

Exact command:

```text
.venv/bin/python benchmarks/physical_delta_calibration.py \
  --device cuda --envs 64 --steps 3000 \
  --seeds 271828,314159,161803 \
  --out docs/internal/perception-belief-phase1-calibration.json
```

This was 192 independent 5v5 Frontline trajectories on the exact `rl` profile:
30 decisions/s, 10 fields, 1024 px team sight, zone occlusion, spawn reveal,
scripted-v-scripted control, and the normal delayed pending-action scheduler.
It collected 5,760,000 candidate ship transitions, retained 5,753,703, and
excluded 6,297 transitions whose destination was a respawn (0.1093%). There
were 6,270 environment-decisions containing at least one respawn. No episode
reached the 9,000-tick limit or Frontline terminal condition within a 3,000-step
seed trajectory, so no environment reset entered the sample.

Hardware/runtime: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), Python 3.13.11,
torch 2.13.0+cu130, CUDA runtime 13.0. Wall time was 444.10 s, 12,970 candidate
ship transitions/s; peak allocated/reserved CUDA memory was 13,962,752 /
29,360,128 bytes. This is an offline calibration rate, not end-to-end PPO SPS.

## Distributions and selected constants

The scale is the divisor: `normalized_delta = physical_delta / scale`. There is
no centering or moving state, so physical zero maps to bit-exact zero.

| physical channel | signed std | zero frac | abs p99 | abs p99.9 | max | scale |
|---|---:|---:|---:|---:|---:|---:|
| position x (px) | 2.4743 | 0.0005 | 6.2031 | 8.0430 | 11.3203 | **2.5** |
| position y (px) | 2.5011 | 0.0005 | 6.4531 | 8.1016 | 11.2070 | **2.5** |
| velocity x (px/s) | 4.0907 | 0.0000 | 17.9016 | 32.5793 | 86.9939 | **4.0** |
| velocity y (px/s) | 4.0307 | 0.0000 | 17.2032 | 32.1222 | 78.8555 | **4.0** |
| attitude (rad) | 0.0847 | 0.0001 | 0.2618 | 0.4821 | 1.8317 | **0.1** |
| angular velocity (rad/s) | 2.7039 | 0.6925 | 7.8540 | 15.7080 | 15.7080 | **7.8540** |
| shield delay (s) | 0.2474 | 0.4639 | 0.1000 | 5.0000 | 5.0000 | **5.0** |
| health | 1.0747 | 0.7086 | 6.3389 | 9.9999 | 28.5378 | **10.0** |
| power | 0.7432 | 0.1488 | 2.3745 | 3.2095 | 5.5619 | **0.75** |
| cooldown (s) | 0.0209 | 0.8523 | 0.1000 | 0.1000 | 0.1000 | **0.1** |
| natural log-index | 0.0105 | 0.9158 | 0.0546 | 0.1296 | 0.3321 | **0.05** |

Dense continuous channels use rounded signed RMS conditioning. Position uses
one pooled rounded scale despite its small x/y distribution difference;
velocity does the same. Attitude rounds its 0.0847 rad RMS to 0.1 rad.

Sparse/event-like channels deliberately do not use their all-sample RMS:

- angular-velocity changes are zero 69.2% of the time and quantized by turn
  commands; 2.5π rad/s is its conditional-nonzero p90;
- shield delay's meaningful event is its exact 5 s damage reset;
- health uses the nominal 10-health projectile event, also its nonzero p99;
  the 28.54 tail represents compound damage, not a typical unit event;
- cooldown uses its exact 0.1 s firing reset;
- local log-index is zero away from field interfaces 91.6% of the time, so 0.05
  is a rounded representative crossing change (conditional-nonzero p90 0.0599),
  rather than the zero-dominated 0.0105 all-sample standard deviation.

The artifact contains p50/p90/p95/p99/p99.9/p99.99, mean, and maximum for both
all absolute samples and conditional-nonzero absolute samples, plus signed
mean/std and sign frequencies for every channel.

## Implementation decisions

- `local_log_index` means `log(physical local_index)` in natural-log units.
  It is not the current observation's division by `2*log(field_index_step)`;
  the belief is meant to store physical state before normal input encoding.
- Pre-step snapshots alias only the needed state tensors. This is safe because
  `TensorState` physics advances by tensor reassignment, and a regression test
  pins that assumption. It avoids cloning bullets and map state per decision.
- Respawn exclusion uses the destination state's authoritative
  `ship_respawned`, exactly matching `transition_contiguous` semantics. The
  excluded 0.1093% empirically demonstrates that the filter was active.
- Position uses minimum-image toroidal displacement. Attitude uses
  `angle(next * conj(current))`, producing the signed shortest rotation.
- The calibration function is pure tensor work and CUDA-compilable. A manual
  Inductor comparison matched eager to max absolute error 2.38e-7 over 440
  values. PyTorch warned that complex operators lack specialized Inductor
  codegen; Phase 3 should measure the integrated compiled path rather than infer
  speed from this correctness probe.

## Materially changed components

- `src/boost_and_broadside/train/rl/physical_deltas.py` (new)
- `benchmarks/physical_delta_calibration.py` (new)
- `tests/train/test_physical_deltas.py` (new)
- `src/boost_and_broadside/train/rl/ppo.py`
- `docs/internal/perception-belief-phase1-calibration.json` (new)
- this handoff

## Validation

Development checks:

| command | exact result |
|---|---|
| focused physical-delta tests after definition | 3 passed in 1.36 s |
| CUDA smoke calibration (`2 envs × 3 steps`, seed 7) | completed; 60/60 transitions retained |
| physical deltas + belief subsystem + belief alignment | 17 passed in 29.42 s |
| Ruff on all changed Python files | clean |
| `git diff --check` | clean |
| compiled CUDA vs eager delta probe | agreement within atol 2.38e-7; 440 values |

Final broad validation:

| command | exact result |
|---|---|
| `pytest -q -p no:randomly` | **1,723 passed**, 6 warnings, 539.36 s |
| `bnb smoke` | **all 16 isolated cases passed**; checkout unchanged |

## Compatibility, caveats, and deferred work

- No model input, model head, checkpoint tensor, or checkpoint schema changed.
  Existing checkpoints remain compatible. The constants are dormant until the
  Phase 3 physical-delta head consumes them; only the diagnostic threshold uses
  one now.
- The calibration is intentionally the one-time scripted 5v5 production
  calibration required by the plan. It is not evidence that distributions are
  invariant under a learned policy, alternate physics, another decision rate,
  or future game-rule changes. Re-run the committed tool if those change; do
  not introduce an online scaler.
- Velocity and attitude maxima are genuine collision/dynamics tails after
  respawn removal. The chosen scales condition the bulk; they are not clamps.
- The outlier diagnostic threshold moved from 60 px under the current 30 Hz
  profile (`10 * max_speed * dt`) to 25 px. The sample's per-axis maxima imply a
  worst possible norm below 16 px, leaving >1.5× empirical norm headroom. It is
  still only interpretable as a leak rate after visible-cell model error is
  small relative to 25 px.
- No end-to-end SPS or PPO VRAM benchmark was repeated: this phase adds an
  offline tool and constants, not rollout/model computation. Phase 3 must
  benchmark the integrated model against Phase 0's baseline.
- Phase 2 pending-action representation/head work, and every Phase 3 physical
  belief/head/composition change, remain intentionally untouched.

Known bugs/blockers: none for Phase 1. The repository-wide pre-existing Ruff
debt described by Phase 0 was not changed; changed files are clean.

## Exact prerequisites for Phase 2

Start from the handoff commit and read the overarching plan plus the Phase 0
evaluation's carried items. In particular, Phase 2 must implement the 42-way
joint pending-action view and reveal/respawn null-action override, preserve
physical ship indexing across canonical views, test t/t+1 alignment, and measure
gradient interaction and throughput. It does not need to consume the physical
delta scales; they are committed now for Phase 3.

Do not begin Phase 3 physical belief work inside Phase 2.

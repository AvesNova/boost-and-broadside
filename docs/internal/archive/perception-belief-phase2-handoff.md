# Phase 2 handoff — joint pending-action belief

Phase 2 of [`perception-belief-overarching-plan.md`](perception-belief-overarching-plan.md).
September 28, 2026.

**Outcome: COMPLETE. The Phase 2 exit gate is satisfied.**

## Scope and history

| | |
|---|---|
| branch | `fix/seat-symmetry-pending-action` |
| starting commit | `fb8242f` (`docs: complete Phase 1 calibration handoff`) |
| last code/docs commit before this handoff | `cf3a95c` (`docs: define joint pending-action semantics`) |
| handoff commit | the commit containing this document |
| completed block | Phase 2 only |
| recommended next block | Phase 3 only |

Ordered Phase 2 commits:

| commit | established |
|---|---|
| `9a6326b` | The 42-way pending-action observation, dedicated prediction head, belief storage/composition, privileged cross-entropy objective, metrics, gradient diagnostics, call-site integration, schema v18, and focused regression coverage. |
| `b1f7202` | Explicit CUDA eager/compiled parity coverage for enemy-action logits. |
| `0ef7589` | The belief compile-parity harness now drives and compares action-belief logits/state, not only physical belief. |
| `3e598ba` | A clean-revision CUDA parity artifact for both team views. |
| `f041c59` | A measured experiment made the enemy head linear and guarded unrequested execution. The linear architecture was later rejected; only the useful no-request guard and correct compile warmup remain. |
| `86dbda6` | Recorded the linear-head profile. This artifact was deliberately superseded after restoring actor/head symmetry. |
| `ccd2c35` | Restored the enemy head to exactly the actor head's `D -> 2D -> 42` layer shape and fixed a match recorder to forward the same-pass prediction flag. |
| `ce766d0` | Replaced the superseded profile with the already completed symmetric-head CUDA result. |
| `cf3a95c` | Updated the public architecture, environment, runtime, training, and schema documentation. |

The intermediate linear-head commits remain in history because they record the measured
engineering decision. The final tree uses two separate, identically shaped heads:
`Linear(D, 2D) -> RMSNorm -> GELU -> Linear(2D, 42)`.

## What changed

- `previous_action` remains the serialized key, but its policy-facing value is now one
  42-float joint-command vector per entity token instead of three categorical indices.
- Own/allied ship commands are exact one-hot vectors. Initial spawn and respawn expose
  the exact null-command one-hot to both teams for that reveal decision.
- Ordinary enemy slots are zero in the raw legal view regardless of physical visibility.
  `BeliefTracker.compose` fills them only from its stored 42-way policy prediction. No
  authoritative enemy pending action enters policy input.
- Each team owns an independent `BeliefTracker` with an explicit observer team. Physical
  ship slots are never reordered; Team 1 canonicalization relabels perspective while
  retaining physical indices.
- The dedicated enemy-action head has the same architecture as the ally actor head but
  wholly disjoint parameters. It is returned from the same trunk forward, so Phase 2 adds
  no second policy pass.
- At decision `t`, the policy predicts enemy commands and selects commands. After commit,
  those logits are stored in belief. Their softmax distribution is visible in observation
  `t+1`; the supervised target is the actual joint command committed at decision `t`.
- PPO cross-entropy is masked to canonical enemy ships that are alive and have
  `decision_committed=true`. Human overrides and other uncommitted decisions do not become
  false labels.
- The action-persistence baseline uses the actual previous committed action over contiguous
  transitions, with its own denominator. It does not use the model's previous prediction.
- Rollout, recurrent sequence re-evaluation, Elo, interactive play, evaluation, and
  benchmark call sites consume the same-pass logits.
- The rollout buffer stores the float distribution in bfloat16 rather than treating the
  channel as a small categorical integer tensor.
- Checkpoint observation schema is now `joint_pending_belief_v18`.

## Implementation decisions

### Legal-view ownership

The environment/action scheduler owns only exact public facts: allied queued commands and
the spawn/respawn null override. It writes zeros for ordinary enemies. The belief layer is
the sole owner of predicted enemy values. This leaves one auditable boundary between legal
observation construction and model memory and prevents physical visibility from becoming
an action-information side channel.

### Temporal contract

The prediction head reads observation `O_t`. Its logits predict the command selected and
committed as `D_t`. `BeliefTracker.advance` stores them after selection, and the next
compose writes their distribution into `O_(t+1)`. The training label is therefore `D_t`,
not the command that was pending in `O_t` and not `D_(t+1)`. Focused tests pin this contract
through real scheduler transitions.

### Symmetric but independent heads

The final enemy and actor heads have identical layer shape and initialization policy but do
not share parameters. A temporary `D -> 42` classifier saved 12.5 MiB peak allocation in
one matched profile but did not improve SPS; it was reverted in favor of architectural
symmetry. Conditional execution remains for callers that do not request enemy logits, and
the CUDA compile warmup now warms the actual requested production signature.

### Baselines and diagnostics

New training metrics are:

- `loss/enemy_action` and `loss_proxy/enemy_action`;
- realized-action probability, entropy, multiclass Brier score, and top-1 accuracy;
- action-persistence accuracy and smoothed persistence cross-entropy;
- analytic uniform cross-entropy, probability, accuracy, and entropy;
- whole-model and shared-trunk enemy-action gradient norms;
- shared-trunk policy-versus-enemy-action gradient cosine.

Uniform values are computed from the authoritative 42-command count. Persistence compares
actual consecutive committed actions and excludes discontinuities. PPO smoke coverage
verifies that the dedicated head's parameters update. Metric-key tests verify emission
through the normal update path; gradient tests verify finite norms and bounded cosines.

## Materially changed components

- observation/action representation: `constants.py`, `runtime/actions.py`,
  `env/observation.py`, `env/wrapper.py`, and `train/rl/features.py`;
- policy/belief: `models/yemong/policy.py`, `train/rl/belief.py`, and policy-I/O call paths;
- training: `train/rl/ppo.py`, `train/rl/opponents.py`, `train/rl/elo_eval.py`, config
  schema/defaults, and the production RL profile;
- evaluation/modes: agent dispatch, match runner, interactive mode, and next-state
  evaluation;
- six benchmark entry points plus the compile-parity and end-to-end profile artifacts;
- public architecture, environment, runtime-semantics, and training documentation;
- focused environment, model, runtime, mode, buffer, policy-I/O, PPO, belief, and gradient
  regression tests.

The complete Phase 2 diff from `fb8242f` is 46 files, 1,229 insertions, and 378 deletions
before this handoff.

## Validation during development

Representative incremental checks, in execution order:

| check | exact result |
|---|---|
| perception/environment/spawn suite | 86 passed in 8.08 s |
| seat-symmetry suite | 7 passed in 8.50 s |
| belief/copy/uncertainty suite | 32 passed |
| configuration suite | 157 passed in 0.85 s |
| runtime production/contract suite | 25 passed in 1.58 s |
| interactive/agent-factory suite | 19 passed in 19.73 s |
| dedicated-head model tests | 2 passed in 1.99 s |
| CPU ego-pass PPO smoke | 1 passed in 99.67 s |
| CPU shared-pass plus metric coverage | 2 passed in 55.60 s |
| top-level gradient diagnostics | 1 passed in 34.91 s |
| final head architecture plus match-mode regression | 10 passed in 2.17 s |
| final config/runtime contract checkpoint | 174 passed in 0.30 s |

One combined 467-test checkpoint initially produced 466 passes and one failure because the
runtime architecture allowlist did not include `belief.py` as the intentional raw pending
action owner. After updating the allowlist, its focused rerun passed 2/2. A later broad run
found two match recorder failures because the test double did not forward
`return_enemy_action`; after that contract fix, all 8 match tests passed and the final full
suite was clean.

## Final validation

| command | exact result |
|---|---|
| `.venv/bin/pytest -q -p no:randomly` | **1,729 passed**, 7 warnings, 975.40 s |
| `.venv/bin/bnb smoke` | **all 16 isolated cases passed; checkout unchanged** |
| `.venv/bin/python -m compileall -q src benchmarks tests` | passed, no output |
| Ruff over every Phase 2 changed Python file | **All checks passed** |
| `git diff --check` | passed |

Repository-wide `.venv/bin/ruff check .` still reports 14 pre-existing violations, all in
`docs/internal/next-state-head-audit-sep2026-probe.py` and
`docs/internal/seat-symmetry-bug-sep2026-experiment.py`. Phase 2 did not modify either file.

## CUDA and compiled/eager validation

Hardware/runtime for Phase 2 CUDA checks:

- NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB reported by `nvidia-smi`;
- driver 595.84;
- Python 3.13.11;
- torch 2.13.0+cu130, CUDA runtime 13.0, compute capability 8.9.

| CUDA check | exact result |
|---|---|
| existing second-shape compile and CUDA-graph output-lifetime tests | 2 passed in 67.92 s |
| explicit enemy-action eager/compiled logit comparison | 1 passed in 13.38 s |
| real host-backed CUDA PPO update plus numerical precision suite | 4 passed, 1 warning in 254.40 s |
| `belief_compile_parity.py --device cuda --seed 271828 --steps 12` with action logits/state | `parity_passed=true`, `error_count=0`, both team views, retained-output ownership passed |

The committed parity artifact is
[`perception-belief-phase2-compile-parity.json`](perception-belief-phase2-compile-parity.json).
It was recorded from clean revision `0ef7589`, uses `torch.compile(mode="default",
dynamic=False)`, and compares float state at `atol=rtol=2e-6` plus exact discrete state.

A separate exploratory 24-step run before the harness was extended to pass action logits
found one Team-1 decoded `data.pos` mismatch at step 20 (`max_abs=0.0078125`) while final
discrete state and predicted-target hashes still agreed. The exact baseline-comparable
12-step command passes after action-belief state was added to the comparison. This is a
long-horizon compiled physical-decode numerical caveat, not an action-belief mismatch; it
is not silently claimed as a 24-step pass.

## End-to-end performance

Exact final command:

```text
.venv/bin/python benchmarks/rl_pipeline_profile.py   --profile rl --timing wall --updates 2 --warmup 1 --no-checkpoint   --out docs/internal/perception-belief-phase2-benchmark.json   --label phase2-rl-wall
```

The committed result is
[`perception-belief-phase2-benchmark.json`](perception-belief-phase2-benchmark.json).
It ran the production 960-environment, 128-step, four-rollout logical update on the GPU
above. The two measured updates completed 4 and 2 PPO epochs respectively.

| metric | Phase 0 `rl-wall` | Phase 2 symmetric head | change |
|---|---:|---:|---:|
| SPS | 2,809.6 | **2,087.0** | **-25.7%** |
| seconds/update | 174.95 | **235.51** | **+34.6%** |
| rollout seconds/update | 103.84 | **140.15** | **+35.0%** |
| update seconds/epoch | 21.39 | **28.82** | **+34.7%** |
| peak allocated | 3,461.3 MiB | **3,766.0 MiB** | **+304.7 MiB / +8.8%** |
| peak reserved | 4,656.0 MiB | **5,052.0 MiB** | **+396.0 MiB / +8.5%** |
| perception/belief phases | 24.05 s/update | **32.06 s/update** | **+33.3%** |

The symmetric-head measurement was made at `3e598ba`, before the later no-request execution
guard and compile-warmup correction. Every measured production hot path requests the head,
so its architecture and measured work are the same as the final tree. A matched linear-head
experiment measured 2,065.9 SPS and 3,753.5 MiB allocated, showing that replacing the
required symmetric head did not recover end-to-end throughput. The final tree and committed
artifact therefore use the requested symmetric architecture and its real measured cost.

The profiler unexpectedly wrote an asynchronous best checkpoint despite
`--no-checkpoint`; the two profile-created directories were moved to `/tmp` rather than left
as repository artifacts. TorchInductor also emitted its known complex-operator codegen
warning and reported too few SMs for max-autotune GEMM on this laptop GPU.

## Compatibility and schema implications

- `joint_pending_belief_v18` is intentionally checkpoint-incompatible with v17 and older:
  the observation width changed, encoder weights changed shape, and a new prediction head
  was added. There is no faithful weight-only migration; retraining is required.
- New checkpoints contain enemy-head weights and optimizer moments. The checkpoint schema
  explicitly describes dedicated prediction plus spawn-null override semantics.
- The rollout observation buffer now stores `previous_action` as bfloat16 probability data,
  not uint8 categorical indices. Old cached rollout tensors are not compatible.
- Runtime physical commands remain the compact three-index tuple; only the policy
  observation is 42-way. No probability tensor reaches physics.
- Evaluation and league policies must be v18-compatible when finite-vision beliefs request
  enemy logits. Generic test wrappers must forward policy keyword arguments.

## Exit gate

| requirement | result |
|---|---|
| No action-information leakage | **met** — raw ordinary enemy slots are zero; focused tests plant distinct authoritative actions and verify policy inputs cannot recover them |
| Canonical indexing | **met** — Team 0/Team 1 tests retain physical ship slots while relabeling perspective |
| Spawn override | **met** — initial spawn and respawn are exact null one-hot for both teams, including held-action transitions |
| Temporal alignment | **met** — decision `t` prediction/commit and observation `t+1` use are explicitly tested |
| Auxiliary task versus trivial baselines | **met** — uniform and causal persistence baselines are emitted, the supervised head updates in PPO smoke tests, and probability/entropy/Brier/accuracy diagnostics are emitted |
| Recurrent rollout/re-evaluation agreement | **met** — dedicated logits agree across step rollout and full causal sequence evaluation |
| Compiled/eager agreement | **met for the baseline-comparable 12-step/action-head checks** — the 24-step physical decode caveat is recorded above |
| Throughput cost measured | **met** — exact end-to-end SPS, phase time, and VRAM comparison recorded above |

Known Phase 2 bugs/blockers: none. The throughput regression is a material measured cost,
not an unresolved correctness bug. The 24-step strict-tolerance physical decode drift and
repository-wide historical Ruff debt are known caveats.

## Intentionally deferred

Phase 2 does not implement any Phase 3 physical-belief representation, Gaussian next-state
head, direct legal-view composition rewrite, or Phase 4 global/game-token and density work.
The existing point-estimate physical belief and next-state head remain in place solely until
their assigned phase replaces them.

## Exact prerequisites for Phase 3

Start from the handoff commit and read this document, the overarching plan, and Phase 1's
calibration handoff/artifact. Preserve these Phase 2 invariants:

1. `previous_action` is exactly 42 floats in policy observations.
2. Raw ordinary enemy slots are zero; only belief composition writes predicted values.
3. Spawn/respawn is exact public null one-hot for both teams.
4. Team canonicalization never reorders physical ship slots.
5. Prediction from decision `t` is the belief consumed in observation `t+1`.
6. The enemy-action head remains separate from, but identically shaped to, the actor head.
7. Do not add an extra policy forward for any belief output.

Proceed with Phase 3 only. Use the calibrated physical delta definitions/scales from Phase 1
and carry the measured Phase 2 SPS/VRAM regression into Phase 3's end-to-end performance
comparison.

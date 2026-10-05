# Decision runtime semantics

Status: **Canonical runtime contract.** The Phase 3 implementation is exercised by the
reference and production conformance suites in `tests/runtime_semantics/`. Recurrent
identity across respawn and causal `decision_committed` PPO masking are approved runtime
semantics, not provisional proposals.

Phase 5 validation is complete. The full migration audit and test record are in
[`engineering/runtime-phase5-validation.md`](engineering/runtime-phase5-validation.md).

The purpose of the decision runtime is to make a game transition mean the same thing in
training, behavior cloning (BC), evaluation, Elo, interactive play, watch mode, and
diagnostics. High-throughput and low-latency executors may schedule work differently,
but they must implement the same observable semantics.

## Vocabulary and ownership

For decision boundary `t`:

| Symbol | Meaning | Owner |
| --- | --- | --- |
| `S_t` | Physical simulator state | Environment |
| `Q_t` | Pending actuator command | Runtime |
| `O_t^k` | Policy observation for team/perspective `k` | Observation composer |
| `C_t^k` | Controller state, including recurrent state | Controller session |
| `D_t` | Newly selected autonomous decision | Controller |
| `U_t` | Command actually consumed by physics | Runtime |
| `R_t` | Reward produced by advancing physics | Environment/reward system |
| `L_t` | Lifecycle events: death, respawn, termination, or truncation | Environment |

The physical environment accepts actuator commands and advances physics. It must not
decide what a policy is allowed to observe. A controller selects decisions but must not
patch observations or advance physics. The runtime owns pending commands and transition
ordering. The observation composer owns perspective, visibility, and private-information
masking.

`pending_action` is the policy-facing concept. The serialized observation key remains
`previous_action` for now, but its canonical meaning is the command physics will consume
next, not historical action. Internally, the simulator may retain `prev_action` where it
unambiguously means the command most recently consumed by physics.

## Core invariants

At every autonomous decision boundary, for every ship controlled by that policy:

```text
policy_observation.pending_action
    == action_that_physics_will_consume_next_for_that_policy
```

This equality applies to neural, scripted, and random controllers. Opponent action
fields are the explicit unknown/private value instead; privacy takes precedence over
revealing the opponent's physical queue.

The following invariants are equally normative:

1. `Q_t`, not `D_t`, is consumed during the normal physical advance from `S_t`.
2. `D_t` becomes `Q_(t+1)` only when that decision is committed.
3. Reset, truncation, death/respawn, and human override can cancel a decision. Canceled
   decisions never enter the pending actuator queue.
4. A team can observe its own permitted pending commands. An opponent's pending command
   is private regardless of whether that opponent is physically visible.
5. Private action information is represented as unknown, not as a valid default action.
6. Full reset initializes every pending command to neutral before constructing the first
   policy observation.
7. A ship's respawn initializes its pending command to neutral and cancels any held
   command for the remainder of the current repeated-action interval.
8. Belief and recurrent state follow explicit lifecycle rules; neither is implicitly
   reset because a mode happened to reconstruct an observation.
9. Rewards describe the physical transition driven by `U_t`. Auxiliary next-state
   targets describe the same transition.
10. Optimized executors must pass the same semantic conformance suite as the reference
    executor.

Unknown/private is not neutral. Neutral is a real command that physics may consume;
unknown/private is an observation-only value that must never reach physics.

## Canonical autonomous transition

An autonomous decision transition is logically:

```text
O_t^k = compose(S_t, Q_t, perspective=k, belief=B_t^k)
D_t, C_(t+1)^k = controller_k(O_t^k, C_t^k)

U_t = Q_t
S_(t+1), R_t, L_t = physics(S_t, U_t)

committed_t = actuator_survives(L_t)
Q_(t+1) = where(committed_t, D_t, neutral)

B_(t+1)^k = update_belief(...)
O_(t+1)^k = compose(S_(t+1), Q_(t+1), perspective=k,
                      belief=B_(t+1)^k)
```

This is a logical ordering, not a required serial schedule. In the batched executor,
controller inference for `D_t` may overlap the physical advance under `Q_t` on separate
CUDA streams. A synchronization dependency is required only before `D_t` is installed
as `Q_(t+1)` or otherwise consumed.

Composition, however, *is* downstream of both: `B_(t+1)^k` is updated by the next-state
forecast the controller produced at `t`, so `O_(t+1)^k` cannot be built alongside the physics
that produced `S_(t+1)`. A scheduler therefore advances physics and runs inference
concurrently, joins, updates each observer's belief, and only then composes.

The transition result must distinguish at least:

- `selected_action`: `D_t`, useful for diagnostics;
- `applied_action`: `U_t`, useful for replay and correctness checks;
- `decision_committed`: a per-environment, per-ship boolean saying whether `D_t` became
  pending for the continuing actuator identity;
- termination, truncation, and per-ship lifecycle masks;
- reward and optional auxiliary targets.

This distinction prevents code from inferring timing from one overloaded action tensor.

## Initial, terminal, and reset boundaries

### Initial state and full reset

For a newly created or fully reset episode:

- physical state is the episode's initial `S_0`;
- `Q_0` is neutral for every ship;
- controller recurrent state is initialized;
- belief state is empty/reset, then populated only through the normal permitted initial
  observation process;
- each team observes its own neutral queue and an unknown/private opponent queue.

No action selected from the previous episode may be patched into `O_0`.

### Termination and truncation

If the advance under `Q_t` terminates or truncates an episode, `D_t` was selected for
overlap but is not committed. Auto-reset must install neutral before composing the next
episode's first observation.

Termination and truncation are separate result flags even though both reset actuator,
belief, and recurrent state. Bootstrapping rules remain the responsibility of the
training collector and must distinguish the two where the algorithm requires it.

An executor that does not auto-reset may expose the terminal physical state for rendering
or diagnostics, but it must not present it as another policy decision boundary.

### Respawn and actuator identity

A respawn creates a new actuator lifetime even when the stable ship/entity slot remains
the same. The entity identity may persist for recurrent and bookkeeping purposes, but:

- the respawned ship's pending command becomes neutral;
- a decision selected by the old life is canceled;
- any remaining substeps of an action-repeat interval consume neutral for that ship;
- the first observation of the new life reports neutral as its own pending command;
- the new life selects normally at its first decision boundary, and that selection is
  applied one autonomous decision later.

An action already consumed by a physics substep cannot be retroactively canceled. If a
respawn is reported after repeated substep `i`, neutral begins at substep `i + 1`.

Per-ship recurrent state persists across respawn. This is intentional: stable entity
identity survives, while actuator state does not. Full episode reset still clears the
recurrent state.

## Action repeat

An autonomous controller selects once per decision boundary. `Q_t` is held for every
physical substep in the repeat interval unless a per-ship lifecycle event invalidates
it. Rewards are accumulated using the commands actually applied on those substeps.

The executor must expose enough lifecycle information to neutralize a respawned ship
without synchronizing unrelated ships or environments. A vectorized in-place mask is
preferred in the batched executor.

## Perspective and belief state

The observation composer produces observer-specific policy views from shared physical and
runtime state, selecting one legal source per ship slot: authoritative truth where the
observer owns or can see the slot, that observer's own belief where it cannot, and zero where
nothing has ever been observed. Belief and truth are the same physical quantities, so this is
a selection rather than a substitution, and it happens before any encoding.

Each controller session composes its own view. Two controllers watching one game remember it
differently; both views are legally sourced, so sharing one leaks nothing, but it attributes
one player's memory to the other and its behaviour stops being a function of its own weights.

Pending action follows the same table:

| Observed ship | Pending-action value |
| --- | --- |
| Own/allied ship | Exact one-hot of its actual `Q_t` |
| Ordinary opponent, visible or hidden | Stored 30-way prediction from the preceding decision |
| Any newly reset or respawned ship | Exact null-command one-hot for both teams |

Belief state must never reconstruct or retain an opponent's authoritative pending command.
It carries only the policy's stored 30-way prediction, alongside the physical means and the
uncertainty terms the next-state head reported. A future visualization may
independently expose raw team perception, belief, and omniscient truth, but debug rendering
must not change the policy view.

Team canonicalization and private-information masking are authoritative observation
operations. Modes must not obtain one team's observation by copying another team's view
and manually patching action fields.

## Controller semantics

| Controller | Action timing | State rules |
| --- | --- | --- |
| Neural | One-decision delayed | Recurrent and belief state follow this document |
| Scripted | One-decision delayed | No privileged pending-action bypass |
| Random | One-decision delayed | Samples the same 30 physical commands |
| Human | Immediate override | Overridden ship's autonomous pending queue becomes neutral |

Using the same delay for autonomous controllers keeps BC demonstrations, evaluation,
and Elo comparisons in the same decision process. A scripted policy may use different
features or computation, but it does not receive immediate actuator semantics.

### Human override

Human input is the sole intentional immediate path. At a decision boundary, an active
human command replaces `U_t` for selected ships. The command may be held over physical
substeps according to the interactive input policy. A respawn during that interval
cancels the held human command after the substep that produced the respawn, just as it
cancels a held autonomous command.

For an overridden ship:

- the autonomous `D_t` is discarded;
- `decision_committed` is false;
- `Q_(t+1)` is neutral;
- team neural inference, belief updates, and recurrent updates may continue so takeover
  and release do not erase context.

When override is released, the neutral queue is consumed while a new autonomous command
is selected. Normal delayed control resumes on the following decision. This visible
neutral handoff is preferable to silently applying a stale command.

## Action representation

Physical commands remain the Cartesian product:

```text
3 power choices * 5 turn choices * 2 shoot choices = 30 commands
```

The authoritative joint ID is:

```text
joint_id = ((power * 5) + turn) * 2 + shoot
power = joint_id // 10
turn = (joint_id // 2) % 5
shoot = joint_id % 2
```

The actor emits one 30-way categorical distribution. The runtime/environment command
remains a compact `(power, turn, shoot)` triple, with one authoritative encode/decode
implementation at the policy boundary. PPO log probability and entropy are those of the
joint distribution, not the sum of three independently sampled heads.

The pending-action observation is a 30-float vector using the same joint-ID ordering.
Own/allied commands are exact one-hot vectors. Ordinary enemy slots carry the dedicated
prediction head's prior softmax distribution, whether or not the ship is in sight. On initial
spawn or respawn both teams instead receive the exact null-command one-hot vector. The two
facts are written together, so neither can overwrite the other. The physical runtime still
consumes only the compact triple `(0, 0, 0)` for neutral; probability vectors never enter the
actuator.

This is an intentional policy/observation schema break. Checkpoints from the factored
12-logit actor and the previous observation width are not silently compatible.

## Training semantics

The delayed runtime is an augmented Markov process whose state includes `Q_t`. A PPO
row may therefore store `(O_t, D_t, log_prob_t, value_t, R_t)` even though `R_t` was
produced by `Q_t`: `D_t` affects future state through `Q_(t+1)`, and `Q_t` is present in
`O_t`.

`decision_committed` is the causal action-effect mask:

- PPO policy-gradient terms include only autonomous decisions that entered the continuing
  actuator queue. A canceled decision cannot have caused the sampled transition, so
  excluding it removes variance without withholding causal credit;
- BC supervision may include a teacher decision selected at a visited observation even
  if lifecycle handling later canceled it. The label still describes the teacher policy
  at that observation;
- entropy regularization may likewise use the ordinary controller/alive mask rather than
  `decision_committed`, because it shapes the policy distribution instead of assigning
  transition credit;
- a decision canceled by termination, truncation, respawn/death, or human override is
  nevertheless recorded as uncommitted for diagnostics and any consumer that needs
  actuator-effect semantics;
- value loss and transition reward accounting remain valid for the completed physical
  transition and use their own masks;
- auxiliary next-state targets are invalid across full reset and for a ship whose
  actuator lifetime ended. Other targets must document their lifecycle mask.

The BC teacher and neural student both select one of the same 30 joint commands under
the same delayed timing. The scripted teacher may continue to compute compact independent
marginals internally. BC storage should retain the current 12 marginal probabilities
and construct their 30-way outer product transiently in the loss, avoiding a persistent
30-way rollout buffer. This represents the current teacher distribution exactly. If a
future teacher models correlated factors, it must provide a true joint distribution.

Training configurations that disable BC must not allocate or populate BC-only buffers.
Evaluation must not pay for training-only rewards, targets, or log probabilities unless
requested as an explicit capability.

## Belief and recurrent lifecycle

Each neural controller session owns recurrent state and, when required, a belief state
for its policy perspective. The lifecycle is:

| Event | Recurrent state | Belief state | Pending action |
| --- | --- | --- | --- |
| Normal transition | Advance | Predict/update | Commit `D_t` |
| Ship respawn | Preserve stable-slot memory | Update from permitted perception; clear invalid per-life facts | Neutral for that ship |
| Human override | Advance, discard overridden output | Update normally | Neutral for overridden ship |
| Full termination | Reset | Reset | Neutral |
| Truncation | Reset | Reset | Neutral |
| Controller identity change | Reset unless explicitly transferred between identical identities | Reset | Neutral at episode boundary |

Belief prediction must use the action known to have driven the physical transition,
not the newly selected decision. Observation composition occurs only after belief is in
the state defined for that boundary — which, because the forecast is this decision's
controller output, is after the physical advance and the inference have joined.

## Episode-stable league identity

An environment's opponent identity is fixed for its episode. Rollout shard boundaries
must not silently replace a controller, recurrent state, or belief state mid-episode.

The bounded replacement design uses `K` logical league slots and at most one draining
generation:

1. Mark one slot generation for replacement.
2. New episodes assigned to that slot use the replacement generation.
3. Existing episodes finish with the old generation and its controller state.
4. Release the old generation after its final episode drains.
5. Do not begin another replacement until the previous generation has drained.

This bounds actively forwarded policy generations to approximately `K + 1` while
preserving episode identity. Checkpoint weight residency is a separate concern: the
roster's LRU cache may retain more inactive policies than the active slot generations.
Batched gather/scatter by generation must be measured; very small draining groups can
be disproportionately expensive. Any drain timeout must be an explicit truncation
policy with reported metrics, never an invisible controller swap.

## Mode requirements

| Mode | Required composition |
| --- | --- |
| PPO rollout | Batched runtime, neural/scripted/league sessions, rewards, GAE inputs, optional auxiliary targets |
| BC | Same batched runtime and delay; scripted teacher capability; no mandatory PPO-only bookkeeping |
| Match evaluation | Runtime plus result aggregation; no independent observe-act-step loop |
| Live Elo | Same match runtime and controller semantics; episode-stable identities |
| Interactive/watch | Low-latency runtime, rendering, belief-enabled neural sessions, optional human override |
| Diagnostics/replay | Runtime trace and optional snapshots; no semantic reimplementation |

Elo values are calibrated only within a run. A run that changes this semantic contract
must start a fresh live-rating state; historical rating-scale compatibility is not a
requirement.

## Execution strategies and performance contract

The architecture should provide shared semantic primitives with at least two schedulers:

- a large-batch scheduler that preserves preallocated tensors, batched inference,
  CUDA-stream overlap, and minimal synchronization/kernel launch overhead;
- a low-latency scheduler that preserves compiled perception/belief paths and may use
  CUDA graphs or different batching choices for one or a few games.

A small reference executor is the executable semantic oracle, not a production fast
path. Production schedulers may fuse operations or update preallocated buffers in place.
They must not call the reference executor in a hot loop.

Shared primitives should be tensor operations that schedulers can compose without
allocating Python objects per transition. Optional result fields and capabilities prevent
evaluation from carrying training bookkeeping. Semantic equality is judged from boundary
inputs/outputs and traces, not from identical internal call graphs.

Performance validation must separately cover:

- large batched 5v5 rollout throughput, policy inference, and overlap;
- low-`N` 5v5 frame latency and achievable FPS;
- a supported larger single game approaching 50v50;
- allocation count, host/device synchronization, and tiny league-generation tail cost.

Use warmup and repeated measurements. Compare staged changes against the same commit and
configuration. The runtime migration and the joint-action/schema migration should be
benchmarked separately so regressions can be attributed.

## Conformance and regression plan

The test-only reference model in `tests/runtime_semantics/` freezes the boundary
semantics before production migration. Its distinctive sequence is:

```text
initial pending: neutral
decisions:       A, B, C
physics applies: neutral, A, B
next pending:    A, B, C
```

Every production scheduler will receive an adapter to the same conformance cases. The
complete validation matrix is:

1. The `neutral, A, B, C` trace for both team perspectives.
2. Own pending action visible and opponent pending action private for visible and hidden
   opponents.
3. Initial state, termination, truncation, auto-reset, and non-auto-reset terminal
   snapshots.
4. Respawn at the first, middle, and final repeated substep, including neutralization of
   all later held substeps.
5. Neural, scripted, and random controllers sharing delayed timing.
6. Human immediate override, discarded autonomous output, and neutral release handoff.
7. Belief/recurrent reset and persistence rules.
8. Team-0/team-1 symmetry and canonicalization.
9. PPO root and attached-opponent views, reset boundaries, rewards, actor masks, and
   auxiliary target timing.
10. BC teacher/student joint-action parity and absence of PPO-only allocations.
11. Match, live Elo, interactive/watch, replay, feature-statistics, autoregressive,
    next-state, and noise-calibration adapters.
12. Episode-stable league identity across rollout shards and two-generation drain.
13. All 30 joint-action codec round trips; probability observations never enter physics.
14. Cross-scheduler trace parity under deterministic seeds.

Known current divergences should first be captured as strict expected failures or as
adapter-level failing tests. As each path migrates, remove the expected-failure marker;
an unexpected pass must fail CI so temporary markers cannot become permanent.

The completed migration's architectural guards require controller-bearing advance loops
to declare a canonical runtime mechanism and restrict raw pending-action observation
access to its explicit owners. Fixed-action physics/render microbenchmarks remain direct
by design because they do not represent controller decision semantics.

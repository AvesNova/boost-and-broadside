# Phase 0 evaluation — correctness, lifecycle, and measurement foundation

Independent evaluation of Phase 0 of
[`perception-belief-overarching-plan.md`](perception-belief-overarching-plan.md),
against its handoff
[`perception-belief-phase0-handoff.md`](perception-belief-phase0-handoff.md).
September 27, 2026.

**Outcome: CONDITIONAL PASS.**

The phase's core is correct and holds up under falsification harder than the
tests it shipped with. One half of one exit-gate line is knowingly unimplemented
and deferred to Phase 2, and the one new diagnostic this phase added is
documented as measuring something it does not measure — on both of its terms.
Neither blocks Phase 1, and the diagnostic's defects are in its prose and its
choice of constant, not in its arithmetic.

## Scope

| | |
|---|---|
| phase evaluated | 0 |
| branch | `fix/seat-symmetry-pending-action` |
| implementation commits | `590a25b..34415e3` — `375cf12`, `c41063f`, `b933da0`, `62c31d4`, `34415e3` |
| evaluation commits | `af0bffb` (CUDA-graph latch regression) and `37aa553` (initial evaluation handoff); this cleanup follows in the evaluation-handoff commit |
| hardware | RTX 4070 Laptop (8 GB), torch 2.13.0+cu130, CUDA 13.2, driver 595.84 |

CUDA was available throughout. Nothing in this evaluation is CPU-only by
necessity; the limits that remain are stated in **What is still unverified**.

## Exit-gate checklist

| requirement | handoff | this evaluation |
|---|---|---|
| Spawn reveal lasts exactly one decision in every relevant stepping path | met | **met**, and in a fourth path the handoff did not name |
| Respawn pending action is null and visible as specified | partly | **partly** — confirmed; the queue is null everywhere, the enemy-visible half is absent |
| Fog resumes normally after reveal | met | **met**, measured end to end rather than at the latch |
| Seat-symmetry tests are clean | met | **met**, mutation-checked independently |
| Belief diagnostics are temporally aligned | met | **met** |
| Baseline performance numbers are recorded | met | **met** — the artifact is complete; VRAM reproduced exactly, SPS did not reproduce on one contended reading |

## Semantic correctness

### The spawn-reveal latch

`TensorEnv.step` now clears `state.ship_spawned` at the top of a decision and
re-fills it from each tick's respawns, gated by `running` — three lines
identical to the two copies `YemongEnvWrapper` already owned. Inspected against
`wrapper.step` and `wrapper.step_interactive`: the three agree, including the
`& running.unsqueeze(1)` that keeps an environment which terminated earlier in a
held action from latching.

Consumers traced, all of which reach the fix through `TensorEnv.step`:
`advance_autonomous_decision` (and therefore `MatchRunner`, every `bnb`
evaluation mode, and `train/rl/elo_eval.py`'s live evaluator) and
`modes/interactive.py`'s `state_only` render path. PPO's rollout steps the
wrapper only (`train/rl/opponents.py:503,520`), so the handoff's claim that
training data was never affected is correct — but live Elo runs inside training
and did read a fog-free world, so that series moves across `375cf12`.

**Falsification, 4000 decisions on CUDA through `MatchRunner` on the `rl`
profile's Frontline geometry** (`probe_reveal`, 16 envs, 5v5, 1024 px sight,
opaque zones, real respawns and episode resets):

```
latched ship-decisions              : 462
latched on two consecutive decisions:   0   (must be 0)
latched but hidden from enemy team  :   0   (must be 0)
latched ships with a live queue     :   0   (must be 0)
mean enemy visibility, overall      : 0.7585
enemy sight of a revealed ship      : 1.0000   (n=299)
enemy sight two decisions later     : 0.0000   (n=299)
```

The last two lines are the contract: a revealed ship is seen by the other side
on exactly its reveal decision, and by the decision after next it is back behind
whatever geometry says. A shorter 400-decision run of the same probe before
contact read 0.1609 mean enemy visibility, against the 1.0000 a stuck latch
produces.

**Mutation check.** With `src/boost_and_broadside/env/env.py` reverted to
`590a25b`, `pytest tests/env/test_spawn_lifecycle.py` gives **6 failed, 1
passed**. The tests bind. (The handoff records "4 of 5" from an intermediate
state of the file; the shipped file is stronger.)

**A fourth entry point, not named in the handoff.** `step_interactive`'s physics
can run through `CapturedTick.replay`, which rebinds `env.state` to fixed
storages and copies every `TensorState` field back from inside the captured CUDA
graph. Had `ship_spawned` been captured by value rather than aliased, every
replay would have overwritten both the clear and the re-fill and play/watch
would have kept the omniscient world. Measured on the GPU: correct. Now pinned
by `test_the_captured_interactive_tick_still_carries_the_latch` (`af0bffb`),
which is CUDA-gated and passes on this machine.

### Null pending action at spawn and respawn

The queue half is correct and now measured over real lifecycle events rather
than asserted: 0 of 462 latched ship-decisions carried a live command in the
`MatchRunner` probe above, and `PendingActionState.commit` zeroes wherever
`actuator_contiguous` is false.

The visibility half is **not implemented**. `write_pending_action_view`
privatises every opponent unconditionally, so during a spawn/reveal decision the
enemy team reads `(3, 7, 2)` rather than the null command the decided semantics
say both teams know. The handoff states this plainly and defers it to Phase 2,
on the grounds that Phase 2 replaces the factorized representation with the
42-way joint one and would delete the work. That reasoning is sound and the
gap is in the safe direction — it under-informs an observer, it does not leak —
so it is not a blocker. It is carried below as a **required Phase-2 item**, not
as Phase-0 rework.

## Information-flow findings

No leak found.

* The reveal is applied to the operative mask only
  (`perception.py:219`); `range_only_ship` and `los_ship` stay pure geometry, so
  the fog diagnostics are not reporting the reveal back as a perception result.
* The reveal broadcasts `state.ship_spawned` over both observer teams, which is
  the intended semantics, and grants no channel beyond ordinary visibility.
* The remaining pending-action asymmetry under-informs rather than over-informs.
* `belief/visible/position_beyond_legal_frac` read 0.0000 on run 747's
  checkpoint in the phase's own probe, i.e. no lifecycle discontinuity is
  currently reaching the visible cell. See the caveat under **Defects**.

## Symmetry findings

The live policy remains Team 0 and nothing randomizes the seat, as the plan
requires.

`test_the_mirrored_game_hands_each_seat_the_other_seat_s_view` and
`test_the_mirrored_game_gives_the_policy_the_same_answers` are a genuine
strengthening: they play one recorded command script twice with the two sides'
labels exchanged, assert the two runs stayed physically in lockstep first, and
then require seat 1 of the game and seat 0 of its mirror to be equal in **every**
observation channel and in action, logprob, value, `pred_next` and hidden state.
Because the comparison puts `flip_team_obs(team1_view)` against a native team-0
view, it also exercises the flip.

**Mutation check, run here.** Replacing `opponent = team_id != observer_team`
with `team_id != 0` in `write_pending_action_view` fails both new tests (and two
older ones): **4 failed, 3 passed**. The handoff's claim is reproduced.

**Coverage narrowing, worth recording.** The mirror runs on `_env_config()`,
which sets no `frontline` and leaves `num_fields` at its default of 0. So the
mirrored comparison covers ships and bullets, not zones and not field tokens.
The Frontline omission is deferred explicitly and for a good reason — `flip_team`
relabels roles without reflecting space, so a ships-only relabelling of a
Frontline board is not a mirrored game, and the mirror would have to swap zone
roles and the front sign at state level. The field-token omission is not named
anywhere; fields are neutral (`team_id == 2`) so the risk is low. Seat symmetry
for zones and fields continues to rest on the older channel-level tests in
`tests/env/test_frontline.py` and `tests/env/test_field_observation.py`, which
are targeted and good but are not a whole-game comparison.

## Temporal-alignment findings

Verified, not inferred from code.

`tests/train/test_ppo.py::TestBeliefDiagnosticAlignment` passes and is the right
shape: it replaces the forecast with the believed target vector — the identity,
which for an absolute Fourier-moment predictor is not a zero vector — and pins
that `obs[t]` agrees with `privileged_targets[t]` to a quantisation floor while
`t+1` is a decision of real motion; that scrambling `privileged_targets[0]`
changes nothing and scrambling index `T` changes the statistic; and that a
planted teleport raises the new series.

The handoff's central diagnostic claim is that the published misalignment
hypothesis is **refuted** — 0.019 px / 8.77 px / 17.61 px at offsets 0/1/2 —
and that the historical 22× was 0.2% of tokens averaging 1726 px each. The
evidence artifact is complete and the probe is committed and re-runnable. I did
not re-run the 3000-decision checkpoint probe; the unit tests pin the property
the exit gate actually asks for, and the historical explanation concerns a run
being discarded.

## Defects

### D1 — `position_beyond_legal_frac` is not a leak rate early in training (low, non-blocking)

`train/rl/ppo.py` and
[`belief-diagnostic-alignment-sep2026.md`](belief-diagnostic-alignment-sep2026.md)
both state that on the **visible** cell the fraction beyond ten decisions'
legal travel "*is* the leak rate", because that distance is unreachable by
physics. The statistic compares a *forecast* against truth, not truth against
truth, so it is only a leak rate once the head's visible-enemy error is well
under the threshold. The head predicts absolute Fourier moments, so an untrained
one is bounded by the world, not by a ship's speed.

Measured on a rollout with a randomly initialised policy and **zero** tokens
removed by `transition_contiguous`, i.e. no lifecycle discontinuity anywhere:

```
threshold                                 =  30.00 px
belief/visible/position_px                = 371.79     (n=128)
belief/visible/position_beyond_legal_frac =   1.0000   (n=128)
tokens excluded by transition_contiguous  =   0
```

So on a fresh run the series pins at 1.0 and reports nothing about leaks. The
metric itself is fine and additive; only its documented interpretation is
overstated. Repair is a comment and a sentence of prose: say that the visible
cell's series is a leak rate *conditional on* the visible-cell mean being small
against the threshold, and that until then it reads head error.

Reproduction: `probe_leakfrac` — build the `test_ppo` trainer with
`EnvConfig(num_ships=4, max_bullets=4, max_episode_steps=400, vision_range=600,
spawn_reveal=True)`, collect one full buffer through `_collect_primary_step`,
`store_final_obs`, then `_precompute_belief_diagnostics`.

### D1b — the threshold is built on a spawn parameter, not a speed limit (low, robustness)

`leak_threshold = 10 · max_speed · dt · action_repeat`. **`max_speed` is not a
speed cap.** It lives in `ShipConfig`'s "Spawn settings" block beside
`min_speed`, `default_speed` and `random_speed`, and its only consumer in the
environment is [`env.py:270`](../../src/boost_and_broadside/env/env.py#L270),
inside `reset_envs`, reached only when `random_speed` is on — which the `rl`
profile leaves off, so on the production profile physics never reads it.
Nothing clamps velocity to it. Real terminal velocity comes out of
`boost_thrust` against quadratic drag, plus lift; and configured speeds are
*proper* speeds `u = n·v` while `ship_vel` is a world velocity, so a refractive
index of `n ≠ 1` separates them further.

Measured over 1,920,000 contiguous ship-decisions of scripted play on the `rl`
profile's Frontline geometry (64 envs × 3000 decisions, respawns excluded):

| | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| speed, px/s | 100.73 | 198.81 | 211.72 | 214.89 | **218.82** |
| travel, px/decision | 1.679 | 3.313 | 3.529 | 3.582 | **3.646** |

Ships exceed `max_speed` **22.08%** of the time. The assumed 3.00 px of legal
travel understates the real maximum of 3.646 px by 21%.

**The 10× slack absorbs it: 0 of 1.92M transitions exceed 30 px.** So this
produces no false positives today and the threshold is sound in practice — but
by luck rather than derivation, and the true headroom is 8.2×, not the 10× the
expression reads. It is unguarded against anything that raises speed: a trained
policy that boosts harder than the scripted agent, a different `boost_thrust`,
a profile with `action_repeat > 1`, or a low-index medium.

The clean repair is to source the threshold from **Phase 1's measured
position-delta scale** rather than from a config constant. Phase 1 collects
precisely this quantity — truth→truth one-decision deltas from scripted
Frontline trajectories with respawns excluded — so the number is free; the
3.646 px above is that measurement in miniature.

Reproduction: `probe_speed` — step `TensorEnv` under
`StochasticScriptedAgent`, accumulate `toroidal_displacement` of `ship_pos`
across each decision under `ship_alive & ~ship_respawned`.

### D2 — the recorded parity command does not run (low)

The handoff records
`benchmarks/belief_compile_parity.py --device cuda --seed 271828 --steps 12`.
`--out` is a required argument, so that command exits on a usage error. With
`--out` supplied the claim reproduces exactly: `parity_passed: true`,
`error_count: 0`, and both teams' `predicted_targets_sha256` identical between
the compiled and eager trackers over 12 steps.

### D3 — the belief diagnostic selects its enemy cell from the observation's team id (low, inherited)

`_precompute_belief_diagnostics` uses `buf.obs[ObsKey.TEAM_ID] == 1`. The
phase's own seat-symmetry write-up names the observation's `TEAM_ID` as
non-authoritative, and the handoff lists `evaluation/next_state.py` and
`modes/ar_report.py` as deferred consumers of that pattern — but not this one,
which the phase touched. The phase's own probe already measured the cost:
16,962 tokens against 17,070 with the state's team id, a 0.6% cell-membership
difference, and the mean is unchanged (0.2083 px against 0.2076 px). Correct
today because `buf.obs` is composed, which restores `TEAM_ID`; not correct by
construction. Fold into Phase 3's direct view composition with the other two.

### D4 — the reveal latch is now three verbatim copies (low, quality)

`TensorEnv.step`, `YemongEnvWrapper.step` and `YemongEnvWrapper.step_interactive`
each carry the same clear-and-re-fill. The duplication is deliberate and
commented on both sides, and
`test_the_wrapper_and_direct_step_agree_on_the_latch` guards two of the three
against divergence — but not the third, and a future change to the reveal rule
has to land in three places. A shared helper taking `(state, respawned,
running)` would collapse it. Not urgent; noted for whoever next edits the
decision loop.

## Claims I could not reproduce as written

D2, and the SPS figure (**Benchmarks** — measurement conditions, not a
regression; VRAM reproduced exactly). Every other number in the handoff that I
checked reproduced:

| handoff claim | this evaluation |
|---|---|
| full suite: 1718 passed | **1718 passed** in 327.79 s |
| `bnb smoke`: all 16 isolated cases passed, checkout unchanged | reproduced, all 16 |
| `tests/env/test_spawn_lifecycle.py` 7 tests | 7, now 8 with `af0bffb` |
| `tests/env/test_seat_symmetry.py` 7 tests | 7 |
| mutation: fixed-team labelling fails both new mirror tests | reproduced — 4 failed, 3 passed |
| mutation: reverting `env.py` fails the lifecycle tests | reproduced, stronger — 6 failed, 1 passed |
| `belief_compile_parity`: `parity_passed: true`, `error_count: 0` | reproduced, with `--out` added |

## Final validation

| command | result |
|---|---|
| `pytest -q -p no:randomly` (full suite) | **1718 passed**, 6 warnings, 327.79 s |
| `pytest tests/env/test_spawn_lifecycle.py tests/env/test_seat_symmetry.py tests/train/test_ppo.py` | 136 passed, 282.52 s |
| `pytest tests/env/test_spawn_lifecycle.py` after `af0bffb` | 8 passed (the CUDA case ran, not skipped) |
| `bnb smoke` | **all 16 isolated cases passed**, checkout unchanged |
| `ruff format --check` / `ruff check` on the touched test file | clean |
| `ruff check .` / `ruff format --check .`, whole repo | 15 errors and 4 unformatted files — **identical at `590a25b` and at `af0bffb`**, so the phase added no lint debt |

## CUDA/GPU validation

The GPU was present and used. `torch.cuda.is_available()` true; RTX 4070 Laptop
GPU, 8188 MiB, torch 2.13.0+cu130.

| what | result |
|---|---|
| `belief_compile_parity.py --device cuda --steps 12 --seed 271828 --out …` | `parity_passed: true`, `error_count: 0`; identical `predicted_targets_sha256` per team between compiled and eager |
| 4000-decision `MatchRunner` reveal probe on `cuda` | table above; all three must-be-zero counters zero |
| 400-decision `MatchRunner` fog probe on `cuda` | 0.1609 mean enemy visibility |
| 1.92M-sample per-decision travel probe on `cuda` | max 3.646 px/decision, 22.08% of speeds above `max_speed` — see D1b |
| captured interactive CUDA-graph tick, latch behaviour | correct; now a CUDA-gated test |
| `rl_pipeline_profile.py --profile rl --timing wall` | VRAM exact, SPS 1,748 against a recorded 2,810 — see **Benchmarks** |
| `bnb smoke` (exercises the CUDA training and mode paths end to end) | all 16 passed |

## Benchmarks

One reproduction of the recorded baseline, same command and same shape —
`--profile rl --timing wall --updates 2 --warmup 1 --no-checkpoint`, 960
environments, 128 steps, 4 rollouts per update, 491,520 environment steps per
update.

| | recorded `rl-wall` | this run |
|---|---|---|
| SPS | 2,810 | **1,748** |
| s/update | 174.95 | 310.22, 252.30 |
| rollout s/update | 103.84 | 179.74 |
| update s/epoch | 21.39 | 30.49 |
| epochs per update | not recorded | 4, 2 |
| peak allocated | 3,461 MiB | **3,461.29 MiB** |
| peak reserved | 4,656 MiB | **4,656.0 MiB** |

**VRAM reproduces to the decimal. Throughput does not**, and this evaluation
cannot say why from one reading. What is known:

* the run was contended — other shells of this evaluation were polling, and a
  `git checkout` round trip landed inside it;
* this is a thermally-limited laptop GPU whose clocks depend on the power
  profile in force, which was not pinned for either measurement;
* the epoch count per update differs (4 and 2 here), and the recorded artifact
  does not carry the baseline's, so `s/update` is not a like-for-like quantity
  between them.

Against that: **Phase 0 changed no code on the training rollout path.** The
`TensorEnv.step` fix is not reached by PPO, which steps the wrapper
(`train/rl/opponents.py:503,520`), and the `ppo.py` change adds one elementwise
comparison inside `_precompute_belief_diagnostics`, under `no_grad`, once per
update. The recorded baseline was itself taken at `62c31d4`, i.e. with both
changes already present. There is no mechanism in this phase that could cost
38% of throughput, so the gap is a property of the measurement rather than of
the code — but that is an argument, not a second measurement, and it is
recorded here as such.

The phase attribution does reproduce structurally. Perception and belief in
this run, as per-update wall averages:

| phase | s/update | share of update |
|---|---|---|
| `01m_rollout/belief_compose` | 23.65 | 8.4% |
| `01n_rollout/belief_advance` | 9.10 | 3.2% |
| `01o_rollout/privileged_obs` | 4.70 | 1.7% |
| `01p_rollout/target_vector` | 3.22 | 1.1% |
| **total** | **40.67** | **14.5%** |

against the recorded sync run's 26.00 s and 12%. Different timing mode, so the
totals are not comparable, but the ordering and the magnitude are — and so is
the finding that matters for Phase 3: `01k_rollout/elo_step` is the largest
line in the profile either way (94.03 s/update and 33.4% of the wall-attributed
update here; 81.8 s/update and 59% of the rollout in the recorded sync run),
several times anything perception does, and it is not Phase 3's to fix.

**Recommendation for whoever next compares against this baseline:** pin the
power profile, run on an otherwise idle machine, and record `epochs_completed`
alongside SPS. The existing artifact omits the epoch counts, which is what makes
`s/update` unusable as a cross-run comparison here.

## What is still unverified

* The 3000-decision run-747 checkpoint probe behind
  `belief-diagnostic-alignment-sep2026.json` was not re-run. The unit tests pin
  the alignment property the exit gate asks for; the probe's extra value is the
  historical explanation for a run that is being discarded.
* The recorded **SPS** figure. See **Benchmarks**: one contended reading came in
  at 1,748 against 2,810, VRAM reproduced exactly, and the phase structure
  reproduced. The exit gate asks that baseline numbers be *recorded*, which they
  are; it does not ask that they be reproduced, and this evaluation does not
  claim to have done so.
* The `--perception eager` arm of the baseline was not re-run. The handoff's own
  reading of it is the honest one: two measured updates cannot separate 1.9%
  from noise, so the conclusion is that compiled perception is not load-bearing,
  not that eager is faster.
* Seat symmetry as a whole-game mirror on Frontline, and on a config with field
  tokens. See **Symmetry findings**.
* Long-horizon training behaviour of any of this. Phase 0 changed no training
  data path, so there is nothing here that a training run would reveal that the
  probes did not.

## Prerequisites for Phase 1, and whether to proceed

**Safe to proceed to Phase 1.** Nothing found here changes what Phase 1
measures. The handoff's own preconditions are correct and I confirm them:

* collect through the wrapper or through `TensorEnv.step` at `375cf12` or later
  — a pre-fix `MatchRunner` measures a fog-free world, which changes what the
  scripted agents do even though it does not change truth→truth deltas;
* exclude respawn transitions with `transition_contiguous` (equivalently
  `ship_respawned` after the step) and report the excluded fraction — Phase 0's
  finding that 0.2% of tokens moved a mean eighteen-fold is the argument;
* one shared scale across position x/y and one across velocity vx/vy, zero
  physical delta to exactly zero normalized delta.

Two additions from this evaluation:

* Phase 1 should **not** reuse `belief/visible/position_beyond_legal_frac` as a
  leak check on a fresh or untrained model without reading D1 first.
* Phase 1's position-delta calibration should be **fed back into the leak
  threshold** (D1b). The threshold currently derives from `max_speed`, a spawn
  parameter that ships exceed 22% of the time; Phase 1 is measuring the real
  distribution anyway, so sourcing the threshold from its constant costs
  nothing and removes a fragile assumption. A useful sanity check for Phase 1's
  own output: scripted Frontline play on the `rl` profile gives a per-decision
  position-delta max of **3.646 px** and p50 of **1.679 px** over 1.92M
  samples, respawns excluded.

## Carried into later phases

**Phase 2 (required):** the spawn override in the pending-action observation, so
both teams read the null command for a spawning ship. This is the unmet half of
Phase 0's exit gate and it is Phase 2's to close, together with the 42-way joint
representation.

**Phase 3:** D3, alongside the two other observation-`TEAM_ID` consumers the
handoff lists; and the handoff's own suggestion of a self-relative team feature
(ego/ally/enemy), which would delete the whole zero-collides-with-a-legal-value
class rather than test around it.

**Any time:** D1 (a comment and a sentence), D2 (one flag in a table), D4 (a
shared helper), and a Frontline state-level mirror for seat symmetry. D1b is
best done *with* Phase 1, since that phase produces the constant it needs.

**Performance target Phase 3 inherits:** 26.00 s/update of sync-attributed
perception and belief work, 12% of the update, against `01k_rollout/elo_step`'s
81.8 s/update, which is 59% of the rollout and not Phase 3's to fix.

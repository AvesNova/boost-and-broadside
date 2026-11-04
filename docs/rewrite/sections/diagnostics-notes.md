# Diagnostics notes (section 9)

Branch `section/diagnostics`. Spec: `diagnostics-spec.md`; audit: `../diagnostics.md`.

## Progress

Branch `section/diagnostics` (round 1), then `section/diagnostics-2` (round 2).

Round 1 (2026-11-03, merged into v2): the collector, the sinks, the recorder, the
method-rule tests, the first-run series, the 21 trigger rules, the skipped-step action, the
dashboards and the cost script. Details below and in the git history.

Round 2 (2026-11-04), on the CPU while a BC run held the GPU, in the brief's order:
1. **The cadence groups** (`bnb.learn.gradients`, `Learner.cadence`), every 10 updates
   (`metrics.due`), after the update's steps, as their own compiled function
   (`Learner.cadence_terms`, the runtime's stage of `Learner._terms`): the gradient
   decomposition over the first minibatch's views in two halves (`gradient/norm`, `share`,
   `cosine`, `agreement`, `coherent_fraction`, `coherent_share`, the actor/critic split)
   and `representation/` per trunk block on one micro-batch (hooks on each block's
   temporal sublayer, removed after). Timed by a device stopwatch,
   `train/time/cadence_seconds`.
2. **The auxiliary heads** (`bnb.learn.head_statistics`): next state (gain per population
   and code part against "nothing changes", KL per population, token shares, supervised
   share), enemy action (cross-entropy, accuracy, entropy, per-factor cross-entropy, gains
   over the batch's repeat and marginal baselines, repeat rate, supervised share, all by
   sight), density (deviance and explained deviance per team, the null the batch's
   per-cell mean, exact from three sums), shots (trajectory KL per part, the three rates
   realised and predicted, supervised share), measured in the first epoch inside the
   compiled loss; the belief (group 8) from the stored forecasts against the stored truth
   by hidden age, with the sigma calibration, in the view pass. **The outcome ledger**
   (group 6) in the rollout: per learner view and second of elapsed time, the sums of
   P(win | resolved) and its square, scored at the episode's end into Brier skill per
   time-to-end bucket, the expected calibration error, the bias, and the reliability
   table at the cadence.
3. **`input/`** (C1): the non-finite count of the raw observation and the share of visible
   living ship rows outside the state code's envelope, per feature, in the view pass.
4. **Fog, physics, frontline and map scale** (groups 13-14) and 5. **the reward's event
   share and zero-sum residual** and 6. **lifespan, events, reward extremes and side
   results**: per-environment sums the rollout adds every tick inside its graph
   (`analytics.resolve_tick`, `perceive_tick`), taken once per update
   (`Rollout.take_games`) and turned into series per slice (`Analytics.games`). The side
   balance rule now has its series. The never-seen invariant (`fog/never_seen_share`)
   comes from the beliefs (a hidden living enemy the view holds no belief of).
7. **Fog slicing**: a pool with fogged and clear games has every per-pool series also as
   `{pool}/fogged` and `{pool}/clear`.
8. **The replay (H1): not built** (see Requests).
9. **`critic/explained_variance_fit`**: the learner's own values during the last completed
   epoch, from moments the loss sums.

Then: the dashboards cover every new quantity, the rules read `representation/`, the cost
script measures the cadence and the rollout's game sums.

Round 2, GPU (2026-11-04, a second agent, under the lock, alone on the GPU):
- the `gpu` tests: 100 passed, 4 failed, all in `test_learner_streaming.py`; fixed in
  `2b63a1da` (see Test runs), then the streaming, cadence and diagnostics files: 34 passed;
- the cost at 4096 environments: out of memory in the first update's cadence; measured at
  2048 (see Cost). The per-update series cost 3.0 %, over the 2 % cap;
- `counters/recompiles`: 0 on every update of a 22-update compiled run (see Cost);
- the full suite: see Test runs.

## Test runs

- CPU (CUDA hidden), quick tier at `b7a3f1ca`'s parent state: 1090 passed, 3 failed: the
  docstring and reference checks (fixed before the commit) and
  `tests/contract/test_layering.py::test_process_wide_settings_belong_to_the_entry_points`,
  which fails on `src/bnb/play/sides.py` (play's code, untouched here). Load 15-20 (other
  agents), 31 min.
- `tests/sections/test_diagnostics.py` and `test_triggers.py`, all tiers, CPU: 54 passed.
- GPU, under the lock: the learner's, BC's, the streaming and precision `gpu` tests: 9
  passed, the no-sync test among them.
- After merging v2 (fix/resume-integrity): `tests/runs`, the trigger, BC and diagnostics
  tests on the CPU, 146 passed and 1 failed (the damaged-draws resume test: dropping
  `pool/slots` from the league's metrics hid the restarted draws); fixed by the entries
  table's `slot` column, then the damaged-resume and league tests: 6 passed.
- The full suite under the lock: not run yet in this round (next agent: run it once).

- Round 2, CPU (CUDA hidden), targeted files while the BC run trained (load 2-3): the
  diagnostics, heads, cadence, games, trigger, learner, BC, learner-shot, reward, mixed,
  loop and runs tests, at `2ad85d9b` plus the fixes in the next commit: `tests/loop`,
  `tests/runs` (the CPU resume tests among them: 7 passed, 4 GPU skipped), the
  diagnostics, heads, cadence, games and trigger tests, the learner, BC, shot, reward,
  mixed, Yemong-runtime and league-metrics tests, `tests/docs` and `tests/contract`: all
  pass (623 + 32 + 7 in the last runs). The first run found the record test's patch of
  `reward_step`, the resume worker's new timing (`train/time/cadence_seconds`, now left
  out like the other times) and two short words in names (`at_most`, `in_field`): fixed.
- Round 2, GPU, under the lock at `dca120c2`: `pytest -m gpu`, 104 tests, 59 min: 100
  passed, 4 failed in `test_learner_streaming.py`:
  - both no-sync tests: the belief errors (`head_statistics`) indexed by a Python list,
    a host-to-device copy of the index per quantity per micro-batch; now slices, with a
    module check that each quantity's columns are a run;
  - the cadence no-sync test, then: `torch.linalg.svdvals` (the effective rank)
    synchronises on CUDA, as every cuSOLVER routine does (svdvals, `gesvdj`, `gesvda` and
    `eigvalsh` all measured); the sampled tokens are now copied to pinned host memory
    without a wait and the rank taken on the host after the readback, through a new
    series value, `bnb.contracts.series.Deferred`, that the collector evaluates after its
    transfer (`test_a_deferred_series_is_evaluated_on_its_copy_after_the_readback`);
  - both pinned-memory equality cases unpacked `Learner.gradient` as four values (it
    returns five since round 2).
  After `2b63a1da`: the streaming, cadence and diagnostics files on the GPU, 34 passed.

## Broken by hand

Each bug was introduced, the named test failed, and the change was reverted:

| Bug | Test that failed |
|---|---|
| switch rate counts a pair across an episode end (`analytics`, `~ended` dropped) | `test_the_rollout_series_match_a_hand_computed_shard` |
| power and stick marginals swapped (`statistics.marginal_entropies`) | `test_uniform_logits_give_the_factors_uniform_entropies` |
| a `Mean` over no samples reported (`Collector.read`) | `test_the_collector_reads_every_device_series_in_one_transfer` |
| rule: warm-up ignored (`Triggers.check`) | `test_a_rule_is_quiet_during_its_warm_up` |
| rule: the entropy-collapse z-score compares the wrong way (`ZScoreBelow`) | `test_each_rule_fires_on_its_case_and_stays_quiet_otherwise[entropy_collapse]` |
| rule: `Above` breaches at its threshold (`>=`) | `test_a_value_at_the_threshold_does_not_breach` (added after the case tests missed it) |
| round 2: the repeat baseline spreads over 42 actions, not the other 41 (`head_statistics.series`) | `test_the_enemy_action_series_match_a_hand_computed_case` |
| round 2: the belief's position error without the minimum image | `test_the_belief_errors_match_a_hand_computed_case` |
| round 2: the coherent fraction divided by half A's square only (`gradients.decomposition`) | `test_halves_that_disagree_are_incoherent` (strengthened after equal-size halves missed it) |
| round 2: a life left running at its episode's end not counted (`analytics.resolve_tick`) | `test_the_resolve_sums_match_a_hand_computed_episode` |

## Cost

`uv run --no-sync python -m bench.diagnostics_cost --environments 2048 --blocks 2
--per-block 2`, under the lock, the mixed 5v5 preset with Yemong (bf16, kernels), RTX 4070
Laptop, commit `f235fcd5` plus the cost script's progress lines; load average 1.7 / 3.2 /
3.2 (a CPU test run of mine overlapped the second half). Whole updates (collect, the
rollout's series, learn, the readback, the record), the learner's diagnostics on and off
alternating in blocks:

| | median of 4 |
|---|---|
| update, diagnostics on | 56.59 s |
| update, diagnostics off | 56.97 s |
| per-update overhead | -0.7 % (within the run-to-run spread of about 4 s) |
| the record (rules, sinks) | 1.8 ms |

- The per-update series are under the 2 % cap; the measurement cannot resolve them. "Off"
  still runs the rollout's series (they feed BC's stop) and the skipped-step guard.
- The cadence groups (5 % cap) are not built, so not measured.
- 4096 environments (the 40,960-ship budget) did not finish 30 updates in 50 minutes on
  this laptop; the run was stopped and repeated at 2048. Re-measure at 4096 on a quiet,
  larger GPU.

### Round 2

`uv run --no-sync python -m bench.diagnostics_cost --environments 2048` (4 blocks of 3,
3 cadences), under the lock, commit `2b63a1da`, RTX 4070 Laptop, load average 1.2 / 1.2 /
1.4, nothing else on the GPU:

| | median |
|---|---|
| update, diagnostics on (12) | 58.05 s |
| update, diagnostics off (12) | 56.36 s |
| per-update overhead | **+3.0 %** (spread within each arm about 0.6 s) |
| the record (rules, sinks) | 4.8 ms |
| the cadence groups alone | 3.40 s (in a 61.3 s update) |
| the cadence, amortised over 10 updates | 0.60 % |
| the rollout's game sums | 2.0 ms per tick, 0.45 % of an update |

- The per-update series are over the 2 % cap at this cell; round 1 measured -0.7 %, so
  round 2's per-update additions (the heads' series in the first epoch, the belief and
  input sums, the fit moments) cost about 1.7 s. Not yet split by group.
- The cadence is well under its 5 % cap; the game sums are small.
- **4096 environments run out of memory** on this 8 GB GPU in update 1's cadence (the
  forward of `cadence_terms` in `_gradient_decomposition`, 7.57 GiB in use, 1 GiB of it
  the CUDA graphs' pool). Round 1's 4096 run fit (it was only slow), so the cadence's
  memory is what overflows; at 512 environments the cadence updates peak no higher than
  the plain ones (5.23 GB from update 2 on). A 4096 training run on this GPU would stop at
  update 1. Not fixed here (see Requests).
- Recompiles: `bnb train --preset mixed --environments 512 --updates 22` (compiled, the
  diagnostics on): `counters/recompiles` 0 on every record, through the cadence updates
  1, 10 and 20; the first update (803 s) compiles the heads' loss and the cadence's
  terms, and the recorder's baseline is taken after it.

The cost script also
reports the cadence groups on their own (`cadence_seconds`, its update, and
`cadence_amortised_overhead` = cadence seconds / 10 / the plain update) and the rollout's
per-tick game sums timed alone on its buffers (`game_sums_per_update_overhead`), since
"off" keeps them. What is expected, to check:
- per update, the heads' series run in the first epoch only (one epoch's worth of extra
  head maths on top of the loss), the belief and input sums in the view pass, the fit
  moments in every micro-batch;
- the cadence: one forward and 14 backwards per micro-batch of one minibatch, two
  accumulators of 14 gradient vectors (about 2 x 14 x the parameter count in float32;
  hundreds of MB for Yemong's default size), two Gram products, and one eager unroll of a
  micro-batch for the representation;
- the rollout: the game sums and the outcome ledger every tick; the ledger's scoring
  reads its (views, seconds, 3) buffer each tick (about 2.5M floats at 8,192 views), the
  likeliest of these to show.

## Deviations and requests

- **The learner skips non-finite steps.** The spec's skipped-step action presumes section
  6's handling (decisions 4d: skipped, counted, dumped), which the learner did not have: it
  counted and stepped, so one NaN gradient ruined the weights. `Learner._step` keeps the
  weights and Adam's state where the norm is not finite, on the device. A finite step is
  bitwise unchanged; the cost is one copy of the weights and the optimizer's state per
  step (measured with the rest, see Cost).
- **Names in full words over the audit's spellings:** `counters/environment_steps` (spec:
  `counters/env_steps`), `train/environment_steps_per_second`, `train/gradient_norm`,
  `train/time/*_seconds`, `critic/explained_variance` (audit `ev`), `returns/advantage_rms`
  (`adv_rms`), `train/memory_peak_gigabytes`, the rule `behaviour_mismatch` (audit "C4
  mismatch"). The stick factor keeps the contract's name (`stick`, audit "turn").
- **The timings at the source:** after fix/resume-integrity merged, `Trainer.update`
  returns `train/time/rollout_seconds` and `train/time/learner_seconds` itself, and the
  resume worker leaves out those two names.
- **`policy/kl`** is the last completed epoch's mean (the early stop's quantity, audit);
  it was the first epoch's. `policy/clip_fraction` is over every step (was the first
  epoch's). BC's `bc/kl` stays the first epoch's (the stop rule reads it).
- **Episode counts** come from the rollout's per-environment sums (a new carry buffer,
  `episodes`, not checkpointed: every update takes it); results stay per learner view.
- **The anomaly rules' baselines** leave out a breaching value, so a lasting collapse stays
  flagged past the hysteresis.
- **`rating/live`** is the gauge's last finite value, as `live_elo` was.
- **Memory** is the allocator's reserved peak per update (the audit's "compiled peak" is
  XLA's; PyTorch has none at compile time).
- **Request:** side balance (`episode/result/team_*`) needs each view's seat in the
  records or a rollout sum; the rule is built and idle.

- **Round 2 names in full words:** `gradient/` (audit `grad/`), `representation/` (audit
  `repr/`; the rules read the new names), `gradient/cosine` (`cos`),
  `density/explained_deviance` (audit `global_density/r2`), units spelled out
  (`belief/position_metres`, `physics/speed_metres_per_second`), `outcome/expected_calibration_error`
  (`ece`). The name check now allows "of" (`angle_of_attack`, the contract's name).
- **The heads' series are the first epoch's**, all of the batch, under the weights as
  they move through that epoch; the other epochs skip them (a second variant of the
  compiled loss). The audit's "per update" holds; the cost is a quarter of every epoch's.
- **Groups 7 and 8 read the model:** the next-state baseline recomputes the model's input
  code (`Yemong.tokens`) inside the diagnostics; the belief reads the replay record, so
  it measures the behaviour policy's forecasts (as the audit's source note says).
- **The belief's sigma** is the per-axis RMS for position (normalised square error =
  distance² / (2 sigma²)) and the covariance for velocity (the Mahalanobis distance over
  two); "within one sigma" uses that per-axis normalised square, so for the two-axis
  quantities it is not the 68 % of one axis.
- **Episode rewards and events cover every ship of every game** (opponents included, as
  the reward function would pay them), not the learner's alone; the learner's own reward
  is `reward/rate`.
- **`physics/shield_recharge_per_ship_second`** is under `physics/` (the audit's
  `frontline/`): shields recharge in every mode. **`physics/energy_mean` and
  `out_of_energy_share`** use the contract's `energy` (audit "power").
- **`frontline/front_moves_per_environment_minute`** replaces `front_speed_m_s`: the v2
  front is a count of captures, not a distance.
- **The gradient decomposition's coherence and cosines are in the trunk scope** (encoder and
  trunk parameters); `coherent_fraction` = <g_A, g_B> / mean(|g_A|², |g_B|²), halves by the
  minibatch's views (contiguous views are neighbouring environments).
- **`rewards.reward_step_events`** gives each event's weighted reward beside the levels;
  `reward_step` is it without the events (its result unchanged).
- **The rollout's carry** holds the games' running state (`game`, the outcome ledger),
  saved in checkpoints, and the update's sums (`game_sums`, `outcome_tables`), not saved.
- **Request (H1, the replay):** not built. `bnb.play.replay` re-simulates from keyframes
  with the match's own CPU generator; a training environment's ticks draw from the
  batched device generator, so the recorded actions would not reproduce its states, and
  a one-environment keyframe needs the game API to slice the opaque state. Options for
  section 10: record each tick's draws in the format, or a keyframe per tick with playback
  that restores rather than re-simulates; plus an API function for one environment's
  state.
- **Request (perception):** `fog/range_visible_share`, `field_occluded_share` and
  `individual_visible_share` need the reason a ship is hidden and who saw it, which the
  observation does not carry.
- **Not built, by source:** the policy gradient per reward level (needs per-level
  advantages through the update), the belief's persistence and dead-reckoning baselines
  (need the last-seen state), the enemy action's opponent-kind slice (the learner does not
  know a view's pool), `bc/loss/heldout` and `bc/loss/{own,teacher}_states`.
- **Request (the cadence's memory):** at 4096 environments on an 8 GB GPU the cadence's
  first update runs out of memory (see Cost). Options: measure it on a smaller slice than
  a whole minibatch's micro-batches, release the update's staging and graph memory before
  it, or skip it when the probe's budget says it will not fit.
- **Request (the per-update cost):** 3.0 % at 2048 environments, over the 2 % cap; the
  heads' first-epoch series are the likeliest share. A split by group (each switched off
  in turn) would say which to move to the cadence.
- **`Deferred` series** (`bnb.contracts.series`): a host function of device tensors
  copied to pinned memory without a wait, evaluated by the collector after its readback.
  Only the effective rank uses it: no CUDA eigen- or singular-value routine runs without
  a synchronisation.

## Open questions

- Should `bnb runs upload` also replay a run's `metrics.jsonl` into W&B for runs trained
  offline? It uploads checkpoints only, as before.

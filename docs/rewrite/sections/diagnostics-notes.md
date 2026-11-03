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

**GPU work left (the next agent or the coordinator, under the lock, once):**
- the `gpu` tests, among them the two new ones in `tests/sections/test_learner_streaming.py`
  (`test_the_cadence_groups_add_no_synchronisation`,
  `test_the_rollouts_game_sums_are_taken_without_a_synchronisation`) and the existing
  no-sync and capture tests, which now run the game sums inside the captured graph;
- the cost: `uv run --no-sync python -m bench.diagnostics_cost --environments 4096` (and
  2048 if 4096 does not finish), filling the Cost section below;
- the full suite under the lock;
- check `counters/recompiles` stays 0 after update 1 on a compiled run (the heads' variant
  of the loss and the cadence's compiled terms both compile in update 1; the repr hooks are
  removed before the next compiled call).

## Spec items

| Spec | Done |
|---|---|
| 1 Collection | `Collector`: producers return `{name: tensor or Mean}`; one stacked float64 transfer per update (`read_back`); a `Mean` is divided on the host and absent at count 0. Producers: `Learner.update` (and BC's `report`), `bnb.runtime.analytics.Analytics` (store shards and the rollout's per-environment episode sums), the league's `metrics`, the recorder. The cadence helper `metrics.due` exists; no cadence group is built yet. |
| 1 Method rules | One readback per update in `Trainer.update`; the per-epoch KL `.item()` stays. Raw values: the only EMAs logged are the return scale's (`returns/scale*`) and the rating. Names quantity first, full words, checked by a test. |
| 2 Sinks | `metrics.jsonl` per run (always), W&B live with `--wandb` (project `boost-and-broadside-v2`, the run's name, its id kept in run.json so a resume continues it, x-axis `counters/environment_steps`). The fired-events table (`alerts/events`) goes to both. `bnb runs upload` unchanged. |
| 3 Trigger rules | 8 alerts and 13 warnings, as the audit lists them, with warm-up, hysteresis and cooldown; placeholder thresholds. Outputs `alerts/{rule}`, `alerts/active`, `alerts/events`. Rules whose series are not built yet stay idle. State in the training checkpoint's record (`triggers`), through
`bnb.runs.checkpoint.training_state`, so the resume test's exact-restore check covers it
(the resume worker attaches the rules and logs their flags); restored on `--resume` with
`bnb.runs.checkpoint.triggers_state`. |
| 3 Skipped step | The learner skips the step on the device (weights and Adam state kept where the gradient norm is not finite); the rule fires per update with a skip; the session writes a checkpoint and `runs/<run>/dumps/step_<steps>/` (the store's whole batch and the record), listed in run.json. A BC run only dumps (it keeps no checkpoints). |
| 4 Dashboards | `bnb.runtime.dashboards`: the 12-panel headline and the curated view (open sections Alerts, Strength, Health, Policy, Value, Learning signal; collapsed detail sections; a `bc/` section open in BC runs) as `wandb-workspaces` objects; one panel per quantity, slices by regex. `bnb runs dashboard RUN` saves both. |
| 5 No host syncs (A6) | The streaming test counts the update's synchronisations under `set_sync_debug_mode("warn")`: one per epoch (the early stops' reads and the readback). |
| 5 Determinism (A8) | `test_diagnostics_do_not_change_the_update` (CPU, bitwise, on and off) and the existing streams test. |
| 5 Resume (A9) | Unchanged (another branch owns `tests/runs/test_resume.py`); the CPU resume tests pass with the new records. |
| 5 Joint action | Entropy marginals and usage over power 3, stick 7, shoot 2 of the joint. |
| 5 Per pool and mode | Pools are named game/opponent, so every per-pool series is per mode and opponent; fog is not sliced yet. |
| 6.1 Hand-computed cases | See the coverage table's test column. |
| 6.2 Method rules | Name check, no-smoothing check, no-sync test. |
| 6.3 Trigger rules | Each rule's trip and quiet case, warm-up, hysteresis, cooldown, the event row, the state's checkpoint round trip (`tests/sections/test_triggers.py`). |
| 6.4 Skipped-step action | `test_a_non_finite_step_writes_the_checkpoint_and_the_dump_and_training_continues`. |
| 6.5 Dashboards | `test_the_code_builds_the_headline_and_curated_views` (offline: built and serialised, not saved). |
| 6.6 Cost | See Cost. |
| 6.7 Coverage | Below. |

## Coverage

Producer abbreviations: L `bnb.learn.ppo.Learner`, H `bnb.learn.head_statistics` (through
L), D `bnb.learn.gradients` (L's cadence), B `bnb.learn.bc.BehaviourCloning`, A
`bnb.runtime.analytics` (store shards, the rollout's episode sums and its game sums), G
`bnb.league.league.League.metrics`, R `bnb.runs.recording`, T `bnb.runs.triggers`. Test
files: `d` `tests/sections/test_diagnostics.py`, `dh` `..._heads.py`, `dc` `..._cadence.py`,
`dg` `..._games.py`, `ls` `test_learner_streaming.py` (gpu).

| Audit group | Series | Built | Test |
|---|---|---|---|
| 1 histograms | dropped; D2 quantiles and D3 usage replace them | `policy/entropy_quantile/p10,p50,p90` (L), `policy/usage/{factor}/{option}` (A) | d: histogram quantiles; hand shard |
| 2 overview | dropped | the dashboards | d: dashboards build |
| 3 losses | per-head losses | `next_state/loss`, `enemy_action/loss`, `density/loss`, `shot/loss/{trajectory,outcome}` (L) | d: named and finite; `test_learner`'s independent computation |
| 4 policy | KL split, clip, far ratio, entropy, coupling, epochs, D2-D4 | `policy/*` (L), `policy/usage/*`, `switch_rate/*` (A) | d: policy sums; marginals; hand shard |
| 5 critic, returns | EV, EV fit, value KL, bias, return mean, support clip, percentiles, advantage, normaliser, mixed | `critic/*`, `returns/*` (L), `critic/explained_variance_fit/{level}` (L, last epoch). Not built: `returns/mixed` (team spirit is 0 at every level) | d: targets hand case; dh: fit moments |
| 6 outcome | ledger: Brier skill per time-to-end bucket, ECE, bias, reliability table, KL | `outcome/brier_skill/{bucket}`, `outcome/expected_calibration_error`, `outcome/bias` (A, every update), `outcome/reliability` (table, at the cadence), `critic/value_kl/outcome` | dg: ledger hand case, Brier skill, reliability rows, a rollout's ledger |
| 7 next state | gain per population and part, KL per population, token share, supervised share | `next_state/gain/{ally,enemy_visible}[/{part}]`, `kl/{ally,enemy_visible,enemy_hidden}`, `token_share/*`, `supervised_share` (H). The residual/absolute output slice: the model has one output, so no slice | dh: zero change gives zero gain |
| 8 belief | errors by hidden age, sigma calibration, baselines | `belief/{quantity}/{age bucket}` (10 quantities), `belief/{normalised_square_error,within_one_sigma,within_two_sigma}/{quantity}/{bucket}` (H, stored forecasts). Not built: the `visible` slice (a visible ship's stored belief is its truth), `position_beyond_legal_share`, the persistence and dead-reckoning baselines (need the last-seen state, which no record keeps) | dh: belief hand case across the seam |
| 9 enemy action | CE, accuracy, entropy, per-factor CE, gains over repeat and marginal, by sight; opponent-kind slice; supervised share | `enemy_action/{cross_entropy,accuracy,entropy,repeat_rate}/{sight}`, `factor_cross_entropy/{factor}/{sight}`, `gain/{repeat,marginal}/{sight}`, `supervised_share` (H). Not built: the opponent-kind slice (the learner has no view's pool) | dh: uniform-logit hand case |
| 10 density | deviance and explained deviance per team | `density/deviance/{ally,enemy}`, `density/explained_deviance/{ally,enemy}` (H) | dh: batch mean explains 0, exact counts 1 |
| 11 shot | trajectory KL per part, rates realised and predicted, supervised share | `shot/trajectory_kl/{part}`, `shot/{enemy_hit,ally_hit,clear_miss}_rate/{realised,predicted}`, `shot/supervised_share` (H) | dh: rates and KL hand case |
| 12 episode | length, lifespan, reward mean/min/max/level, events, alive share, results by side, end, count | `episode/count`, `length_seconds`, `end/*`, `wins`, `draws`, `losses`, `alive_share` (round 1); `episode/lifespan_seconds`, `reward_mean`, `reward/{level}`, `reward_min`, `reward_max`, `event/{name}` per slice, `episode/result/{team_0,team_1,draw}` (A, game sums; every ship of every game) | dg: hand episode; a rollout's sums against its counters |
| 13 fog | visible share (all, by scale), hidden age, reacquisitions, occlusion, never seen, fogged share | `fog/visible_share/{all,scale_*}`, `mean_hidden_age_seconds`, `reacquisitions_per_slot_minute`, `occlusion_share/{bucket}`, `fogged_environment_share` (A, fogged environments), `fog/never_seen_share` (H). Not built: `range_visible_share`, `field_occluded_share`, `individual_visible_share` (the observation does not say why a ship is hidden or who saw it; needs a perception output) | dg: perceive hand case |
| 14 physics, frontline, map scale | rates in readable units | `physics/damage_per_ship_second/*`, `deaths_per_ship_minute/*`, `speed_metres_per_second`, `energy_mean`, `out_of_energy_share`, `in_field_share`, `shield_recharge_per_ship_second`; `frontline/front_moves_per_environment_minute`; `map_scale/mean`, `slot_share/*` (A). Not built: simultaneous captures (no per-zone capture event in the observation) | dg: resolve and perceive hand cases |
| 15 reward | event share, level share, zero-sum residual | `reward/event_share/{event}` (13 events, `rewards.REWARD_EVENTS`), `reward/zero_sum_residual/{level}` (A, reward per environment-second), `level_share` (round 1) | dg: hand episode |
| 16 seat | dropped | | |
| 17 schedule | every scheduled hyperparameter | `schedule/learning_rate` (L), the only schedule | |
| 18 train, counters | throughput, time, skipped, memory, recompiles, gradient and parameter norms, counters | all (L, R), plus `train/time/cadence_seconds` | recorder; parameter groups; skip test |
| 19 gradient | the decomposition at the cadence | `gradient/norm/{model,trunk}/{term}`, `share/{scope}/{term}`, `cosine/trunk/{a}/{b}` (every pair), `agreement/{scope}/{terms,value,next_state}`, `coherent_fraction/{term}`, `coherent_share/next_state_{population}`, `actor_share`, `actor_norm`, `critic_norm` (D). Terms: the eight top-level, the value per level, the next state per population. Not built: the policy per reward level (needs per-level advantages kept through the update) | dc: hand Gram case; incoherent halves; the learner's cadence; ls (gpu): no sync |
| 20 rating, pool | live rating and error, averages, window games, score and game share, size, entropy, draining, entries | all (G) | `test_league_metrics` |
| ungrouped | `repr/` (B6, F2), `input/` (C1), C5, C6, `bc/` | `representation/{dormant_share,activation_rms,effective_rank}/trunk_block_{i}` (D, at the cadence); `input/nonfinite_count`, `input/envelope_violation_share/{feature}` (H); `batch/token_share/*` (A); `*/supervised_share` (H); `bc/*` (B). Not built: `bc/loss/heldout`, `bc/loss/{own,teacher}_states` | dc: rank and dormant cases, the taps; dh: input case; `test_bc` |
| H1 replay | one environment every N updates | not built (Requests) | |
| triggers | 8 alerts, 13 warnings | all (T); every rule now has its series except the ones on unbuilt series | `test_triggers` |

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
  loop and runs tests; results in the commit messages and below.

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

Not measured: the GPU was held by the BC run for the whole round. The cost script now also
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

## Open questions

- Should `bnb runs upload` also replay a run's `metrics.jsonl` into W&B for runs trained
  offline? It uploads checkpoints only, as before.

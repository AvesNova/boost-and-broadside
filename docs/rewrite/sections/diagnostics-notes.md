# Diagnostics notes (section 9)

Branch `section/diagnostics`. Spec: `diagnostics-spec.md`; audit: `../diagnostics.md`.

## Progress

Done and pushed (first round, 2026-11-03):
1. The collector (`bnb.runtime.metrics`), the series types (`bnb.contracts.series`), the
   sinks (`bnb.runs.sinks`: `metrics.jsonl` always, W&B with `bnb train --wandb`), the
   recorder (`bnb.runs.recording`), with every existing series moved onto them under
   quantity-first names.
2. The method-rule tests (`tests/sections/test_diagnostics.py`): the name check, the
   no-smoothing check, diagnostics on and off give the same update; the no-sync test is
   `tests/sections/test_learner_streaming.py::test_an_update_synchronises_only_to_read_its_kl_and_its_results`
   (GPU), now reading the series back.
3. Series, first-run groups: the learner (losses per head, policy and its KL split,
   entropy joint and marginals, quantiles, near-certain share, factor coupling, far-ratio
   share, clip, ratio drift, epochs), the critic (explained variance, value loss and KL,
   bias, return mean, support clip), returns (percentiles per level, advantage mean and
   RMS, the scale and its EMAs), training health (gradient norm, clip fire rate, skipped
   steps with the offending terms, parameter norm and update ratio per module), BC
   (`bc/kl`, entropies, agreement per factor, per-pool KL, scores, the stop's window), the
   rollout's series per pool (episode counts, results, length, timeout and decided shares,
   reward rate per level, level shares, token shares by game and opponent, action usage
   and switch rate), the league (live rating and error per mode, the averages, window
   games, score and game share by kind, sampling entropy, pool size, the entries table),
   throughput, time breakdown, memory, counters.
4. The 21 trigger rules (`bnb.runs.triggers`), their state in the training checkpoint, the
   skipped-step action (checkpoint and batch dump). The learner now skips a non-finite
   step (see Deviations).
5. The dashboards in code (`bnb.runtime.dashboards`, `bnb runs dashboard RUN`).
6. The cost script (`bench/diagnostics_cost.py`).

Not built yet (next agent, in this order): the cadence groups (`grad/`, `repr/`, the
actor/critic split), the auxiliary heads' baselines and gains (groups 7-11), the outcome
ledger (group 6), `input/` (C1), the game analytics (groups 13-14), `reward/event_share`
and `zero_sum_residual`, `episode/result/team_*` (side balance; needs the seat per view),
`episode/lifespan_seconds`, `episode/event/*`, `reward_min/max`, the fog slicing of the
episode and reward series, the replay every N updates (H1), `critic/explained_variance_fit`
and `returns/mixed/*`. The coverage table below lists each.

## Spec items

| Spec | Done |
|---|---|
| 1 Collection | `Collector`: producers return `{name: tensor or Mean}`; one stacked float64 transfer per update (`read_back`); a `Mean` is divided on the host and absent at count 0. Producers: `Learner.update` (and BC's `report`), `bnb.runtime.analytics.Analytics` (store shards and the rollout's per-environment episode sums), the league's `metrics`, the recorder. The cadence helper `metrics.due` exists; no cadence group is built yet. |
| 1 Method rules | One readback per update in `Trainer.update`; the per-epoch KL `.item()` stays. Raw values: the only EMAs logged are the return scale's (`returns/scale*`) and the rating. Names quantity first, full words, checked by a test. |
| 2 Sinks | `metrics.jsonl` per run (always), W&B live with `--wandb` (project `boost-and-broadside-v2`, the run's name, its id kept in run.json so a resume continues it, x-axis `counters/environment_steps`). The fired-events table (`alerts/events`) goes to both. `bnb runs upload` unchanged. |
| 3 Trigger rules | 8 alerts and 13 warnings, as the audit lists them, with warm-up, hysteresis and cooldown; placeholder thresholds. Outputs `alerts/{rule}`, `alerts/active`, `alerts/events`. Rules whose series are not built yet stay idle. State in the checkpoint's `extra.triggers`; restored on `--resume`. |
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

Producer abbreviations: L `bnb.learn.ppo.Learner`, B `bnb.learn.bc.BehaviourCloning`, A
`bnb.runtime.analytics`, G `bnb.league.league.League.metrics`, R `bnb.runs.recording`, T
`bnb.runs.triggers`. Tests are in `tests/sections/test_diagnostics.py` unless named.

| Audit group | Series | Built | Test |
|---|---|---|---|
| 1 histograms | dropped; D2 quantiles and D3 usage replace them | `policy/entropy_quantile/p10,p50,p90` (L, from a 64-bin histogram), `policy/usage/{factor}/{option}` (A) | histogram quantiles; hand shard |
| 2 overview | dropped | the dashboards | dashboards build |
| 3 losses | per-head losses | `next_state/loss`, `enemy_action/loss`, `density/loss`, `shot/loss/{trajectory,outcome}` (L) | named and finite; the loss terms keep `test_learner`'s independent computation |
| 4 policy | KL split, clip, far ratio, entropy joint and marginals, coupling, epochs, D2-D4 | `policy/kl`, `kl_first`, `clip_fraction`, `far_ratio_share`, `entropy/{joint,power,stick,shoot}`, `factor_coupling`, `near_certain_share`, `entropy_quantile/*`, `epochs_completed`, `minibatch_steps`, `stopped_early`, `ratio_drift/{max,mean}` (L); `usage/*`, `switch_rate/*` (A) | policy sums hand case; uniform marginals; hand shard |
| 5 critic, returns | EV, EV fit, value KL, bias, return mean, support clip, percentiles, advantage, normaliser, mixed | `critic/explained_variance`, `value_loss`, `value_kl`, `value_bias`, `return_mean`, `support_clip` (per-ship levels); `returns/p5,p50,p95/{level}`, `advantage_mean`, `advantage_rms`, `scale`, `scale_p5`, `scale_p95`, `scale_floored` (L). Not built: `explained_variance_fit` (needs the last epoch's moments from the loss), `returns/mixed` (team spirit is 0) | targets hand case; target entropy |
| 6 outcome | ledger: Brier skill per time-to-end bucket, ECE, bias, reliability table, KL | `critic/value_kl/outcome` is the outcome KL. The ledger is not built | |
| 7 next state | KL and gains per population and group | not built | |
| 8 belief | physical errors by sight and hidden age, baselines | not built | |
| 9 enemy action | realised probability, entropy, Brier, accuracy, persistence baselines, supervised share | loss only | |
| 10 density | deviance and R2 per team | loss only | |
| 11 shot | rates and per-channel CE | losses only | |
| 12 episode | length, lifespan, reward mean/min/max/level, events, alive share, result by side, end decided/timeout, count | `episode/count`, `length_seconds`, `end/{decided,timeout}` (A, from the rollout's sums), `wins`, `draws`, `losses`, `alive_share`, `reward/rate/{level}` per pool (A). Not built: lifespan, events, reward min/max, side results | hand shard; episode sums; rollout counts against its records |
| 13 fog | visibility, occlusion, hidden age, never-seen | not built (the never-seen rule is idle) | |
| 14 physics, frontline, map scale | rates in readable units | not built (the rollout's `counters` hold shots, hits, deaths, resets) | |
| 15 reward | event share, level share, zero-sum residual | `reward/level_share/{level}` (A). Not built: event share, residual | hand shard |
| 16 seat | dropped | | |
| 17 schedule | every scheduled hyperparameter | `schedule/learning_rate` (L), the only schedule | |
| 18 train, counters | throughput, time, skipped, memory, recompiles, grad norm, clip fire, parameter norm, update ratio, counters | all (L, R); memory is the allocator's reserved peak per update against the probe's budget; recompiles are graphs compiled after the segment's first update | recorder throughput; parameter groups; skip test |
| 19 grad | the decomposition at the cadence | not built | |
| 20 rating, pool | live rating and error, averages, window games, score and game share by kind, size, sampling entropy, draining, entries table | all (G); per mode | `test_league_metrics`; sampling entropy; entry kinds |
| ungrouped | `repr/` (B6, F2), `input/` (C1), C5 token shares, C6 supervised shares, `bc/` (I1-I5) | `batch/token_share/{mode,opponent}/*` (A); `bc/kl`, `bc/entropy/{student,target}`, `bc/agreement/{power,stick,shoot,joint}`, `bc/kl_by_pool/*`, `bc/score/*`, `bc/stop/*` (B, cloning). Not built: `repr/`, `input/`, supervised shares, `bc/loss/heldout`, `bc/loss/{own,teacher}_states` | `test_bc` |
| triggers | 8 alerts, 13 warnings | all (T) | `test_triggers` |

## Test runs

- CPU (CUDA hidden), quick tier at `b7a3f1ca`'s parent state: 1090 passed, 3 failed: the
  docstring and reference checks (fixed before the commit) and
  `tests/contract/test_layering.py::test_process_wide_settings_belong_to_the_entry_points`,
  which fails on `src/bnb/play/sides.py` (play's code, untouched here). Load 15-20 (other
  agents), 31 min.
- `tests/sections/test_diagnostics.py` and `test_triggers.py`, all tiers, CPU: 54 passed.
- GPU, under the lock: the learner's, BC's, the streaming and precision `gpu` tests: 9
  passed, the no-sync test among them.
- The full suite under the lock: not run yet in this round.

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

## Cost

To be measured under the lock at the 5v5 mixed cell with Yemong
(`uv run --no-sync python -m bench.diagnostics_cost`): pending at this commit.

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
- **Temporary:** `Trainer.update` still returns `rollout_seconds` and `learn_seconds`
  under those names, because `tests/runs/resume_worker.py` (owned by fix/resume-integrity)
  filters them out by name; the recorder renames them `train/time/*`. Rename at the source
  once that branch merges.
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

## Open questions

- Should `bnb runs upload` also replay a run's `metrics.jsonl` into W&B for runs trained
  offline? It uploads checkpoints only, as before.

# Seat-symmetry bug in the pending-action privacy mask — September 26, 2026

Subject: the rollout's observation construction at commit `30790f6`, found while
reading the expected-value curves of run 747 (`bright-forest-747`, profile `bc`,
221.2M steps). Fixed on branch `fix/seat-symmetry-pending-action`.

Evidence in this directory:
[`seat-symmetry-bug-sep2026.json`](seat-symmetry-bug-sep2026.json) (every number
below) and
[`seat-symmetry-bug-sep2026-experiment.py`](seat-symmetry-bug-sep2026-experiment.py)
(the four-way matchup harness; `--driver train|eval` selects the path).

## The defect

`_mask_hidden_ships` zeroes every channel of a ship the observer cannot see,
`ObsKey.TEAM_ID` included. Zero is also Team 0's real id, so in the team-0 view a
hidden enemy is indistinguishable from an ally, and once every enemy is out of
sight the channel is uniformly zero.

The rollout passed that channel into the pending-action privacy mask:

```python
# opponents.py, _collect_primary_step — before
action_state.write_observation(step.obs, step.obs["team_id"][:, :num_ships], num_ships)
```

`write_pending_action_view` computes `opponent = team_id != observer_team`. With
`team_id` all zeros:

* `observer_team=0` → `opponent` all **False** → nothing hidden → **Team 0 saw
  every enemy's pending command**
* `observer_team=1` → `opponent` all **True** → everything hidden → **Team 1 saw
  no pending commands at all, including its own**

`MatchRunner.observe` passed `self.env.state.ship_team_id` and was never
affected, which is why the Elo evaluator's numbers stayed plausible throughout.

The same call appeared in `modes/interactive.py`, so play and watch ran with the
same broken mask.

## Measured effect

Four matchups as four blocks of one 960-environment batch, run 747's final
checkpoint, 12,000 decisions. `a` is the policy, `s` the scripted controller
built as the trainer builds it.

| condition | driver | t0 win | t1 win | draw | t0 score | ep len |
|---|---|---|---|---|---|---|
| a vs a | train, before | 0.997 | **0.000** | 0.003 | 0.999 | 3070 |
| a vs s | train, before | 0.253 | 0.398 | 0.349 | 0.427 ± .032 | 8836 |
| s vs a | train, before | 1.000 | **0.000** | 0.000 | 1.000 | 2983 |
| s vs s | train, before | 0.349 | 0.336 | 0.315 | 0.506 ± .032 | 8774 |
| a vs a | eval, before | 0.267 | 0.333 | 0.400 | 0.467 ± .032 | 8854 |
| s vs a | eval, before | 0.347 | 0.339 | 0.314 | 0.504 ± .032 | 8819 |
| a vs a | train, after | 0.347 | 0.357 | 0.297 | 0.495 ± .029 | 8747 |
| a vs s | train, after | 0.237 | 0.463 | 0.300 | 0.387 ± .028 | 8724 |
| s vs a | train, after | 0.443 | 0.327 | 0.230 | 0.558 ± .029 | 8620 |
| s vs s | train, after | 0.330 | 0.360 | 0.310 | 0.485 ± .029 | 8892 |

Three readings:

* **The environment was never at fault.** `s vs s` scores 0.506 before the fix —
  no seat bias in the physics, the front mechanic or the win condition.
* **The policy could not play Team 1 at all**, winning 0 of 1593 games from that
  seat across the two blocks that used it, while scoring 0.427 against scripted
  from the Team-0 seat.
* **After the fix the seats agree.** The policy scores 0.387 as Team 0 and 0.442
  as Team 1 (`1 − 0.558`), within noise of each other and of the Elo evaluator's
  long-standing 0.44. Episode lengths in the two collapsing blocks go from ~3000
  to ~8700, matching the others: the fast resolutions were Team 1 folding.

A per-seat confidence measure agrees. Sampling the same dual-perspective batch
twice and comparing, the policy self-agreed 0.637 in the Team-0 seat against
0.348 in the flipped seat — it was roughly half as certain playing a view it had
effectively never been trained on.

## Why nothing caught it

`_combine_actions` returns `actor_mask = team0_mask` under `ego_pass`, and BC
gates its loss on it (`bc_f = bc_valid & mb_actor_mask & mb_alive`), so **only
Team-0 tokens are ever supervised**. `elo_eval` seats the live policy on Team 0
only (`elo_eval.py:14`). Neither the loss curves nor the evaluator ever looked at
the broken seat. `loss/behavioral_cloning_kl` fell to 0.127 measuring one side.

`episode/team0_win_rate` sat at 0.996 for the whole run and was logged the entire
time. Runs 741, 744 and 745 used the same BC profile and ran 0.30–0.64; run 747
is the only one pinned, and it is the first after `138f540`, which introduced
`write_pending_action_observation` and the `PRIVATE_*` categories.

## Second defect found while fixing

`reset_envs` reshuffles slot-to-team assignment on every reset ("Independent
random permutation per env → any slot can be any team"). The two reads are
therefore not interchangeable: `_combine_actions` needs the **pre-step**
assignment its actions were selected under, while `write_observation` runs after
the wrapper's auto-reset and must match the observation it writes into. The old
code was accidentally consistent on this axis because `step.obs["team_id"]` was
also post-step, so a single-read fix would have introduced a subtler bug on
episode boundaries. Both sites now read the state at the correct point.

## What changed

* `train/rl/opponents.py` — `_collect_primary_step` reads `state.ship_team_id`
  before the step for the action merge and after it for the privacy mask.
* `train/rl/ppo.py` — same for `_collect_aux_steps`; plus
  `_seat_symmetry_metrics`, logged once per update.
* `modes/interactive.py` — same call, fixed.
* `tests/env/test_seat_symmetry.py` — five observation-level invariants, exact
  rather than distributional.
* `tests/train/test_ppo.py::TestSeatSymmetryOfPendingActions` — the regression,
  driven through the real rollout, plus a test that the new metric fires on the
  reconstructed bug.

`seat/own_pending_hidden` and `seat/enemy_pending_leaked` are both zero on a
healthy run. Either going positive means the two seats are playing different
games.

## Consequences

**Run 747's 221M steps should be discarded.** BC supervised Team-0 tokens under
an observation where Team 0 could read every enemy's pending command — the policy
learned against information that disappears with the fix. Every earlier BC run
carries the same contamination from `138f540` onward.

[`next-state-head-audit-sep2026.md`](next-state-head-audit-sep2026.md) measured
that checkpoint and should be re-read with this in mind: the belief head was
trained against a corrupted input distribution, so its 93x dead-reckoning gap is
not cleanly attributable to the head's own design.

## Left in place

Two consumers still merge by the observation's team id. Both are correct today
because they receive *composed* observations, where `BeliefTracker.compose`
restores `TEAM_ID` from its cache — but both rely on that without stating it:

* `evaluation/next_state.py` — a pure observation→observation transform for
  `imagine_trajectory`, with no state in scope.
* `modes/ar_report.py:345` — replay analysis.

The durable fix is a self-relative team feature (ego / ally / enemy) emitted per
perspective, which would delete `flip_team`'s `TEAM_ID` swap and make seat
identity unrepresentable in model input rather than merely absent. A precedent
for this bug class already exists in `BeliefTracker.compose`:

```python
# Ship tokens have no zone role. Five is the explicit NONE value;
# leaving the masked zero would falsely describe a Team-0 spawn.
```

Someone hit the same zero-collides-with-a-legal-value problem for `ZONE_ROLE` and
patched it downstream. Two instances of one root cause is a pattern.

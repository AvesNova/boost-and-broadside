# Overnight log, 2026-11-14

This log records the changes and decisions made while the author was away, with the evidence for each. Times are KST. The newest entries are at the bottom.

## 02:00 — mandate

1. Finish bc-v19 so it yields data for balancing the geometry head.
2. Build Yemong v8 with the knee raised.
3. Decide whether to make settings tensors.
4. BC to about 1000 Elo against Baekho, with a budget of at most 150M steps.
5. Then calibrate RL from the BC run, aiming at about 2000 Elo.

## 02:00 — decision: settings stay baked-in numbers (not tensors)

- **What the change would buy:** about 7 minutes of compile saved per run that differs only in numbers. On 8 GPUs the arms compile in parallel, so that is about 7 minutes of wall-clock time per round, under an hour for 10 rounds.
- **What it would cost:** plumbing through the returns, losses, map sampling and play tuning, plus tests. Values that switch work on or off, such as a zero weight skipping a head, can't be tensors without computing both paths. CUDA graphs need in-place updates.
- **Verdict:** the bug risk outweighs about an hour of compile. I'll revisit if compile time becomes a real bottleneck, for example cold compiles on shared-CPU hosts.

## 02:00 — geometry head: total weight kept at 0.007

- **The design rule:** set in commit e334d032, the weight targets a trunk gradient norm of about 0.02, the level the next-state head reaches in trained BC.
- **bc-v19 measurements:**
  - `gradient/norm/trunk/geometry` is 0.0205 at update 10 and 0.0218 at update 20.
  - next_state is 0.0224 and 0.0132; density 0.076 and 0.092; policy 0.029 and 0.072.
- **Verdict:** on target, so it stays.
- **The balance of its five outputs needs per-output gradients.** Those are being added to the diagnostics in v8 and measured on a short CPU run, using the method from learner.md.

## 02:00 — bc-v19 flags at update 17 (Sonnet triage): none is a run killer

- **`sigma_miscalibrated`:** a monitoring artifact. It is a mean of error²/σ² over near-constant quantities. Robust calibration is fine, with 60–80% of errors within 1σ. The fix to the trigger goes into v8.
- **Next-state loss:** peaked during the combat ramp-up and is now falling.
- **Correction loss:** rising slowly and flattening; I'm watching it.
- **Damage critic explained variance:** recovering; 0.19 at update 11, 0.44 at update 17.
- **To watch:** BC KL has been flat at 0.46–0.47 since update 10 (v1 reached 0.40). The stick head is the weak part, at 76% agreement.

## 02:00 — Yemong v8 (agent, in progress)

- **Velocity code:** knee 100, spacing 5, about 97 bins. Reach stays about 1006 m/s.
- **Diagnostics:** the per-output gradient split for the geometry head.
- **Alert:** the `sigma_miscalibrated` trigger made robust.

## Live Elo (averaged policy against Baekho = 1000)

| step | deathmatch | elimination | frontline |
|---|---|---|---|
| 10M | 548 ± 35 | lost every game | lost every game |

## 01:56 — bc-v19 crashed at update 27 (27M steps): NaN in a rollout belief decode

- **The error:** a device-side assert, `index out of bounds < 3457`, in the compiled `codes.invert_mean` table lookup. That lookup is the delta head's decode of a 55-bin family, run during the rollout. A NaN mean bin produces a garbage index.
- **What it rules out:** the learner was clean. It had 0 non-finite steps, 0 skipped steps, 0 non-finite inputs and finite losses. So the NaN entered through a rollout-time input or game state, or from an op hit by an extreme input. The weights were not corrupted.
- **What I've ruled out by reading the code:**
  - `arena_code`: log(0) at a closed boundary gives −∞, which clamps.
  - The boundary inputs: they are ratios over the fixed arena radius.
  - The boundary damage ramp: it is clamped.
- **A possibly related signal:** `belief/normalised_square_error/position_metres/*` has been NaN every update.
- **Action:** an agent is reproducing the rollout on the idle 5090 with per-tick finiteness checks to find the exact op. The v8 BC run won't launch until the cause is fixed. Snapshots up to 25M and the live Elo for 10M and 20M are kept.

## 02:50 — root cause: a game physics 0/0 (fixed, b17a43e8)

- **The op:** `numerics.softmin(0, 0)` = 0/0 in `flight._vertical_demand`.
- **When it happens:** a slow ship with a nearly empty store, climbing, with the stick neutral. At p=16 the held climb rounds to the whole lift budget, so the turn budget is exactly 0, and the lateral command is 0.
- **What follows:** the ship's state goes NaN, the NaN reaches the model inputs, and the belief decode asserts.
- **How it was found:** a rollout on the 5090 reproduced it deterministically (seed 2, env 1259, tick 8436). Replaying the frozen state found the exact op.
- **The fix:** softmin(0, 0) = 0, plus a test with a finite gradient. Every other value is unchanged, so the game version stays the same. The agent's fixed rerun ran clean past the ticks that had failed.
- **Not done, a proposal:** a rollout-side non-finite detector, counting non-finite ship state per update and failing fast with the env and tick. It would have turned a 30-minute hunt into a log line. It isn't needed for the run.
- **Still open:** the NaN in `belief/normalised_square_error/position_metres/*` is probably a separate metric issue: a near-singular covariance determinant overflows.

## 03:06 — bc-v20 launched (ac77680c): Yemong v8 + the softmin fix, 150M

- **Yemong v8** (d43a4b7d): the velocity code is knee 100, spacing 5, **99 bins**, with a reach of 1059 m/s.
  - 97 bins reach only 1002 m/s, short of the 1006 target. Every past count has been odd, so I accepted 99.
  - The state code is 544 wide (616 in v7).
- **The `sigma_miscalibrated` alert** (dd797c9c): it now fires when any `belief/within_one_sigma/*` falls below 0.4.
  - The agent warns that it still fires on a short CPU run, where shield_recharge_wait is within one σ 0% of the time.
  - I'll check this in bc-v20 before trusting the rule.
- **The geometry head's per-output gradient split** (4a3fe524). The short CPU run, all in learning-rate warm-up, gave these trunk norms:

  | output | trunk norm |
  |---|---|
  | range | 1.2e-4 |
  | closure | 7.5e-5 |
  | antenna train angle | 1.6e-5 |
  | aspect | 1.6e-5 |
  | heading crossing angle | 1.2e-5 |

  - Range and closure carry about 80% of the term, and the five outputs' gradients point the same way (cosines 0.77–0.94).
  - **Decision:** no reweighting from a warm-up CPU run. bc-v20 logs the split at production scale. I'll set the per-output weights from it before RL, which recompiles anyway.
- **The launch line** takes NAME and BUDGET. bc-v20 runs 150M steps, finishing about 05:10 if steps/s matches bc-v19.
- **Live Elo** is repointed: the `bnb-live-elo` unit with RUN=bc-v20, and the bnb-elo worktree at ac77680c.

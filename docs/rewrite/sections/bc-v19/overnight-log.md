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

## 03:25 — bc-v20 update 18: the BC KL flag judged benign

- **The flag:** BC KL rose from a best of 0.40 to 0.49, then fell back to 0.468.
- **Why it's benign:** over the same updates the teacher's entropy rose from 0.80 to 0.95 as combat ramped up. A harder target is the same distribution shift that bc-v19's triage found behind the next-state and correction rises.
- **Speed:** unchanged, with 3.1 s rollout and 44.7 s learning per update.
- **Inputs:** no envelope violations, no clipping, nothing non-finite.
- **The new within-one-σ alert:** fired at update 16, then cleared.
- **Elo at 5M:** deathmatch 598 ± 31, elimination −84 ± 142, frontline −275 ± 246.

## 04:15 — bc-v20 stopped by its plateau rule at 84.9M

- **Why it stopped:** BC KL held within the plateau band for 10.5M steps. KL was 0.35, against bc-v19's floor of 0.40.
- **Live Elo** of the averaged policy against Baekho (1000):

  | step | deathmatch | elimination | frontline |
  |---|---|---|---|
  | 40M | 809 | 350 | −3 |
  | 50M | 858 | 610 | 469 |
  | 60M | 925 | 791 | 760 |
  | 70M | 953 | 872 | 885 |

  Elo was still rising at 70M even though the KL plateau had fired. The KL rule doesn't track strength, as the old-game handoff curve also showed. A resume can raise the budget but cannot change the stop rule. I'm waiting for 84.9M's ratings before choosing between resuming and handing off to RL.

## 04:20 — geometry head, production per-output split: no reweighting

- **Trunk norms** from bc-v20, medians over updates 30–80:

  | output | median trunk norm |
  |---|---|
  | antenna train angle | 0.0090 |
  | aspect | 0.0088 |
  | closure | 0.0056 |
  | range | 0.0033 |
  | heading crossing angle | 0.0030 |
  | geometry total | 0.025 (target 0.02) |
  | next_state, for comparison | 0.0085 |

- **The CPU run was misleading:** it ranked range and closure on top, the opposite of this.
- **Decision:** keep equal nats per target and the total weight of 0.007.
  - The spread is within 3×.
  - In the CPU run the outputs' gradients all pointed the same way.
  - The total is near target.
  - Gradient size also reflects how much is left to learn, so equalising it has no clear benefit.

## 04:26 — handoff: RL `rl-v20` launched from bc-v20's averaged policy (5b85b6c2)

- **bc-v20's final averaged policy** at 84.9M: deathmatch 970 ± 25, elimination 930 ± 21, frontline 893 ± 28. At 70M it was 953, 869 and 869.
- **Why hand off now:**
  - Gains are slowing.
  - Every mode is within about 110 of Baekho.
  - A resume can't lift the plateau stop.
  - The earlier v2 handed off at a similar level.
- **RL settings** (rl.sh), the champion's PPO:
  - a warm start: weights and return scale, with a fresh optimizer;
  - 2048 environments, about 1.05M steps per update;
  - **32 minibatches**, 4 epochs;
  - per-level γ/λ, target KL 0.02, learning rate 3e-4 with a 10M warm-up;
  - 256 rating lanes;
  - one map size per mode;
  - 143 updates, about 150M steps.
- **Why 32 minibatches:** the author locked 256 for BC. For RL, 32 is what trained the champion, under the build-on-the-winner rule. The minibatch comparison is still a candidate for the arms.
- **Live Elo** is repointed to rl-v20.

## 04:47 — rl-v20 update 12: `gauge_runaway` judged a warm-up artifact (recheck at about 22M)

- **What the alert counts:** league entries that have played training games but whose rating error is still infinite.
- **Why it fires now:** after 12 updates, the deathmatch rating lanes have finished no rated games, so every deathmatch entry, including the scripted anchor, is at stderr ∞. Frontline has rated one or two games.
- **Not a runaway fit:** elimination is rated normally (live 1068 ± 26).
- **Expected to clear** as long deathmatch and frontline games finish. If the deathmatch lanes still have no rated games by about 22M, that is a real bug in the lanes.
- **Otherwise healthy:**
  - 6.1 s rollout and 38–43 s learning per update, so about 1.8 h for 150M steps;
  - policy KL 0.006–0.008, clip fraction 0.07–0.08, entropy 1.44;
  - advantage shares: damage 0.08, kill/death 0.14, objective progress 0.10, objective 0.20, outcome 0.49, with every level's gradient cosine to the total between 0.65 and 0.86;
  - inputs clean.

## 04:57 — rl-v20 at 27M: the alert cleared, Elo climbing, fighting more

- **`gauge_runaway`:** every mode now has rated games, and the anomalies are 0. Confirmed as a warm-up artifact.
- **The run's own rating lanes** put the live policy against Baekho (1000) at deathmatch about 1660, frontline about 1545, elimination about 1429. These are noisy, with ±40–160.
- **The laptop's live Elo** of the averaged policy at 10M: deathmatch 999 ± 25, elimination 953 ± 25, frontline 1094 ± 26.
- **Combat rose with RL:** deathmatch hits per game went from 417 to 927 and frontline from 569 to 1061 by 12M. That is the opposite of the failed old-game RL check, which learned to stop fighting.
- **Entropy** fell slowly from 1.44 to 1.32. KL is 0.006 and the clip fraction 0.06.

## 05:28 — rl-v20 at 70M: `rating_drop` judged noise near the ceiling; deathmatch averaging lags

- **The laptop's live Elo** of the averaged policy against Baekho (1000):

  | step | frontline | elimination | deathmatch |
  |---|---|---|---|
  | 30M | — | 1252 | 1082 |
  | 35M | 1739 | 1329 | 1097 |
  | 45M | 1946 | 1428 | 1116 |
  | 55M | **2110** | **1480** | **1142** |

- **`rating_drop`:** the run's own frontline rating of the live policy fell from about 1966 to about 1760. Its error is ±130–180, because the live policy wins almost every game against Baekho, which leaves the rating poorly determined. The laptop's frontline rating of the averaged policy rose over the same stretch. I judged it noise and am not acting on it.
- **To watch:** in deathmatch the averaged policy, 1142 on the laptop, lags the live policy, about 1880 in the run's lanes, far more than in the other modes. I'll rate the final live and averaged policies side by side to see whether averaging costs deathmatch strength.
- **Health:** entropy fell from 1.32 to 1.11 and KL is 0.008. Advantage shares: outcome 0.44, objective 0.29, objective progress 0.11, kill/death 0.11, damage 0.06.

## 05:59 — rl-v20 at 113M: frontline 2369, elimination 1727; the deathmatch "lag" is a scoring-scale effect

- **The laptop's live Elo** of the averaged policy:
  - frontline 2179 at 65M, then 2268, 2353, **2369 at 95M**;
  - elimination 1584 at 65M, then 1618, 1676, **1727 at 95M**;
  - deathmatch **1162 at 65M**, up from 1097 at 35M.
- **Why deathmatch looks low:** the laptop rates deathmatch on kill share, a continuous score, as the author chose. A dominant policy scores well under 1, so its Elo gap is compressed. The run's own win/loss lanes rate the same averaged policy about 1585, the uniform average 1539, and the live policy about 1944. So averaging is not costing deathmatch strength. The kill-share Elo scale is simply not comparable with the other two modes.
- **`gauge_runaway` again:** a new league snapshot (86M) has played but has no rated games yet, the same warm-up artifact as before. Benign.
- **To watch:**
  - Entropy is falling: 1.11 at 70M, 0.81 now. KL is steady at 0.008 and the clip fraction 0.08.
  - Deathmatch boundary deaths in league games rose from 0.14 to 2.9 per game, against about 41 combat deaths. Small, but rising.

## 06:30 — rl-v20 finished 150M; resumed to 300M (a budget extension)

- **Final averaged-policy live Elo:**
  - frontline **2585 ± 37** at 145M, still rising: 2457 at 115M, 2535 at 135M;
  - elimination **1693 ± 33**, flat since about 95M, where it peaked at 1727;
  - deathmatch is still being rated, on the kill-share scale.
- **At 143 updates:** entropy 0.66, KL 0.007, clip fraction 0.07, no skipped or non-finite steps.
- **Why resume to 286 updates, about 300M** (same settings, a budget extension only):
  - The 5090 would otherwise sit idle.
  - Frontline is still climbing.
  - The 500M publication run needs evidence that nothing collapses late, such as entropy collapse or the rising deathmatch boundary deaths.
  - The cost is about $1.

## 06:41 — rl-v20 at 161M: `reward_hacking` warning and the density loss judged benign

- **`reward_hacking`** (a warning: reward per second rising while the rating is flat):
  - Against Baekho the live policy wins nearly every game, so its rating is at a ceiling while its margin, and so its reward, can still grow.
  - The reward event shares have been unchanged since 100M:
    - damage dealt 0.41 against damage taken 0.47;
    - kills 0.43 against deaths 0.48;
    - friendly fire 0.033 → 0.027;
    - recharge and charge-back up to 0.004.
  - No single event source is growing, which an exploited reward would show.
- **Density loss** rose from 1.13 to 1.46. That is game-distribution drift: positions are harder to predict as the play changes.
- **The real watch item:** entropy is 0.57 and still falling. KL is 0.008 and the clip fraction 0.07.
- **The laptop's live Elo** has not written a deathmatch rating since 65M even though the games are reused. That needs a look.

## 06:50 — live Elo: deathmatch was cut from latest.txt (fixed, a0805606); the deathmatch series

- **The bug:** `latest.txt` kept the log's last 30 rating lines. With 16 or more snapshots × 3 modes, deathmatch, which prints first, was cut off. The ratings themselves were always computed and logged.
- **The fix:** keep the last finished cycle's whole table. The service is restarted on it; finished game batches are kept per batch, so little was lost.
- **Deathmatch, averaged policy, kill-share scale:**
  - 997 at 10M, 1082 at 30M, 1142 at 55M, 1182 at 75M, 1205 at 95M, 1210 at 115M, 1204 at 145M, 1204 at 150M.
  - Like elimination, it has plateaued since about 100M. Frontline is the only mode still climbing.

## 06:50 — the author's question: why is deathmatch Elo about 1200 when v2.1 reached about 2200?

**What the data show (rl-v20 at about 160M):**

1. **Deathmatch gets a 4–5× smaller advantage.** Mean |advantage| summed over the levels:

   | mode | summed mean \|advantage\| | the largest part |
   |---|---|---|
   | deathmatch | 0.011 | outcome 0.0038 |
   | frontline | 0.054 | objective 0.022, outcome 0.018 |
   | elimination | 0.041 | outcome 0.035 |

   One return scale normalises all modes, so frontline and elimination dominate the policy gradient. The cause is the outcome level. Since b2bea713, deathmatch's outcome is the kill share (2·share − 1), which sits near 0 in even games. v2.1 used a ±1 win/loss.
2. **The rating scale is compressed.** The 145M averaged policy took 76.6% of the kills against Baekho, which is +206 on the kill-share scale. On win/loss, a policy that takes 77% of the kills would win most games and rate far higher. So the kill-share number and v2.1's win/loss deathmatch Elo aren't comparable.

**Experiments, each from bc-v20's averaged policy:**

- **E2 `rl-dm`:** deathmatch only, the same settings, 72 updates (about 75M). It tests whether deathmatch learns when not crowded out. Started 06:50. The rl-v20 extension was stopped at update 166 (172M; checkpoints kept) to free the GPU. Entropy had fallen to 0.4 by then, after 0.57 at update 154.
- **E1 (agent):** a rating switch, `--deathmatch-score win_loss`, so deathmatch can also be rated as v2.1 did and matched with it.
- **E3 (agent, then the GPU):** a training switch for deathmatch's outcome, kill lead as win/loss, and a mixed run with it.
- **Attention shares (agent):** the attention mass to self, allies, enemies, zones, fields and the super token, by mode, for the BC final and RL at 150M.

## 07:10 — switches in; win/loss rating of rl-v20 running; E3 queued

- **Rating switch** (c6fc6d88): `bnb ratings --deathmatch-score win_loss`. Records name their scoring and are never reused under the other.
- **Training switch** (1eaebc74): `--set reward.deathmatch_outcome=kill_lead`. It sits on RewardConfig, so existing policy files still pass the game check. The default is unchanged.
- **E1 is running on the laptop:** rl-v20's averaged snapshots at 10, 30, 55, 95 and 150M are being rated in deathmatch on win/loss against Baekho. The live Elo service is paused while it runs.
- **rl-dm flags:** the objective and objective-progress advantage shares are about 0. That is expected, since deathmatch has no objective.
- **E3 `rl-kl`** is queued on the box. It is mixed modes with deathmatch's outcome on kill lead, 143 updates. It starts when rl-dm ends, from a second checkout, /workspace/game2 at 4475e893, so rl-dm's code isn't touched mid-run.
- **07:12, rl-dm update 23:** `critic_useless` fired on the objective and objective-progress critics, whose returns are always 0 in deathmatch, so their explained variance is meaningless. The real signal is that the **outcome critic explains only 0.135 of the kill-share outcome's variance**. The value the policy learns against is noisy, which fits deathmatch's weak advantage. Entropy is 1.16.

## 07:17 — ANSWER: deathmatch is the strongest mode; the low number was the kill-share Elo scale

rl-v20's averaged policy rated in deathmatch on **win/loss** against Baekho, which is how v2.1 rated it. The column is `--deathmatch-score win_loss`, 768 games.

| step | deathmatch, win/loss | frontline, win/loss | deathmatch, kill share |
|---|---|---|---|
| 10M | 987 ± 35 | 1342 (20M) | 997 |
| 30M | 1668 ± 89 | — | 1082 |
| 55M | 2196 ± 121 | 2110 | 1142 |
| 95M | **2825 ± 164** | 2369 | 1205 |
| 150M | **2719 ± 147** | 2585 | 1204 |

- **Deathmatch leads frontline by a few hundred Elo at matched steps,** the same pattern as v2.1.
- **Kill-share scoring compresses a dominant policy to about +200:** a 77% kill share is about +206.
- **Deathmatch's 4–5× smaller advantage** did not stop it outpacing the other modes.
- **For the author's decision** (not changed): kill share is a fair measure of margin, but its Elo is not comparable with the other modes or with v2.1, and it saturates near +200. Rating deathmatch on win/loss, or reporting both, would restore comparability.
- **Follow-ups:**
  - rl-dm, deathmatch only, finishes about 07:45.
  - rl-kl, mixed with the kill-lead outcome, is trimmed to 72 updates, a matched 75M comparison with rl-v20, since the rating question is answered. It still tests whether a ±1 outcome helps deathmatch learn faster.
  - The live Elo service is unpaused.

## 07:30 — attention shares (the author's diagnostic; bench/attention_shares.py, 1ac40156)

Each living own ship's attention mass, in %, averaged over heads, ticks and the 4 spatial layers, with the policy against Baekho. The recomputed softmax is checked against each attention call's output: 4–5e-3 under bf16, 6e-7 in float32.

| policy | mode | self | allies | enemies seen | enemies hidden | global | fields | zones (or absent-zone slots) |
|---|---|---|---|---|---|---|---|---|
| BC | deathmatch | 9.3 | 12.5 | **23.6** | 4.2 | 4.2 | 32.0 | (12.6) |
| BC | elimination | 8.7 | 18.7 | 10.9 | 5.7 | 6.2 | 34.3 | (14.1) |
| BC | frontline | 10.9 | 14.4 | 14.3 | 5.3 | 3.0 | 34.5 | 16.2 |
| RL 150M | deathmatch | 9.5 | 15.3 | 15.3 | 5.4 | 5.2 | 33.3 | (13.3) |
| RL 150M | elimination | 9.3 | 19.6 | 9.1 | 7.6 | 6.1 | 33.4 | (13.6) |
| RL 150M | frontline | 8.2 | 18.4 | 10.5 | 7.5 | 3.7 | 30.0 | 20.7 |

The dead-ship slots get about 1–3%. Self-play is within 2–3 points, and the two ship types within 1.

**Per key:** a visible enemy and the ship itself each get about 7–11%. Fields take a third of the total, but each field gets only about 3%.

**By layer:**
- the global token peaks in layer 1 (6–12%);
- allies peak in layer 2 (up to 29%);
- layer 3 sends 20–31% to the zones, or to the absent-zone slots.

**RL against BC:** RL attends less to visible enemies, most clearly in deathmatch (23.6% → 15.3%), and more to allies and hidden enemies in every mode.

**Striking: the absent-zone slots are an attention sink.** In deathmatch and elimination, one learned token repeated in five slots takes 13–14% of the mass, about 20% in the last layer. It is probably an "attend to nothing" sink, which transformers commonly learn and use, so it isn't necessarily waste. A proposal for the author, not done: mask absent zones and give the model one explicit sink token instead, then compare with an arm.

## 08:00 — E2 result (rl-dm, deathmatch only, 75M): no crowding-out; deathmatch training lifts elimination but forgets frontline

| steps | rl-dm deathmatch, win/loss | rl-v20 (mixed) deathmatch, win/loss |
|---|---|---|
| 30M | 1697 ± 95 | 1668 ± 89 |
| 50–55M | 2297 ± 118 | 2196 ± 121 |
| 75M | **2866 ± 151** | 2825 ± 164 (95M) |

- **The mixed run matches deathmatch-only at equal total steps** while giving deathmatch only 31% of them. Per deathmatch step, mixed training is about 3× as efficient. The other modes help deathmatch rather than crowd it out.
- **The smaller deathmatch advantage is not limiting it.** The suspicion from 06:50 is ruled out.
- **Transfer:** the live Elo of rl-dm's averaged policy in the modes it never trained on:

  | step | elimination | frontline |
  |---|---|---|
  | 30M | 1387 | 1439 |
  | 50M | 1625 | 1548 |
  | 75M | **1847** | **914** |

  - Elimination rose above the mixed run's best, about 1727. Its fighting is deathmatch's fighting.
  - Frontline collapsed after 50M. That is forgetting: zone and objective play decays without frontline training.
- **Deathmatch on kill share** reached 1221 at 75M. Again, it's compressed.
- **Live Elo** has moved to rl-kl.

## 08:05 — launch slip: rl-kl is deathmatch-only, so it became the clean outcome A/B; the mixed E3 is queued

- **The slip:** rl-dm was launched as `NAME=rl-dm PRESET=real_scale_deathmatch UPDATES=72 tmux new ...`, which started the box's tmux server. Those variables went into tmux's global environment, so the queued rl-kl session inherited PRESET=real_scale_deathmatch.
- **The fix:** the variables are cleared (`tmux set-environment -g -u`), and later queues set PRESET explicitly.
- **Kept:** rl-kl, deathmatch only, kill-lead outcome, 72 updates. Against rl-dm, which is deathmatch only with the kill-share outcome, at the same steps, only the outcome reward differs. That is the cleanest test of the reward shape.
  - Early sign: the outcome critic's explained variance is 0.161 at update 18; rl-dm had 0.135 at update 23.
- **Queued:** rl-mix-kl, mixed modes with the kill-lead outcome, 72 updates. It starts after rl-kl and compares with rl-v20 at 75M.

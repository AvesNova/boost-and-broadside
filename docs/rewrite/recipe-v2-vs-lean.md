# Training recipe: v2's release run against lean

2026-11-23. Report only; nothing here is a decision.

**Sources.** v2's release run (`rel-jujak-s1`, 2026-11-17) ran from branch
`feat/release-jujak-elim` at commit `4cfd1e86`, launched by `box/release_run.sh`: a chain of BC
for 100M environment steps, then PPO warm-started from the clone, with the preset
`release_jujak_elimination` (real-scale elimination alone, no fog, hemisphere spawn, Jujak only,
belief heads off). Every v2 `file:line` below is in that commit (`git show 4cfd1e86:<path>`).
The `src/bnb/` in this checkout (branch `core`) is an older v2 tree; its learner and reward
numbers are the same, but it lacks the release preset and `runs/averaging.py`, so its line
numbers differ. Lean is `lean/` at `HEAD` (`bd24365c`), except the league row, which describes the league change made the same day in the working copy.

The launch command (`box/release_run.sh:36-39`) sets `--environments 2048 --seed 0
--micro-batch-rows 480 --set host_store=false --set update_policies=all`, and BC
`lanes.environments=0`. The preset's settings are `src/bnb/runtime/train.py:102-120`. Neither
changes a PPO number, so the RL stage runs on `LearnerConfig`'s defaults
(`src/bnb/learn/ppo.py:172-212`), plus the warm start's entropy override (see the entropy row).
According to the 2026-11-17 notes the RL stage was extended on the box to about 200M RL steps
(a 301M chain). That extension is not recorded in the repository.

Both games tick at 30 Hz with one decision per tick (v2 `src/bnb/game/rules.py:129`; lean
`lean/learn/config.py:13-17`), so the per-tick discounts compare directly.

## Table

| Item | v2 (release run) | lean (today) | Mark |
|---|---|---|---|
| Reward levels | Five levels, each split into offence and defence parts (10 value parts): damage, kill_death, objective_progress, objective, outcome (`src/bnb/contracts/trajectory.py:12,18,28`). In elimination the two objective levels are always zero (`src/bnb/game/modes/phase.py:69-75`). | Four levels: outcome, kill, damage, shaping (`lean/learn/config.py:5-6`). | differs |
| Discount per level | damage 0.982, kill_death 0.990, objective_progress 0.995, objective 0.999, outcome 0.9997 (`ppo.py:172`) | outcome 0.998, kill 0.990, damage 0.982, shaping 0.951 (`config.py:20`) | kill and damage the same; outcome differs (horizon about 3,300 ticks, 111 s, against 500 ticks, 17 s) |
| GAE lambda | One per level: 0.81, 0.90, 0.95, 0.97, 0.97 (`ppo.py:173`). The outcome head's targets are lambda targets with the outcome's γ and λ (`ppo.py:546-553`, `models/yemong/critic.py:112-146`). | 0.95 on every level (`config.py:21`) | differs |
| Bootstrap at the segment's end | 128 lookahead ticks are played after the last shard and never trained on; the returns run through them before bootstrapping (`runtime/train.py:369`, `ppo.py:507-525`) | Bootstraps from the value at the end of the 128-tick segment (`collect.py:280`, `returns.py:16-40`) | differs |
| Damage level: weight and components | Weight 2.74 per 100 damage points (`learn/rewards.py:74,79`). Events in units of 1: taken −1, dealt +1, friendly fire −1 (and +2 to the other team), recharge +1, charge-back −1 to the enemy attacker, friendly-fire refund +1, boundary and unattributed healing to the other team (`rewards.py:226-270`). Zero-sum, dealt equal to taken. | Dealt +0.548, taken −0.274, friendly −0.548 per `damage_unit` 1.0 (`config.py:50-53`). `damage_unit` is unmeasured against the game's units (NOTES.md). 0.5 own term + 0.5 team mean (`config.py:54`, `rewards.py:84-93`). No recharge charge-back. Not zero-sum. | differs |
| Kill/death level | Weight 2.83 (`rewards.py:75`). Death −1. Kill +1, shared over the attackers by the final tick's hits (half) and by outstanding damage (half), so assists earn a share. An ally kill blames the shooter and pays the other team 2. A boundary death pays the other team (`rewards.py:195-223`). | Kill +0.283 and death −0.283, killer only; a friendly kill −0.283 (`config.py:48-49`, `rewards.py:74-82`). Same 0.5 own + 0.5 team mean. No credit for assists or boundary deaths. | differs (the kill:outcome ratio is the same, 0.283) |
| Objective levels | objective_progress 20.0, objective 20.0 (`rewards.py:76-77`); zero in the release game | none (no objective modes) | lean lacks it (no effect in the release game) |
| Outcome level | Weight 10.0 (`rewards.py:78`). Win +1, loss −1 (offence: win; defence: −loss), paid on the final tick to the living and the dead (`rewards.py:184-192`). | `win_reward` 1.0, ±1 to every ship that existed (`config.py:47`, `rewards.py:69-72`) | same shape; v2's weights are all ×10 (`rewards.py:60-66`), which the return scale cancels |
| Shaping | none | facing 0.09, closing 0.08 toward the nearest enemy; scale 1 to 100M steps, then decaying to 0.05 at 400M (`config.py:55-60`, `rewards.py:95-98`) | v2 lacks it |
| Team mixing | `team_spirit` 0 on every level, so a ship's advantage is its own (`ppo.py:174`, `learn/returns.py:92-112`) | Mixing is in the reward instead: `individual_share` 0.5 (`config.py:54`) | differs |
| How levels combine; advantage normalisation | A = Σ over the 10 parts of w·A_part, with w = 1 because `offence_premium` stays 1 (`ppo.py:175`, `returns.py:28-32`). Divided by the return scale: an EMA (decay 0.99) of the 5th–95th percentile gap of the summed λ-returns, floor 1.0 (`ppo.py:176-177`, `critic.py:147-170`, `ppo.py:1173`). Mean not subtracted (`losses.py:36-51`). Computed once per update. | Σ over the levels with weight 1, divided by the std over the update's valued tokens (`returns.py:43-61`, `config.py:22`). Mean not subtracted. Recomputed from the current critic at every epoch (`update.py:81-83`). | differs |
| PPO clip | 0.2, symmetric (`trust_region` "clip"; `ppo.py:178-179`) | 0.2 (`config.py:26`) | same |
| Policy term coefficient | `policy` 27.0 on the surrogate (`ppo.py:182`, `ppo.py:720`) | 1 (implicit; `update.py:113-119`) | differs (this sets the balance against every other term) |
| Epochs | 4 (`ppo.py:208`) | 4 (`config.py:27`) | same |
| Update size | 2,048 environments × 4 shards × 128 ticks = 1,048,576 environment ticks (`cli.py:111,113`, `runtime/train.py:365-368`). Learner views: self-play trains both seats, the league one, so 1,024×2 + 1,024 = 3,072 views per shard and 12,288 per update (`runtime/lineup.py:34-40`, `train.py:122,426`). One view is 128 ticks × 5 ships (league views 3–5 under the release team sizes). | 256 environments × 128 ticks = 32,768 environment ticks (`lean/train.py:21`, `config.py:29`). One team per environment trains: 4 ships at 4v4 (`config.py:63-65`). | differs (32× more per update) |
| Minibatches and minibatch size | 32 minibatches (`ppo.py:209`) of 384 views = 245,760 ship-steps. Each minibatch is run as micro-batches of 480 ship rows (96 views; `box/release_run.sh:38`, default 160 at `ppo.py:210`), with the gradient accumulated, so memory follows the micro-batch. Up to 128 optimizer steps per update. | 4 minibatches (`config.py:28`) of 64 whole environments = 32,768 learner ship-steps. No micro-batching. 16 optimizer steps per update. | differs |
| Learning rate and schedule | 3e-4 (`ppo.py:204`). Linear warm-up from 1e-7 over 10M environment steps, about 10 updates (`ppo.py:205-206,302-305`). No decay. The RL stage starts a fresh optimizer with its own warm-up (`cli.py:69-70,87-90`). | 3e-4. Linear warm-up over 10 updates (327,680 environment ticks), then constant. No decay (`config.py:30-31,91-93`). | same rate and both constant after warm-up; the warm-up is about 30× longer in steps in v2 |
| Entropy coefficient | Default −0.0081 on each of the three factor entropies (power, stick given power, shoot given both), which together weigh the joint entropy (`ppo.py:183-185`). The release run's RL stage is a warm start, which sets all three to `WARM_START_ENTROPY` = 0.0 (`ppo.py:263-268`, `cli.py:1072-1082,1465`). No schedule. Entropy floor off (`ppo.py:186`). | 0.003 on the joint entropy, constant (`config.py:34`, `update.py:99,116`) | differs (the release RL stage had no entropy bonus) |
| Value coefficient | 0.007 (`ppo.py:188`) on the **sum** of nine head losses: eight per-ship part distributions plus the outcome head (`ppo.py:795`, `losses.py:106-132`) | 0.05 on the **mean** cross-entropy over the four levels (`config.py:35`, `update.py:102-104`) | differs |
| Value loss form and targets | Categorical cross-entropy with HL-Gauss targets (σ 0.75 bins; `models/yemong/codes.py:19`). 77 bins evenly spaced in symlog over ±3 (±19 reward units), plus 2 padding bins each side, 81 in all (`models/yemong/model.py:291-295`, `train.py:394`, `critic.py:29-52`). Targets are the realised λ-returns of each part in reward units, not normalised, from the rollout's values. The outcome is a separate 4-class head (win, tie, loss, unresolved) trained on λ-targets per environment-step. Tokens: ships alive or waiting to respawn. | Categorical cross-entropy with HL-Gauss targets (σ 0.75 bins), 64 linear bins per level (`lean/model/policy.py:58`) over outermost centres ±1.5 / ±2 / ±2 / ±4 (`config.py:42-44`). Targets are the λ-returns in reward units, recomputed each epoch; targets outside the support are clamped. Outcome is a scalar level like the others. Tokens: living own ships. | differs |
| Gradient clip | Global norm 1.0 (`ppo.py:207`, `ppo.py:1494`). A step with a non-finite norm is skipped. | 1.0 (`config.py:33`, `update.py:122`) | same |
| Adam | `torch.optim.Adam`: lr only, default betas (0.9, 0.999), default eps 1e-8, no weight decay, `fused=True` on CUDA (`ppo.py:437-439`) | `torch.optim.Adam`: eps 1e-5, default betas, no weight decay, not fused (`update.py:12-14`, `config.py:32`) | differs (eps, fused) |
| KL early stop | `target_kl` 0.02: after each epoch but the last, stop if the epoch's mean approximate KL exceeds it (`ppo.py:211,1519-1526`) | none; KL is reported only (`update.py:129`) | lean lacks it |
| Old log-probabilities | The rollout's (`ppo.py:212`) | The rollout's (`update.py:94`) | same |
| Weight averaging | Two in-run averages, folded after every update (`league/pool.py:107-158`): an EMA with decay 0.95 per update from the start (half-life about 13.5 updates, about 14M steps), and a uniform average from the end of warm-up, 10M steps (`runtime/opponents.py:188-190`, `train.py:521-524`). Both are league entries and both are rated in the rating lanes against Imugi (`runtime/lanes.py:94-102`). BC hands the EMA to RL (`runs/session.py:43,46,207-224`). With `update_policies=all`, every update's live, uniform and exponential files are saved, and `bnb average` builds further averages after training (`runs/averaging.py:1-40`). Which file was shipped as the release agent is not in the code. | none (NOTES.md "Departures") | lean lacks it |
| BC / distillation | A separate BC stage before RL, 100M environment steps (`learn/bc.py:414`, `cli.py:85-96`). Cross-entropy to Imugi's full action distribution (temperature 1, floor 0), coefficient 0.025 in place of the surrogate. No entropy term, no KL stop. 256 minibatches. BC's own discounts (0.982, 0.990, 0.995, 0.999, 1.0) and traces (0.98, 0.98, 0.99, 0.99, 1.0) (`bc.py:44-69,72-81`). BC lineup: 0.5 self-play, 0.5 against Imugi (`runtime/cloning.py:32`). RL has **no** BC term; it starts from the clone's EMA weights and return scale. | No BC stage. A BC term inside PPO: cross-entropy to the scripted agent's sampled action, coefficient 2.0 × max(0, 1 − win_rate / 0.45), where win_rate was an EMA (momentum 0.9) of the win rate against scripted at HEAD (`config.py:37-38,72,75-77`, `update.py:100,118`, `lean/train.py:59-62`); since the league change it is the league's decaying tally against the scripted member (0 until a game is played). | differs |
| Auxiliary losses | Release: next-state, correction and enemy-action heads off (`train.py:114-118`). Kept: density 0.021, shot trajectory 0.007, shot outcome 0.019, geometry 0.007 (`ppo.py:196-198,203`; `train.py:386` turns the geometry head on; shot head "trajectory", `model.py:334`). Defaults when the belief is on: next_state 0.0033, correction 0.0033, enemy_action 0.11 (`ppo.py:191-195`). | Next-state squared error, coefficient 0.1 (`config.py:36`, `update.py:106-117`). No others. | differs |
| Opponent mix / league | Half of each batch is self-play with both seats trained; half plays the league's seat (`train.py:122`). The league has 4 network slots, and one slot is redrawn every 2 updates. Sampling is "hard" PFSP: weight (1 − p)² on the predicted chance p that the live policy wins, no floor (`opponents.py:181-187`). Entries: snapshots every 5M steps to 50M, then at intervals that double as the steps double, all kept (`runtime/snapshots.py:104`); the two averages; full-strength Imugi; six cheese agents (`controllers/cheese.py:276`, `league/pool.py:1-10`). Release team sizes in the league pool: 5v5 at 0.5, 4v5 and 3v5 at 0.25 each, the learner always the smaller side (`train.py:102-108`). Also 1,024 rating-lane environments, not trained on (`runtime/lanes.py:121`). | (Changed 2026-11-23, working copy, the author's design.) Half the environments self-play with both teams trained; half are league games with the live side trained. The league is the scripted agent plus up to 8 snapshots (one every 10 updates, oldest replaced). Opponent drawn per environment at each reset: 7/8 by (1 − p)², 1/8 uniform, p from AlphaStar-style tallies decaying 0.99 per game, 0.5 unplayed (`lean/learn/pool.py`, `collect.py`). No averages, no cheese agents, no rating lanes, no lookahead. | differs |

## Memory note

With the new model, lean's default size runs out of 8 GB at 4 minibatches. It fits at 8
minibatches (4.8 GB) and at 16 (2.5 GB). That makes lean's minibatch 32 environments × 128 ticks
× 4 ships = 16,384 ship-steps at 8 minibatches, and 8,192 at 16.

v2 used 32 minibatches of 384 views (245,760 ship-steps each) over a 12,288-view update. It
bounded memory separately, with micro-batches of 480 ship rows (96 views) and gradient
accumulation, so its minibatch count was not a memory setting. In lean the minibatch count is
currently the only memory lever. Moving to 8 or 16 minibatches also doubles or quadruples the
optimizer steps per update (to 32 or 64), with no KL stop to bound the drift.

Since the league change, lean trains 1.5 seats per environment (self-play environments train
both teams): 384 seats at 256 environments, so a minibatch holds 1.5 times the rows it did
when the figures above were measured. Expect 8 minibatches to need about 1.5 x 4.8 GB (an
estimate, not measured); 12 or 16 minibatches keep the measured row counts.

## Main differences

1. **BC is a stage in v2 and an auxiliary term in lean.** v2 cloned Imugi's full distribution
   for 100M steps, then started PPO from the clone's EMA weights with no BC term and no entropy
   bonus. Lean starts PPO from random weights, with a gated cross-entropy to the scripted
   agent's *sampled* action.
2. **Batch scale and update control.** v2's update is 1.05M environment ticks (12,288 views),
   with 32 minibatches of 384 views and a KL early stop at 0.02. Lean's update is 32,768
   environment ticks, with 4 minibatches of 64 environments and no early stop. v2 takes about
   8,192 environment ticks per optimizer step; lean takes 2,048.
3. **Outcome horizon.** v2's outcome γ is 0.9997 with λ 0.97 (about 111 s), plus a separate
   4-class outcome head on λ-targets. Lean's is 0.998 with λ 0.95 (about 17 s) on a scalar
   level. The damage and kill discounts are the same. v2's λ varies by level (0.81 to 0.97);
   lean's is 0.95 throughout.
4. **Advantage scaling and term balance.** v2 divides the summed advantage by a slow EMA of the
   returns' 5–95% gap (floor 1), computed once per update. Lean divides by the advantage's own
   std, recomputed every epoch. v2 also weights the surrogate by 27 against value 0.007 (summed
   over 9 heads). Lean weights it by 1 against value 0.05 (averaged over 4 levels). So the
   critic's and the auxiliary heads' relative pull on the trunk differs by more than an order
   of magnitude. How much depends on the two normalisations, which were not measured here.
5. **Critic support.** v2 uses 81 symlog bins reaching ±19 reward units, per offence/defence
   part. Lean uses 64 linear bins over narrow, partly unverified supports (damage ±2 depends on
   the unmeasured `damage_unit`), and clamps targets beyond them.
6. **Reward accounting.** v2's levels are zero-sum, with a damage ledger: healing charges back
   the attacker, kills are shared by damage (assists), and boundary deaths and friendly fire pay
   the other team. There is no shaping and no team mixing. Lean pays dealt damage twice the
   taken charge, gives kills to the killer only, mixes half the team mean into kill and damage,
   and adds facing and closing shaping.
7. **Weight averaging.** v2 keeps an EMA (0.95 per update) and a uniform average (from 10M
   steps) during the run, uses them as league opponents, rates them, and hands the EMA from BC
   to RL. Lean keeps no average.
8. **Opponents.** v2 uses 50% self-play with both seats trained, plus a PFSP league of
   snapshots, the averages, Imugi and six cheese agents. Lean now has the same 50/50
   split with both self-play seats trained and (1 − p)² sampling with a 1/8 uniform floor over
   the scripted agent and up to 8 snapshots; it has no averages, cheese agents or rating lanes.
9. **Smaller items.** Adam eps is 1e-8 and fused in v2, 1e-5 in lean. v2's warm-up is 10M
   environment steps, lean's 10 updates (0.33M ticks). v2 plays a 128-tick lookahead so the
   returns are not cut at the segment's end. Lean has a next-state loss at 0.1, which the
   release run had turned off.

## Not determined

- Which of the release run's policy files (live, uniform or exponential average, at which step)
  was shipped or presented as the release agent. The code keeps all three; the choice was made
  in evaluation, outside the repository.
- The exact RL budget after the on-box extension (the notes say about 200M RL steps). The
  committed script says a 200M chain.
- BC's minibatch size in views. `bc.py:57-58` says 32 views of an 8,192-view batch. The lineup
  code gives 12,288 views (self-play stores both seats), which would make 48 views per BC
  minibatch. I did not run it to check.
- The effective size of each loss term in either system: the coefficients are comparable only
  together with the advantage normalisations, and no gradient shares were measured for lean.

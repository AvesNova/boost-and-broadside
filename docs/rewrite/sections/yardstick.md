# Yardstick (gate G3): v1's recipe, v2's comparable numbers, proposed requirement numbers

Collected 2026-11-22 from existing records only. Nothing was trained or run on a GPU; the
only computation is reading stored JSON and JSONL files from `origin/main`.

**How to read the sources.**
- `OM:` means the git path on `origin/main` (read with `git show origin/main:<path>`).
  `OM:src/` is short for `OM:src/boost_and_broadside/`.
- `R/` means `docs/rewrite/` in the working tree (v2's notes). `S/` means
  `docs/rewrite/sections/`.
- **[record]** a file states it. **[code]** read from code, not run. **[derived]** my
  arithmetic on recorded numbers, shown. **[inferred]** my reading; not stated anywhere.
- `cfg731` is `OM:checkpoints/wild-universe-731/config.json`, the stored configuration
  of run 731. `hist731` is
  `OM:checkpoints/wild-universe-731/artifacts/wandb-export/20260906T072950Z-b9848587/history.jsonl`
  (674 rows, one per update). `fan` is `OM:docs/results/v1-731/fan_731_n{4,8,16,32}.jsonl`.
  `xover` is `OM:checkpoints/wild-universe-731/artifacts/figures/crossover.json`.

**Limits of this collection.**
- The checkpoints on `origin/main` are git-LFS pointers, so no weights were read. The
  parameter count is inferred from a file size.
- v2's run directories are not in the repository; v2 numbers come from the notes.
- "Step" means one environment advancing one decision (1/30 s of game time) in both
  versions. v1 has 8 ships per environment, v2 has 10.

---

## Summary

| Requirement | Proposed number | Rests on | Confidence |
|---|---|---|---|
| Clone | Score at least 0.40 against the scripted opponent (wins plus half the draws, lower 2-SE bound over at least 1,000 games) within 85M steps and 5.6 laptop GPU-hours | v2 bc-v20 (84.9M steps); v2 laptop BC speed | Steps: recorded. Hours: derived, never run on the laptop |
| RL at the training size | Break-even at least 1.5 (at least 50% wins against 1.5N scripted ships, ties as losses, at least 1,024 games) within 90M RL steps and 9 laptop GPU-hours after the hand-off; total from random weights reported against v1's 120M steps and 12.3 h | v1 run 731 ladder checkpoints | Recorded as a bracket (79.6M to 120.4M steps) |
| Size curve | v1's protocol at 1, 2, 4, 8, 16, 32, 64; ties as non-wins; world fixed; pass = wins at least half its games at N against N at every size, with the ratio reported | v1 `crossover.py` | Protocol: recorded. See Part 3 for what "above parity" has to mean at 1 ship |

**Three things the reader should know before using these.**
1. **v1's run 731 had no separate BC stage.** It trained from random weights with a BC
   term inside PPO, switched off by the win rate against the scripted agent. The "clone"
   requirement has no v1 number of the same kind.
2. **No v2 run recorded a break-even of 1.5.** The one figure (7.46 scripted for 5
   network ships, 1.49) is in deathmatch with respawn, in the older 2,600 m game, after
   301M steps on a rented 3090. No v2 break-even exists in elimination.
3. **v1's game is not "elimination with the shrink off" on a v2 map.** It is a 1024 x 1024
   torus with both teams spawned uniformly at random, interleaved, a 17 s cap, and a much
   simpler scripted opponent. Part 4 ranks the differences.

---

## Part 1. The v1 recipe for run 731

### 1.1 Provenance of the run

| Fact | Value | Source |
|---|---|---|
| Command | `bnb train --profile rl --gradient-diagnostics reward_full --gradient-diagnostics-interval 1`; no `--pretrain-from` | [record] `.../wandb-export/.../files/wandb-metadata.json` `args` |
| Commit | `3d1a2e55`, clean | [record] `OM:checkpoints/wild-universe-731/run.json` |
| Started from | Random weights. Live rating 119 at update 10 | [record] `.../files/output.log` line 15; no pretrain argument |
| Steps, updates | 671,846,400 steps, 674 updates, status `interrupted` (target 1,000M) | [record] `run.json`; `OM:docs/evaluation.md:129-131` |
| Wall time | 253,060 s = 70.3 h | [record] `run.json` `elapsed_seconds` |
| GPU | One RTX 4070 Laptop | [record] `OM:docs/evaluation.md:131` |
| Steps per second | 2,655 over the whole run; 21,239 ship-tokens/s | [record] `OM:docs/evaluation.md:131-134` |

The current `OM:src/profiles/rl.py` differs from what ran (it now sets 500M steps,
`rl.py:74`). `cfg731` is what ran, as `OM:docs/training.md:30-31` also says.

### 1.2 The game as trained

| Item | Value | Source |
|---|---|---|
| Mode | Two teams; the episode ends when one team has no ship alive. No respawn exists in the code | [code] `OM:src/env/physics.py:981-987` |
| Time limit | 1,024 physics ticks = 17.1 s = 512 decisions; then truncated | [record] `cfg731` `max_episode_steps`; [code] `OM:src/env/env.py:410-414` |
| Ties | Reaching the cap with both teams alive, or both teams dying on one tick, is a tie. Evaluation counts a tie as a non-win | [code] `OM:src/evaluation/match.py:237-243`; `OM:src/modes/crossover.py:148` |
| World | 1024 x 1024, toroidal (wraps on both axes), no boundary | [record] `cfg731` `world_size`; `OM:docs/environment.md:3-4` |
| Tick rate | Physics 60 Hz (`dt` 1/60) | [record] `cfg731` `dt` |
| Action repeat | 2, so decisions at 30 Hz. Honoured by the environment's `step`, so evaluation uses it too | [record] `cfg731` `action_repeat`; [code] `OM:src/env/env.py:325-333` |
| Action latency | One decision. The observation carries each ship's action about to be applied, and every ship can read every other ship's, opponents included | [record] `OM:docs/training.md:99-116` |
| Team sizes in training | 8 ships, always 4 against 4. One scale; no size is drawn | [record] `cfg731` `scales` (one entry, `num_ships` 8); [code] `OM:src/env/env.py:155-156` |
| Spawn positions | Each ship uniform over the whole world, heading uniform, **teams interleaved**: team membership is a random permutation of the slots | [code] `OM:src/env/env.py:170-178`, `252-264` |
| Spawn speed | 100 (proper speed), fixed (`random_speed` false) | [record] `cfg731` |
| Spawn resources | Health and power each uniform in 75% to 100% of maximum, cooldown uniform in 0 to 0.1 s (`spawn_resource_spread` 0.25) | [record] `cfg731`; [code] `OM:src/env/env.py:231-242` |
| Ship type | One type | [code] `OM:src/config/core.py:31-111` |
| Speeds | Spawn 100; maximum 180; minimum 1 (stall); thrust 8 base, 80 boost, -80 reverse | [record] `cfg731` `ship_config` |
| Turning | Normal 5 degrees, sharp 15 degrees (sideslip, with lift and drag); air brake | [record] `cfg731`; `OM:docs/environment.md:44-46` |
| Hit points | Health 100; power 100, regenerating 10 per s | [record] `cfg731` |
| Weapon | Bullets: speed 500, damage 10, lifetime 1.0 s, cooldown 0.1 s, energy cost 2, spread 12, 10 slots per ship | [record] `cfg731` |
| Frontal armour | A head-on hit lands 30% of its damage; fades with hit angle (`bullet_min_damage_frac` 0.3) | [record] `cfg731`; [code] `OM:src/env/physics.py:898-907` |
| Friendly fire | On. No ship-to-ship collision | [record] `OM:docs/environment.md:206` |
| Fields | 4 refractive fields per map, radius 30 to 490, interface damage 10 (standard) or 20 (severe), nesting probability 0.35 | [record] `cfg731` `num_fields`, `field_map`, `ship_config` |
| Map randomisation | Field maps regenerated every rollout from a 512-map cache, translated randomly per reset | [record] `OM:docs/training.md:438-443`; [code] `OM:src/env/env.py:180-196` |
| Fog | None. Every ship sees the full state | [record] `OM:docs/architecture.md:3-4` |

### 1.3 Observation and actions

| Item | Value | Source |
|---|---|---|
| Per ship | Position x, y; velocity; attitude; angular velocity; health; power; cooldown; team one-hot (3); alive; previous (pending) action one-hots; radius; local log index; local index gradient | [record] `OM:docs/architecture.md:83-98`; [code] `OM:src/train/rl/features.py:770-931` |
| Position encoding | Absolute x and y as 4-frequency Fourier features over the world's period. Nothing relative, no rotary, no random offset | [code] `OM:src/train/rl/features.py:770-783`; `OM:src/models/yemong/attention.py:132-147` |
| Fields | 4 extra tokens, always alive, team id 2 | [record] `OM:docs/architecture.md:65-68` |
| Bullets | Every ring-buffer slot (10 per ship) as a key/value-only token: position on the same Fourier basis, velocity, damage, lifetime, shooter's team | [record] `OM:docs/architecture.md:110-119`, `154-171`; `cfg731` `n_bullet_cross_per_block` 1 |
| Perspective | The policy acts as team 0; team 1 gets a team-flipped view (`ego_pass`) | [record] `cfg731` `paradigm`; `OM:docs/training.md:139-153` |
| Actions | Per ship, three factored categoricals: power 3, turn 7, shoot 2 (12 logits). Joint log-probability is the sum | [code] `OM:src/constants.py:38-48`; `OM:docs/architecture.md:209-215` |

### 1.4 The model

| Item | Value | Source |
|---|---|---|
| Width, heads | 128, 4 heads | [record] `cfg731` `model_config` |
| Trunk | 2 blocks, each 2 spatial sublayers then 1 temporal: `S S T | S S T` | [record] `cfg731`; `OM:docs/architecture.md:41-44` |
| Encoder | 2-layer MLP (to 256, then 128) with RMSNorm, shared by ships and fields | [code] `OM:src/models/yemong/encoder.py:82-90` |
| Spatial sublayer | Pre-norm self-attention over ships and fields, then a gated MLP (4x). The first spatial sublayer of each block also cross-attends to bullets, as a separate softmax summed in | [code] `OM:src/models/yemong/attention.py:40-99`, `149-166` |
| Dead ships in attention | Masked out as keys by an additive bias. Still computed as queries. No null key. The alive flag is also a token feature | [code] `OM:src/models/yemong/attention.py:142-147`; `OM:src/train/rl/features.py:853` |
| No active bullets | The bullet contribution is multiplied by zero | [code] `OM:src/models/yemong/attention.py:158-165` |
| Temporal sublayer | Griffin block per ship: causal depthwise convolution (kernel 4), RG-LRU, gated MLP. Fields take a non-recurrent path through the same weights | [code] `OM:src/models/yemong/griffin.py:22`, `143-167`, `222-246`, `359-368` |
| Action head | MLP 128 to 256 to 12 | [code] `OM:src/models/yemong/policy.py:185-190` |
| Value head | One scalar per ship per reward component (16 active). Win components read a team-pooled embedding (attention pooling over the team's live ships); the others read the ship's token | [code] `OM:src/models/yemong/policy.py:78-113`, `193-208`; `OM:docs/training.md:307-308` |
| Auxiliary head | Next state per ship (phase or additive deltas) | [code] `OM:src/models/yemong/policy.py:57-74`, `209` |
| Parameter count | **Not recorded.** A weights-only ladder snapshot is 7,965,377 bytes, which bounds it at about 2.0M float32 parameters | [inferred] LFS pointer of `OM:checkpoints/wild-universe-731/ladder_step_000247836672.pt` |
| Precision | bf16 autocast for the forward | [code] `OM:src/train/rl/ppo.py:1598` |

### 1.5 Training

**PPO settings** (all [record] `cfg731` unless marked).

| Setting | Value |
|---|---|
| Environments | 2,592, of which 1,296 self-play and 1,296 league ([derived] from `league_fraction` 0.5) |
| Rollout | 128 decisions, 3 rollouts per update: 995,328 steps per update |
| Epochs | 4 (mean completed 3.96, [derived] `hist731` `train/epochs_completed`) |
| Minibatches | 32 |
| Clip | 0.15 |
| KL early stop | 0.1, tightened to 0.02 once the win rate against the scripted agent passes 0.8 |
| Entropy coefficient | 0.005, constant, on each of the three action factors |
| Learning rate | 1e-7 rising linearly to 4.5e-4 at 5M steps; held to 100M; exponential decay to 1.5e-4 at 500M; then held |
| Optimiser | Adam, eps 1e-5, other values PyTorch defaults ([code] `OM:src/train/rl/ppo.py:480-482`) |
| Gradient clip | 1.0 |
| Discount, per decision | Win 0.998; kill and death 0.990; damage 0.982; shaping 0.951 (per tick 0.999, 0.995, 0.991, 0.975: `OM:src/config/defaults.py:140-171`) |
| GAE lambda, per decision | Win 0.941; death 0.903; kill shot 0.757; kill assist 0.941; damage 0.81; shaping 0.64 |
| Advantage | Per-component GAE, scaled per component by a running RMS, combined through the weights, then divided by the whole buffer's RMS ([code] `OM:src/train/rl/ppo.py:1622`; `OM:docs/training.md:176-186`) |
| Value loss | Squared error to 1 normalised unit, linear beyond (`value_huber_delta` 1.0); returns normalised per component ([code] `OM:src/train/rl/ppo.py:1633`) |
| Value coefficient | 1.0 |
| Next-state coefficient | 0.2, plus a windowed drift loss at 0.1 |
| Weight averaging | A uniform running mean of live snapshots, started when the BC term ends ([record] `OM:docs/training.md:483-484`) |

**Rewards** ([record] `cfg731` `rewards`; table at `OM:docs/training.md:211-229`).

| Tier | Components and weights |
|---|---|
| Outcome | Win 1.0 to each surviving teammate; the enemy's win seen as -1 |
| Kill and death | Death -0.283 (combat or field); kill shot, kill assist +0.283 each; friendly kill -0.283 each; enemy field death +0.283 |
| Damage | Taken -0.274; dealt to enemies +0.548; dealt to allies -0.548; enemy field damage +0.548 |
| Shaping | Facing 0.09, closing speed 0.08, both scaled by 1.0 until 100M steps, then decaying exponentially to 0.05 at 400M |
| Normalisation | Every reward divided by the total ship count ([code] `OM:src/env/wrapper.py:382-386`) |

The side that causes an event is paid twice what the victim is charged
(`kill_payout_ratio` and `damage_payout_ratio` 2.0).

**BC.** There was no BC stage. A cross-entropy term toward the scripted agent's action
probabilities ran inside PPO at coefficient 2.0, multiplied by
`max(0, 1 - win_rate / 0.45)`, and switched off for good after the win rate held at 0.45
for 3 updates ([record] `cfg731` `behavior_cloning_coef`, `bc_winrate_target`; [code]
`OM:src/train/rl/ppo.py:116`, `1295`). In `hist731` the coefficient first reads 0 at
update 18 (17.9M steps, 1.6 h) and stays 0 from update 33 (32.8M steps, 3.1 h).

**League and self-play.**

| Item | Value | Source |
|---|---|---|
| Split | Half the environments are self-play (live weights on both sides, both sides train); half are league | [record] `cfg731` `league_fraction` 0.5; `OM:docs/training.md:473-493` |
| League slots | 4 contiguous slots of 324 environments; each draws one opponent per rollout | [record] `cfg731` `league_slots`; [code] `OM:src/train/rl/opponents.py:198-232` |
| Pool | The scripted agent (rated 1000), nine semi-random rungs (rated 1000 x p), the averaged policy, and frozen checkpoints saved at every 200 points of live rating. The random agent is excluded | [record] `OM:checkpoints/wild-universe-731/roster.json`; `cfg731` `elo_milestone_gap` |
| Sampling | Weight `exp(-abs(opponent rating - live rating) / 200)` | [code] `OM:src/train/rl/roster.py:345-375` |
| Frozen checkpoints | Nine, at 10.9M, 12.9M, 14.9M, 24.9M, 55.7M, 79.6M, 120.4M, 153.3M and 247.8M steps | [record] `roster.json` |
| Share against the scripted agent | Not logged. By construction: at step 0 every league draw is the scripted agent, so 50% of games; it falls as the live rating leaves 1000 | [inferred] from the sampling rule |

**Docs and code disagree here.** `OM:docs/training.md:495-497` says the semi-random rungs
are league opponents. In the training rollout a drawn rung has no policy, and a slot with
no policy plays the full scripted agent's action
(`OM:src/train/rl/opponents.py:187-196`, `281-320`, `340-362`). The blend with random
actions exists only in the evaluation battery (`OM:src/train/rl/elo_eval.py:92-109`). So
in training a rung draw is a scripted-agent game. [code], not run.

**In-training evaluation.** Five slots of 512 environments advance with every rollout
step (2,560 evaluation environments beside 2,592 training ones): live against a fixed
anchor, a floating checkpoint, the scripted agent, the averaged policy, and the floating
checkpoint against the anchor. The win rate against the scripted agent is a 100-game
window. ([record] `OM:src/config/defaults.py:34-43`; `OM:docs/training.md:513-522`.)
The reported steps per second count training steps only.

### 1.6 Budget and throughput

| Item | Value | Source |
|---|---|---|
| Total | 671.8M steps in 70.3 h | [record] `run.json` |
| Steps per second | 2,655 whole run; 2,670 median of `train/sps` | [record] `OM:docs/evaluation.md:131`; [derived] `hist731` |
| Early speed | 3,808 at update 10, 3,051 at update 30 | [record] `output.log` lines 15, 47 |
| Known overheads | Full gradient diagnostics every update (1.17x on the update phase); 4 field tokens; the evaluation battery | [record] `OM:docs/evaluation.md:135-138`; `OM:docs/training.md:860-865` |
| Pure game step | 1.19M environment steps/s with 4 fields, 9.49M with none (4,096 environments, 8 ships, no policy) | [record] `OM:docs/environment.md:221-226` |

### 1.7 The learning curve

**In-training gauge** (4v4 win rate against the scripted agent, 100-game window, ties as
non-wins; first update at or above each level; [derived] from `hist731`
`eval/win_rate_vs_scripted`, `_runtime`).

| Level | Steps | Hours |
|---|---|---|
| 0.45 | 17.9M | 1.6 |
| 0.50 | 34.8M | 3.3 |
| 0.75 | 46.8M | 4.6 |
| 0.90 | 64.7M | 6.5 |
| 0.95 | 86.6M | 8.8 |
| 0.99 | 114.5M | 11.7 |

The last update below 0.50 is at 36.8M. `OM:docs/evaluation.md:392-396` states the 95%
and 99% rows.

**Frozen checkpoints, 4 network ships against 4 to 8 scripted** (win rate, ties as
non-wins, 1,024 games a point; final live and averaged 4,096; [record] `fan` n4; hours
[derived] from `hist731`).

| Checkpoint | Hours | 4v4 | 4v5 | 4v6 | 4v7 | 4v8 |
|---|---|---|---|---|---|---|
| 10.9M | 0.9 | 0.113 | 0.039 | 0.015 | 0.007 | 0.000 |
| 14.9M | 1.3 | 0.393 | 0.122 | 0.037 | 0.018 | 0.005 |
| 24.9M | 2.3 | 0.462 | 0.186 | 0.063 | 0.016 | 0.009 |
| 55.7M | 5.5 | 0.774 | 0.458 | 0.236 | 0.104 | 0.032 |
| 79.6M | 8.0 | 0.909 | 0.689 | 0.406 | 0.200 | 0.094 |
| 120.4M | 12.3 | 0.966 | 0.828 | 0.601 | 0.344 | 0.180 |
| 153.3M | 15.8 | 0.985 | 0.923 | 0.729 | 0.471 | 0.267 |
| 247.8M | 25.7 | 0.992 | 0.954 | 0.790 | 0.554 | 0.354 |
| 671.8M live | 70.3 | 0.996 | 0.938 | 0.782 | 0.550 | 0.353 |
| 671.8M averaged | 70.3 | 0.998 | 0.966 | 0.833 | 0.616 | 0.409 |

**Time to strength at the training size** ([derived] from the table; each is a bracket
between two checkpoints, not a point).

| Milestone | First checkpoint that has it | Previous checkpoint |
|---|---|---|
| Even numbers (4v4 at least 50%) | 55.7M, 5.5 h | 24.9M (0.462) |
| Break-even 1.25 (4v5) | 79.6M, 8.0 h | 55.7M (0.458) |
| **Break-even 1.5 (4v6)** | **120.4M, 12.3 h** | 79.6M (0.406) |
| Break-even 1.75 (4v7) | 247.8M, 25.7 h | 153.3M (0.471) |

**Other sizes at the frozen checkpoints** (win rate at N against N; [record] `fan`).

| Checkpoint | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| 24.9M | 0.462 | 0.390 | 0.288 | 0.090 |
| 55.7M | 0.774 | 0.754 | 0.717 | 0.577 |
| 79.6M | 0.909 | 0.935 | 0.947 | 0.923 |
| 120.4M | 0.966 | 0.982 | 0.997 | 0.990 |

Even numbers hold at every size from 4 to 32 by 55.7M steps (5.5 h). The 247.8M
checkpoint matches the final live checkpoint at every size and ratio in `fan` within a
few points, so the size curve was complete by about 250M steps (25.7 h); `OM:docs/evaluation.md:45`
says the same of 4 ships.

### 1.8 The size-curve protocol (`crossover.py`)

| Item | Value | Source |
|---|---|---|
| Checkpoint | The final **live** one, `step_000671846400.pt`. The averaged weights were not swept | [record] artifact `recipe.subjects.trained`; `OM:docs/evaluation.md:160`, `482-484` |
| Sides | Network always team 0, scripted always team 1; no side swap | [code] `OM:src/modes/crossover.py:3-4`; `OM:src/evaluation/match.py:217-222` |
| Sizes | Network 1, 2, 4, 8, 16, 32, 64 | [record] artifact `parameters.trained_counts` |
| Games per matchup | `max(48, min(256, 4,000,000 // n^2))`, n the total ships. 256 up to 125 ships; 204 at 64 v 76 | [code] `OM:src/modes/crossover.py:38-48`; [record] `xover` |
| Ties | Count against the network | [code] `OM:src/modes/crossover.py:148` |
| Search | Integer scripted counts. The first size brackets by doubling and bisects; later sizes start at the previous size's crossover and walk one ship at a time. Assumes the win rate falls as scripted ships are added | [code] `OM:src/modes/crossover.py:237-288` |
| Reported | `beats_up_to` = largest scripted count with at least 50% wins; `crossover` = the next count. The ratio is `beats_up_to / N` | [code] `OM:src/modes/crossover.py:181-192` |
| World | **Fixed at 1024 x 1024 at every size.** Only `num_ships` is replaced | [code] `OM:src/evaluation/match.py:206-212`; `OM:src/modes/crossover.py:101-108` |
| Environment | The checkpoint's own: 4 fields from the same map distribution, 1,024-tick cap, action repeat 2, spawn spread 0.25, uniform interleaved spawns | [record] artifact `parameters.environment`; [code] `OM:src/evaluation/environment.py:12-45` |
| Scripted opponent | The default configuration, recorded in the artifact | [code] `OM:src/evaluation/agents.py:100-102`; [record] artifact `recipe.subjects.scripted` |
| Seed | None passed to the environment in this path | [code] `OM:src/evaluation/match.py:227` |
| Cap | 320 ships in total | [code] `OM:src/modes/crossover.py:93` |

**The stored result** ([record] `xover`).

| Network | Beats up to | Win rate there | First below 50% | Win rate there | Ratio | Win rate at N v N |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.512 | 3 | 0.113 | 2.00 | 0.992 |
| 2 | 3 | 0.742 | 4 | 0.363 | 1.50 | not played |
| 4 | 7 | 0.559 | 8 | 0.336 | 1.75 | 0.996 |
| 8 | 13 | 0.648 | 14 | 0.488 | 1.62 | 0.996 |
| 16 | 25 | 0.617 | 26 | 0.488 | 1.56 | 1.000 |
| 32 | 46 | 0.523 | 47 | 0.477 | 1.44 | 0.996 |
| 64 | 76 | 0.549 (204 games) | 77 | 0.493 | 1.19 | 0.934 (244 games) |

The curve is not monotone in the stored data (32 ships: 0.648 at 43 scripted, 0.707 at
44), which the walk tolerates only because it stops at the first count below 50%.

**The finer study** (`fan`, 4,096 games a point, linear interpolation between ratios 1.0
to 2.0 in steps of 0.25; [record] `OM:docs/results/v1-731/crossover_by_size.json`).

| Network | Final live | Final averaged |
|---|---|---|
| 4 | 1.81 | 1.89 |
| 8 | 1.73 | 1.82 |
| 16 | 1.63 | 1.71 |
| 32 | 1.44 | 1.55 |

**Two earlier v1 runs, same protocol** ([record]
`OM:checkpoints/good-leaf-719/artifacts/crossover/.../result.json`,
`OM:checkpoints/resilient-resonance-682/artifacts/crossover/.../result.json`;
conditions `OM:docs/training-runs.md`).

| Run | Conditions | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 731 | 4 fields, armour 0.3, 30 Hz | 2.00 | 1.75 | 1.62 | 1.56 | 1.44 | 1.19 |
| 719 | 4 fields, full damage, 30 Hz, 999M steps | 1.00 | 1.50 | 1.50 | 1.44 | 1.38 | 1.23 |
| 682 | **No fields**, armour 0.1, 60 Hz, 7,808 environments, 999M steps | 1.00 | 1.25 | 1.38 | 1.50 | 1.50 | 1.38 |

All three stay above even numbers from 4 to 64 ships. The run without fields does too,
so fields are not needed for the size result (one run each; different physics).

### 1.9 The scripted opponent

| Item | Value | Source |
|---|---|---|
| Target | Each ship's nearest living enemy, by toroidal distance. No shared target (the team-target blend is off by default) | [code] `OM:src/agents/scripted_utils.py:160-196`; `OM:src/agents/stochastic_config.py:85-89` |
| Aim | First-order lead: both ships' positions advanced by distance / bullet speed | [code] `OM:src/agents/scripted_utils.py:47-78` |
| Turn | Probability of turning rises from 0.068 at 0.03 rad off target to 0.95 at 0.12 rad; sharp turn from 0.05 at 0.24 rad to 0.95 at 0.42 rad. Never uses the air brake | [code] `OM:src/agents/stochastic_scripted.py:71-116`; `stochastic_config.py:65-69` |
| Power | Reverse with probability 0.5 inside 10 units, falling to 0.05 at 50. Boost when slow (0.68 below speed 16, 0.04 above 32) or when far with power to spare. Otherwise coast | [code] `stochastic_scripted.py:118-146`; `stochastic_config.py:58-62` |
| Fire | Probability = aligned x in range. Aligned: 0.95 inside the target's angular size, 0.05 at 1.5 times it. In range: 0.95 inside 200, 0.05 beyond 500 | [code] `stochastic_scripted.py:148-173`; `stochastic_config.py:74-79` |
| Stochastic | Every decision samples the three factors independently from those probabilities. It is never greedy | [code] `stochastic_scripted.py:248-254` |
| Ignores | Fields, allies in the line of fire, incoming bullets, health and numbers | [record] `OM:docs/training.md:416-418`; [code] the functions above read none of them |
| Size | `stochastic_scripted.py` 261 lines, `stochastic_config.py` 174, `scripted_utils.py` 196 (two of its three functions matter) | [record] file lengths |
| Operations per call | 170 by `R/core-brief.md:134`. `R/simplification.md:266` says v2's 4,256 is "12.6x v1's", which gives 338. The two do not agree | [record], unresolved |
| Unused setting | `prob_offset` 0.1 is defined and read nowhere | [code] `stochastic_config.py:55`; grep |

---

## Part 2. v2's comparable numbers

### 2.1 The games behind the v2 numbers

| | Champion line (2026-11-09/10) | rl-v20 line (2026-11-14/15) |
|---|---|---|
| Game | Older game, 2,600 m arena, 5v5 as 3 fighters and 2 heavies | Real-scale set (Yongma, Jujak), 7.8 km preset maps, 5v5 |
| Training modes | Frontline and deathmatch mixed | Frontline, deathmatch and elimination mixed |
| Maps | Fixed size (map scale 1.0 in every game) | One map size per mode |
| Scripted opponent | The v2 scripted agent ("scripted", later "Baekho", then "Imugi") | Same family, later version |
| Sources | `S/gpu-box-log.md:829`, `833-836` | `S/bc-v19/overnight-log.md:144-151`; `S/elimination/progress.md:20` |

**Which results are in elimination.** Only the rl-v20 line. v2's elimination has no
respawn (`S/elimination/progress.md:5-8`) and a boundary that shrinks: it holds for 36 s
of a 180 s clock at real scale, then closes so every ship dies before the clock
(`docs/explanation/game/modes.md:155-170`). Every v2 elimination result with a network
has the shrink on. The only shrink-off figures are for the scripted agent against
itself: time to wipe p50 52 s, p80 90 s, p90 120 s, and the shrink causing 2.4% of
deaths (`R/arms-todo.md:153`). Spawns are by team side, not interleaved
(`docs/explanation/game/modes.md:148-153`). **No v2 record is in elimination with the
shrink off.**

### 2.2 BC

"Score" in v2 is wins plus half the draws against the raw scripted agent over the last
1,000 finished games in the rating lanes (`docs/explanation/behaviour-cloning.md:134-136`).
Ratings fix the scripted agent at 1000, so a rating of 1000 is a score of 0.5.

| Run | GPU | Steps | Wall time | Result against the scripted agent | Source |
|---|---|---|---|---|---|
| Champion BC (deadreckon-split-hiddenphysics) | 3090 | 120.6M | 3.6 h | Pool score 0.458; rating 1010 frontline, 979 deathmatch (uniform average) | `S/gpu-box-log.md:769`, `711` |
| Null BC (absolute) | 3090 | 120.6M | About 1.45 h for the last 59.8M (21:12 to 22:39) | Pool score 0.393; rating 948 / 898 | `S/gpu-box-log.md:769`, `711`, `632`, `641` |
| Nearest-neighbour MLP BC | 3090 | 120M | 19 min | Pool score 0.438; live rating 972, uniform average 611 | `S/gpu-box-log.md:769`, `823` |
| bc-v20 | Rented 5090 | 84.9M (plateau stop) | 69 min, 03:06 to 04:15 | Rating 970 +- 25 deathmatch, 930 +- 21 elimination, 893 +- 28 frontline | `S/bc-v19/overnight-log.md:69`, `101`, `138` |
| Laptop BC arm (tau10) | Laptop | 30.4M to 60.8M | About 2.0 h, 16:42 to 18:41 | Lane window 0.260 +- 0.013 at 60M | `S/gpu-box-log.md:601`, `604` |
| An earlier BC run | Not stated | 66M to 151M checkpoints | Not stated | Live clone 0.272; uniform average of checkpoints 0.350 | `S/bc-averages-brief.md:12-13` |
| Toy BC run | Laptop | 91.2M at the stop | Not stated | 0.401 over 512 fresh games | `S/bc.md:73-75` |

**Champion BC over steps** (uniform average; frontline / deathmatch rating, scripted
1000; `S/gpu-box-log.md:700-711`).

| Steps | 10.5M | 21.0M | 30.4M | 50.3M | 70.3M | 90.2M | 110.1M | 120.6M |
|---|---|---|---|---|---|---|---|---|
| Frontline | 465 | 728 | 803 | 924 | 997 | 976 | 1033 | 1010 |
| Deathmatch | 275 | 727 | 838 | 902 | 947 | 972 | 1002 | 979 |

Its pool score read 0.000, 0.006 and 0.180 at about 10M, 21M and 30M steps
(`S/gpu-box-log.md:767`). The log's reading: "BC plateaus at the teacher (~1000 by
70-100M)" (`S/gpu-box-log.md:694`).

**bc-v20 over steps** (averaged policy; `S/bc-v19/overnight-log.md:104-111`, `138`).

| Steps | 40M | 50M | 60M | 70M | 84.9M |
|---|---|---|---|---|---|
| Deathmatch | 809 | 858 | 925 | 953 | 970 |
| Elimination | 350 | 610 | 791 | 872 | 930 |
| Frontline | -3 | 469 | 760 | 885 | 893 |

Still rising when the plateau rule stopped it. The stop rule reads the imitation loss,
which "doesn't track strength" (`S/bc-v19/overnight-log.md:113`). BC was later fixed at
100M steps with the plateau rule logging only (`R/arms-todo.md:187`). v2's default score
target is 0.40, chosen because "v1 stopped BC at about 0.40" (`S/bc.md:69-71`).

### 2.3 RL

| Run | GPU | Started from | RL steps | Wall time | Strength against the scripted agent at 5v5 | Source |
|---|---|---|---|---|---|---|
| Champion RL | 3090 | Champion BC 120.6M | 180.4M | Not stated in total; the last 58.8M took about 1.9 h (12:02 to 13:54) | Rating 1997 frontline, 2404 deathmatch. **Break-even: 6.36 scripted for 5 network ships in frontline (1.27); 7.46 in deathmatch (1.49)** | `S/gpu-box-log.md:755`, `765`, `861`, `875` |
| Null RL | 3090 | Null BC 120.6M | 180.4M | Planned at about 1.6 h for the last 58.8M | Rating 1875 / 2235. Break-even 5.87 (1.17) frontline; 7.70 (1.54) deathmatch, "tied" with the champion | `S/gpu-box-log.md:698`, `861`, `875` |
| Nearest-neighbour MLP RL | 3090 | Its BC | 180.4M (best at 40.9M) | 23 min | Rating 1303 / 1386. Break-even 5.44 (1.09) / 5.67 (1.13) | `S/gpu-box-log.md:809`, `861`, `875` |
| rl-v20 | Rented 5090 | bc-v20 84.9M | 150M (extended to 172M) | 2.1 h for 150M, 04:26 to 06:30 | Rating at 145M to 150M: frontline 2585 +- 37, **elimination 1693 +- 33** (peak 1727 at 95M), deathmatch 2719 +- 147 on win/loss. **No break-even recorded** | `S/bc-v19/overnight-log.md:136`, `202-207`, `272-278` |
| rl-v20-seed1 | Laptop 4070 | bc-v20 84.9M | 149.9M | Not recorded | 70 to 150 rating points weaker than rl-v20 head-to-head | `R/arms-todo.md:108` |
| rl-dm (deathmatch only) | Rented 5090 | bc-v20 | 75M | Not stated | Elimination 1847, deathmatch 2866 +- 151, frontline 914 at 75M | `S/bc-v19/overnight-log.md:315-331` |
| rl-mix-kl | Rented 5090 | bc-v20 | 75M | Not stated | Elimination 1685 at 70M | `S/bc-v19/overnight-log.md:399-403` |

**The champion's break-even, in full** (256 games a pair; `S/gpu-box-log.md:862-875`).
Frontline win rates at 5 against 5, 6, 7, 8: 0.994, 0.676, 0.186, 0.021. Deathmatch: 5
against 10 still 0.221. Scored as win plus half a draw, sides swapped, interpolated
(`S/gpu-box-log.md:848-850`). The deathmatch target scaled with the enemy fleet.

**rl-v20 in elimination over RL steps** (averaged policy, laptop rating service;
`S/bc-v19/overnight-log.md:171`, `179-184`, `192-194`).

| RL steps | 10M | 30M | 35M | 45M | 55M | 65M | 95M | 150M |
|---|---|---|---|---|---|---|---|---|
| Elimination rating | 953 | 1252 | 1329 | 1428 | 1480 | 1584 | 1727 | 1693 |

**Champion RL over steps** (uniform average, frontline / deathmatch;
`S/gpu-box-log.md:712-723`, `821`).

| RL steps | 10.5M | 30.4M | 60.8M | 90.2M | 121.6M | 180.4M |
|---|---|---|---|---|---|---|
| Frontline | 1293 | 1436 | 1573 | 1711 | 1825 | 1937 |
| Deathmatch | 1313 | 1597 | 1855 | 2080 | 2219 | 2386 |

Seed-to-seed spread in v2 is about 110 rating points at 30M BC and 70 to 150 for the one
RL pair (`S/model-audit/history-learner.md:245-282`).

### 2.4 Steps per second and compile time

| GPU | Stage | Steps per second | Conditions | Source |
|---|---|---|---|---|
| 3090 | BC | 7,700 at defaults; 12,900 to 13,100 tuned; 11,400 for the champion's config | Older game, 2,048 environments | `S/gpu-box-log.md:31-34`, `73`, `126-127`, `658` |
| 3090 | RL | About 9,100 | Champion, 320-row micro-batches | `S/gpu-box-log.md:658` |
| 3090 | BC, large batch | 22,300 | Half the learning per sample; dropped | `S/gpu-box-log.md:173-180` |
| Rented 5090 | BC (bc-v20) | 3.1 s rollout + 44.7 s learn per update of about 1.05M steps: about 21,900 [derived] | Real scale | `S/bc-v19/overnight-log.md:96`, `146` |
| Rented 5090 | RL (rl-v20) | 6.1 s + 38 to 43 s per update: about 21,400 to 23,800 [derived]; "about 1.8 h for 150M" | Real scale | `S/bc-v19/overnight-log.md:162` |
| 5060 Ti 16 GB | RL | 5,747 plain, 5,323 with the diagnostic update one in ten | Real scale, 1,024 environments | `S/steady-speed/progress.md:85-86` |
| Laptop | BC | About 232 s learn per 1.05M-step update: about 4,200 to 4,350 [derived] | Older game | `S/gpu-box-log.md:32`; `R/simplification.md:276` |
| Laptop | RL | Learn 189.5 s + collect 18.0 s per update: about 5,050 [derived] | Real scale, 160-row micro-batches, after the learner work | `S/compile-overhaul-notes.md:526-533`; `R/simplification.md:277` |

**The "about 5,000" in the brief is the last row.** It is a steady-update figure from a
benchmark of 3 updates, with no compile, no rating games and no diagnostic update in it.
v1's 2,655 is a whole-run figure with all of those in. `R/simplification.md:279-281` says
the two are not like for like. The one full laptop RL run (rl-v20-seed1) has no wall time
in the notes.

**The notes name the laptop GPU three ways:** RTX 4070 Laptop (most notes, and
`OM:docs/evaluation.md:131`), "laptop 3070" (`S/gpu-box-log.md:32`), "the laptop's 4060"
(`S/compile-overhaul-notes.md:765`, `S/cleanup-notes.md:376`).

| Compile | Time | Source |
|---|---|---|
| Laptop, cold, per stage | Act stage 143 s, loss 64 s, advance 12.6 s, play tick 25.8 s | `R/simplification.md:646-648` |
| Laptop, compiled tiny config | Still compiling the BC stage after 40 min | `S/launch/progress.md:70-73` |
| Rented boxes, cold | About an hour per stage; 73 min on one box | `S/launch/progress.md:74-77` |
| 5060 Ti, RL, cold to first update | 3,216 s; warm 449 s | `S/steady-speed/progress.md:72-78` |
| 3090, a new BC config | First update about 11 min | `S/gpu-box-log.md:114` |
| 3090, RL at a new micro-batch size | About 36 min before running out of memory | `S/gpu-box-log.md:657` |
| 3090, evaluation at 40v40 | 7 to 17 min compile, then 111 s per 256 games | `S/gpu-box-log.md:837-838` |

### 2.5 Every size-curve measurement in v2

`R/model-audit.md:83-114` and `S/model-audit/history-model.md:753-820` already hold
these. I re-read the sources and they agree. In one table:

| Agent | Mode | Map | Sizes | Result | Source |
|---|---|---|---|---|---|
| Champion (trained 5v5) | Deathmatch, fixed odds, 128 games a ratio | Training arena, 2,600 m | 10, 20, 40 | Win rate at 1:1, 1.2:1, 1.5:1, 2:1 scripted to network. 10: 1.00, 0.95, 0.73, 0.43. 20: 0.97, 0.91, 0.69, 0.22. 40: 0.97, 0.53, 0.03, 0.00 | `S/gpu-box-log.md:902-910` |
| Champion | Frontline rating | Training arena | 5, 10, 20, 40 | 1997, 1875, 1134, 526 (scripted 1000) | `S/gpu-box-log.md:861`, `874`, `911-919` |
| Champion | Deathmatch rating | Training arena | 5, 10, 20, 40 | 2404, 2778, at least about 2551 (won 1,024 of 1,024), saturated | `S/gpu-box-log.md:861`, `874`, `905-913` |
| Champion | Both | Density arena, 3,677 m | 10 | 1499 frontline, 2317 deathmatch: about -375 and -460 against the training arena | `S/gpu-box-log.md:876` |
| Champion | Frontline break-even | Training arena | 10 | 11.19 scripted for 10 network (1.12) | `S/gpu-box-log.md:881-882` |
| Champion | Frontline, smoke | Density arena | 40 | "every trained agent ~140-270 vs scripted 1000" | `S/gpu-box-log.md:836-838` |
| rl-v20 | Frontline head-to-head, 64 games | Preset 7.8 km | 5, 10, 20 | Every game at 5 and 10; 26 wins and 14 draws of 64 at 20 | `R/evaluation-suite.md:132-138` |
| rl-v20 | Frontline head-to-head | Density maps 11, 13, 15.6 km | 10, 14, 20 | 58 of 64; 15 of 64; 0 of 64 | `R/evaluation-suite.md:135-138` |
| rl-v20 | Not stated | Not stated | 2 | "only even with Imugi at 2v2" | `R/arm-review.md:31` |
| rl-dm | Deathmatch | 20-a-side density map | 20 | Won 5 of 32 | `R/evaluation-suite.md:142` |
| rl-v20 against rl-mix-kl | Elimination 1v1, 2v2, 5v5, 10v10 | Density maps except 5v5 | 1, 2, 5, 10 | Network against network only (+0.013, +0.014, +0.024, -0.007); nothing against the scripted agent | `R/evaluation-suite.md:106-120` |

**Gaps.** No v2 measurement of a network against the scripted agent in elimination at
any size other than 5v5. No v2 measurement at 1v1 against the scripted agent with
conditions stated. No unequal-team training. The "1.4x" map-scale collapse has no table
behind it (`S/model-audit/history-model.md:809-811`).

---

## Part 3. Proposed numbers

### 3.1 Clone

**Proposed:** the cloned policy scores at least **0.40** against the scripted opponent
at the training size, within **85M environment steps** and **5.6 laptop GPU-hours**,
compile included.

- **Score:** wins plus half the draws, as v2 defines it, read as the mean minus two
  standard errors over at least 1,000 fresh games (v2's rule, `S/bc.md:60-71`).
- **0.40** is v2's default target (`S/bc.md:69-71`). bc-v20's elimination rating of 930
  at 84.9M corresponds to an expected score of 0.40 [derived: 1 / (1 + 10^(70/400))].
  The champion's BC reached 0.458 at 120.6M.
- **85M steps** is bc-v20's length, the only real-scale BC run recorded.
- **5.6 hours** is 85M steps at the laptop's recorded BC speed of about 4,250 steps per
  second [derived: 30.4M steps in about 2.0 h, `S/gpu-box-log.md:601-604`].

**What the records do not support.**
- No v2 BC run reached the plateau on the laptop. The hours are steps divided by speed.
- The laptop speed is from the older game; bc-v20 ran the real-scale game on a 5090.
- v1 has no clone number. Run 731 passed 0.45 on a 100-game window at 17.9M steps with
  the policy gradient also on. That is a hint that 85M is loose, not a measurement of
  cloning.
- v2's clones were scored against Imugi; the core's opponent is not yet chosen.

**So the number to measure in M1b** is the score over steps of the first clean clone
run. I would set the requirement from it and keep 85M and 5.6 h as a ceiling that must
not be exceeded.

### 3.2 RL at the training size

**Proposed:** break-even of at least **1.5** within **90M RL steps** and **9 laptop
GPU-hours** after the hand-off from the clone.

- **Break-even 1.5:** at least 50% wins for N network ships against 1.5N scripted ships
  (4 v 6 if the training size is 4), ties as losses, at least 1,024 games, final live
  weights.
- **90M steps, 9 hours** is run 731 from the end of its BC term (32.8M steps, 3.1 h) to
  the first frozen checkpoint with 4 v 6 above 50% (120.4M steps, 12.3 h) [derived:
  87.6M steps, 9.2 h].
- **Also report the total from random weights** against v1's 120.4M steps and 12.3 h.
  A clone of 85M steps plus 90M of RL is 175M steps, more than v1's total. At the
  brief's floor of 5,000 steps per second that is 9.7 h, less than v1's time. The
  author should decide which of the two totals binds.

**Caveats.**
- The v1 figure is a bracket. 4 v 6 was 0.406 at 79.6M (8.0 h) and 0.601 at 120.4M
  (12.3 h); there is no checkpoint between.
- v1's hours include full gradient diagnostics on every update, the evaluation battery
  and 4 field tokens.
- v1's 1.5 is against its simple scripted agent on a torus with interleaved spawns. If
  the core uses Imugi, v2's side spawns or a bounded arena, 1.5 is a different number.
- v2 supports no budget: its one 1.49 took 301M steps on a 3090 in deathmatch.

### 3.3 The size-curve protocol

Adopt v1's, with two additions.

| Item | Adopt |
|---|---|
| Sizes | Network 1, 2, 4, 8, 16, 32, 64 |
| World | The training map, unchanged at every size |
| Sides | Network on one fixed side, as v1; add a side-swapped repeat only if the core's game is not symmetric |
| Ties | A tie is a non-win for the network |
| Cap | The training episode's cap |
| Search | Integer scripted counts, v1's walk from the previous size's crossover |
| Games per cell | v1's 256 for the comparison with v1's table. **Add** 1,024 games at the two counts either side of the boundary; v1's boundary cells carry about 3 points of standard error (`OM:docs/evaluation.md:181-188`) |
| Weights | Final live for the comparison with v1. **Add** the averaged weights; v1's held 0.08 to 0.11 more ratio at every size |
| Report | `beats_up_to / N` per size; the win rate at N v N; games per cell |

**"Above parity at every size" needs a definition at 1 and 2 ships.** With integer
counts, a ratio above 1 at 1 ship means beating 2 scripted ships. v1 did, at 0.512 over
256 games, which `OM:docs/evaluation.md:195` calls "within noise of even". Runs 719 and
682 did not (ratio 1.00). I propose the requirement reads: **at every size the network
wins at least half its games at N v N, and the ratio is above 1.0 at every size from 4
up.** The ratio at 1 and 2 is reported.

**Not supported by the records:** how many games the laptop can hold at 64 a side in the
core, and how long the sweep takes. v1's sweep took about 1.6 h [derived: its artifact was
created 00:27 and last updated 02:01 UTC, on the training machine].

---

## Part 4. What v1 did that the plan does not yet say

Compared with `R/core-brief.md:202-248`. Ranked by how likely each is to matter for
reproducing the size curve. The ranking is my judgement; v1 has no ablation of any of
these (`OM:docs/evaluation.md:491-492`).

| # | v1 ingredient | The plan | Why it may matter |
|---|---|---|---|
| 1 | **Torus with interleaved uniform spawns.** No boundary; both teams scattered over the whole world; fights start everywhere at once | "Elimination on one fixed map with no shrink". v2's elimination is a bounded arena with the teams on opposite sides | A large fleet on a torus is many small local fights that look like training. Two massed sides in a bounded arena are not |
| 2 | **The scripted opponent is simple.** Nearest enemy, lead, fire when aligned; no shared target, no evasion; about 170 operations | Open question: "how far to simplify it" | 1.75 is measured against this agent. Imugi is 4,256 operations with doctrines; the same network would score a different ratio |
| 3 | **Trained at 4v4 only.** No size was drawn. Smaller fights were seen only through attrition | "Team sizes drawn per environment, unequal teams included" | Drawing sizes is a different claim from v1's zero-shot one, and probably an easier one. The range drawn is not stated |
| 4 | **17 s cap, ties as losses.** Mean game 5 to 9 s | Shrink off, no clock stated | Without the shrink something must end the game. The cap and the tie rule set the win rates near the boundary (ties reach 10% of games) |
| 5 | **Bullets are observed.** Every bullet is a key in cross-attention, 10 per ship, with the shooter's team | Not mentioned | The count of bullet keys grows with the fleet. v1's result includes this path; whether it is needed is unmeasured |
| 6 | **Absolute position only.** 4 Fourier frequencies over the world's period; no relative or rotary term | Hexagonal Fourier at 6 scales with a random offset, plus rotary on position and heading | v1 generalised in size without any relative position. The plan changes the mechanism the result came from |
| 7 | **Half self-play, half league, sampled by rating distance.** Every early league game is the scripted agent; frozen checkpoints join every 200 rating points | "A pool of past snapshots and the scripted opponent"; rating-based sampling deferred | The share of scripted games over time is set by the rating rule. Without ratings the share has to be chosen |
| 8 | **Friendly fire on, no collisions.** The scripted agent does not check its line of fire | Not stated | Crowded scripted ships hit each other. This moves the break-even at large sizes |
| 9 | **Refractive fields the scripted agent ignores.** 4 per map, crossing costs 10 or 20 of 100 health | "One fixed map" | Part of 731's edge may be the scripted agent's field damage. Run 682, without fields, still stayed above parity, at lower ratios at small sizes |
| 10 | **Every ship sees every ship's pending action,** opponents included, one decision ahead | Not stated | An information edge over an opponent that reads only the state |
| 11 | **Action repeat 2** over 60 Hz physics | v2's tick is 30 Hz | Same decision rate; collision and bullet integration differ |
| 12 | **Reward weights and shaping.** Offence paid twice the charge; facing and closing-speed shaping until 100M steps; rewards divided by ship count | "Reward levels with a discount each"; no weights | Shaping "exists to stop early passive collapse". Affects reaching 1.5 in budget more than the curve's shape |
| 13 | **No BC stage.** A BC term inside PPO, gated off at a 0.45 win rate | A BC stage, then a decaying distillation term | Changes the starting policy and how long the scripted agent's style persists |
| 14 | **The averaged weights are stronger outnumbered** (1.89 against 1.81 at 4 ships) | Not stated which weights are measured | The headline numbers are the live weights'. The averaged ones are a free margin |
| 15 | **Dead ships masked as keys, no null key** | Key mask plus a learned null key | Consistent; the null key is new |
| 16 | **Spawn resources randomised** (health and power 75% to 100%) | Not stated | Training sees damaged states from the first tick |
| 17 | **Factored action** (3, 7, 2) | A joint categorical | Low for the curve |
| 18 | **Scalar value per component, 16 of them, team-pooled for the win** | Categorical critic per reward level | Low for the curve; matters for the budget |
| 19 | **Batch and schedule.** 2,592 environments, 995k steps per update, 4 epochs, 32 minibatches, clip 0.15, learning rate 4.5e-4 falling to 1.5e-4, entropy 0.005 | None of these numbers | Needed to reproduce the time to 1.5 |
| 20 | **Model size.** Width 128, 4 heads, 4 spatial and 2 temporal sublayers, about 2.0M parameters | "The audit's baseline" | Low, if the baseline is similar |

**Open items the collection could not settle.**
- The parameter count needs the weights (git LFS).
- The operations per call of v1's scripted agent: 170 or 338.
- The share of training games against the scripted agent over time was never logged.
- rl-v20-seed1's wall time on the laptop, the one full laptop RL run, is not in the notes.
- The laptop GPU's model is given three ways in the notes.

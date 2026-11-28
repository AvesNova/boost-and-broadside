# Correctness review of the lean tree (core 5b54fccb), 2026-11-28

Queue item 7. A read-only review by an agent that built none of this code, looking for silent
mistakes a dry run would train straight through. No speed work, no style points, nothing changed.
CPU checks used the review worktree with the `v2-docs` virtual environment, 2 threads, no GPU.
The check scripts are described inline; they were scratch files and are not committed.

Severity: **bug**, **likely bug**, **design question** (for the author), **doubtful**. Ranked by
severity x impact on training.

## Ranked findings

| # | Severity | Where | One line |
|---|---|---|---|
| 1 | likely bug / design question, high impact | `lean/learn/returns.py:44-73`, `lean/learn/rewards.py:132-138`, `lean/learn/rollout.py:593` | A ship that dies loses almost all of its team's outcome: after its death the values are 0, so the lambda-return discounts the outcome by (gamma lambda)^k, not gamma^k |
| 2 | design question (an author decision not built yet) | `lean/learn/training.py:211-215`, `rollout.py:647-649` | Next-state and hit losses train on own living ships only. No pairwise loss exists. The 2026-11-27 ~22:00 decision (all visible enemies, hidden ones masked) is not in the code |
| 3 | design question, medium | `lean/learn/pool.py:537-542` | The EMA and uniform slots take new weights after every update, draining slots too, so a running league game's opponent changes part way through (games last up to about 4 updates) |
| 4 | design question (E5), medium for the teacher and yardstick | `lean/scripted/opponent.py:928-993`, `paths.predict` | The scripted agent's trigger and candidate paths ignore the one-tick action delay. In 22% of fire decisions the aim moves by more than the target's half width before the shot leaves |
| 5 | doubtful, low | `lean/learn/config.py:163-169`, `lean/model/policy.py:45-59, 104` | Constants left over from the old defaults (21-point hits, before the egg): the damage knee, the damage range and the hit-class prior. Measured effect is small, except the hit prior, which is 8x off |
| 6 | doubtful, low | `lean/learn/rewards.py:123-129` | Team-shared payments (boundary deaths, friendly-fire compensation) are split over present ships, the dead included. The dead ships' shares never reach training |
| 7 | doubtful, low | `lean/learn/training.py:201` | The value loss averages over all 5 levels, 2 of them the objective placeholders with target 0, so the live levels get 3/5 of `value_coefficient` |
| 8 | doubtful, negligible (not checked on a GPU) | `lean/train.py:427, 464` | The same seed goes to the explicit generator and to torch's default generators. On a compiled GPU run the first segment's deals replay `start()`'s draws |

### 1. Dead ships lose the outcome (likely bug / design question)

**What.** `outcome_reward` pays every ship of the team on the last tick, the dead included
(rewards.py:132-138). The values are zero outside `own` (rollout.py:593). GAE runs through the
dead rows with V = 0 (returns.py:67-73). For a dead row, delta = r, so the reward reaches the
ship's last living tick only through A_t = delta_t + gamma lambda A_(t+1): discounted by
(gamma lambda)^k. A living ship's later values carry the outcome at gamma^k. The docstring
(returns.py:58-60) says the outcome "still reaches its earlier ticks through the lambda-return".
It does, but at outcome lambda 0.97 the effective horizon falls from 1/(1 - 0.9997) = 3,333 ticks
to 1/(1 - 0.9697) = 33 ticks.

**Evidence** (`generalised_advantages` on CPU: two ships, +10 outcome on the last tick of 512,
a perfect critic while alive, ship 1 dead from tick d; the target at ship 1's last living tick):

| Death before the end | Target, survivor | Target, ship that dies | Its advantage |
|---|---|---|---|
| 30 ticks (1 s) | 9.91 | 3.97 | -5.94 |
| 100 ticks (3.3 s) | 9.70 | 0.46 | -9.24 |
| 300 ticks (10 s) | 9.14 | 0.001 | -9.14 |

**Failure scenario.** In 4v4 elimination most deaths come well before the end, so the outcome
level (weight 10, the largest) becomes "alive at the end and on the winning team". On a losing
team, dying early avoids the -10. Surviving to the loss earns it, so the outcome level
rewards an early death there. On a winning team it adds a survival bonus. The zero-sum outcome
does not hold for what the critic learns. The kill and assist shares a ship earns after its
death (and the team payments, finding 6) are attenuated the same way.

**Confidence.** High on the mechanics. Whether v2 did the same, and which rule is wanted (pay
the outcome at death as its expected value, give dead ships a terminal value, or accept
"survive to win"), is for the author.

### 2. What each head's loss trains on today (design question)

| Head | Loss mask (rows that train) | Source |
|---|---|---|
| Policy surrogate, entropy, BC / teacher cross-entropy | `own` = the seat's team and alive (the view's alive) | training.py:190-193 |
| Value (5 levels, HL-Gauss) | `own` | training.py:201 |
| Next state (6 columns) | `next_valid` = `own` and alive after the step and the episode not ended on that tick | rollout.py:647-649, training.py:211 |
| Hit (3 classes) | `own` and label >= 0. The label is the outcome of the bullet the ship emitted on that tick (ghost or real), ended within this segment or the next. Rows of the update's last 11 ticks are unlabelled, and so are rows whose bullet was cut off by the episode's end | training.py:110, 215; rollout.py:527-557 |
| Pairwise | none exists | |
| Enemies (visible or hidden), dead and absent slots | never trained by any head | |

The author decided on 2026-11-27 ~22:00 (queue.md) that next-state, hit and pairwise losses
train on all visible enemies, with hidden enemies masked. None of this is built: item 4 was to
report it and item 5 to fix it. For an enemy row the next-state target exists already (the game
computes `next_state_target` for every slot). The mask would be `visible` (Obs.visible at t
and at t + 1, the latter for the target), alive after the step, and not done. One trap for
whoever builds it: an enemy's hit label uses the enemy's ghost, and it should stay a plain game
label, not a view label.

### 3. The averages' opponents change mid-game (design question)

`after_update` copies the new EMA and uniform weights into every slot whose source is
("ema",) or ("uniform",). That includes draining slots (pool.py:537-542). A league game lasts up
to clock + segment = 1,928 ticks, about 4 updates at 512 ticks each. So a running game against
an average sees its opponent's weights change 1 to 3 times, and the opponent's recurrent carry
was built by the old weights. The module docstring says "an average is one moving player" on
purpose. The author's rule for replaced snapshots (they keep playing their games in a retiring
slot) does not cover averages. The tallies for "ema" also mix several versions (minor). Low
risk to learning. Raised because the brief asks whether a running game's opponent ever changes:
it does, for the averages only. Snapshot slots are never reloaded while they have games. Loads
go only into free slots, and a slot frees only with 0 rows (checked in `allocate`, `_unload`,
`free_drained`, `prepare`'s orphan check).

### 4. E5: the scripted agent ignores the one-tick action delay (design question)

The game applies the action chosen on tick t at t + 1 (the pending action). The scripted
agent's trigger (`_trigger`) gates on the aim error from tick t's pose. The real bullet leaves
from tick t + 1's pose, after the pending stick has turned the nose for one more tick. The
candidate paths (`paths.predict`) also start each candidate stick at once, not after the pending
tick. The agent knows the pending action (it reads it for stickiness), so it could predict
that tick.

**Evidence** (scripted 1v1, 32 environments x 450 ticks, standard spawn, CPU). At fire
decisions (p_fire > 0.5, 263 rows), the change of the nose-to-target angle over that one tick
divided by the target's half width (the egg's side-on 2.375 m) was: median 0.36, 75th
percentile 0.87, 90th percentile 1.42. **22% of decisions move by more than the half width.**
The nose moves a median 10 mrad per tick against a median half width of 14 mrad (at 165 m).
This lowers the yardstick's and the BC teacher's hit rate. Fire labels are effectively one tick
late in turning fights. It may explain why `lead_error` 0, 0.25 and 1 all score the same
(questions.md 14: the aim gate dominates).

The rest of the scripted NOTES matched the code where I checked:
- Information: `enemy_view` only, dead reckoning 2 s, then straight. No read of bullets, true
  enemy state or enemy pending actions. Team-mates and own ships read true state, as the team
  may.
- Enemies: the 4 nearest known, with roam and cutoff weights.
- Enemy table power: from straight flight with the yardstick's rows.
- `hit_worth`: divides by the shooter's hit damage, so the trigger does not count it twice.
- Hit chance: erf(half / (sqrt 2 sigma)) with sigma = flight x sqrt(spread^2 + lead^2), in
  metres.
- `store_crowd` 1 with `store_weight` 0.5.
- The joint's order is stick x 6 + power x 2 + fire, as the contract.
- Never fires through a team-mate. Randomness only from sampling.
- Decision 5.5 (a random side drawn per ship for a symmetric head-on pass) has no code. It
  belonged to the replaced rule-based agent. The look-ahead agent breaks the tie by sampling,
  which matches "randomness only from sampling".

### 5. Constants left over from the old defaults (doubtful, low)

- **`value_knee` damage** (config.py:165-169): the comment's rule (one event x e^-3) with a
  21-point hit gives 0.0286. With hit damage 8 the rule gives 8 / 100 x 2.74 x e^-3 = 0.0109.
  Effect: a single hit's return sits 23 bins from 0 with the old knee and 27 with the rule's.
  **Negligible.**
- **Damage support +-5.4** (config.py:163): re-measured on the current game (scripted 4v4,
  training spawn, 48 environments x 1,800 ticks, 51 finished episodes, 409k living ship-ticks).
  - Damage returns: 0.1st / 99.9th percentiles -1.65 / 1.67 (2026-11-25: +-2.1); extremes
    -2.42 / 2.42 (2026-11-25: -3.9 / 5.4). In symlog bin space about 85% of the support's
    span is reachable. Harmless; the range monitor would flag a narrow support.
  - Kill_death returns: -2.89 / 5.58, inside +-7.5.
  - A small sample. The proper re-measure is `target_ranges.py` (model NOTES "Open").
- **Hit-class prior** (policy.py:104, 0.9374 / 0.0547 / 0.0080): on the current game (scripted
  4v4, 32 environments x 600 ticks, 141k ended bullets of living owners) it is 0.9914 / 0.0070
  / 0.0015. The egg cut the hit width from 24 m to 2-5 m. The enemy class's initial bias is
  about 8x too high. Only the initial loss is affected, but the class is now very rare (0.7%):
  watch whether the hit head learns anything beyond the prior.
- **Shield next-state range -90..0** (policy.py:49-51 says "four hits of about 21"): bullets now
  deal at most 8 x 4 = 32 per tick at 4v4. Collisions (up to the shield held) still pass -90
  rarely, and those clamp into the buffer columns. The comment is wrong; the range is fine.
  The feature scales are from the 2026-11-23 game (game NOTES "Open").

### 6. Team-shared payments go to dead ships too (doubtful, low)

`enemy_team_share` divides a team payment (boundary deaths, the 2x friendly-fire compensation,
unattributed boundary healing) by the team's present ships, the dead included
(rewards.py:127-129). The dead ships' shares are never trained (finding 1). So the living get
n_alive / n_present of it. The sums stay zero, as the tests check. What training sees does not.

### 7. Objective placeholders in the value loss (doubtful, low)

`value_loss` is divided by `own_count x 5` (training.py:201). The two objective levels (target
0, values 0) take 2/5 of the term. The three live levels get an effective coefficient of 0.03
instead of 0.05. Harmless, but the coefficient means less than it says. Dropping the dead
levels in elimination, or dividing by the live levels, would make it exact.

### 8. One seed for both generators (doubtful, negligible, not checked on a GPU)

`train.py` seeds the explicit `generator` (train.py:427) and torch's default generators
(train.py:464, `torch.manual_seed`) with the same seed. Compiled on a GPU, `start()` draws the
initial deals from the explicit CUDA generator. The first segment's `draw_next_episodes` draws
from the default CUDA generator: the same Philox seed and offset, and nearly the same call
order. So the first games to end in segment 0 are likely re-dealt with environment k's opening
(fields, spawns, team sizes, offsets). On the CPU, the weight-initialisation stream equals the
generator's. A one-off repeat of a few scenarios. Seeding the generator with `seed + 1` would
remove it.

## Checked and found fine

- **Ghost-only bullets and labels.**
  - Every ship writes slot `tick mod 12` from the tick-start pose. Ghosts touch but never
    damage, and only real hits reveal.
  - `_bullet_ends` reports each bullet once. A fresh deal's slots are CONTACTED, so a new
    episode never reports old bullets.
  - `hit_outcomes_by_age` uses the pre-step `state.tick`, as `bullet_ages` does.
    `assemble_hit_labels` puts age a at row T + t - a.
  - The cross-segment merge in `end_segment` is a max over -1 or one label. An episode cut
    leaves rows unlabelled, not marked missed.
  - int8 is enough for -1..2.
  - Quick tests pass: `test_hit_head`, `test_ghost_bullets`, `test_rewards`, `test_learn`,
    35 passed.
- **The egg.** The turn's sign is away from the centre. The relative frame matches between
  `_hit_test`, `_deflect` and `_damage` (a turned hit's direction, impact on the remainder). A
  path starting inside the egg is not turned. CONTACTED bullets are not turned. One egg per
  tick, as documented.
- **Rewards.**
  - Damage, kill_death and outcome signs are right, team by team, and the levels are zero-sum.
  - Assists: half from the last tick's hits, half from the ledger.
  - Ally kills and friendly fire: -share to the shooter, 2x to the other team.
  - A boundary death pays the other team.
  - Collision damage is credited to the other ship.
  - The ledger clears on death and at the episode's end.
- **GAE, bootstrap and chaining.**
  - The scan computes A_t = delta + gamma lambda (1 - done) A_(t+1) correctly (reversed, decay
    first). The Triton forward and backward match the stated gradients. No int32 overflow at
    parity sizes.
  - Segments chain per seat (`chain_index` / `join_segments` order). The bootstrap comes only
    from the last segment.
  - The clock end is a true termination. The update's end bootstraps from `final.value` with
    `own`.
  - Advantages are normalised over own tokens.
- **Rollout.**
  - Seats: self-play seat E + e is 1 - learner_team.
  - The `reset` and `deal_row` records line up with `act`.
  - `next_state_target` is computed before the re-deal and masked on done ticks.
  - The rows written to the store and their compact dtypes are fine: positions and PPO numbers
    float32, teacher float16, pending and labels int8.
- **Fog.** `observe` builds every enemy row from `team_view` and `seen`. `fired` is the true
  flag, but a shot always reveals its ship that tick. Never-seen and seen-dead enemies are left
  out of `key_mask`. Rotary and the position code read last-seen positions. The scripted agent
  reads only `enemy_view` and dead reckoning for enemies.
- **League.**
  - Level 2's capacity and overflow work, and duplicates are fine.
  - `route_rows` and `merge_routed` ranks agree.
  - Network slots run on W = C rows, and `prepare` raises if a slot holds more.
  - Scripted per-row parameters: the yardstick for the live side, the persona's row for the
    league side.
  - `record` credits the live side's win to the member drawn at the deal. RETIRED games are
    skipped. Dropped members' games are set RETIRED before their index is reused for a new
    snapshot (train.py:271-274).
  - PFSP (1 - p)^2 on the live side's win rate, with the floor.
  - Checkpoint and resume: tallies, debts, slots, the averages' weights rebuilt, generators
    restored. The eager CPU resume test compares real runs.
- **Learner.**
  - Per-minibatch denominators over micro-batches. A non-finite step is skipped through
    `found_inf`.
  - Warm-up per stage. The teacher term's gate uses the yardstick's tally.
  - The BC teacher comes from the full joint over every ship. The EMA and uniform averages and
    the hand-off are fine.
- **Model.**
  - Temporal reset and keep semantics match between `step` and `sequence` (the zero-carry rule
    for sqrt(1 - a^2)). The model has no slot-index input (seat symmetric).
  - Rotary turn in degrees is converted to radians.
  - Value and next-state HL-Gauss and the decode use the same knees.
- **New defaults.** Hit damage 8 and spread 10.42 m/s (= 5 mrad x 2,084 m/s) are what the
  constants hold at runtime.
- **Static compile.** A recompile raises (loud, not silent). The routing graph sees the same
  shapes through loads and drains. A league with no network slot takes the eager scripted path
  with the same shapes (`league_rows` clones).

## Not covered

The evaluation package beyond its seat and scripted call (lean/eval), the tool and sandbox, the
flight model's physics, and the GPU-only paths (the Triton scripted kernel against its twin, the
compiled league tick, streaming copies). Those were read only where the training path passes
through them.

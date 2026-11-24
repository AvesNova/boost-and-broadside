# The lead turn and the yo-yo: review and diagnosis (2026-11-24)

The screening (scripted-screening.md) found that the lead turn (`lead_turn_scale`) and the
yo-yo's pursuit half (`yo_yo_time`) lower skill at every setting above 0, and carry the
largest total Sobol indices. This review asks whether they are built wrong or are bad ideas
in this game. Code: lean/scripted/opponent.py (line numbers are of core a0909ac8, before
the fix). Doctrine: shaw-bfm-notes.md sections 5 and 6, Shaw Figs 2-6, 2-8, 2-9.

## Verdict

- **Both were built wrong in the same way.** Inside gun range they took the nose off the
  gun's lead point. The merge's gun weight was zero under the two-circle plan, and a high
  yo-yo discounted it. That was the whole of each feature's cost in 4v4 brawls (below).
- **The high yo-yo also fired in the wrong place.** It had no aspect condition, so it lagged
  in beam and crossing passes at the 3/9 line, not in the overshoot from behind that it is
  for.
- **Fixed** (this branch): the gun owns the nose inside gun range for both tactics, and
  the high yo-yo lags only with him in my rear quarter. The yo-yo's pursuit half is now
  neutral at every setting. The lead turn is neutral in mixed spawns but still costs skill
  in sides spawns.
- **Neither gains anything.** No setting of either, before or after the fix, and no
  rewritten lead turn tried here scores above the yardstick.
- **Recommendation:** keep both off in the yardstick. Delete the lead turn, or rebuild it
  only together with the pre-merge offset (not built). Keep the fixed yo-yo pursuit half as
  a persona trait if persona variety is wanted; it costs nothing now.

## How it was measured

- **1v1 situations** (CPU, 512 environments each, 60 s, no fog or fields, 4,200 m arena).
  The subject flies the variant and the opponent the yardstick. One of each was built:
  - an overshoot-prone approach from behind: 1.2 v_c against a target at 0.6 v_c, 560 m
    ahead, crossing, turning at the sustained rate;
  - a slow tail chase out of range: both at 0.6 v_c, 1.6 gun ranges apart, the target
    turning;
  - head-on merges with lateral offset d = 1, 2, 3 R_st, 10 R_st apart, both at v_c. The
    subject has store 0.8 against 0.3, so its plan is two-circle and the lead turn can fire.
- **4v4 brawls** (CPU): one team flies the variant, the other the yardstick. The default
  game (fog, 10 fields, mixed spawn, 4,200 m, 5,400 ticks), 256 deals in each seat, so 512
  games with an error of about 0.022. Every variant plays the same deals. Each brawl records
  where the feature fired: the ship-ticks with |pursuit time| above 0.05 s, or the rewritten
  lead turn on.
- **Ablations** (scratch copies of opponent.py with switches, not committed). Each switch
  removes one part of the feature, to show where the cost comes from.
- **The screening's skill measure** (GPU), for the re-measured sweep. Its store was not
  reusable. An agent's identity is its parameter vector, not its code, and the panel's BFM
  member and personas fly both features, so the fix changed the panel's behaviour under
  the same identities. A fresh store holds the new sweep
  (data/scripted-screening/leadyoyo-fixed.sqlite, 10,368 games, about 4 minutes on the
  laptop GPU). Script: sections/lead-turn-yo-yo/measure.py.

## Defects

### The lead turn

| # | where | defect |
|---|---|---|
| L1 | 360-362 | gun weight `merge x (1 - two_circle_plan)`: under the two-circle plan the merge never uses the gun, even in range |
| L2 | 334-357 | not a turn started at a time but lead pursuit toward his position min(time to pass, lead time) ahead, on every merge-quadrant tick |
| L3 | 351 | at τ = time to pass the point is where he will be at the pass: a collision course with his track, which uses up the offset and puts me in front of his nose (Shaw p. 75) |
| L4 | 414-423 | the pursuit point is the gun's lead point moved by v_t τ, not his position (design B4). The bullet's flight time and my own velocity count on top of τ |
| L5 | 350 | θ / ω_st, though the merge is flown at merge speed with full stick, where the rate is higher (the design's P-51D table: 1.19 against 0.65 rad/s) |

- **L1 and L2 together are the measured cost.** In the 4v4 brawls at scale 1 the lead turn
  is active on 24% of ship-ticks. On most of its ticks inside gun range the merge weight is
  0.9 and the gun weight is at most 0.02, and the nose points 0.24 to 0.35 rad off the gun's
  lead point.
- **L3 and L4 were not scored on their own,** but `explain` shows their size in the d = 2
  R_st start (two-circle plan, both at v_c, with the committed code):

  | range | his bearing | gun | wanted at scale 1 |
  |---|---|---|---|
  | 6 R_st (2.1 gun ranges) | 18 deg | 28 deg | 58 deg |
  | 4 R_st (1.4 gun ranges) | 27 deg | 38 deg | 71 deg |

  A turn of 30 deg or more off the gun, beyond gun range, is the likely source of what is
  left in sides spawns, where every game opens with a long head-on approach.
- **L5 is not fixed.** It no longer matters with the gun owning the nose.

### The yo-yo

| # | where | defect |
|---|---|---|
| Y1 | 355-357 | the high yo-yo's lag has no aspect condition (Shaw p. 71: aspect about 30-60 deg, in pure pursuit from behind) |
| Y2 | 360-361 | gun weight `offensive x (1 - high_yo_yo)`: a high yo-yo drops the gun in range |
| Y3 | 315-318 | the wanted closure κ* = (r - nose-tail) / (closing time x v_c) is about 0.8 v_c at gun range against a scale of 0.1 v_c, so y ≈ -1 at almost any range in a chase |

- **Y1 and Y2 are the measured cost.** At yo_yo_time 0.5 the high yo-yo is active on 4.6% of
  ship-ticks. 92% of those are inside gun range and half inside 0.5 gun ranges. The aspect
  there is 1.5 to 1.8 rad, at the 3/9 line, with merge weight 0.5 to 0.7. The nose points
  0.08 to 0.41 rad off the gun inside gun range, 0.41 inside half of it.
- **Y3 makes y a sign of the geometry, not of the overshoot risk.** In the brawl, y averages
  -0.97 over the low yo-yo's ticks. In the 1v1 overshoot-prone approach from behind (closing
  at 0.3 v_c at 0.6 gun ranges), y is -0.5: the ship leads in exactly the situation the
  high yo-yo is for.
- **Y3 is not fixed.** y also sets the yardstick's speed target and stick cap (the yo-yo's
  energy half), and the yardstick must not change. That energy half probably acts in the
  same beam passes (it reads the same y), but it was not measured here.

### The low yo-yo

- **No defect was found.** Its lead acts mostly beyond gun range, where the nose is at most
  0.09 rad off the gun on average in each range band. Alone it scores 0.512 against 0.500.

## Evidence

### 4v4 brawls against the yardstick

512 games each, error ±0.022. "Gun kept" means the gun weight is not discounted in range.
The scale-1 lead-turn rows are rescored with the committed code in the next table.

| variant | score |
|---|---|
| both rows 0 (control) | 0.500 |
| yo_yo_time 0.5 | 0.355 |
| yo_yo_time 0.125 | 0.457 |
| 0.5, low half only | 0.512 |
| 0.5, high half only | 0.354 |
| 0.5, gun kept | 0.418 |
| 0.5, high half in rear quarter | 0.490 |
| 0.5, rear quarter + gun kept | 0.494 |
| 0.125, rear quarter + gun kept | 0.496 |
| lead_turn_scale 1 | 0.350 |
| lead_turn_scale 0.25 | 0.463 |
| 1, gun kept at the merge | 0.506 |
| 1, only beyond gun range | 0.496 |
| 1, timed turn (below) | 0.355 |
| 0.25, timed turn | 0.506 |

- **The high yo-yo alone costs as much as the whole yo-yo.** It fires on 4.6% of
  ship-ticks, against 33.5% for the low half.
- **Keeping the gun recovers the lead turn completely** (0.506). For the yo-yo it recovers
  part (0.418), and the rest needs the aspect condition.

### The lead turn rewritten as a timed turn (Shaw Fig 2-9)

- **The attempt.** Fly the gun until the time to the pass falls below the turn's start
  time, then turn at full stick toward his side, so that the turn ends on his track behind
  him.
  - The start time is scale x (R_st sin θ + v_t θ / ω) / closing speed, with θ = arccos(1 -
    min(d / R_st, 2)) and ω = v / R_st.
- **It lost as badly as the original at scale 1** (0.355). At 0.25 it fired on 1% of
  ship-ticks and was neutral (0.506).
- **It fails because the head-on pass is a shot for both ships in this game.**
  - In the d = 2 R_st merge (yardstick against yardstick), both fire gates reach 1.0 at the
    pass.
  - Both ships point their guns, which uses up the 2 R_st (446 m) offset. The closest
    approach is 0.08 gun ranges (51 m).
  - After the pass the 1v1 becomes repeated head-on jousts at about 1 gun range, with
    angles off near π.
  - Any turn before the pass gives up the snapshot, and the turn that Shaw's geometry needs
    has no offset left to use.
- **This game differs from an earlier one.** The v2 study's 0.5 s lead turn
  (sections/maneuvers/progress.md, round 3) won in a different game, against a different
  agent.

### 1v1 head-on merges (60 s, 512 environments)

The subject's damage dealt and taken. Before the fix:

| d | off | lead turn 0.25 | lead turn 1 | timed 1 |
|---|---|---|---|---|
| 1 R_st | 3.7 / 1.2 | 3.7 / 1.7 | 2.3 / 5.6 | 3.4 / 1.3 |
| 2 R_st | 3.8 / 1.3 | 3.7 / 1.7 | 2.3 / 5.6 | 3.4 / 1.2 |
| 3 R_st | 3.6 / 1.2 | 3.7 / 1.7 | 2.4 / 5.7 | 3.3 / 1.2 |

- **Nobody was killed** in these 60 s in any row.
- **At scale 1 the lead turn was active for 17.6 s of the 60, from the first tick** (10
  R_st out). It cut the subject's shot-open time from 9.3 to 8.0 s and multiplied the
  damage it took by 4.6.

### 4v4 brawls with the committed code

Same set-up as above (mixed spawn, 512 games, ±0.022). The scale-1 lead turn and
yo_yo_time 0.5 are the same settings as the "before" rows above.

| setting | before | after |
|---|---|---|
| lead_turn_scale 1 | 0.350 | 0.492 |
| lead_turn_scale 0.25 | 0.463 | 0.484 |
| yo_yo_time 0.5 | 0.355 | 0.475 |
| yo_yo_time 0.125 | 0.457 | 0.518 |
| BFM persona (both, plus graded defence) | 0.295 | 0.451 |
| lead_turn_scale 1, sides spawn | - | 0.473 |
| both rows 0, sides spawn | - | 0.500 |

### 1v1 situations with the committed code

- **Head-on merge, d = 1 R_st** (the other offsets were not rerun; the run hit its time
  bound). The subject's damage dealt and taken was 3.7 / 1.2 with the lead turn off, 3.5 /
  1.3 at scale 0.25 and 3.7 / 1.3 at scale 1. Before the fix, scale 1 gave 2.3 / 5.6.
- **The slow tail chase still loses damage when either feature is on, fixed or not.**
  - The damage dealt was 31.0 off, 15.2 with the lead turn at 0.25 and 3.4 at scale 1. The
    target turns around, the merge weight rises, and the lead pursuit beyond gun range
    turns the chase into head-on jousts.
  - With the yo-yo it was 28.6 at 0.125 and 21.8 at 0.5: the low yo-yo's lead beyond gun
    range.
  - The 4v4 brawls do not show this cost (the low yo-yo alone scored 0.512), so it is a
    property of this one start. It is reported, not explained.
- **The overshoot-prone approach:** every variant killed within 2.08-2.16 s, against 2.14 s
  with both off.

### 1v1 tail chase and overshoot (before the fix)

- **Tail chase:** the damage dealt was 31.0 with the yo-yo off, 29.4 at yo_yo_time 0.125
  and 23.0 at 0.5. The feature was active for 8.8 and 20.0 s of the 60.
- **Overshoot-prone approach:** every variant killed within 2.4 s. The yo-yo slowed the
  kill from 2.14 s to 2.30-2.36 s, because y < 0 there (Y3), so it led instead of lagging.

## Fixes (this branch)

- **The gun owns the nose in range.** The gun weight is now `in_range x (offensive +
  merge)` (L1, Y2). The lead turn and the yo-yo's lag and lead now act only beyond gun range.
- **The high yo-yo lags only in his rear quarter.** Its pursuit half is multiplied by a
  logistic gate on the aspect (Y1). The gate is centred on π/3, Shaw's upper bound, with
  the quadrant gates' scale of 0.1 log units, tempered like them.
  - These are module constants (`HIGH_YO_YO_ASPECT`, `QUADRANT_SCALE`), not table rows. A
    new row would change every agent's vector and identity and break the screening's
    panel.json.
  - This is a judgement call for the author: it could become a fixed row when the panel is
    next rebuilt.
- **The yardstick is unchanged.** With both rows at 0 the pursuit time is 0, so the wanted
  point is the gun's lead point whatever the gun weight. The scripted tests pass (see
  Checks).
  - One test was changed: the lag case in range now checks that the gun owns the nose.
  - One test was added: no high yo-yo lag at the beam.
- **Not fixed:** Y3 (it would change the yardstick), L4 and L5. L2 and L3 are the lead
  turn's own law. Fixing them means a different manoeuvre, and the one tried here lost.

## Re-measured sweep

The screening's skill measure, through the yardstick, 1,152 games a point (error about
±0.015). The panel's BFM member and personas fly the fixed code too, so the yardstick's own
skill moved. Each row is therefore compared with the yardstick measured in the same store.

| lead_turn_scale | before | after |
|---|---|---|
| 0 (yardstick) | 0.609 | 0.551 |
| 0.25 | 0.556 | 0.543 |
| 0.5 | 0.503 | 0.477 |
| 0.75 | 0.464 | 0.495 |
| 1.0 | 0.434 | 0.475 |

| yo_yo_time | before | after |
|---|---|---|
| 0 (yardstick) | 0.609 | 0.551 |
| 0.125 | 0.540 | 0.524 |
| 0.25 | 0.511 | 0.567 |
| 0.375 | 0.455 | 0.510 |
| 0.5 | 0.426 | 0.544 |

- **Yo-yo: flat within the noise after the fix.** The point's drop from its yardstick goes
  from 0.18 to 0.01 at 0.5.
- **Lead turn: the drop from its yardstick at 1.0 goes from 0.18 to 0.08.**
  - The rest is in the sides spawn. By spawn rule, scale 1 against 0 is 0.510 against 0.538
    in mixed spawns and 0.439 against 0.564 in sides spawns.
  - Sides spawns open with a long head-on approach on every ship, where the lead pursuit
    beyond gun range (L2-L4) still acts.
  - Head to head with the yardstick in 4v4 sides spawns, though, the fixed scale-1 lead
    turn scores 0.473 ± 0.022, little below even. Most of the panel's sides-spawn drop
    comes from the other members, the team sizes and the arena radii. Which of these it
    is was not isolated.

## Recommendation

| feature | verdict |
|---|---|
| lead turn | keep off; delete unless rebuilt with the pre-merge offset |
| yo-yo pursuit half | keep off in the yardstick; harmless as a persona trait |
| yo-yo energy half (y in the speed target and cap) | open: y ≈ -1 at most ranges (Y3); review with the yardstick's next search |

- **Why the lead turn does not pay here.** A lead turn needs lateral separation at the pass
  (Shaw Fig 2-9), and gives up the forward-hemisphere shot. In this game the head-on pass is
  a shot for both ships, and the yardstick uses up the separation by pointing its gun.
  - The pre-merge offset of the two-circle plan (shaw-bfm-notes.md, "add the pre-merge
    offset by plan"; lean/scripted/NOTES.md "Open") is what would give it something to use.
  - The yardstick's best two-circle play is already the committed sense after the pass,
    which banks and cashes the store (the author's win against the old agent).
- **Why not to keep tuning the yo-yo's pursuit half.** The yardstick already flies the
  yo-yo's energy half (climb and cap when closing, dive and corner speed when falling
  behind). The pursuit half (lag and lead) gains nothing measurable, even fixed.
- **The CMA-ES search (scripted-screening.md) needs no rerun for these rows.** It ran on the
  old code and left both rows at 0. The fixed rows are now flat or neutral, not harmful, so
  that choice stands.

## Checks on the change

The scripted opponent is on the hot path (the evaluation runner's tick, the BC teacher), so
the change also ran the checks for hot-path code.

- **Operations** (game_ops.py). `act` for both teams: 433 compute operations, against 439
  in lean/scripted/NOTES.md. The rear-quarter gate is skipped when the yo-yo time is the
  number 0, as the lead turn already was.
- **Compile tripwire** (compile_check.py):
  - 4 graphs, 0 graph breaks.
  - 5,763 traced operations against the baseline's 5,721. `act` has 1,721 (baseline
    1,746), `advance` 2,287 (2,220); both are within 10%.
  - **It failed on the cold compile time: 430 s against the 260 s budget.** The laptop's
    load average was about 40 during the run (four CPU brawl jobs and another agent's
    job), and every backend compile took 1.5 to 2.4 times its baseline time. It was not
    repeated, per the measuring policy.
- **One rough speed run** (train_gpu.py): the steady loop ran at 6,565 steps/s (collect 50,117, update 7,554), against the
  2026-11-24 parity row's 6,559 (collect 49,796, update 7,554). Peak memory was 5.95 / 6.79
  GiB. The machine was not quiet.
- **Tests:** `pytest lean/scripted -m "not slow"` gave 51 passed, and ruff is clean on the touched
  files.

## Files

- Fix and tests: lean/scripted/opponent.py, lean/scripted/tests/test_tactics.py.
- Measure: docs/rewrite/sections/lead-turn-yo-yo/measure.py and its output, sweep.json.
- Store: data/scripted-screening/leadyoyo-fixed.sqlite.
- The 1v1 and brawl harnesses and the ablation copies were scratch tools and were not kept
  in the repository.

# lean/game against v2's game: parity check (2026-11-23)

Question: is `lean/game` (core.py, card.py) the same game as v2's `real_scale_elimination`
(`src/bnb/game/`), and is every difference known? Branch `core`. CPU only, one thread.
`lean/game/core.py` and `card.py` were not changed.

**Answer: not yet the same. Three differences in lean's code change the game:**

1. **The lift slope past the stall** (flight). Lean clamps it at 0; v2 keeps it negative
   between one and two stall angles. Any ship past the stall angle flies differently.
2. **The clock's health rule reads the shield before the last tick's recharge**; v2 reads
   it after the recharge.
3. **The clock's health rule takes one signed sum over both teams**, so an exact tie can come
   out as a win; v2 compares the two team sums.

A fourth difference changes only the `killer` output: on equal tau, v2 credits the closer
pass and lean the lower bullet index. A further set of differences are configuration and
scope choices: the default arena, the boundary ramp, no shrink, no random fields, the
"mixed" spawn, the observation and fog. These are listed in the table with where each one
comes from. Everything else, including fields, games that end, the result, unequal and
large team sizes, the "sides" spawn and every card number, matched v2 to float32 rounding.

## What was run

The new test file is `lean/game/tests/test_parity_v2.py`, with 13 tests. It drives v2
through `bnb.game.api` (`move`, `resolve`, `reset`), as `game-audit.md` section 5.2
describes, and uses the same tolerances as `test_against_v2.py` (section 5.5 check A; the
bullet bounds come from the same table). Every run used random sticky actions (each ship
redraws with probability 0.1 per tick) and v2's spread. Each tick it ran:

- **Check A:** one lean step from v2's state, compared with v2's next state. Continuous
  values were compared for ships alive at the start of the tick and not dying that tick.
  Post-stall rows (|angle of attack| above the stall angle) were reported as their own
  group, as section 5.5 asks.
- **Discrete checks:** `alive`, `died`, `fired`, `done`, `result`, `killer`, each bullet's
  `moves_left`, `cooldown` and `next_slot` had to be equal.
- **Check B:** an open-loop lean rollout on v2's actions. An environment was dropped from
  the check at its first discrete difference, at its end, or once one of its ships went
  post-stall (the known difference 1 grows in the open loop; see below).
- **Ends:** for every environment v2 ended, the tick and the result had to match lean's
  open loop.

How v2 was set up:

- The shrink was off up to and including the clock's last tick: `shrink_hold_seconds=1e9`,
  `shrink_end_lead_seconds=-10`. The test asserts that the boundary fraction is 1 on every
  tick up to the clock.
- Lean's ramp was set as v2 sets its own: `boundary_ramp_width = 100 m x R / 2600 m`.
- In the scenarios with fields, v2's random map was replaced by the same fixed 5-field map
  in both games, and v2's cached medium was recomputed. The map has overlapping fields,
  levels -2, -1, 1 and 2, bands of 160 m and 80 m, and one field away from the action.
- Unequal teams used v2's `active=(3, 5)`; lean used 8 slots.

Commands, from the repository root:

```
OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python -m pytest lean/game/tests/test_parity_v2.py -q -s
OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python -m pytest lean/game/tests -q
```

**Whole `lean/game/tests` suite: 13 passed, 4 xfailed in 19.3 s** (two runs: 19.4 s and 19.3 s; the last after the card test was rewritten as a list of 28 pairs). The 4 xfails are the
known differences, marked `xfail(strict=True)`. Each one fails today because lean differs
from v2. If lean is made to match v2, the test passes and pytest reports an error, so the
mark has to be removed.

### Results by scenario

All of these passed check A, with zero discrete mismatches and check B within bounds.
Check A is given as (99.9% quantile, max) of |lean - v2|.

| Test | Size, ticks | What it exercised | Check A, worst quantities | Ends |
|---|---|---|---|---|
| `test_fields_match_v2` | 5v5, 32 envs, 150 | 21,350 of 22,447 ship-ticks and 63,221 of 92,643 bullet moves inside a field; 11,294 shots, 213 deaths | x (1.5e-5, 3.1e-5) m; speed (1.5e-5, 6.1e-5) m/s; nose rate (3.6e-7, 3.7e-4); energy (0, 0.016); bullet x (1.5e-5, 1.2e-4) m; bullet speed (2.4e-4, 4.9e-4) m/s | 3 wipes, all equal |
| `test_unequal_teams_match_v2` | 3v5, 16 envs, 150, fields | 4,927 shots, 75 deaths | x (1.5e-5, 3.1e-5); nose rate (3.6e-7, 1.4e-6); bullet speed (2.4e-4, 3.7e-4) | 1 wipe, equal |
| `test_large_teams_match_v2` | 20v20, 4 envs, 60, fields | 1,864 shots, 136 deaths | x (1.5e-5, 1.5e-5); nose rate (3.6e-7, 3.9e-5) | none ended; 1 killer tie (below) |
| `test_wipes_end_games_as_v2` | 5v5, 48 envs, up to 400, fields, shields 4, 80% fire | 11,049 shots, 391 deaths | x (1.5e-5, 1.2e-4); nose rate (2.4e-7, 1.1e-4) | 39 wipes (21 team 0, 18 team 1), ticks 14 to 64: tick and result equal in 39 of 39 |
| `test_boundary_deaths_end_games_as_v2` | 3v3, 32 envs, 300, 1,000 m arena (ramp 38.5 m), ships at the edge, shields 3 | 537 boundary-damage ship-ticks, 115 deaths | x (0, 6.1e-5); shield (0, 1.9e-6) | 14 wipes, ticks 137 to 296: 14 of 14 equal |
| `test_simultaneous_wipe_is_a_draw_as_in_v2` | 3v3, 8 envs, 1 tick, every ship 100 m out, shield 0 | 48 deaths on one tick | (no living rows) | 8 draws (v2 0, lean 2): 8 of 8 equal |
| `test_clock_ends_games_as_v2` | 5v5, 64 envs, 90-tick clock, shields 0.2 to 1 of full, no fire | the clock and the health rule | x (0, 4.9e-4); speed (1.5e-5, 3.1e-5) | 64 clock ends at tick 90 (38 team 0, 26 team 1): 64 of 64 equal |
| `test_sides_spawn_matches_v2_in_distribution` | 4,096 envs each, 5v5, 4,200 m | v2's `side_placement` and the edge rule | KS statistic against the critical value at alpha 0.001: distance from centre 0.0059 / 0.0136; heading off the outward normal 0.0063 / 0.0136; slot 0 to its team-mate 0.016 / 0.043; to an enemy 0.024 / 0.043; heading cosine with that enemy 0.016 / 0.043. Spawn speed, energy, shield and types exactly equal | |
| `test_card_and_rules_match_v2` | | every entry of `card.CARD` against v2's real-scale card as float32, exactly; 28 shared constants against `rules_of`; stick and power order; quiet sticks; K = 11; tick; clock 5,400; ramp 300 m at 7,800 m | all equal | |

Open-loop drift (check B, maximum over the environments kept):

- Fields: at tick 150, position 4.6e-4 m and speed 4.6e-5 m/s, 21 of 32 environments
  kept (8 dropped as post-stall, 3 ended).
- Wipes: at tick 150, position 2.7e-4 m, 8 environments kept.
- Boundary: at tick 150, position 2.7e-4 m, 31 of 32 environments kept.

The bounds are 0.25 m and 0.05 m/s.

Killer ties: 1, in the 20v20 run (env 3, tick 10, ship 31). Two bullets hit at tau = 0.
v2 credited the closer pass, bullet 381 of ship 34, with a miss of 35.6 m². Lean credited
the lower bullet index, bullet 274 of ship 24, with a miss of 110.1 m². The test checks
that every such mismatch is a tie of this kind and reports it apart. No killer mismatch of
any other kind was seen. There were 0 killer differences in the other new runs, which had 842
deaths between them.

### The known differences, measured (the xfail tests)

| Test | What it sets up | v2 | lean |
|---|---|---|---|
| `test_post_stall_matches_v2` | 320 ships at angles of attack from -1.9 to 1.9 stall angles, one step, coasting | (reference) | max error: angle of attack 7.4e-4 rad (bound 5e-4), nose rate 7.4e-3 rad/s (bound 5e-3), speed 1.8e-3 m/s, x 1.2e-3 m. Pre-stall rows 3.0e-8 rad and 1.2e-7 rad/s |
| (scratch, not a test) | the same states with lean's slope replaced by v2's form, in a copy of core.py | | post-stall angle 1.2e-7 rad, nose rate 4.8e-7: the clamp is the whole difference |
| (scratch) | `test_wipes_end_games_as_v2` without the post-stall drop, lean as is against the patched copy | | open-loop drift at tick 30: speed 0.020 m/s (bound 0.01) against 3.1e-5 with v2's slope; at tick 150 position 0.19 m against 2.7e-4 m |
| `test_clock_counts_the_last_ticks_recharge` | 1v1, one-tick clock, both shields at half, team 0 quiet with its wait over, team 1 boosting | 4 of 4 games won by team 0 | 4 of 4 draws |
| `test_clock_mirrored_health_is_a_draw` | 5v5, 30-tick clock, team 1's shields mirror team 0's (random fractions), nobody fires or recharges | 64 of 64 draws | 53 of 64 called as wins (11 draws) |
| `test_mixed_spawn_matches_v2_deathmatch_in_distribution` | lean "mixed" against v2's nearest rule, team deathmatch's spawn at a reset, 4,200 m, 5v5, 4,096 envs each | | KS statistic / critical value: distance 0.52 / 0.014; off-normal 0.11 / 0.014; to a team-mate 0.15 / 0.043; to an enemy 0.32 / 0.043; facing 0.023 / 0.043 |

Post-stall frequency in these random-action runs:

| Run | Post-stall ship-ticks | Largest angle error | Largest nose-rate error |
|---|---|---|---|
| Fields | 42 of 22,447 | 3.7e-4 rad | 3.6e-3 rad/s |
| 3v5 | 7 of 10,449 | | |
| 20v20 | 11 of 3,716 | | |
| Wipes | 15 of 14,124 | | |

## Every difference, side by side

v2 paths are under `src/bnb/game/`; lean paths under `lean/game/`.

**Changes behaviour?**

- yes: the game plays differently.
- output only: the state is the same and an output differs.
- rare: only in a case that seldom or never occurs.
- no: rounding at most.

**Origin:**

- *bug*: lean's code departs from v2 without a record.
- *builder*: the builder's choice, recorded in lean/game/NOTES.md ("Choices where the audit
  left it open" or "Known differences").
- *author*: an author decision in core-decisions.md.
- *agreed fix*: the NOTES.md list of agreed waste fixes.
- *scope*: not yet built (a later step).

### Behaviour-changing

| # | Area | v2 | lean | Test | Changes behaviour? | Origin |
|---|---|---|---|---|---|---|
| 1 | Flight: lift slope past the stall | `physics/flight.py:88-95` `lift_slope`: (C/alpha_s)(2 - 2r) for r < 2, so negative for 1 < r < 2; used in kappa at `flight.py:331-333` | `core.py:274-276`: `.clamp_min(0.0)`, so 0 for r > 1 | `test_post_stall_matches_v2` (xfail) | yes: every post-stall ship-tick; nose oscillator stiffness and path rate; grows in the open loop | bug: not in NOTES, which reads the post-stall spread as rounding |
| 2 | Clock rule: when health is read | `physics/step.py:309-315`: health = shield after `apply_damage` (damage and the tick's recharge) | `core.py:546`: `shield` after damage, before `restored` (`core.py:534-536`) | `test_clock_counts_the_last_ticks_recharge` (xfail) | rare: only when the lead at the clock is within the last tick's recharge, which is 0.8 shield points per quiet ship (24/s x 1/30 s; 0.0056 of a Yongma's full shield, 0.010 of a Jujak's, by arithmetic) | bug: NOTES Q3 says "v2's health rule" |
| 3 | Clock rule: exact ties | `modes/phase.py:157-158`: sign(sum over team 0 - sum over team 1) | `core.py:547-548`: sign of one sum of +health (team 0) and -health (team 1), which leaves a float residue on an exact tie | `test_clock_mirrored_health_is_a_draw` (xfail) | rare: exact ties with fractional health (53 of 64 mirrored ties became wins). Full or empty shields sum exactly | bug |
| 4 | Killer on equal tau | `physics/combat.py:233-241` with `contact_order` (142-148): earliest by tau, then the smaller miss, then the bullet index | `core.py:512-521`: earliest tau, then the lowest bullet index (miss ignored); the lower index is the lower slot, so team 0 is favoured | counted as `killer_ties` in every run (1 of 136 deaths at 20v20; 0 of 842 in the other new runs) | output only: `StepOut.killer`, so kill credit in rewards; the game state is the same | builder (NOTES "Known differences") |
| 5 | Default arena radius | `modes/rules.py:44, 66-73`: `real_scale_elimination` at 7,800 m | `core.py:45`: 4,200 m | the tests pass R explicitly | yes, by configuration: spawn distance 6,720 m apart instead of 12,480 m | builder (NOTES Q4, the audit's recommendation); not in core-decisions |
| 6 | Boundary ramp width | `physics/combat.py:280`: 100 m x map scale (R / 2,600 m), which is 300 m at 7,800 m and 161.5 m at 4,200 m (`layout.py:105-113`; `ships/real-scale/rules.toml:12`) | `core.py:46, 527`: 300 m at any radius | `test_boundary_deaths_end_games_as_v2` (with v2's ramp passed in), card test (300 m at 7,800 m) | yes at lean's default 4,200 m arena: the ramp is 300 m where v2's rule gives 161.5 m; equal at 7,800 m | builder (NOTES Q4: "assumed, not checked against v2") |
| 7 | Shrinking boundary | `layout.py:242-254`, `step.py:294-296`: holds 36 s (`rules.toml:52`), then shrinks past 0 eight seconds before the clock, so every ship dies before the clock | none (`core.py:524`: constant radius) | v2 held off in every test | yes: v2's elimination never reaches the clock with survivors; lean decides at the clock by health | scope (audit section 4.1, step 1 "elimination without shrink"; NOTES Q3) |
| 8 | Fields | 10 random point-symmetric fields per episode (`physics/spawning.py:89-122`; radii lognormal median 400 m x map scale, band 160 m, `rules.toml:8-20`) | `core.py:49`: a fixed tuple for the run, empty by default (no medium) | `test_fields_match_v2` (a fixed map in both: the medium itself matches) | yes, by configuration: lean's default game has no fields | scope / agreed fix 11 ("fields drawn per environment", not yet done); the author's decision (per environment at reset) is not yet built |
| 9 | Lineup per episode | game default `rules.py:184-185` slot i is type i mod 2 (same as lean); training draws mirrored lineups per episode (`runtime/train.py:255` `lineup_draw = True`, `modes/rules.py:191`) | `core.py:50, 616-617`: a fixed cycle (Yongma, Jujak) | the card test checks the types at reset | yes for training (one lineup instead of drawn ones); no against the game's own default | builder (NOTES Q6) |
| 10 | "mixed" spawn | none in elimination; v2's nearest rule is team deathmatch's (`modes/spawns.py:160-237`): best of 8 uniform candidates in R, farthest from enemies, heading drawn uniformly over the edge-allowed directions (`_heading`, 75-95) | `core.py:634-637`: uniform in 0.85 R, uniform heading, then `_edge_safe` | `test_mixed_spawn_..._distribution` (xfail: KS distance 0.52) | yes, when chosen: a new spawn rule, not v2's | builder (NOTES "mixed" spawn; audit Q4's "v1-like" option) |
| 11 | Fog | `perception/sight.py:94` `visibility`; fog on in training by default (`contracts/pools.py:49` `fog_share = 1.0`) | none: full information (`core.py:717`) | not comparable | yes for what policies see | scope; the author's fog rule (core-decisions, NOTES item 10) is not yet built |
| 12 | Observation: velocity inside a field | world velocity, proper speed / n (`perception/observe.py:120-127`) | proper speed, not divided by n (`core.py:733-734`) | not tested (different contracts) | yes for the policy's input inside fields; not for the game | builder (lean's new contract) |
| 13 | Observation: "fired last tick" | the stored flag `ships.fired` (`observe.py:145`) | inferred from the bullet ring (`core.py:737-739`): a shot that hit on its first move is missed | not tested | yes for the policy's input, in that case | builder (NOTES "Known differences"); agreed fix 12 (store it) not yet done |

### Same behaviour (rounding, or never reached)

| # | Area | v2 | lean | Test | Changes behaviour? | Origin |
|---|---|---|---|---|---|---|
| 14 | A dying ship's speed and nose rate | `step.py:343-344`: speed floor, nose rate 0 | `core.py:559-563`: kept | rows excluded in check A; `alive`/`died` compared | no for the game (dead ships never fly, collide or score); yes for observation rows of dead ships, which lean masks from attention (`key_mask`) but shows | builder (NOTES "Known differences"); agreed fix 3 |
| 15 | Bullet first contact on near-equal tau | `combat.py:142-159`: exact tau order, then the miss | `core.py:467-473`: tau within 1e-6 of the earliest, then the miss | `moves_left`/`died`/damage equal in every run | rare: only two contacts within 1e-6 in tau but not equal; none seen | builder (NOTES Q10) |
| 16 | The medium cached in the state | `flight.py:440-450`, `bullets.py:52-62`, `spawning.py:211`: stored, read next tick | `core.py:396, 343, 360, 369`: evaluated at tick start at the same points | `test_fields_match_v2` | no (same points, same values; 4 evaluations per tick instead of 2) | builder (NOTES); agreed fix 11 caches it |
| 17 | `softmin` | `physics/numerics.py:282-295`: guarded division | `core.py:131-134`: clamp form | flight in every run | no (equal for a, b >= 0; 0 at 0) | builder (NOTES) |
| 18 | Arming distance; classic closeness; bleed exponent; coast exchange | `combat.py:100-117, 203-212`; `flight.py:275-281`; `flight.py:152-156` | dropped | card test: arming 0, core 5 m, bleed 2, `direct_charge` true | no: inert at the real-scale card's values | builder (NOTES) |
| 19 | Result coding | `phase.py:71`, `step.py:314-316`: +1, -1, 0 (0 also "not done") | `core.py:545-551`: 0, 1, 2, -1 not done | every test maps and compares | no (a relabelling) | builder (NOTES) |
| 20 | Unequal teams | benched slots (`step.py:147-159`; `state.py:285-289`), win target from `active` (`phase.py:133-142`) | team-contiguous slots and an `exists` mask (`core.py:607-617`); a wipe of the living ships | `test_unequal_teams_match_v2` | no | builder |
| 21 | A team of size 0 | `active` 0 makes the other team's target 0: it wins on tick 1 | `core.py:542-545`: that team has no living ship, so the other wins on tick 1 | not tested (by reading) | no by reading; NOTES lists it as "not handled" | builder |
| 22 | What the step emits | `contracts/events.py`: damage (E, N, N), boundary damage, shield lost and restored, aspects, win/loss | `StepOut`: (E, N) sums; `damage_taken` includes boundary damage before the clamp | the step's state is compared | no for the game; the learner's ledger cannot be rebuilt | builder (NOTES Q9) |
| 23 | Resets | inside `resolve`, built every tick and selected by `done` (`step.py:347-381`) | `reset_done`, called by the caller (`core.py:684-703`); bullets cleared, tick 0 | spawn state equal in the sides test | no | builder |
| 24 | Observation contract in general | 29 ship columns in metres and joules, field rows, 15 globals, zones, accelerations, team-relative order (`perception/observe.py`, `contracts/obs.py:99-115`) | 27 scaled features, no field rows, no medium or accelerations, 3 globals (`core.py:717-784`) | not comparable | for the policy only; field tokens are needed before training with fields (NOTES item 11) | builder / scope (audit 1.2: "a new, smaller contract") |

Read and found equal by reading, with a test covering each:

- **Flight** (`flight.py` against `core.py:192-344`): the setting with vertical demand and
  direct charge, the rates with drag rise and separation, the kick with its 2x2
  exponential, and the drift along the nose arc with the field correction.
- **Bullets** (`bullets.py` against `core.py:350-373, 399-450`): fire, launch velocity,
  spread, half drag and half bend.
- **Hit test, damage and shields** (`combat.py` against `core.py:453-537`): the hit test,
  kinetic damage, cored closeness, aspect, friendly fire, shields, the quiet rule (stick
  indices 0 to 2, coast, no shot), the recharge and the wait.
- **Boundary damage** at a given ramp.
- **The win rule** (score against the active count, which with no respawn is a wipe) and
  the draw on a simultaneous wipe.
- **The clock tick** (`tick + 1 >= clock`).
- **The "sides" spawn**, including `_edge_safe` (`spawns.py:98-110, 240-272` against
  `core.py:594-633`).
- **The spawn state**: speed x n, energy, shield, wait 0, angle 0.

## What could not be compared, and why

- **Check C**, the distribution of whole games under one controller, was not run. It needs
  v2's `Chaser` ported to lean, or a shared controller, and about 2,048 games per side
  (audit 5.5). Ends were compared game by game instead, within the scenarios above. Those
  were open loop, at most 400 ticks, and on hand-made conditions: low shields, a short
  clock, a small arena.
- **Check D**, the card's designer numbers flown by lean, was not run.
- **Long open-loop horizons**: none beyond 400 ticks. The clock rule was tested on 1-, 30-
  and 90-tick clocks, not at 5,400 ticks.
- **Post-stall in random play**: random actions rarely stall. The difference was measured
  on constructed states and in the open loop, not as a rate in trained play.
- **Random maps, the shrink and fog**: lean does not have them, so there is nothing to
  compare (rows 7, 8 and 11).
- **The observation**: lean's is a new contract, with no column-for-column v2 counterpart.
  The differences were read, not tested (rows 12, 13 and 24).
- **GPU and `torch.compile`**: not run; this task was CPU only.
- **Exact frequencies of the rare differences** (rows 2, 3, 4 and 15) in real play were not
  measured. The tests show that each one exists and where.

## Notes for the author

- **Row 1 (lift slope).** The fix is in `core.py:275`:
  `torch.where(ratio < 2.0, p.lift_over_stall * (2.0 - 2.0 * ratio), 0.0)`. In a scratch
  copy this brought post-stall to rounding (above). It is not applied here, as instructed.
- **Row 2 (recharge before the clock read).** The fix reads health from the shield after
  the recharge: `shield + restored`.
- **Row 3 (exact ties).** The fix sums each team separately, then compares.
- **Row 4 (killer).** Matching v2 means ordering by tau, then the miss, then the bullet
  index.
- **Test bookkeeping.** When any of rows 1 to 3 or the mixed spawn is changed on purpose
  to match v2, its xfail test starts to pass and pytest reports an error until its mark is
  removed.
- **NOTES.md** says the post-stall spread is rounding. It is not; it is row 1.

## Fixed (2026-11-23)

The three behaviour-changing differences were bugs against "lean computes v2's mathematics"
and are fixed in `lean/game/core.py`:

1. Lift slope past the stall: v2's form, negative between one and two stall angles, zero
   beyond (`torch.where(ratio < 2.0, ..., 0.0)`).
2. The clock's health reads the shield after the last tick's recharge (`shield + restored`,
   the value the state stores).
3. The clock compares the two teams' health sums. A lead within float32 rounding of the
   sums (1e-5 relative) is a draw: lean adds each team's ships in a different order inside
   one masked reduction, so an exact mirrored tie can differ by a rounding residue; v2's
   ties are exact only because its layout adds both teams in the same order. Within
   rounding, not bitwise, is the bar.

Their three strict xfails are now plain tests and pass. `pytest lean -q` on the CPU, one
thread: 48 passed, 1 xfailed (the "mixed" spawn, a recorded configuration choice, not a
bug), in 32 s.

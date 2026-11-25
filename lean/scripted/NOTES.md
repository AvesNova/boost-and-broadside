# lean/scripted

The scripted opponent: the yardstick that the evaluation rates against, the BC teacher, and a
population of personas around it. Plus a weaker deterministic chaser. Design and review:
`docs/rewrite/scripted-bfm-design.md`; Shaw's rules against it: `docs/rewrite/shaw-bfm-notes.md`.
Decisions: `docs/rewrite/core-decisions.md` ("Scripted opponent", "Scripted BFM"). History:
`docs/rewrite/core-worklog.md`, judgement calls in `docs/rewrite/morning-notes.md` ("E2b").

## Entry points

`from lean import scripted` (API in `lean/CONTRACT.md`):
`ScriptedConfig()` (the yardstick; `replace(name=value)`), `personas`, `fresh_memory`,
`draw_uniforms`, `act`, `decide` (also returns the `Marginals`, the BC teacher's distribution),
`sample`, `chaser_act`. The sandbox flies against it (`World(opponent="scripted" | "chaser")`).

- `parameters.py`: every parameter is a named row of `PARAMETERS` (default, reference, sampling
  law, half-width at spread 1), plus the gates and `personas`. A reference is a game constant by
  its name in `state.constants` (corner, stall and sustained speed, gun range, sustained radius,
  quarter-turn time, sustained rate, the store) or radians, a store fraction, a probability, a
  ratio. So a persona transfers across ships and time scales. Adding a parameter is one row.
- `situation.py`: `assess` (bearing, aspect, angles off, line-of-sight rate, turn senses,
  flow, energy advantage, store).
- `opponent.py`: `engage` (target), `explain` (one decision with its reasons), `sample`, `act`,
  `decide`, the memory.
- `chaser.py`: the memoryless floor.

## One decision (`explain`, one pass over (E, N) tensors, no data-dependent branch)

1. Target: the nearest known enemy, kept with factor 0.8 (its team's view under fog). A
   remembered enemy's distance adds its roam (seconds since seen x its sustained speed); past
   one arena radius of roam it is dropped (25 s at 4,200 m). No target: fly to the centre.
2. `assess`, and the gun's lead point (drag-law flight time in the medium): the stick's
   wanted point, unless the boundary steer blends in, or the merge's miss offset (`merge_miss`,
   author 2026-11-25, collisions): of the other ships, enemies or team-mates, closing on me
   faster than my own speed (head-on or crossing, not a chase) within `merge_miss_range` (1.25
   sustained radii, 301 m) on a straight-line pass nearer than `merge_miss_offset` (10/3 hit
   radii, 40 m), the nearest; the stick aims the offset beside its centre on the side it is
   already on. The gun keeps the lead point. `ally_clearance` only blocks the gun through a
   team-mate; it does not steer.
3. The fight's gates in one logistic: engaged (3 gun ranges), front and rear on the 3/9 line,
   scissors, the stick caps, the gun. (The spend for a near shot, the two-circle cash, the
   graded defence and the overshoot reversal were removed after the reach check,
   docs/rewrite/scripted-reach.md "Trim".)
4. Tactic weights (sum 1): disengaged, merge, offensive, turning, defensive.
5. Commands mixed as scalars: speed target v* (log space), stick cap. Offensive, v* is the
   chase speed: the fight speed, raised by corner over sustained speed against a turning target.
   Turning: full stick, no cap (the hard turn after the pass, author 2026-11-26); two-circle
   geometry at the fight speed, one-circle down to the speed floor inside the scissors range. Defensive: a break (full stick, corner speed).
6. The engagement's plan and the committed turn sense (memory; author 2026-11-26, see "Fight
   plan" below): the plan is chosen at the first pass against a target and held across passes;
   the sense is committed at each pass by the plan, held while he is aft, released when he
   comes forward or the target changes.
7. Control laws: stick (nested gates, the cap, the side by the sense or the short way, the
   boundary steer), power (climb above v* + band, dive below v* - band, each faded at its store
   edge), fire (aligned, in range at the lead point, never through a team-mate, and only when the
   hit at the predicted aspect, by the game's `aspect_factor` (the nose cone: nothing within 30
   deg of his nose), is worth a tick of the shooter's recharge or empties the target's shield).
8. Sample: one uniform per head, the only randomness.

Memory, carried by every caller (`fresh_memory`, a fresh one per match): `target`, `side` (drawn
once per match, the dead-astern tie-break), `plan_lean` (uniform on [-1, 1), drawn once per
match), `sense`, `plan` (plan x angles off when it was chosen or last switched; 0 with no plan),
`plan_passes` (passes flown under it).

## Fight plan (author 2026-11-26, branch scripted-fight-plan)

- **Choice, once per engagement.** At the first pass against a target (the first commit of the
  turn sense): two-circle when energy advantage + `plan_lean` x 2/3 store fractions is above
  `plan_threshold` (-0.33, BFM-CMA2), else one-circle. The lean is drawn per ship per match in
  `fresh_memory` (as the side is), so the yardstick keeps a mix too; personas also sample the
  threshold. At equal energy a quarter of the ships choose one-circle. Two-circle turns his
  way (toward his side when he is straight), one-circle the other way. Mixed plans fly a
  one-circle fight, equal plans a two-circle one (both one-circle planners both turn away).
- **Held**, through the release of the sense when he comes forward, until the target changes.
- **Switched only when clearly wrong**, at a pass: losing (angles off grown by more than
  `switch_loss`, 45 deg, since the plan was chosen), or stalled (`stall_passes`, 3, passes
  flown and less than `switch_gain`, 10 deg, gained since the choice). The stall rule was
  added after measuring: without it two equal ships in a two-circle plan at full stick meet
  head-on after every turn for the whole clock (1v1 draws 41 to 78 of 256; 98% of the drawn
  games' planned ticks had both ships two-circle).
- **Hard turn after the pass**: the turning tactic has no stick cap (the two-circle middle
  level is gone); the stick's level gates on the error carry full stick at once.
- Numbers: `docs/rewrite/sections/fight-plan/progress.md`.

## The yardstick's defaults

The defaults are BFM-CMA2 (author 2026-11-24, provisional): the values in
`docs/rewrite/sections/scripted-screening/cma_candidate_round2.json`, a CMA search around the
earlier defaults. Six rows still in the table differ from those (plan threshold, fight speed,
aligned, in range, lead gain, temperature); `BFM_DEFAULT_2026_11_24` in `parameters.py` holds
the earlier values of exactly those rows
(`ScriptedConfig().replace(**BFM_DEFAULT_2026_11_24)`). The test persona `BFM` in
`tests/test_tactics.py` (the design's values) stands on the earlier defaults and does not move.
The default BC teacher is the single narrow yardstick (spread 0); the persona population is an
experiment arm.

## Numbers (2026-11-24, core game, 4v4, laptop CPU/GPU)

- Against the agent it replaced (code removed; 4v4, seats alternated, 640 games): 0.413 +- 0.019.
  Against itself the side with more energy wins nearly every offset head-on start.
- Brawl, both teams the yardstick, 256 environments x 5,400 ticks (the clock then; it is 1,800 now): 6.45 kills per
  environment-minute, mean speed 0.72 corner speeds, mean store 0.23, firing 10% of ship-ticks,
  874 games ended, mean length 1,314 ticks, 0.03% of ship-ticks beyond the arena radius.
- Operations, `act` for both teams, E = 256, 5v5: 475 operations (2026-11-26, the fight plan:
  +14 on core's 461); 459 operations (2026-11-26, with the merge miss
  offset: +73); 386 (2026-11-25, with the ghost
  roam and the head-on gun gate; 343 before them, 470 before the lead turn, the yo-yos and the
  no-effect mechanisms went)
  (`docs/rewrite/sections/scoreboard/game_ops.py`).

## Tests

`tests/test_scripted.py` (well-formed actions, spread 0 draws nothing, the power cases, kept
target, drawn side, every number from the game, no shot through a team-mate, one call for both
teams equals two, competence at spread 1, beats the chaser), `test_situation.py`,
`test_tactics.py` (one hand-built case per tactic, the commit, hold, release; the full-stick
turn after the pass; the plan held across passes, switched when losing or stalled; the mix of
plans over leans and personas), `test_scripted.py`'s in-play check of the plan rule,
`test_callers.py` (every caller carries and renews the memory); `lean/game/tests/test_tuning.py`
holds the references equal to the quasi-steady analysis.

## Open

- Not built (Shaw notes): jinks; keeping two-circle after a big bite; the pre-merge offset and
  merge speed by plan; no banking close to a higher-energy opponent.
- Removed (author 2026-11-25): the lead turn and the yo-yos
  (`docs/rewrite/lead-turn-yo-yo-review.md`; `docs/rewrite/author-review-2.md`, "Later
  experiments") and, after the reach check, the overshoot reversal, the spend terms, the
  two-circle cash, the break relaxation and the defend gate. What survives of the yo-yo is the
  chase speed above and `no_bank` against a straight target.
- The in-range gate uses the gun range without fields. No retreat or extend (decision 5.7).
- BFM-CMA2 stays the yardstick, no further search round (morning-notes.md, step 3). The search
  script `scripted-screening/search.py` no longer imports; a round needs new search rows.

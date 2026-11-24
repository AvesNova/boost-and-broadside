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
- `situation.py`: `assess` (bearing, aspect, angles off, closure, line-of-sight rate, turn
  senses, flow, energy advantage, store).
- `opponent.py`: `engage` (target), `explain` (one decision with its reasons), `sample`, `act`,
  `decide`, the memory.
- `chaser.py`: the memoryless floor.

## One decision (`explain`, one pass over (E, N) tensors, no data-dependent branch)

1. Target: the nearest known enemy, kept with factor 0.8 (its team's view under fog).
2. `assess`, and the gun's lead point (drag-law flight time in the medium).
3. The fight's gates in one logistic: engaged (3 gun ranges), front and rear on the 3/9 line,
   defend, shot near, cash, relax, scissors, the stick caps, the gun.
4. Tactic weights (sum 1): disengaged, merge, offensive, turning, defensive.
5. Commands mixed as scalars: pursuit time, gun weight, speed target v* (log space), stick cap.
6. The committed turn sense (memory): commit at the pass by the plan, hold while he is aft,
   release when he comes forward or the target changes, reverse on his overshoot.
7. Control laws: stick (nested gates, the cap, the side by the sense or the short way, the
   boundary steer), power (climb above v* + band, dive below v* - band, each faded at its store
   edge), fire (aligned, in range at the lead point, never through a team-mate).
8. Sample: one uniform per head, the only randomness.

Memory, carried by every caller (`fresh_memory`, a fresh one per match): `target`, `side` (drawn
once per match, the dead-astern tie-break), `sense`, `last_pass` (plan x angles off at the last
commit; a pass that gained under 10 degrees switches the plan).

## The yardstick's defaults

The defaults are BFM-CMA2 (author 2026-11-24, provisional): the values in
`docs/rewrite/sections/scripted-screening/cma_candidate_round2.json`, a CMA search around the
earlier defaults. Ten rows differ from those (plan threshold, lead turn scale, nose-tail
distance, closing time, yo-yo time, fight speed, aligned, in range, lead gain, temperature);
`BFM_DEFAULT_2026_11_24` in `parameters.py` holds the earlier values of exactly those rows
(`ScriptedConfig().replace(**BFM_DEFAULT_2026_11_24)`). The test persona `BFM` in
`tests/test_tactics.py` (the design's values) stands on the earlier defaults and does not move.
The default BC teacher is the single narrow yardstick (spread 0); the persona population is an
experiment arm.

## Numbers (2026-11-24, core game, 4v4, laptop CPU/GPU)

- New yardstick against the old one (4v4, seats alternated, 640 games): scores 0.413 +- 0.019.
  The old agent dives its store away in the first seconds and wins equal starts; the new one
  banks it. Against itself the side with more energy wins nearly every offset head-on start.
- Brawl, both teams the yardstick, 256 environments x 5,400 ticks: 6.45 kills per
  environment-minute, mean speed 0.72 corner speeds, mean store 0.23, firing 10% of ship-ticks,
  874 games ended, mean length 1,314 ticks, 0.03% of ship-ticks beyond the arena radius.
- Operations, `act` for both teams, E = 256, 5v5: 439 compute + 109 view
  (`docs/rewrite/sections/scoreboard/game_ops.py`).

## Tests

`tests/test_scripted.py` (well-formed actions, spread 0 draws nothing, the power cases, kept
target, drawn side, every number from the game, no shot through a team-mate, one call for both
teams equals two, competence at spread 1, beats the chaser), `test_situation.py`,
`test_tactics.py` (one hand-built case per tactic, the commit, hold, release and plan switch),
`test_callers.py` (every caller carries and renews the memory); `lean/game/tests/test_tuning.py`
holds the references equal to the quasi-steady analysis.

## Open

- Not built (Shaw notes): jinks; the high yo-yo's AOT condition; keeping two-circle after a big
  bite; the pre-merge offset and merge speed by plan; no banking close to a higher-energy
  opponent; a lead turn with little offset for the slower ship.
- The lead turn and the yo-yo's pursuit half (`docs/rewrite/lead-turn-yo-yo-review.md`):
  since 2026-11-24 the gun owns the nose inside gun range, and the high yo-yo lags only in
  his rear quarter. The yo-yo's pursuit half is then neutral. The lead turn still costs skill
  in sides spawns (it is lead pursuit to his pass point, not a timed turn). Neither gains;
  both stay off in the yardstick. The yo-yo signal's wanted closure saturates (y ≈ -1 at
  most ranges), which the yardstick's speed target also reads.
- The plan alternates at every neutral pass (Shaw switches after one or two turns).
- The in-range gate uses the gun range without fields. No retreat or extend (decision 5.7).
- The yardstick is still the weaker of the two agents; the screening and search that choose it
  are the next step (`docs/rewrite/core-backlog.md`).

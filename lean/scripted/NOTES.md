# lean/scripted: notes

What the package is today. Design and review: docs/rewrite/scripted-bfm-design.md (E2);
Shaw's rules against it: docs/rewrite/shaw-bfm-notes.md. Decisions: docs/rewrite/core-decisions.md
("Scripted opponent", "Scripted BFM"). Judgement calls of the build: docs/rewrite/morning-notes.md
("E2b").

## Files

- `parameters.py`: every parameter a named row of `PARAMETERS` (default, reference, sampling
  law, half-width at spread 1), the gates (`GATES` before the stick, `STICK_GATES` on the
  stick's error), `ScriptedConfig` (the values; `replace(name=value)`), `personas`. A
  reference is a game constant by its name in `state.constants` (corner, stall and sustained
  speed, gun range, sustained radius, quarter-turn time, sustained rate, the store), or
  radians, a store fraction, a probability, a ratio. Adding a parameter is one row.
- `situation.py`: `assess`: bearing, aspect, angles off (alpha + |lambda|), closure, line of
  sight rate, each ship's turn sense, the flow, the energy advantage, the store.
- `opponent.py`: `engage` (target), `explain` (one decision with its reasons: the marginals),
  `sample`, `act`, `decide`, the memory.
- `chaser.py`: the deterministic memoryless floor (`--opponent chaser` in the sandbox).

## One decision

`explain(config, state, team, persona, memory)`, one pass, (E, N) tensors, no data-dependent
branch (a few Python branches on whether a row is the number 0 for the yardstick):

1. Target: the nearest known enemy kept with factor 0.8 (its team's view under fog).
2. Assessment (`assess`) and the gun's lead point (drag-law flight time in the medium).
3. The fight's gates in one logistic: engaged (3 gun ranges), front and rear on the 3/9 line,
   defend, shot near, cash, relax, scissors, the stick caps, the gun.
4. Tactic weights (sum 1): disengaged, merge, offensive, turning, defensive.
5. Commands mixed as scalars: the pursuit time tau (the yo-yo's lag or lead, the merge's lead
   turn), the gun weight, the speed target v* (log space), the stick cap (full level folded
   into the sustained one).
6. The committed turn sense (memory): commit at the pass by the plan, hold while he is aft,
   release when he comes forward or the target changes, reverse on his overshoot.
7. Control laws: stick (three nested gates in radians, the cap, the side by the sense or the
   short way, the boundary steer), power (climb above v* + band, dive below v* - band; the
   dive only offensive, turning or defensive; each faded at its store edge), fire (aligned x
   in range at the lead point, never through a team-mate).
8. Sample: one uniform per head (the only randomness).

Memory (every caller carries it; `fresh_memory`, a fresh one per match): `target`, `side` (drawn
once per match; the dead-astern tie-break), `sense` (+1 left, -1 right, 0 none), `last_pass`
(plan x angles off at the last commit; a pass that gained under 10 deg switches the plan).

## The yardstick's defaults, set by measurement

Against the old agent the design's values scored 0.18 (morning notes, "E2b NEW AGAINST
OLD"). The yardstick flies the gun's solution: pursuit point = the lead point (yo-yo time 0,
lead-turn scale 0), and breaks whenever defensive within 3 gun ranges (relax angle pi).
Personas draw yo-yo time up to 0.5 t90, lead-turn scale up to 1.0, relax angle pi +- 0.3,
defend range x e^+-0.3. The design's values are the test persona `BFM` in
tests/test_tactics.py.

## Measured (2026-11-24)

Brawls: the core game (`GameConfig()`: fog, ten fields), 4v4, both teams scripted in one call,
256 environments x 5,400 ticks, seed 0, finished games re-dealt; shares of living ship-ticks;
kills per environment-minute at game time. Before: core 484c5d10's opponent.

| | before | new | new, spread 1 | old, spread 1 |
|---|---|---|---|---|
| mean speed (corner speeds) | 0.709 | 0.719 | 0.713 | 0.703 |
| share below 0.75 corner | 0.648 | 0.642 | 0.660 | 0.666 |
| mean store | 0.050 | 0.232 | 0.220 | 0.068 |
| share diving / climbing / coasting | 0.219 / 0.055 / 0.727 | 0.160 / 0.132 / 0.708 | 0.163 / 0.140 / 0.697 | 0.209 / 0.065 / 0.726 |
| share firing | 0.084 | 0.101 | 0.085 | 0.082 |
| kills per environment-minute | 5.44 | 6.45 | 5.78 | 5.14 |
| games ended; mean length (ticks) | 710; 1,370 | 874; 1,314 | 773; 1,436 | 686; 1,548 |
| share beyond the arena radius | 0.0000 | 0.0003 | 0.0004 | 0.0002 |

New against old: 4v4 elimination in the core game, both yardsticks, seats alternated, batches
of 128 to a standard error of 0.02: the new agent scored 0.413 +- 0.019 over 640 games (264
wins, 376 losses, no draws; 130 of 320 as team 0, 134 of 320 as team 1).

Hand-built 1v1 offset head-on starts (no fog or fields; ship 0 at -3 R_st, ship 1 at +3 R_st,
lateral offset d, both at corner speed; 32 games each; wins of ship 0, the new agent):

| d (R_st) | stores 0.5 / 0.5 | 0.8 / 0.3 | 0.3 / 0.8 |
|---|---|---|---|
| new against new: 1, 2, 3 | 12, 14, 20 | 31, 29, 31 | 2, 1, 3 |
| new against old: 1, 2, 3 | 0, 3, 2 | 21, 16, 11 | 0, 0, 0 |

New against new: the side with more energy (store plus speed) wins nearly every game. New
against old: the old agent dives its store away in the first seconds (its store 0.01 at 10 s)
and wins nearly every equal start; the new one banks (0.45 to 0.48 at 10 s) and loses. The
passes recur every 4 to 6 s (about 28 per game) with a mean gain in angles off near zero
(within 0.001 rad new against new), so the plan switches at nearly every pass.

Operations (docs/rewrite/sections/scoreboard/game_ops.py, CPU, one thread, E = 256, 5v5, the
game with fields and fog): `act` for both teams 439 compute (one team 446); before 208. By
stage in the morning notes.

## Checks

- `tests/test_scripted.py`: actions and marginals well formed; spread 0 draws nothing; no
  hold; the power cases; the kept target; the drawn side at a dead-astern pass; the lead in a
  field; every number from the game; the boundary margin in turn radii; no shot through a
  team-mate; one call for both teams equals two; competence at spread 1 (none past the
  boundary ramp, 97 in 100 long-lived ships fire); every sampled row alone at both ends of
  its box competent (one batch); beats the chaser.
- `tests/test_situation.py`: quadrants, flow, closure, line of sight, energy advantage.
- `tests/test_tactics.py`: one hand-built case per tactic (disengaged, tail chases, lag and
  lead, defensive break and grade, merge and turning, two-circle bank and cash, one-circle
  scissors), the commit, the hold and release, the overshoot, the plan switch, the lead turn.
- `tests/test_callers.py`: every caller carries the memory and renews it per match.
- lean/game/tests/test_tuning.py: the references equal the quasi-steady analysis.

## Open

- Not built (Shaw notes): jinks; the high yo-yo's AOT condition; keeping two-circle after a
  big bite; the pre-merge offset by plan; merge speed by plan; no banking close to a
  higher-energy opponent; a lead turn with little offset for the slower ship.
- The plan alternates at every neutral pass (Shaw switches after one or two turns).
- The in-range gate uses the gun range without fields (a bullet's reach in a field is
  shorter). Retreat or extend: none (decision 5.7).

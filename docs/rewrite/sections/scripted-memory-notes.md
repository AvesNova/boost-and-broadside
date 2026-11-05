# Scripted memory: notes

Brief: `scripted-memory-brief.md` (with the author's amendment of 2026-11-05: one parameter
set per game mode). Round one: `scripted-redesign-notes.md`.

## The memory's form

`bnb.controllers.scripted_memory.ScriptedCarry`, per view: each own ship's committed target
(an enemy row, or none), turn side per layer (combat, Frontline), Frontline goal (defend or
attack), recovery, a `fresh` flag and an agent flag (a mixture rung's draw); each enemy's
last seen position and velocity and whether it is known. One code path: memory off
(`Scripted(memory=False)`) makes every ship fresh and forgets every hidden enemy at every
decision, which is the memoryless agent. On the CPU over round one's own games (three
segments of 64 ticks, 8 games per preset, T = 1 and 2), memory off with round one's
parameters gives round one's distribution exactly (largest difference 0.0).

- **Switches.** A plan choice switches to the best alternative with probability
  `1 − exp(−rate dt)`, `rate = r_max σ((advantage − centre) / (scale T))`. Advantages: the
  target's, the log ratio of its distance over its weight to the best's; the turn side's,
  how much shorter the other way round is (radians); the Frontline goal's, the other goal's
  rule margin in ship strengths; recovery's, the other choice's log margin. A fresh ship or a
  lost target (dead, forgotten) re-chooses outright. The Frontline/combat blend is a ramp
  already, so it takes no hysteresis.
- **The turn side** is committed only past an angle: the committed side outweighs the
  error's sign by a logistic in the error (`side_commit_in_stall_angle`).
- **Dead reckoning.** Last seen position plus last seen velocity times the time since seen,
  held inside the arena; its weight as a target a falling logistic in log time
  (`memory_fade_in_seconds`, centre bounded to 0.5–2 s); forgotten below a weight of 1%, on
  its death, or when the prediction falls inside an own living ship's vision.
- **Shooting at a prediction** is allowed, P(shoot) times the target's weight: the same
  fade, so a target just lost (behind a core) still draws a shot and a stale one does not;
  beyond vision range (1024 m, the gun's range about 210 m) the in-range logistic already
  gives a hidden target almost none.
- **BC labels** are the teacher's distribution given its carry: before each update the
  learner runs the teacher over the stored shards in time order, on from the last update's
  carry, and keeps the carry per tick in the update's derived tensors (re-reading a carry
  without switching gives the same distribution; `tests/sections/test_bc.py`).
- **Per mode** (amendment): `ModeParameters`, a set per training preset, picked per view
  from the observation (Frontline's flag; the wide set beyond the geometric mean of the two
  deathmatch arenas). Every use plays the mode's set.

## The search

`bench/scripted_search.py --preset <mode>` from a scratch wrapper whose references are round
one's agent (the anchor, 0; frozen from fd0de232) and the earlier raw agent (frozen from
672364f9, as in round one), one search per preset (amendment): 32 candidates, 128 games per
candidate, a quarter against the references, candidate seats at T drawn from
{1, 1.5, 2, 3}, the top 8 kept, sigma 0.25 x 0.85 per generation, the start round one's
defaults with the memory's starting values. New bounds: each switch's rate in 1-30 per
second, its scale at or above its floor (`SWITCH_SCALE_FLOORS`) and at most 10 floors; the
fade's centre in 0.5-2 s; the side commitment's scale at or above the turn logistics' floor.

The plateau rule on the top rating (patience 4, 10 points) stopped the wide search at
generation 4, held by a lucky top at generation 0 (-99 +- 30); every search was then resumed
with the rule on the median (`--plateau-on median`; the wide one with patience 8). About 23 s
a generation (4,096 games).

| Preset | Generations | Top (gen 0 -> last) | Median | Last |
|---|---|---|---|---|
| frontline | 20 | -190 -> -51 | -251 -> -97 | -465 -> -139 |
| deathmatch | 16 | -195 -> -101 | -308 -> -151 | -837 -> -231 |
| deathmatch_wide | 30 | -99 -> -59 | -211 -> -108 | -680 -> -164 |

(Ratings of candidates at the drawn temperatures, against round one's agent at T = 1.) Five
finalists per preset (kept at least twice, the best mean rating) replayed at T = 1 against
round one's agent in their own preset, 384 games each (+-0.022 to 0.025): frontline 0.408,
0.449, 0.474, 0.478, 0.492; deathmatch 0.385, 0.383, 0.424, 0.406, 0.400; deathmatch_wide
0.504, 0.490, 0.438, 0.438, 0.456. The best of each became its mode's default
(`MODE_DEFAULTS`). What moved: in Frontline the side commitment out of play (its centre 4 ->
about 22 stall angles, beyond any error); in both deathmatches it stayed in play. The
target's switch centre stayed near 0.3 (a third nearer), the fade's centre at 1.1-1.4 s.

## Acceptance

128 games per preset (384 per row), seeds above the search's, the seat drawn per game;
score from the first player's side, +- its standard error. "new" is the agent with its
per-mode defaults; "memory off" the same parameters with `memory=False`.

| First | Second | Total | frontline | deathmatch | deathmatch_wide |
|---|---|---|---|---|---|
| new, T = 1 | round one's agent | 0.454 +- 0.024 | 0.539 | 0.363 | 0.461 |
| new, T = 1 | the earlier raw agent | 0.464 +- 0.024 | 0.594 | 0.312 | 0.484 |
| new, T = 1 | new, memory off, T = 1 | 0.374 +- 0.023 | 0.387 | 0.375 | 0.359 |
| new, T = 1.5 | new, T = 1 | 0.522 +- 0.024 | 0.512 | 0.500 | 0.555 |
| new, T = 2 | new, T = 1 | 0.385 +- 0.023 | 0.406 | 0.359 | 0.391 |
| new, T = 3 | new, T = 1 | 0.207 +- 0.019 | 0.277 | 0.164 | 0.180 |
| new, T = 4 | new, T = 1 | 0.074 +- 0.012 | 0.145 | 0.008 | 0.070 |
| mixture T = 2, p = 0.1 | random | 0.648 +- 0.023 | 0.691 | 0.602 | 0.652 |
| mixture T = 2, p = 0.3 | random | 0.897 +- 0.014 | 0.922 | 0.906 | 0.863 |
| mixture T = 2, p = 0.5 | random | 0.980 +- 0.006 | 0.977 | 0.992 | 0.973 |
| mixture p = 0.3 | mixture p = 0.1 | 0.777 +- 0.021 | 0.758 | 0.805 | 0.770 |
| mixture p = 0.5 | mixture p = 0.3 | 0.706 +- 0.023 | 0.730 | 0.691 | 0.695 |
| new, T = 8 | mixture p = 0.5 | 0.693 +- 0.023 | 0.746 | 0.812 | 0.520 |

- **Memory makes the agent weaker:** 0.374 +- 0.023 against itself with memory off, in every
  mode (0.36-0.39). Each part switched off alone (the others on, 384 games each against
  memory off): the target's hysteresis 0.436, the turn side's commitment 0.414, the goal's
  and recovery's switches 0.445, dead reckoning 0.438. No one part carries the cost; each
  costs about 0.05. At the starting values (before the search) memory cost more: 0.339
  against round one's agent, 0.514 with memory off.
- **Against round one** the new agent is level in Frontline (0.539) and loses deathmatch
  (0.363); about the same against the earlier raw agent. The memory-off agent with the new
  parameters is the stronger: the search's re-tuned numbers are worth more than the memory
  costs.
- **The temperature curve** is flat to T = 1.5, then falls (0.385, 0.207, 0.074 at T = 2, 3,
  4), as in round one.
- **The ladder's bottom:** mixture rungs at T = 2, shares 0.1, 0.3 and 0.5 (each ship the
  agent with probability p per game, else random), space random to T = 8 in steps of 0.65
  to 0.78: random -> p 0.1: 0.648; -> p 0.3: 0.777; -> p 0.5: 0.706; -> T = 8: 0.693.
  `LADDER` is random, the three mixtures, then T = 8, 4, 3, 2.5, 2, 1.5 and the agent; the
  lanes' ladder chain calibrates each against the next.

## Softness audit

`/tmp/bnb-scratch-scripted-memory/tools/audit_memory.py` (round one's method): CPU
self-play at T = 1 with each mode's set, 16 games per preset, 3,000 decisions per seat,
425,000 to 443,000 living ship steps per preset; memory on, and again with memory off.

| Preset | Factor | = 1 | Entropy T = 1 | T = 2 | T = 4 |
|---|---|---|---|---|---|
| frontline | power | 0 | 0.351 | 0.421 | 0.591 |
| frontline | stick | 0 | 0.513 | 0.659 | 0.878 |
| frontline | shoot | 0.769 | 0.014 | 0.025 | 0.045 |
| deathmatch | power | 0 | 0.408 | 0.512 | 0.695 |
| deathmatch | stick | 0 | 0.440 | 0.575 | 0.768 |
| deathmatch | shoot | 0.320 | 0.014 | 0.031 | 0.078 |
| deathmatch_wide | power | 0 | 0.443 | 0.576 | 0.737 |
| deathmatch_wide | stick | 0 | 0.392 | 0.504 | 0.678 |
| deathmatch_wide | shoot | 0.342 | 0.015 | 0.036 | 0.088 |

Joint entropy, T = 1 / 2 / 4: frontline 0.88 / 1.11 / 1.51, deathmatch 0.86 / 1.12 / 1.54,
deathmatch_wide 0.85 / 1.12 / 1.50 (round one: 0.92, 0.84, 0.84 at T = 1).

Per plan choice: changes of the committed option per living ship-second (a ship alive on
both ticks; forced re-choices included), and the share of living ship ticks within 0.1 s
(3 ticks) of one. Memory off for comparison: its "committed option" is the memoryless
choice each tick.

| Plan choice | switches / ship-s (fl, dm, dmw) | memory off | within 0.1 s (fl, dm, dmw) | memory off |
|---|---|---|---|---|
| target | 0.220, 0.233, 0.160 | 0.254, 0.264, 0.162 | 0.050, 0.053, 0.036 | 0.050, 0.055, 0.035 |
| turn side, combat | 0.428, 2.046, 1.246 | 0.845, 5.169, 3.390 | 0.086, 0.364, 0.235 | 0.129, 0.598, 0.423 |
| turn side, Frontline | 1.457, 0.066, 0.097 | 3.157, 0.231, 0.240 | 0.278, 0.015, 0.022 | 0.445, 0.047, 0.051 |
| Frontline goal | 0.029, 0.010, 0.009 | 0.036, 0.012, 0.010 | 0.007, 0.002, 0.002 | 0.008, 0.003, 0.002 |
| recovery | 0.002, 0.012, 0.003 | 0.008, 0.023, 0.010 | 0.000, 0.003, 0.001 | 0.002, 0.005, 0.002 |

The memoryless agent's target was not flipping much (0.25 changes per ship-second, deaths
and respawns included), and memory takes few of them away; memory cuts the turn side's flips by half
to two thirds (a side flip at a near-zero error, where the commitment carries no weight,
still counts). Each logistic's 10-90% width and the share of its inputs inside it (combat
site):

| Logistic | width (fl, dm, dmw) | 4.39 scale / std of ln input | inside (fl, dm, dmw) |
|---|---|---|---|
| turn | 0.070, 0.038, 0.031 stall angles | 0.87, 0.45, 0.27 | 0.07, 0.29, 0.16 |
| level two | 0.144, 0.193, 0.086 | 0.50, 0.31, 0.27 | 0.05, 0.03, 0.06 |
| level three | 0.313, 0.341, 0.426 | 0.93, 0.29, 0.29 | 0.09, 0.02, 0.02 |
| side commitment | 21.5, 2.95, 4.86 stall angles | 0.49, 0.27, 0.31 | 0.00, 0.08, 0.18 |
| boost slow | 0.070, 0.084, 0.082 corner speeds | 0.71, 0.55, 0.52 | 0.00, 0.10, 0.07 |
| boost far | 11.9, 6.1, 9.1 store gun ranges | 0.42, 0.33, 0.28 | 0.00, 0.02, 0.07 |
| reverse | 0.231, 0.229, 0.332 gun ranges | 0.56, 0.53, 0.49 | 0.06, 0.08, 0.17 |
| aim | 0.183, 0.311, 0.400 target sizes | 0.40, 0.42, 0.43 | 0.04, 0.22, 0.17 |
| shoot | 0.431, 0.302, 0.264 gun ranges | 0.46, 0.40, 0.29 | 0.09, 0.09, 0.14 |

## Left open

- Memory costs strength (above). The parts each cost a little; whether a differently shaped
  memory (a target score that weighs the threat, not only the distance; a fade tied to the
  enemy's speed) would pay is untested.
- The deathmatch search plateaued lowest: its best finalist scores 0.424 against round one's
  agent in deathmatch.
- `tests/runs/test_resume.py::...[gauges]` fails on the base commit too (its two-second
  episodes all time out, so the damaged gauges change nothing); not this branch's.

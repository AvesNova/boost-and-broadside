# Imugi by behaviour: what each part does and what it costs

One row per behaviour of v2's scripted opponent Imugi (`src/bnb/controllers/imugi*.py`), with
its operations, lines, CPU time and free constants, and whether lean's opponent
(`lean/scripted/opponent.py`) kept it. Written 2026-11-22 so the author can decide,
row by row, what to keep, drop or simplify.

**Where each number comes from.** (m) measured here (2026-11-22, commands at the end);
(a) from the audit, `docs/rewrite/sections/game-audit.md` section 6.2 or
`docs/rewrite/simplification.md`; (e) estimated. Lines and parameters were counted from the
source (counted). Operations are compute ops per call by the `regions.py` rule (eager
PyTorch, views not counted), which does not depend on the batch size. Times are for one call
on one team's views, **E = 256 environments, 5v5 (256 views of 5 own ships), CPU, one
thread**, real-scale elimination with no shrink or fog, the state taken 450 ticks (15 s) into
Imugi-against-Imugi games (96% of ships alive). A row's time is its share of the measured op
time multiplied by the call's measured wall time (5.81 ms), so it includes its share of
Python overhead.

## The table

Totals for the full Imugi call (stored doctrines, memory on), in the same units:
**2,122 ops (m; the audit has 2,124), 5.81 ms (m)**. Keyed draws are split out of the rows
that use them and shown in their own row; their ops are not repeated in the other rows.

| # | Behaviour | What it does in the game | Ops per call | Lines | Time per call, E = 256 | Free constants | In lean's opponent? | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | Target choice | Each ship picks the nearest known enemy by distance divided by its memory weight; the committed target is kept until a switch is drawn (row 10) | 49 (m) | about 42 (counted) | 0.22 ms (m) | 0 of its own (the switch is in row 10) | Simplified: nearest living enemy by true distance; with the optional memory, the kept target wins unless another is closer than 0.8 of its distance (one comparison, no rate, no draw) | Matters in every mode. Imugi chooses from the enemies it remembers (row 7); lean chooses from the true state |
| 2 | Lead (intercept) | Aims where the target will be when a bullet fired now arrives: first-order intercept using the drag law's time of flight, the distance capped | 37 (m) | about 22 (counted) | 0.11 ms (m) | 1 per mode (the cap) | Kept (real-scale cap 4.78 gun ranges), plus a per-persona lead gain | The audit infers that lead, soft gates and boundary steer are where the strength over the chaser comes from (60 of 61 wins) |
| 3 | Stick: turn and level gates | P(turn) and the two stick levels as logistics in the angle error over the stall angle, each floored at 0.02; the side by a committed side for errors near 180 degrees; with no target and near the centre, fly straight | 165 (m): two calls of about 82, the second (the no-target "hold straight") computed for every row and used only by targetless ships near the centre | about 70 (counted) | 0.38 ms (m) | 6 per mode (3 logistics) + 1 (centre hold) + 4 fixed floors | Simplified: the three gates kept with the same centres and floors; side = the error's sign with a 0.02 floor; no committed side; no hold-straight call (targetless ships steer to the centre) | The side's commitment is in row 10, its draws in row 13 (132 ops for the three stick calls' side draws, including Frontline's) |
| 4 | Power: boost, coast, reverse | Boost when slow (in corner speeds) or far with energy in store; reverse when very close (in gun ranges); coast otherwise; floors 0.04 | 48 (m) without doctrines; 67 (m) with them (the doctrine's extend power reuses these functions) | about 30 (counted) | 0.10 to 0.15 ms (m) | 6 per mode (3 logistics) + 2 fixed floors | Kept (no per-type power rule) | Matters in every mode. In the real-scale set the "slow" gate (2.61 corner speeds) is open nearly always (lean NOTES) |
| 5 | Gun gate | P(fire) = aligned (aim error over the target's angular size, clamped 1 to 45 degrees) x in range x the target's memory weight; 0 while the gun cools or the store cannot pay | 48 (m) | about 12 (counted) | 0.13 ms (m) | 4 per mode (2 logistics) + 2 fixed (the size clamp) | Kept, without the memory weight (always 1 there) | No floor on the shot: a per-tick floor wastes energy (imugi_parameters.EPSILONS comment) |
| 6 | Boundary steer | Within a margin of g-limited turn radii of the edge and flying outward, blend a steer to the centre into the wanted direction | 49 (m) | about 22 (counted) | 0.09 ms (m) | 5 per mode packed (margin + a 4-number ramp of which 2 are free) | Kept (real-scale 2.28 turn radii, ramp 0.0554 to 0.52) | Reads the boundary's current radius, so it follows elimination's shrink in v2. Lean reads a fixed arena radius from its config |
| 7 | Dead reckoning and last-seen memory | Remembers each enemy's last seen position and velocity; a hidden enemy is predicted forward, kept inside the arena, faded as a target (centre about 1 s) and forgotten when the fade is below 1%, when it dies, or when its prediction falls inside an own ship's vision | 74 (m) (76 with memory off: the code still runs) | about 85 (counted), plus the carry's 45 shared with row 10 | 0.22 ms (m) | 2 per mode (the fade logistic) + 1 fixed (forget at 1%) | Dropped | Only matters under fog: without fog nothing is hidden. Reads only the team's observation (`time_since_seen` is observed), so it is not cheating. imugi.md measured that the memory as a whole costs strength (0.374 against itself with memory off) |
| 8 | Type doctrines | Per ship type, bends the aim and power against the committed target by an advantage logistic over the type ratios: slash (boost into the attack), bleed (reverse to corner speed in a turn fight), lag (aim behind the target), deny (turn off a head-on line), extend (fly away when out of energy or off the nose) | 312 (m) | about 130 in `imugi_combat.py` + 314 in `imugi_doctrine.py` (shared with row 9) + 182 of JSON (counted) | 0.81 ms (m) | 32 per type (counted from `Doctrine`); 2 types in real scale, so 64 | Dropped | Stored real-scale values: Yongma slash 0.874, lag 0.129, deny 0.022; Jujak bleed 0.816, deny 0.806, lag 0.305; extend 0 for both but computed. The audit measured no strength from doctrines in elimination without fog (0.49 +- 0.07, 64 games); they were tuned for mixed-type duels and Frontline. Reads a seen target's energy store and shield: both are in the observation, so not cheating, but worth checking that a human player can see them too |
| 9 | Manoeuvres | After the merge, picks a turn sense against the target (one-circle or two-circle), crowd extension and duel extension, and blends a commanded stick and power over the combat layer's | 385 (m) | about 175 (counted) | 1.17 ms (m) | 31 per type (counted); 62 for 2 types | Dropped | In the stored real-scale doctrines only the sense is on (sense 1, one-circle, same-type duels only); crowd sense 0 and duel extension 0, yet their logistics are computed every call. Reads the target's nose rate and shield |
| 10 | Switch-rate commitments | Plan choices (target, turn side) are kept and switch with probability 1 - exp(-rate dt), the rate rising with how much better the alternative is; turn-side commitment for large errors; everything in the ship's own time unit | 57 (m), plus their draws in row 13 (176 ops for target and side) | about 55 (counted) + the carry (45, shared with row 7) | 0.13 ms (m) | 9 per mode (2 switches of 3, the side commit's logistic, the time-unit exponent) | Dropped: replaced by row 1's single comparison and a per-persona action hold (redraw every 1 to 6 ticks) | The Frontline goal and recovery commitments are in row 11. imugi.md: memory, of which these are part, loses to the memoryless agent in every mode |
| 11 | Frontline layer | Per Frontline zone strengths (Gaussian kernels over visible ships' health), assigns defenders and attackers, separation from allies, recovery to spawn when hurt and threatened, a capture-lock term, then blends its stick and power with combat by the target's distance | 435 (m) (411 without doctrines; the doctrine's lock term is 24), plus 88 of its own keyed draws (row 13); the audit's 567 includes its draws and stick call | 237 (counted: the whole of `imugi_frontline.py`) | 1.40 ms (m), plus 0.16 ms of its draws | 16 per mode (10 + two switches of 3) + 1 per type (lock press) + 3 fixed | Dropped | Matters only in Frontline mode, but is computed on every row in every mode and selected away (`torch.where(in_frontline, ...)`). Reads only visible enemies and the own team's respawn timer |
| 12 | Mode inference and per-view parameter sets | Infers each view's game mode from the observation (map size in gun ranges, gravity, mode flags) and gathers that mode's 50 parameters per view | 24 (m) (and 53 view ops) | about 90 (counted) + `imugi_defaults.py` 151 + `imugi_real_scale.json` 257 + `imugi_populations.json` 5,170 (unused, read by one test) | 0.07 ms (m) | 0 searched; 2 fixed thresholds; it is what multiplies rows 2 to 11 into 6 sets x 50 = 300 stored numbers | Dropped: one parameter set, game constants passed in | Every parameter becomes an (R, 1) tensor, so every logistic pays a tensor multiply instead of a constant |
| 13 | Keyed random draws | Hashed counter-based uniforms per purpose, with salts kept so the draws match earlier runs: target switch, turn side (three stick calls), sense, Frontline goal and recovery, and the three action draws | 352 (m): side 132, Frontline goal and recovery 88, target 44, sense 44, actions 44. The audit counts 396; simplification.md 616 per tick for two calls | about 10 in Imugi + `bnb/streams.py` (about 45, shared with the rest of v2) (counted) | 0.67 ms (m) | 0 | Dropped: one ordinary `torch.rand` of (E, N, 3), 1 op, 0.018 ms (m) | Exists for repeatability against earlier runs, not for play |
| 14 | Temperature and share (the ladder's rungs) | Temperature divides every logistic and switch's scale, flattening decisions; share makes a ship play uniformly at random with a probability drawn each episode | 1 (m) direct; about 43 (e) inside the logistics (one multiply each of the 43 calls) | about 30 (counted) | about 0.1 ms (e) | 2 per view (set by the league) | Simplified: temperature is a persona field folded into the gate sharpness at reset (no per-call cost); share not written; a persona population added (centres, temperature, stand-off, weak-target preference, lead gain, action hold) | Only matters for a ladder or a BC teacher at several strengths; at share 1 the share costs nothing |
| 15 | Reading the view, per-type lookups, output assembly | Splits own and enemy rows, looks up per-type characteristics (gun range, stall angle, corner speed...), then masks dead ships to the no-op, applies the uniform rung and selects combat or Frontline | 52 (m) (22 reading, 30 assembly) | about 100 (counted) | 0.20 ms (m) | 0 | Simplified: reads the true state; masks uncontrolled rows; characteristics from the game's constants or the config's Vizia fallbacks | Imugi reads only the team's observation (fog applied). Lean reads the true game state, all enemies included, and flies Yongma and Jujak with Vizia's gun range, corner speed and launch speed (a few percent off): harmless without fog, cheating under fog |
| 16 | Action sampling | One action per head by the inverse CDF of each marginal | 15 (m) | about 15 (counted) | 0.07 ms (m) | 0 | Kept (inverse transform, `uniform >= cumulative`) | The marginals double as BC soft labels in both |

The rows sum to the measured 2,122 ops. The logistic gates are inside the rows: 43 logistic
calls per call (m; the audit's count is the same), about 9 ops each (a), 387 ops in all.

**Cross-check by ablation (m).** Imugi without doctrines: 1,340 ops, 3.56 ms. So doctrines
and manoeuvres together cost 782 ops and 2.25 ms, against 2.1 ms from rows 8, 9 and the
sense draw. Without doctrines or memory: 1,342 ops, 3.57 ms (memory off still runs the
memory code).

**Both teams in one call (m):** Imugi on 512 views (both teams of 256 environments),
8.21 ms.

## Lean's opponent as it is

`lean/scripted/opponent.py:act`, one team per call, E = 256 environments, 10 ship slots
(lean computes every slot and masks the other team's, so it does twice the rows of Imugi's
call for the same decisions), same machine, one thread, the state 450 ticks into
yardstick-against-yardstick games in `lean.game`.

| | Ops per call | Time per call, E = 256 |
|---|---|---|
| `act`, spread 0, no memory (the yardstick as tested) | 172 (m) | 0.66 ms (m) |
| `act`, spread 0, with memory (hysteresis and hold) | 183 (m) | 0.76 ms (m) |
| `act`, spread 1 persona, with memory | 183 (m) | 0.77 ms (m) |
| `draw_uniforms` (its one random draw) | 1 (m) | 0.018 ms (m) |
| `chaser_act` (the floor) | 67 (m) | 0.25 ms (m) |
| Both teams, two `act` calls | 344 (m) | 1.36 ms (m) |
| *Imugi, full, for comparison* | *2,122 (m)* | *5.81 ms (m); 8.21 ms both teams* |

Lean's yardstick is about 12 times fewer ops and about 9 times less time per call than
Imugi; per tick (both teams) about 6 times less time. Lean's NOTES give 203 calls (166
without views and indexing) by the profiler; the 172 here is the `regions.py` rule.

## How this was measured

Machine: the development desktop, 16 CPUs, platform profile `performance`, load average
0.1 to 0.3 before the runs; `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`; no GPU, no
compilation. From the repository root (branch `core`):

```
OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python \
  docs/rewrite/sections/imugi/measure_imugi.py 256 450
OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python \
  docs/rewrite/sections/imugi/measure_lean.py 256 450
```

(The scripts and their logs, `imugi_E256.log` and `lean_E256.log`, are in
`docs/rewrite/sections/imugi/`.)

- **Imugi.** `bnb.game.api` with `real_scale_elimination`, shrink held off
  (`shrink_hold_seconds = 1e9`, as in the audit's `match.py`), fog off, E = 256. Imugi
  (`Imugi(characteristics, 5)`, stored doctrines, memory on) plays both teams for 450 ticks;
  then team 0's 256 views and carry are the input to every measured call
  (`Imugi.step(carry, obs, reset=False, key)`).
- **Ops per behaviour.** A `TorchDispatchMode` counts each dispatched op, skipping view ops
  and the metadata ops `regions.py` skips, and assigns it to a behaviour from the Python
  call stack inside `bnb/controllers/imugi*.py` and `bnb/streams.py`: any op under
  `choice_uniforms` or a keyed stream goes to keyed draws (split by its caller); else the
  outermost of `frontline`, `remember`, `mode_index`/`resolved`, `doctrinal`/
  `advantage_weight`, `maneuvers`/`commanded_stick`; else `commit`/`switch_probability`/
  `time_unit` go to commitments; else the innermost named function (`choose_target`,
  `lead`, `stick_marginal`, `slow`/`far`/`power_marginal`, `steer`); else by line range
  inside `combat`, `context` and `decide`. Logistic and floor helpers count toward their
  caller.
- **Time.** Wall time of 40 calls after 5 warm-up calls, median (minimum within 2%). The
  share per behaviour is the sum of each op's own dispatch-to-return time over 5 counted
  calls; a row's time is that share times the median wall time.
- **Lean.** `lean.game.reset`/`step` with the default `GameConfig` (4200 m arena), 5v5,
  E = 256, the yardstick on both teams for 450 ticks; `regions.Counter` counts one `act`
  call; wall time is the median of 60 calls after 5 warm-up calls.
- **Lines** are non-blank source lines, docstrings and comments included, of the functions
  and line ranges that make up each row. **Free constants** count every number a parameter
  set packs (`imugi_parameters.pack`: 50 per mode; `imugi_doctrine.pack`: 64 per type) by
  the row that reads it, and name fixed module constants separately.

## Not measured

- Strength per behaviour: only the audit's 64-game ablations exist (doctrines, memory);
  nothing here measures what any single row is worth in play.
- Behaviour in Frontline or under fog: the operation counts are the same in every mode
  (every layer runs on every row), but the time was taken only in elimination without fog.
- The temperature's cost inside the logistics (row 14) is an estimate; it cannot be
  separated from the logistic it scales without changing the code.
- GPU and compiled costs: everything here is eager CPU.

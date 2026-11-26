# Overnight plan, 2026-11-27 (author's go at ~01:30; soft target 09:00)

Coordinator: the main Claude session. Two workstreams, two fresh agents, run in parallel with one
GPU lock. Nothing is merged into `core` overnight; both branches wait for the author's review.

## Rules for both agents

- `uv run --no-sync`; never `pkill -f` / `pgrep -f`; own worktree, never switch branches in the
  shared v2 folder.
- One GPU job at a time across BOTH agents: every GPU job runs as
  `flock /home/vizia/avesnova/boost-and-broadside/data/gpu.lock systemd-run --user --wait
  --unit bnb-<name> -p MemoryMax=12G ...` (the lock serialises them; `reset-failed` own units).
- One source of truth: the game's own functions and values; tests read real values.
- Speed work order: profile, cut work, remove bloat, fuse/chunk/CUDA graphs, custom kernel last.
- Commit small, push after each step; milestone notes (tried / worked / didn't / next, numbers
  measured only) in the branch's progress.md; ETAs as clock times.
- Testing once at the end of the branch: quick tier on touched tests, ruff, compile check.
- If a step hits a real problem, stop that step, record it with numbers, go on with the rest.

## Stream A: scripted look-ahead, round 3 (branch `scripted-lookahead`, worktree v2-lookahead)

Plan already in docs/rewrite/sections/scripted-lookahead/progress.md "Round 3". In order:
1. Steady-flight lookup table built once from the game's own functions (speed x stick level x
   power; interpolated, as the Sustain table is), replacing the 8 inlined steady-flight
   evaluations. Target: cold compile under the 260 s budget (now 333 s; `act` 7,864 traced
   operations), lower path cost. Re-check path prediction error (path_check.py).
2. Grouped distribution check: tick-to-tick total variation of the stick distribution after
   grouping actions by effect (left / straight / right; steady turn rate), next to the raw
   0.14-0.15. Report; tune nothing for sampled switching.
3. Power and energy rebalance: store runs nearly empty (mean 0.06-0.16 of full, dive 55-65%,
   climb 3-6%). Weight the store more, weaken the preferred-speed term; compare with round 1
   (0.21-0.52); check the slow near-stall empty-store 1v1 (round-2 replay 1v1 seed 1).
4. Final pass: remove the rule-based agent (lean/scripted/rule_based and every use; xfailed
   tests that encoded it are rewritten or removed), tests, compile check (report; do not update
   the baseline), docs (lean/scripted/NOTES.md, core-decisions entry, backlog E6), new replays
   (1v1 fog off no fields x2, 4v4 fields fog x2), re-measure (cost, draws, kills/min, merge turn,
   42-action use, entropy, store, power shares).
Coupling with stream B: the fire-worth rule must read the game's own hit-worth function (today
`aspect_factor` and the hit damage) through one call, so a change of the hit model in `core`
flows into the agent without edits. Do not merge `core` again overnight.
Author notes: 1v1 mirror draws are not a target; judge flicker on distributions, not samples;
1v1 tests run with fog off and no fields; 2 s look-ahead kept.

## Stream B: bullet deflector and a fast hit-model workbench (new branch `bullet-deflector` off `core`)

1. Mechanism (lean/game/core.py, in the bullet hit test / move): a deflector around each ship.
   A bullet whose straight path within the tick passes at closest distance b from a ship's
   centre is pushed out to about b + D, on the side it was passing (smooth sign
   b / sqrt(b^2 + eps^2), eps a fraction of a metre). Damage uses the pushed-out distance in the
   existing core/falloff graze. After the closest point the bullet continues bent by the
   matching small angle; speed unchanged (never adds energy). Applied only on the tick whose
   straight path contains the closest-approach point (exactly once per pass). Ghost bullets
   follow the same code (hit labels stay right). Constant with shield strength (a deflector,
   not the shield). A second ship in the same tick uses the unbent path (rare; noted).
2. Two presets, switchable by settings:
   - "symmetric": D constant at every aspect;
   - "frontal": D at least the hit radius (12 m) in the front cone (30 deg, fading to the side
     value by 45 deg), a few metres elsewhere: every frontal bullet glances off left or right
     ("the samurai sword"), replacing the nose-cone damage rule (aspect_factor) in this preset.
   The deflector function is one game function the scripted agent's fire-worth rule can read.
3. Fast iteration (the author's main ask): every hit-model number (hit radius, core radius, D
   front/side, cone angles, spread, hit damage) is a runtime tensor input, so changing it needs
   NO recompile (check how ship constants reach the compiled tick; Python floats bake into the
   graph). Prove it: change a value between calls of the compiled tick and show no recompile.
4. Workbench tools (CPU-first, using the game's own functions on synthetic bullets):
   - effective radius: given the settings, damage against aim offset at each aspect and range,
     the effective hit radius and apparent angular size of the target (mrad) by aspect/range;
   - hit chance per bullet against aim error (Gaussian aim error sigma, plus the game's spread)
     by range and aspect; damage per second while tracking; time to kill;
   - a one-command report (table + PNG) that re-runs in seconds when a number changes;
   - a live loop: the sandbox (lean/tool) gets the deflector settings as sliders (one source of
     truth: the same settings), applied without restarting or recompiling.
5. Measurements: god-mode scripted games (look-ahead agent from `scripted-lookahead` is NOT
   available on this branch; use the rule-based agent on core, or a fixed-aim shooter, and say
   which) for hits per pass, share of hits in control streaks, frontal hits (should be 0 in the
   "frontal" preset), kills per minute with normal damage; compare off / symmetric / frontal.
   Cost: tick time and compile-check operations, off against on.
6. Rendering: bullets drawn tick to tick show the bend as a kink (no special drawing needed);
   check a replay or still; optionally a small spark at the deflection point if cheap.
7. Tests (quick tier): pushed-out distance, no energy added, exactly one bend per pass, ghosts
   bend like real bullets, frontal preset gives no frontal damage, settings change without
   recompile. Docs: lean/game/NOTES.md section, core-decisions entry (proposal, unmerged).

## Coordinator

Watches both, answers their questions within the author's decisions, keeps this file's status
section current, and writes the morning summary at the bottom.

## Status

(updated overnight)

- 2026-11-27 launch: both agents started. Stream A in v2-lookahead (`scripted-lookahead` at eeabe98b, GPU units `bnb-la-*`); stream B in v2-deflector (`bullet-deflector` off core 146f2d4a, GPU units `bnb-df-*`). Shared lock `data/gpu.lock`. Soft finish 08:30, morning summary 09:00.
- 02:56 stream B done (`bullet-deflector` at b3c7ae19, unmerged). Agent's measurements (rule-based scripted, god-mode 1v1, 4 seeds x 128 games x 60 s), off / symmetric / frontal: hits per pass 12.2 / 9.2 / 9.0; damaging hits within 30 deg of the nose 0 / 0 / 0, contacts there 205 / 158 / 0; kills/min 1v1 0.80 / 0.61 / 0.62, 4v4 4.68 / 4.01 / 3.94. Tick 4096 envs 4v4: off 0.97 ms, frontal 1.21 ms. Compile check: `advance` 3,219 ops for frontal against the 2,856 baseline (+13%, over the 10% tripwire); off 2,998. Separate commit f583eb0c (2x2 matrix exponential written out: step 1.76 -> 1.07 ms with the deflector off) for the author to keep or drop. Workbench `python -m lean.tool.hitbench` re-ran here in 2 s (coordinator check). Sandbox sliders not flown by eye. Six open questions in the branch's questions.md.
- 03:10 stream A done (`scripted-lookahead` at 56f70453, unmerged). Agent's numbers: steady-flight table cut `act` from 7,864 to 4,451 traced ops; cold compile 333.0 -> 280.2 s, still over the 260 s budget. Grouped flicker: raw 0.146/0.148, by side 0.068/0.089 (1v1/4v4). store_weight 0.25: mean store 0.44-0.51 (was 0.07-0.12), dive 23-25% (was 57-60%); beats round 2 68% in 1v1, 57% in 4v4. Weaker speed term tried and rejected (loses to round 2). Rule-based agent removed; hit worth through one call `core.hit_damage`. Cost 5.35 ms per call (was 5.70). Replays in data/replays-2026-11-27-lookahead-r3/ (not watched).
- Coordinator check: both branches pushed and clean; no `bnb-*` units left; the two branches conflict on merge (core.py, opponent.py, test_scripted.py, core-decisions.md, questions.md): both define a hit-worth function (A `core.hit_damage`, B `core.hit_fraction`) and B edits the rule-based agent A deleted.

## Morning summary

Both streams finished early (02:56 and 03:10). Nothing merged into core. Numbers below are the agents' own measurements unless marked.

Stream B, bullet deflector (`bullet-deflector`): the mechanism, presets off/symmetric/frontal, all hit-model numbers as runtime inputs (no-recompile test passes), workbench `PYTHONPATH=. uv run --no-sync python -m lean.tool.hitbench` (re-ran in 2 s, coordinator), sandbox "Hit model (live)" sliders (not flown by eye). Frontal: 0 bullets reach the ship within 30 deg of the nose; hits per pass 12.2 -> 9.0, the 3 m side push costs about a quarter of hits. Tick +0.24 ms. Compile check fails for frontal (+13%). Separate speed fix f583eb0c (step 1.76 -> 1.07 ms).

Stream A, scripted round 3 (`scripted-lookahead`): all four steps done; compile 280 s still over 260 s; store fixed; rule-based agent removed.

Decisions for the author:
1. Deflector: bend angle rule, bend near misses too, dead-centre pass-through, side D 3 m or ~1 m, frontal as default, collision blame angles (branch questions.md).
2. Keep or drop the matrix-exponential speed fix f583eb0c.
3. Deflector compile +13%: accept, or trim first.
4. Scripted compile 280 s: paths kernel, 4 segments instead of 5, or a larger budget.
5. Scripted store_weight: 0.25, or higher (1v1 duels prefer 0.4-0.6, 4v4 even with fewer kills).
6. Merge order: the branches conflict; suggest merging the deflector first, then rebasing the scripted agent so its fire-worth reads the deflector's hit function.

## Morning run (05:20 to noon)

Author decisions 2026-11-27 (deflector discussion): bullets turn on entering an invisible egg around the ship (front 53.3 m, rear 24 m, half-width 24 m), by theta_max x 27/4 x (1 - x)^2 of the impact parameter over 24 m, theta_max 45 deg; speed kept; dead centre straight; hit radius 12 m, core 10 m; nose-cone rule kept as code, off; friendly fire as always (never the owner); no drawing of the egg. Sketches on `bullet-deflector` (docs/rewrite/sections/bullet-deflector/sketch/).
One agent on the new branch `deflector-lookahead` (worktree v2-combined, GPU units `bnb-egg-*`): merge the two branches, replace the deflector, re-measure, then experiments (deflector off/egg x spread x 1v1/4v4) until about 11:30; findings in docs/rewrite/sections/bullet-deflector/egg-experiments.md; report by 11:45.

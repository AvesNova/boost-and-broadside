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

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
Author answers (~05:40): keep the matrix-exponential speed fix; compile budget provisionally raised (revisit); collision blame unchanged (aggressor punished more); time to kill: ~0.4 s only when everything is perfect, typical 0.6-0.8 s, ~1.0 s upper limit when only dispersion and distance cost hits; spread decided from the experiments; store_weight open, try situation-dependent weights (enemies near, ships alive). Relayed to the agent.
- ~07:25 agent done (`deflector-lookahead` at 180d16a8, pushed, unmerged). Agent's numbers: egg hit widths 2.05 m head-on, 4.75 m side/astern (matches the sketch); tick 0.98 -> 1.37 ms (+40%) after one cut; `advance` 3,296 ops (+15% over 2,856), `act` 4,590, cold compile 293.4 s. God mode off/egg: hit chance per bullet 0.75 / 0.28, head-on share of hits 0.09 / 0.01. Play: egg removes head-on kills in 1v1 (45% of kills without it), kills/min 1v1 3.25 -> 1.45, 4v4 11.1 -> 6.6; no degenerate behaviour found. Spread 1.5-20 mrad and hit damage 7-14 barely move play kill rates (the agent's aim gate dominates); theta_max does (30/45/60 deg: 1.56/1.45/1.23). Workbench time-to-kill band: hit damage 8.5 at <= 5 mrad, or 7 at today's spread; 10.5 gives 0.33 s. `store_crowd` (0.5, k = 1, off by default) beats fixed 0.25 and 0.40 in 1v1, even-to-better in 4v4. 11 replays in data/replays-2026-11-27-egg/ (not watched). Write-up: docs/rewrite/sections/bullet-deflector/egg-experiments.md; 7 open questions there and in the branch's questions.md. Quick tier 342 passed, ruff clean.
- ~13:00 hit range tab on branch `hit-range` (off `deflector-lookahead`, head 063819c7): fixed target (Q/E), east-facing always-firing shooter (drag/WASD), range rings, egg, true-size ships, vision outlines, 10 s DPS / time to kill, 2-sigma spreads before and after the egg, apparent half-width. Not merged.
- Tabled (author ~13:40), bullets that fly on after a hit: a hit turns the bullet into a ghost instead of stopping it; drop the moves-left countdown (dead ships keep emitting ghosts, slots start as ghosts, drop the ghost-off ablation, bullet life fixed to the 12 slots); labels either first contact (one state byte: 0 ghost / 1 real / 2 contacted; coordinator's recommendation, keeps ally-first right) or at least one contact. Open: label rule, fixing bullet life globally.
- 13:45 egg tick profiling + speed-ups: agent on branch `egg-speed` (worktree v2-eggspeed, units `bnb-eggspeed-*`), bound 16:30.
- 15:10 egg speed done (`egg-speed` at b70dcd18, off `deflector-lookahead`, unmerged). Agent's numbers, quiet machine: egg tick 1.39-1.43 -> 1.08-1.13 ms, off 0.97 ms unchanged (egg extra ~0.41 -> ~0.13 ms; bitwise for two cuts, within rounding for the third). Whole off tick by feature (ncu): ship flight 371 us, bullet move 339, hit test + damage 182, fog 148, fire 90, bullet counts 44; bullets 55% of the tick; fields ~0.29 ms (28%); ghosts ~0. Ghost-only prototype (`egg-speed-ghostonly` b3a16e9d, measurement only): off 0.975 -> 0.77 ms, egg no gain until the moved block is stored again. Compile: advance 3,296, cold 295 s (same failures as before).
- Author decisions ~15:20: weak firing penalty only (humans fire early and through the pass; spray is fine and looks good; no penalty check); egg's tail-chase statistics look good; `egg-speed` fast-forwarded into `deflector-lookahead` (b70dcd18); finish the ghost-only design (agent, branch `ghost-only`, bound 18:30; first-contact labels as recommended, ghost_bullets setting removed, bullet life = 12 slots). Also launched: per-pass damage distribution + always-fire maximum cell (agent, bound 17:45), hot-path simplification audit (read-only, bound 17:30).
- Author decisions ~16:00 (hot-path audit): compile shapes a per-run choice: training fully static (any dynamic shape in training is a bug), crossover evals dynamic; labels built once per batch; constants per type; reset only finished envs via a fixed cap D per tick (no every-k-ticks batching); measure deal creation before a pre-built deal pool; do the audit's small fixes. Sequential: one agent after ghost-only lands.
- ~16:30 ghost-only done (`ghost-only` 23a4271c, off b70dcd18, unmerged). Agent's numbers: tick off 0.973 -> 0.753 ms, egg 1.078 -> 0.88-0.93 ms; ring is the clock (slot = tick mod 12), uint8 bullet_state, bullet life asserted = 12 ticks, ghost_bullets setting removed; quick tier 343 passed; advance 3,283 ops, cold 282 s. Open: gun-range design input vs fixed 12-tick life (author), +23 us egg pair kernel.
- Launched: training-path audit (read-only, bound 18:00); game-tick audit fixes on `tick-audit` off ghost-only (compile static/dynamic per run, labels once per batch, constants per type, reset cap D 16, deal-creation measure, small fixes; bound 20:30).
- Author decisions ~16:30: hit damage 8; spread 5 mrad sigma (author said "microradian", meant milliradian: today's is ~1.6 mrad); store_crowd the scripted default; shield recharge never triggering is fine for now; one egg per bullet per tick locked; first-contact labels locked; bullet life fixed at the 12-slot ring (the gun-range design input goes for now; a different range later = a different ring length); compile budget still provisionally raised until the audits are done; flight numbers (Q10 / A12) signed off: "I like how they fly"; Q11 moot (new scripted agent). To apply the defaults: one agent after `tick-audit`.
- ~17:00 training-path audit done (docs/rewrite/sections/training-path-audit.md, deflector-lookahead e88cf849). Agent's numbers, BC update 158 s (6,637 steps/s): backward 54%, update forward 35%, act 9%, game sim 1.2%. Static learner compile -8% of the micro-batch step; single-pass scan -21% with it (prototype op). Dynamic-shape bugs: compile_dynamic on all five learner graphs (in tick-audit scope now); the league network table's width changes mid-run (against the 2026-11-23 fixed-capacity decision). No belief state outside the Griffin layer.
- Author decisions ~17:15: turned bullet's speed as it is (relative frame); boundary ramp fixed 300 m regardless of map size; E5 closes with one final review of the scripted agent after all fixes; an independent correctness review of the merged tree before the BC dry run. Coordinator check: Q12 and D2 were already in the code (targets once per update and 32 minibatches per docs/rewrite/literature-ppo-targets-minibatches.md; 1,048,576 ticks per update; v2's five levels with v2's discounts and lambdas, ledger and assists since 2026-11-23) — questions.md and the backlog were stale; updated.
- Author decisions ~17:30 (training-path audit): static compile for training runs (all five learner graphs; in tick-audit); investigate and probably replace the doubling scan; norms keep their reductions in float32 but inputs/outputs may be 16-bit where safe (careful, measured). Queue, sequential: tick-audit (running) -> defaults (damage 8, spread 5 mrad, store_crowd, gun-range input removed) -> training-path fixes (league table fixed width, reset_carry, host syncs, scan, bf16 norm outputs) -> merge to core (author approval) -> docs cleanup -> independent correctness review (incl. final scripted review) and a second training hot-path review (both read-only) -> BC dry run (proposed first).
- 17:45 tick-audit done (`tick-audit` 996fdc10, off ghost-only + deflector-lookahead). Agent's numbers: core.step static 0.766 ms egg / 0.572 off (trainer's old dynamic path 1.777 / 1.713); collect 31.8 -> 25.2 ms per tick (act now 23 of 25 ms); update 141.0 -> 129.8 s (micro-batch 183.6 -> 169.0 ms). Kept: compile_shapes static|dynamic (training static with fail_on_recompile, league width W still dynamic), labels once per segment, constants per type, folded floats, a redundant where. Dropped with numbers: reset cap (a re-deal costs only 0.076 ms per tick), deal pool (0.012 ms per tick), exp2/log2, bullet share once. Cold compile 135.4 s. egg_play noise bug came in with the ghost-only port (not in the morning runs).
- ~17:55 launched in parallel (separate packages): defaults-1127 (damage 8, spread 5 mrad, store_crowd, gun-range input removed; bound 20:00) and train-audit (league table fixed width, reset_carry, host syncs, scan replacement, bf16 norm outputs; bound 22:30).
- ~19:00 launched: look-ahead persona study (divergence, Elo per persona, style, degenerate cases, label consistency for BC persona spread) on defaults-1127, short locked jobs to interleave with train-audit; bound 23:30. Queued after the merge: Muon BC A/B, on-policy (real BC stage shortened), screen then confirm with 3 seeds (docs/rewrite/literature-bf16-residual-muon.md). Author on offline BC: not as the stage (sim data nearly free; covariate shift); an offline arm only if wanted as a measurement.
- ~20:45 train-audit done (`train-audit` aadcad7b, off tick-audit). Agent's numbers: BC 8,255 steps/s (127 s per update; update 129.7 -> 114.5 s); Triton linear scan kept (micro-batch 170.0 -> 146.8 ms, peak 5.12 -> 4.17 GiB, gradients cosine >= 0.99999); reset_carry and host syncs kept; norm outputs bf16 dropped (no gain); bf16 residual measured only (-5.3%, gradient error ~6x the scan's), not adopted; league table fixed width (18 slots x 86 rows) costs the RL tick (30.1 -> 38.6 ms at 1 snapshot). Cold compile 121 s.
- Author ~21:00: "fully static" means as fast as possible with a recipe fixed mid-run (pause/resume/extend aside), not identical calls every tick. So: skip empty league slots (same graph, no recompile): agent on train-audit, bound 22:30. Open for the author: scripted capacity uncapped (early RL league mostly scripted), merge approval.
- Author ~19:40: Muon tabled; merge into core when sensible (coordinator: after the empty-slot skip and the persona study land); once the teacher term is off, the scripted agent runs only where it plays (its league lane) and the teacher record is no longer stored (to do). Open: league capacity (coordinator recommends capacity by members present from a few sizes compiled at the start, scripted capped like the others).
- 19:58 persona study done (defaults-1127 33de64fe; docs/rewrite/sections/lookahead-personas/results.md). Agent's numbers: Elo sd 154 (1v1, 652-1319) and 56 (4v4); label blur 0.10 nats at spread 1, 0.025 at 0.5, <0.01 at 0.25; behaviour driven by temperature, power_stickiness, speed_weight; inert rows boundary_weight, seek_weight, disengaged_energy_weight; no degenerate personas. Skip-empty-slots done (train-audit 53dd0708): RL tick 26.4 ms at 1 snapshot, 32.5 at 12.
- ~20:05 merge agent on merge-1127 (train-audit + defaults-1127 onto core; bound 22:00); core fast-forwards after the coordinator's check. Open for the author: BC persona spread; the two-level league pool (load/unload experts by PFSP shares, duplicates allowed, scripted as an ordinary member); pre-compiling league routing at RL start.

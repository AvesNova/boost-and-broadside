# Lean core: backlog

Every planned item, in order, with its status. Decisions behind the items are in
`docs/rewrite/core-decisions.md`; what was done, when and how is in
`docs/rewrite/core-worklog.md`. Status: **todo**, **doing**, **done** (with the commit),
**blocked** (with what it waits for), **dropped** (with why).

## A. Flight physics (behind a switch; the author flies it before it becomes the default)

| # | Item | Status |
|---|---|---|
| A1 | `GameConfig.flight_model` switch: "v2" (today, the default) and "core" (the new physics) | done (overnight, see worklog 02:43) |
| A2 | Weight on the wing: coast holds 1 g; the turn is limited unless diving (remove the `1 - sink_room` exemption in `_setting`) | done (overnight, see worklog 02:43) |
| A3 | Engine always at full effort, constant power, drives speed only; never feeds the store (no charge term) | done (overnight, see worklog 02:43) |
| A4 | Store of about two vertical loops, start at half; dive and climb equal and opposite at a capped rate | done (overnight, see worklog 02:43) |
| A5 | Smootherstep taper on the exchange over 5% of the band at top and bottom | done (overnight, see worklog 02:43) |
| A6 | Firing cuts the engine (fraction to measure); bullets no longer cost store | done (overnight, see worklog 02:43) |
| A7 | One ship first (Vizia), built so more ships can follow | done (overnight, see worklog 02:43) |
| A8 | Assumption tests for the core model (store constant while coasting straight; dive at an empty store and climb at a full store do nothing; the engine never raises the store; coast at full stick holds altitude above the 1 g speed; firing costs speed) | done (overnight, see worklog 02:43) |
| A9 | Measurements: top speed, corner, best sustained, minimum radius, dive and climb rates, time to cross the band, speed lost per second of firing | done (worklog 02:45) |
| A10 | Before and after charts (doghouse, comparison, the five charts) for "core" | done (worklog 02:57) |
| A11 | Sandbox flies "core" (`--set flight_model=core`); readout shows the model | done (overnight, see worklog 02:43) |
| A12 | Number tuning (engine power, exchange caps, store size, fire fraction): needs the author's flying | blocked: author |

## B. Game corrections and settings

| # | Item | Status |
|---|---|---|
| B1 | Observation velocity in world coordinates (divide by the refractive index, as v2) | done (worklog 02:47) |
| B2 | Boundary ramp: v2 scales it with the arena (161.5 m at 4,200 m), lean fixes 300 m | blocked: author (docs/questions.md) |
| B3 | Fields drawn per environment at reset (agreed fix 11) | done (worklog 02:52) |
| B4 | The agreed waste fixes in lean/game/NOTES.md (constants packed or compiled in, packed state, dead ships free, constant work at reset, velocity once per step, one scripted call, personas at reset, no-fields path, batched resets) | done 2026-11-23 (game side 04:52, learn side 06:34) |
| B5 | Cheaper damage form (agreed fix 14) | done, head-on term only (worklog 03:01) |
| B6 | Ghost-bullet hit labels | game side done (worklog 03:06); the hit head is in C and D |
| B7 | Fog (rules in core-decisions.md) | game side done (fog off by default; on in core()); model must read it (C10) |

## C. Model (port v2's mathematics with the agreed changes)

| # | Item | Status |
|---|---|---|
| C1 | Doubling scan for the recurrence over time (same maths, measured 2.9x faster update) | done (worklog 03:13) |
| C2 | Griffin temporal block (v2's block) with the paper's initialisation | done (worklog 03:13) |
| C3 | Encoder: one first layer per token kind, a shared second layer | done (worklog 03:13) |
| C4 | Field tokens as key/value memory in the same attention operation (needs the Obs contract to carry fields) | done (worklog 02:52, 03:13) |
| C5 | Gated attention, element-wise, released-code initialisation; no null key | done (worklog 03:13) |
| C6 | Rotary: drop bands far larger than the map; same layout in both heads; rotate values | done (worklog 03:13) |
| C7 | Critic: HL-Gauss, linear bins, 64 per level, v2's five levels; range monitoring | bins done (64, linear); v2's five levels with D2 (worklog 03:13) |
| C8 | MLP width 4x as v1 and v2 (departure audit row) | done (worklog 03:13) |
| C10 | The model reads fog: `Obs.visible` and `Obs.since_seen` as ship inputs (a last-seen token plus the time since seen); needed before any training under fog | done |
| C9 | Swappable position code: v2's categorical codes, Fourier hexagonal, grid cells | done (worklog 03:13) |

## D. Learner and league

| # | Item | Status |
|---|---|---|
| D1 | Half self-play (both sides train), half league; scripted agent a league member; (1 - p)^2 with one-eighth uniform floor; AlphaStar's win-rate tally | done (worklog 03:29) |
| D2 | v2's training recipe (discounts, batch size, KL stop, weight averaging); reward levels as v2's five, no shaping | report done (recipe-v2-vs-lean.md); adoption needs the author (BC stage, weight averaging, five levels) |
| D3 | Shaping-field bug (moot once shaping is dropped) | dropped: shaping removed with v2's five levels (worklog 03:49) |
| D4 | Damage reward unit (departure audit row 2) | done: v2's unit, 2.74 per 100 points (worklog 03:49; lean/learn/config.py) |
| D5 | Compilation in training | done 2026-11-23 (merged; v2 game meets 5,000 steps/s, core game 4,640: question 12) |
| D6 | Batched evaluation across sizes | done: all sizes in rounds of one batch, walk from the previous ratios (learn/NOTES.md "Batched crossover"); the GPU gain is not shown yet |

## E. Scripted opponent

| # | Item | Status |
|---|---|---|
| E1 | Behaviour decisions from imugi-behaviours.md (core-decisions.md "Scripted opponent") | partly done: per-type characteristics, line-of-fire block (worklog 03:17, 03:18); fly-straight rule and manoeuvres wait for A12 |
| E2 | Basic fighter manoeuvres for the new physics (after A settles) | blocked: A12 |
| E3 | Reads its team's view, never the state (with fog) | done (worklog 03:45; both teams in one call via `enemy_view`, 06:34) |
| E4 | Retreat or extend: none for now (decision 5.7); only a carefully designed rule, if ever (the game should make a badly timed escape from a committed fight fatal) | backlog |

## F. Documents and gates

| # | Item | Status |
|---|---|---|
| F1 | Rewrite core-brief.md to match core-decisions.md | todo |
| F2 | Provenance table docs/provenance.md from provenance-check.md drafts | first version done (worklog 03:19) |
| F3 | Step 3 scoreboard numbers (operations, compiled against uncompiled, compile time, new team size) | game side done (worklog 03:07); policy and GPU parts after C |
| F4 | Gate G2 (where play runs) from the real game's latency | provisional numbers in scoreboard.md (re-measure on a quiet machine) |

## Build batch 2026-11-23 (author's go, overnight; agents run one at a time)

Each step is one agent in its own worktree; I merge, run `pytest lean -m "not slow"` and the
smoke run, record it in the worklog and push. Decisions: core-decisions.md sections from
"Answers from the author review" on.

| Step | Scope | Status |
|---|---|---|
| G1 | Game sign-off cleanup: v2 flight and damage models removed, core the only game, P-51D lineup, fixed fields and transition tests removed, analysis type vizia_plain removed, `core.step` split, dead StepOut outputs, field_mask, team_view/enemy_view merged, every number the scripted opponent needs in the game's constants | done (18:12) |
| G2 | Game rules: clock tie-break (ships alive, then shield, then draw); spawn rule a setting, default random over the arena without edge-facing spawns, one-side spread over the half disc; training spawns uniform speed, shield, store per ship with one eighth standard; one-tick action delay with allies' pending actions observed; x/R, y/R dropped; feature scales from measured ranges (linear or symlog per feature) | done (18:34) |
| S | Scripted: power rule for the core physics (dive when slow, climb when fast or close, regain the store when far); target memory 0.8 everywhere; head-on side drawn per ship per match; persona fields per ship (temperature included) on a dense competent distribution, spread 0 default; action hold deleted; no retreat; all numbers from the game; review fixes | done (19:07) |
| M | Model: rotary ratio 2 above 16.8 km, irrational turns, 70-75% rotated; fields' rotary heading to the centre; game token at the arena centre; initialisations from the literature, 1 at a reset; next-state head HL-Gauss per column (agreed table); pending-action inputs; dead options removed | done (19:43) |
| L1 | Learner recipe and cleanup: BC stage (full distribution, all heads and critics) then a shrinking teacher term gated at 0.45; EMA and uniform averages; Adam settings, 2,000-step warm-up, non-finite skip; KL monitor; advantage scaling; fixed coefficients and entropy logging; crossover only in evaluation; test stand-ins and duplicates removed | done (20:49) |
| L2 | League: capacity routing with retiring slots; v2's snapshot schedule; checkpoints every 10M; the averages in the league | done (22:30) |
| E2 | Scripted BFM, built natively (author 2026-11-23). First an architecture review of lean/scripted (cohesion, coupling between parameters, how easy it is to change or add a behaviour, how tunable when parameters are sampled per persona) and a short design: situation (engaged or not, one or two circle, closure, angle-off), tactic, and control laws for stick, power and fire, each parameter in physical units and independent of the others. I check the design before it is built. Then: circle detection; one-circle hard turning as now; two-circle at the sustained stick with lead, pure or lag pursuit by geometry; high and low yo-yos through climb and dive; climb to regain the store when not engaged; thresholds from BFM doctrine (Shaw and similar). Measured: store use and speed in scripted brawls, new against old, hand-built two-circle cases, operations per call | running (design merged) |
| C | Cheese agents ported from v2, checked to be reasonable, in the league | todo |
| D | Notes hygiene: one short current notes file per package, CONTRACT and README current, LAPTOP_PROMPT deleted, author-review.md updated | todo |
| A1 | P-51D refined from the references: wing area 21.67 m^2 (NACA), CLmax about 1.6 (AAF turn calculation), g limit 8 (author 2026-11-23, against 6.5 g from the tested weight) | todo |
| A2 | Fw 190 A-8 and Bf 109 G-6 built by the tuning route and checked against their data (drag table; the 109's 360-degree times) | todo |
| A3 | Spitfire IX; then a check of every type's turning against the relative data ("A out-turns B") | todo |
| P | Checkpoint disk use: every snapshot is repeated in every checkpoint (250 to 350 MB each, every 10M steps; about 15 GB per 500M steps against 32 GB free). Write each snapshot once beside the checkpoints and reference it. Before any long run | done (22:37) |
| K | Compile time: the cold compile to the end of the first update doubled (about 200 s to 403 s) between the D5 measurement and step M. Find what grew (graphs, traced operations, recompiles, shapes) by bisecting the batch's merges, fix it, and add a tripwire so a jump is caught when it happens (author 2026-11-23: very concerning if it keeps happening). Also: the one recompile of the routing graph and `advance` at the first league update (guard 'NoneType' has no attribute 'stride'); the per-slot league call's launch cost (CUDA graphs) | todo |
| R | Independent review after the batch, by an agent that did not build it: bloat, overly coupled code, bugs and mistakes; a ranked report, no changes | todo |
| X | Measurements: GPU parity row on the new tree; scripted against scripted wipe times and the clock set at about 80% of games ending | todo |

After the batch: use idle GPU time for speed benchmarks and dry runs (BC first, then RL) to
check everything works together; then training runs. Judgement calls made overnight go in
docs/rewrite/morning-notes.md, briefly.


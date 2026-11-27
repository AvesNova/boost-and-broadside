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
| A12 | Number tuning (engine power, exchange caps, store size, fire fraction): needs the author's flying | done 2026-11-27: author signed the flight numbers off as they are |
| A13 | Post-stall regime (author 2026-11-26): no new stick level; below corner speed full stick goes as deep into the stall as the plane gives, the middle level is about maximum lift before separation, the gradual level less; above corner unchanged (g limit). Gradual lift loss and rapid drag rise past the stall, the nose swinging freely; no asymmetric stall or spin. Literature: docs/rewrite/sections/post-stall/literature.md. Built and signed off (author 2026-11-26): levels Sustain / Max / Snap, deep angle 45 degrees opening over 30% of corner speed below corner, nose spring per level (core-decisions.md "Post-stall flight") | done 2026-11-26 (branch `post-stall`, docs/rewrite/sections/post-stall/progress.md) |

## B. Game corrections and settings

| # | Item | Status |
|---|---|---|
| B1 | Observation velocity in world coordinates (divide by the refractive index, as v2) | done (worklog 02:47) |
| B2 | Boundary ramp: v2 scales it with the arena (161.5 m at 4,200 m), lean fixes 300 m | done 2026-11-27: fixed 300 m regardless of map size |
| B3 | Fields drawn per environment at reset (agreed fix 11) | done (worklog 02:52) |
| B4 | The agreed waste fixes in lean/game/NOTES.md (constants packed or compiled in, packed state, dead ships free, constant work at reset, velocity once per step, one scripted call, personas at reset, no-fields path, batched resets) | done 2026-11-23 (game side 04:52, learn side 06:34) |
| B5 | Cheaper damage form (agreed fix 14) | done, head-on term only (worklog 03:01); the egg deflector (merged 4bc9b2f7, core-decisions.md "The egg deflector") now sets the hit model, the nose-cone rule is kept as code and off |
| B6 | Ghost-bullet hit labels | done: ghost-only bullets on the 12-slot ring with first-contact labels (merged 4bc9b2f7, core-decisions.md "Ghost-only bullets"); the hit head trains on them (`hit_coefficient`) |
| B7 | Fog (rules in core-decisions.md) | done: on by default with the tracking circle (`GameConfig.fog`, `fog_tracking`); the model reads it (C10) |

## C. Model (port v2's mathematics with the agreed changes)

| # | Item | Status |
|---|---|---|
| C1 | Scan for the recurrence over time | done: doubling scan (worklog 03:13), replaced by a Triton linear scan (merged 4bc9b2f7, core-decisions.md "The training-path fixes") |
| C2 | Griffin temporal block (v2's block) with the paper's initialisation | done (worklog 03:13) |
| C3 | Encoder: one first layer per token kind, a shared second layer | done (worklog 03:13) |
| C4 | Field tokens as key/value memory in the same attention operation (needs the Obs contract to carry fields) | done (worklog 02:52, 03:13) |
| C5 | Gated attention, element-wise, released-code initialisation; no null key | done (worklog 03:13) |
| C6 | Rotary: drop bands far larger than the map; same layout in both heads; rotate values | done (worklog 03:13) |
| C7 | Critic: HL-Gauss, linear bins, v2's five levels; range monitoring | done: 63 bins per level (`value.COLUMNS`), v2's five levels with D2 |
| C8 | MLP width 4x as v1 and v2 (departure audit row) | done (worklog 03:13) |
| C10 | The model reads fog: `Obs.visible` and `Obs.since_seen` as ship inputs (a last-seen token plus the time since seen); needed before any training under fog | done |
| C9 | Swappable position code: v2's categorical codes, Fourier hexagonal, grid cells | done (worklog 03:13) |

## D. Learner and league

| # | Item | Status |
|---|---|---|
| D1 | Half self-play (both sides train), half league; scripted agent a league member; (1 - p)^2 with one-eighth uniform floor; AlphaStar's win-rate tally | done (worklog 03:29); now the two-level league (merged c7c525b4, proposal: core-decisions.md "The two-level league"; sizes open, docs/questions.md 16) |
| D2 | v2's training recipe (discounts, batch size, KL stop, weight averaging); reward levels as v2's five, no shaping | done: BC stage, weight averages, five levels with v2's discounts and lambdas, zero-sum ledger with assists, 1,048,576 ticks per update with 32 minibatches (checked 2026-11-27 against lean/learn/config.py and rewards.py) |
| D3 | Shaping-field bug (moot once shaping is dropped) | dropped: shaping removed with v2's five levels (worklog 03:49) |
| D4 | Damage reward unit (departure audit row 2) | done: v2's unit, 2.74 per 100 points (worklog 03:49; lean/learn/config.py) |
| D5 | Compilation in training | done 2026-11-23; static shapes for training (`compile_shapes`, merged 4bc9b2f7); compile check baselines reset ce27fe64 (cold about 121 s, budget 150 s) |
| D6 | Batched evaluation across sizes | done: all sizes in rounds of one batch, walk from the previous ratios (lean/eval/matches.py); the GPU gain is not shown yet |

## E. Scripted opponent

| # | Item | Status |
|---|---|---|
| E1 | Behaviour decisions from imugi-behaviours.md (core-decisions.md "Scripted opponent") | superseded 2026-11-27: the look-ahead agent (E6) replaced the rule-based agent |
| E2 | Basic fighter manoeuvres for the new physics (after A settles) | superseded 2026-11-27 by E6 |
| E3 | Reads its team's view, never the state (with fog) | done (worklog 03:45; both teams in one call via `enemy_view`, 06:34) |
| E5 | Flying straight for a while after the merge before turning (author 2026-11-26; measured on branch `merge-turn` (unmerged), its docs/rewrite/sections/merge-turn/results.md) | superseded 2026-11-27 by E6; merge turn re-measured there (look-ahead 1v1 no fields: 1.17 / 2.97 s); one final review of the scripted agent after all fixes |
| E6 | Look-ahead scoring agent (author 2026-11-26; core-decisions.md "Scripted look-ahead") | done: merged 4bc9b2f7 (2026-11-27); hit worth through `core.hit_damage`, `store_crowd` the default; one final review after all fixes (queue item 7) |
| E4 | Retreat or extend: none for now (decision 5.7); only a carefully designed rule, if ever (the game should make a badly timed escape from a committed fight fatal) | backlog |

## F. Documents and gates

| # | Item | Status |
|---|---|---|
| F1 | Rewrite core-brief.md as a short goals-and-standards document (review item C4) | done (ef69296d); its open points are docs/questions.md 18 |
| F2 | Provenance table docs/provenance.md from provenance-check.md drafts | first version done (worklog 03:19) |
| F3 | Step 3 scoreboard numbers (operations, compiled against uncompiled, compile time, new team size) | game side done (worklog 03:07); policy and GPU parts after C |
| F5 | Renderer-neutral frame API (author 2026-11-26): a plain per-tick frame record (ships, bullets, fields, sight, events) written by the game and read by any renderer, the v2 native view and a web view as consumers; files for replays, a stream for live. After a successful dry run; the v2 view port (branch `view`) works for now | todo: after the dry run |
| F4 | Gate G2 (where play runs) from the real game's latency | provisional numbers in scoreboard.md (re-measure on a quiet machine) |

## Build batch 2026-11-23 (closed)

All steps done and merged (details in core-worklog.md 2026-11-23 to 24) except these:

| Step | Scope | Status |
|---|---|---|
| E2 | Scripted BFM, built natively | superseded 2026-11-27 by the look-ahead agent (E6) |
| C | Cheese agents ported from v2, in the league | future work (tabled 2026-11-24, author) |
| A2 | Fw 190 A-8 and Bf 109 G-6 by the tuning route, checked against their data | future work (tabled 2026-11-24, author) |
| A3 | Spitfire IX; every type's turning against the relative data | future work (tabled 2026-11-24, author) |
| R | Independent review | round 1 done and fixed (2026-11-24 05:55); the next review is queue item 7 |
| X | Measurements: GPU parity row on the new tree; scripted against scripted wipe times; clock set so about 80% of games end | todo; the persona study round 2 measured 40 / 72 / 68% of games ending by elimination at 1v1 / 4v4 / 16v16 on the 60 s clock (clocks per size: docs/questions.md 17) |

Done in the batch: G1, G2, S, M, L1, L2, D, A1 (P-51D refined, 2026-11-24 03:51), P, V, W, K.

## Order

The current order is the work queue, docs/rewrite/queue.md. After it: the BC dry run, then
the RL dry run (proposed first; waits on the author).

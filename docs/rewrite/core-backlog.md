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
| D3 | Shaping-field bug (moot once shaping is dropped) | todo |
| D4 | Damage reward unit (departure audit row 2) | todo |
| D5 | Compilation in training | done 2026-11-23 (merged; v2 game meets 5,000 steps/s, core game 4,640: question 12) |
| D6 | Batched evaluation across sizes | todo |

## E. Scripted opponent

| # | Item | Status |
|---|---|---|
| E1 | Behaviour decisions from imugi-behaviours.md (core-decisions.md "Scripted opponent") | partly done: per-type characteristics, line-of-fire block (worklog 03:17, 03:18); fly-straight rule and manoeuvres wait for A12 |
| E2 | Basic fighter manoeuvres for the new physics (after A settles) | blocked: A12 |
| E3 | Reads its team's view, never the state (with fog) | todo |

## F. Documents and gates

| # | Item | Status |
|---|---|---|
| F1 | Rewrite core-brief.md to match core-decisions.md | todo |
| F2 | Provenance table docs/provenance.md from provenance-check.md drafts | first version done (worklog 03:19) |
| F3 | Step 3 scoreboard numbers (operations, compiled against uncompiled, compile time, new team size) | game side done (worklog 03:07); policy and GPU parts after C |
| F4 | Gate G2 (where play runs) from the real game's latency | provisional numbers in scoreboard.md (re-measure on a quiet machine) |

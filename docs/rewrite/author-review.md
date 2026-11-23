# Lean core: everything open, in one place

Written 2026-11-23 from the notes and reports listed in the coverage table at the end, checked
against the code on branch `core` at fc3e6ba9. Where a note and the code disagree, the code is
what is described here, and the note is listed under "Stale notes" in its section.

## How to use this document

- Each topic section has three parts, always in this order:
  1. **Decisions needed**: numbered items (for example 7.1) you can answer one by one.
  2. **Done or decided, for information**: short. Where something is decided but the code does
     not follow it yet, it says "not built".
  3. **Stale notes**: places where a note says something the code no longer does. No action
     beyond tidying the notes (section 11).
- Each decision item gives: what it is, the value in the code now (file:line), the options, a
  recommendation only where a source makes one (with the source), what it blocks, and its
  references.
- Reference shorthand: "Q9" is `docs/questions.md` item 9; "review 5" is finding 5 of
  `docs/rewrite/lean-review-2026-11-23.md`; "backlog A12" is `docs/rewrite/core-backlog.md`;
  "departure audit" is `docs/rewrite/departure-audit.md`; "recipe" is
  `docs/rewrite/recipe-v2-vs-lean.md`; "literature" is
  `docs/rewrite/literature-ppo-targets-minibatches.md`. File paths without a directory are in
  `lean/`.
- Numbers are copied from the sources; none are new. Two are marked "arithmetic" where they
  multiply numbers that are in the sources.

## The decisions that block the most

In priority order. Each line: the item, then what waits on it.

1. **1.1 Sign off the core flight model** (after flying it; tuning in 1.2 to 1.5). Waits: the
   core game as the training default, removing v2's flight and damage models (the largest
   cleanup: about 80 source and 750 to 850 test lines), the scripted opponent's power rule
   (5.1) and manoeuvres, and the core game's training speed (9.1).
2. **7.1 Behaviour cloning: a stage before RL or a term inside PPO.** Waits: the first real
   training run, the teacher's settings (5.2, 5.3) and the BC gate (8.1).
3. **7.2 Weight averaging and 7.3 the KL early stop.** Waits: the first real run; averaging
   also decides which weights are rated and shipped.
4. **7.4 Advantage scaling and 7.5 the balance of the loss terms.** Waits: the first real run.
5. **What the first run plays: 4.1 (stage 1), 4.2 team sizes, 4.3 lineup, 2.2 arena size,
   2.3 the clock rule.** Waits: the first run and how its evaluation compares with v1 and v2.
6. **3.2 Absolute position in the observation.** Changes the model's input, so it belongs
   before the first real run.
7. **5.1 The scripted opponent's power under the core physics and 5.2 its target memory.**
   Waits: any measurement against the yardstick, and the BC teacher in the core game.
8. **3.1 The five fog choices.** Waits: training in the core game (fog on).
9. **7.8 bf16 as the default.** Already on; waits only your confirmation.

---

## 1. Flight physics and tuning

### Decisions needed

#### 1.1 Sign off the core flight model and make it the default

- **What it is.** Two flight models (and two damage forms) live side by side in
  `game/core.py`: "v2" (v2's, tested equal to v2) and "core" (the agreed new physics: weight
  on the wing, the engine driving speed only, a two-loop store, firing cutting the engine).
  The decisions say the author flies "core" before it becomes the default, and that v2's
  models go once core is signed off.
- **In the code now.** `GameConfig.flight_model = "v2"` and `damage_model = "v2"`
  (`game/core.py:101-102`); `GameConfig.core()` switches both, plus ghost bullets, Vizia
  only, ten drawn fields and fog (`game/core.py:111-117`). `train.py --game` defaults to "v2"
  (`train.py:39`, `:63`). Sandbox: `python -m lean.sandbox --core`.
- **Options.** (a) Fly, tune (1.2 to 1.5), sign off: core becomes the default and the v2
  models, `core()`, inert card columns, `train.py --game`, the sandbox's `--core` and the
  transition tests go. (b) Keep both until the tuning settles.
- **Recommendation.** The review (finding 5) notes that the defaults contradict the decisions
  (training defaults to the v2 game with fog, ghosts and drawn fields off) and that this is
  recorded as "v2 until sign-off". No source recommends switching before you fly it.
- **Blocks.** Review 1 (about −80 source, −750 to −850 test lines), review 5, backlog A12,
  E1 (fly-straight rule) and E2 (manoeuvres), item 5.1, item 9.1.
- **References.** core-decisions "Flight physics" (last line) and "Proposals, not decided";
  review 1 and 5; backlog A1 to A12.

#### 1.2 Store size, exchange rate and the edge band

- **What it is.** The store is the ship's altitude, as energy. Dive trades store for speed,
  climb trades speed for store, at a capped rate (the exchange rate). Near the top and bottom
  of the store a taper fades dive and climb out (the edge band).
- **In the code now.** `store_loops = 2.0` (`game/core.py:103`), store = loops × g × 2 ×
  corner radius (`game/core.py:158-162`); exchange rate = the mean of the card's dive and
  climb rates (38.6 m/s², about 1 g, for Vizia); `store_edge_band = 0.05` (`game/core.py:104`).
- **Measured.** Each half of the store is crossed in about 1.5 s at the full rate (half to full
  1.6 s from top speed; half to empty 1.4 s). The whole band is worth about ±45 m/s around
  cruise (dive peak 368 m/s from 325). At full climb rate the store moves about 36% of the band
  per second at 300 m/s, so the 5% edge band is crossed in about 4 ticks and the taper acts
  almost like a wall.
- **Options.** A larger store; a slower exchange (lower cap); a wider edge band; as is.
- **Recommendation.** None; the sources leave it to your flying.
- **Blocks.** Backlog A12, and through it 1.1 and E2.
- **References.** Q10(a); worklog 02:43 ("Finding for tuning"); core-flight-measurements
  "Reading"; core-decisions "The store sized to about two vertical loops", "the width whatever
  flies best (5% for now)".

#### 1.3 The cost of firing

- **What it is.** Firing cuts the engine on the ticks the ship fires (bullets no longer cost
  store). The fraction of thrust kept while firing sets how much speed firing costs.
- **In the code now.** `fire_thrust_fraction = 0.0` (`game/core.py:105`): the engine is fully
  cut while firing.
- **Measured.** 6.0 m/s lost per second of firing at top speed, 7.3 m/s at corner speed: about
  2 to 3% of speed per second.
- **Options.** Keep 0.0; keep some thrust (a smaller cost). Known quirk: firing doubles as an
  air brake (the only way to cut thrust); recoil drag is the recorded fix if that proves
  exploitable.
- **Recommendation.** None; "for the author's flying".
- **Blocks.** Backlog A12.
- **References.** Q10(b); core-flight-measurements; core-decisions "Guns".

#### 1.4 Engine power

- **What it is.** Under core the engine always runs at full effort and only drives speed.
  Its power is the card's full-store power; v2's extra 7 to 9 kW/kg "surplus", which only
  charged the store, is gone.
- **In the code now.** Vizia 2.04 kW/kg; top speed 325 m/s, unchanged from v2.
- **Related finding.** Under core, Vizia's v2 climb modifications are already inert, apart
  from the dive and climb trade (0.983 g becomes 1 g) (p51d-model "Vizia's climb
  modifications").
- **Options.** Accept; change the power (changes cruise and top speed).
- **Recommendation.** None in the sources.
- **Blocks.** Backlog A12.
- **References.** Q10(c); worklog 02:43 ("Engine power choice"); p51d-model.

#### 1.5 The slow full-stick turn, and the "turn wins" switch

- **What it is.** Because coasting always holds 1 g on the wing, a slow ship at full stick
  turns less; diving frees lift for the turn. You decided "altitude wins", and asked for the
  other option (the turn wins and the ship sinks) to be built behind a switch so you can fly
  both.
- **In the code now.** Altitude wins. **The switch is not built**: `GameConfig` has no such
  setting (`game/core.py:50-117`).
- **Measured.** Full stick while coasting settles at 111 m/s and 25 deg/s (v2: 139, 38.6).
  In real units Vizia's settled full-stick turn is 12.5 deg/s against its v2 sustained-turn
  handle of 19.29 deg/s; meeting the handle under core would need an induced drag scale of
  0.155 (e·AR near 10), which is not physical.
- **Options.** Accept; build the switch and fly both; later, a low-speed lift bonus
  (flaps-like), which the decisions allow "only if flying shows it is missing".
- **Recommendation.** None.
- **Blocks.** Backlog A12.
- **References.** Q10(d); core-decisions "Over-turning when slow", "Required order";
  core-flight-measurements; p51d-model "Vizia's climb modifications" (last paragraph).

#### 1.6 The core model's shared terms and realism

- **What it is.** Four terms of the core model apply to every ship type: a drag rise above
  394 m/s (game units), separation drag that doubles induced drag at full lift, a soft minimum
  of thrust against static thrust, and a soft minimum of lift against the g limit. A real
  aircraft converted directly (the P-51D) loses performance to them.
- **Measured (P-51D, core against its point mass).** Top speed −3.7%, best climb −14.5%,
  sustained turn −17%, turn at corner −8%. With the terms off, the card reproduces the October
  fit exactly. The core P-51D's 15.4 deg/s sits beside the simulator figure, 15.7 deg/s.
- **Options.** Keep the terms (game feel, as today); weaken some of them for realism.
- **Recommendation.** None; the report only measures.
- **Blocks.** Nothing now; matters when real aircraft become ship types.
- **References.** p51d-model "Comparison with real figures".

### Done or decided, for information

- Decided and built (backlog A1 to A11): weight on the wing; bank implicit and instant; no
  climb or dive angle; coast, dive and climb as the power action; the engine never feeds the
  store; store of two loops, start at half; no thinner air; firing cuts the engine; a
  smootherstep taper over 5% at each edge; one ship first (Vizia).
- The required speed order (minimum sustained radius, best sustained rate, corner speed)
  holds for every type and is a test (Vizia 124, 160, 268 m/s). This also answers the
  physics question in core-decisions "Scripted opponent" (are minimum-radius speed and
  maximum-rate speed the same?): under core they separate.
- Fixed 02:57: inside an edge band, dive on an empty store no longer frees lift for the turn.
- Measurements and charts: `docs/rewrite/core-flight-measurements.md`,
  `docs/rewrite/*-core.png`; P-51D card and "vizia_plain" (Vizia without its climb
  modifications): `docs/rewrite/p51d-model.md`. Vizia keeps a g limit of 9 against the
  researched 8.

### Stale notes

- `sandbox/README.md` gives `--set flight_model=new` as a future example; the switch exists
  and its value is "core".

---

## 2. Damage, boundary, arena and the clock

### Decisions needed

#### 2.1 Boundary ramp

- **What it is.** The distance beyond the arena edge over which boundary damage rises from
  zero to its maximum.
- **In the code now.** 300 m fixed (`game/core.py:64`), a builder's assumption.
- **v2.** 100 m at map scale 1 (`src/bnb/game/rules.py:151`), scaled with the map: 161.5 m
  at a 4,200 m arena (Q9).
- **Options.** v2's rule; keep 300 m.
- **Recommendation.** Q9 proposes v2's rule.
- **Blocks.** Nothing urgent.
- **References.** Q9; backlog B2; review 13; game/NOTES "Q4 arena".
- **Note.** game/NOTES says "v2's 100 m times its map scale 3", and the departure audit lists
  the 300 m ramp as matching v2. Both disagree with Q9 and with v2's `rules.py`; Q9 is the one
  consistent with v2's code.

#### 2.2 Arena size

- **What it is.** The arena radius sets fight density, the field sizes (scaled by
  arena / 2,600 m) and how far the rotary bands must reach (6.1).
- **In the code now.** 4,200 m (`game/core.py:63`), deathmatch's size in v2, applied to
  elimination.
- **v2.** Elimination 7,800 m; deathmatch 4,200 m.
- **Options.** Keep 4,200 m (the game audit's recommendation, not a decision of yours); v2's
  7,800 m for elimination; draw sizes per environment (generalising across map sizes).
- **Recommendation.** The game audit (GA:1384-1389) recommended 4,200 m; recorded, not
  approved.
- **Blocks.** 4.1; 6.1 (rotary reach); the scripted opponent's arena bug (section 5, "Known
  weaknesses") matters as soon as the size changes.
- **References.** departure audit "Close behind" and Game row "Arena"; game/NOTES Q4;
  README "Decisions still with the author".

#### 2.3 How a game ends without a shrinking boundary

- **What it is.** With no shrink, a game ends by a wipe or at the clock; at the clock the
  team with more shield over its living ships wins, equal is a draw.
- **In the code now.** `clock_ticks = 5400` (180 s, `game/core.py:65`); the health rule at
  `game/core.py:895-907`.
- **Why it matters.** In v2's elimination the shrink killed every ship before the clock, so
  the health rule never decided a game in practice. In evaluation, v1's "ties count as
  non-wins" now almost never applies, so crossover ratios are not comparable with v1's.
- **Options.** Keep; add elimination's shrink (the scripted opponent's boundary steer is
  decided to handle a linear shrink simply); a shorter clock (v1's was 17 s).
- **Recommendation.** None.
- **Blocks.** 4.1; 8.4 (what a crossover ratio means).
- **References.** game/NOTES Q3; departure audit Game "Clock and win rule" and Loop
  "Crossover games at the clock"; README "What stage 1 reproduces".

### Done or decided, for information

- Cheaper damage (backlog B5): the head-on term in the hit angle's cosine, within 0.024 of
  v2's curve, same half-damage angle. The closeness term stays v2's: the agreed shortcut (the
  hit test's miss distance) read 6.8% less damage, so it is not the same effect.
- Friendly fire on; reported apart (game/NOTES Q5).
- The step reports attacker-by-target damage (`StepOut.hit_damage`), boundary damage and
  shield restored; the reward ledger reads them exactly (worklog 03:51).
- Parity with v2 completed, three bugs fixed (game/NOTES "Parity with v2 completed";
  `docs/rewrite/game-parity.md`).
- Collision code read, no error found; 11 bullet slots per ship is exact.
- Known differences kept: result coding 0/1/2/−1 (v2 +1/−1/0); killer tie-break by lowest
  bullet index; team sizes of 0 not handled.
- The dense (environments, N × 11, N) hit test at 50v50 (your future check): uncompiled it
  runs out of 8 GB at 2,048 environments; compiled it fits (35.8 ms) (scoreboard "GPU").
- Could not be determined from v2 (departure audit): v2's spawn store and shield spread;
  whether v2 resets cooldowns randomly.

### Stale notes

- game/NOTES "Known differences": "a ship that dies keeps its speed and nose rate". Now every
  slot flies and dead ships are only kept finite and masked (waste fix 3).
- README first-cut table: "no game ended in the comparison", "`reset_done` costs a third of a
  step". Games that end are compared (`test_parity_v2.py`); deals are drawn once per segment.
- departure audit Learner rows "damage unit" and "kill assist, boundary credit", top-10 item
  2: the damage unit is v2's (2.74 per 100 points, `learn/config.py:72`, `:77`), and boundary
  deaths pay the other team.

---

## 3. Fog and what the model observes

### Decisions needed

#### 3.1 Five fog choices

- **What it is.** Fog is built to your rule (seen in sight, on a tick of firing, on a tick of
  being hit; not by boundary damage). The builder had to settle five points the rule does not
  state.
- **In the code now.** `perceive` (`game/core.py:1244`), `team_view` (`:1255`), `observe`
  (`:1303`); `GameConfig.fog = False` (`:109`), on in `core()`.
- **The five points.**
  1. **No reveal at a spawn.** Your rule has none; v2 reveals every ship on its spawn tick.
     With the "sides" spawn nobody is seen at the start.
  2. **Deaths are known only when seen** (on the death tick, or a wreck later in sight). A
     bullet kill is always seen, since the hit reveals. An unseen boundary death stays a
     last-seen row and counts as alive in the globals. v2 shows every death.
  3. **Wrecks can be sighted** (v2's sight test ignores the target's life). This is how a
     hidden death becomes known.
  4. **Only real bullets reveal on a hit** (ghost bullets are labels); friendly hits reveal
     too.
  5. **Time since seen is given as log(1 + seconds)** (v2's model input), not raw seconds.
- **Options.** Confirm each, or take v2's behaviour for 1 and 2.
- **Recommendation.** None.
- **Blocks.** Training in the core game.
- **References.** game/NOTES "Fog of war", "Choices for the author"; review 13; worklog 03:45.

#### 3.2 Absolute position in the ship features

- **What it is.** Each ship row carries its position over the arena radius (x/R, y/R), its
  outward direction and its distance inside the boundary, all in the true frame. The model's
  random position offset exists to stop memorised positions, but these features reach the
  encoder unshifted. In a disc arena the boundary features also give absolute position, in
  polar form.
- **In the code now.** `game/core.py:1368-1374`.
- **The plan.** "Position: none absolute. Clipped distance and direction to the boundary"
  (model-audit, MA:347). v1 used absolute Fourier features; v2 an absolute colour code shifted
  by the offset.
- **Options.** Drop x/R and y/R and keep the boundary distance and direction (the plan);
  drop all four and rely on the position code; keep.
- **Recommendation.** The plan's (no absolute position).
- **Blocks.** The model's input width; transfer across map sizes. Best settled before the
  first real run.
- **References.** departure audit top-10 item 8 and Observation row "Absolute position"
  (marked unrecorded).

#### 3.3 The previous action as an input

- **What it is.** Whether a ship sees what each ship did last tick.
- **In the code now.** Only a stored "fired last tick" flag (`game/core.py:1377`).
- **v1 and v2.** v1 gave every ship's pending action, opponents included (listed as a v1
  information edge, YS:594); v2 gave own ships' previous action and a forecast for enemies.
- **Options.** Add own ships' previous action (42-way); add every visible ship's; keep.
- **Recommendation.** None.
- **Blocks.** The model's input width.
- **References.** departure audit Observation row "Pending or previous action".

#### 3.4 Feature scaling

- **What it is.** The 20 ship scalars are linear at fixed scales (speed over 400 m/s, shield
  over 150 and so on, `game/core.py:1291-1375`). The plan said symlog where the range is wide.
- **Options.** Keep; symlog the wide-range ones.
- **Recommendation.** The plan's (symlog where wide); no measurement either way.
- **Blocks.** Nothing.
- **References.** departure audit Observation row "Per-ship features".

#### 3.5 Can a player see an enemy's store and shield?

- **What it is.** The model and the scripted opponent read a seen enemy's store and shield.
  The Imugi report flags this as worth checking against what a human player can see.
- **In the code now.** Both are in every ship row (`game/core.py:1363-1365`), at their
  last-seen values under fog.
- **Options.** Keep (show them to players too); hide them from enemies.
- **Recommendation.** None.
- **References.** imugi-behaviours row 8 (Notes).

### Done or decided, for information

- Fog rule decided and built; the model reads `visible` and `since_seen` (backlog B7, C10).
- The scripted opponent reads only its team's view (enemies as last seen) (backlog E3).
- Field tokens as key/value memory in the attention (Q3, backlog C4); bullets are not tokens
  (Q3).
- Observed speed and velocity in the world frame (backlog B1); "fired" stored by the step
  (waste fix 12).
- **Decided, not built:** "tell, do not infer" (Q4) means the mode and the arena size must be
  inputs; the global token today has only the clock and the two alive fractions
  (`game/core.py:1389-1395`). Needed once there is more than one mode or size.
- **Decided, not built:** a dead ship that will respawn is visible (Q5); there is no respawn
  yet.

### Stale notes

- `CONTRACT.md` describes `Obs` as "full information (no fog yet)" and the unpacked `State`.
- departure audit Game row "Fog", Observation rows "Field and zone tokens", Scripted row
  "Reads": all built since.
- scripted/NOTES "Dead reckoning and the last-seen memory: nothing is hidden without fog".
  Fog exists now; dead reckoning stays dropped by your decision.

---

## 4. Game modes, spawns, lineup and team sizes

### Decisions needed

#### 4.1 What the first training run reproduces

- **What it is.** The README asks whether stage 1 reproduces v1's conditions (mixed spawns,
  a short clock, a simple opponent, 4v4 only) on v2's game, then v2's conditions. Since then
  core-decisions set the goal as "every v2 feature back", with the recipe leaning towards v2's.
- **In the code now.** "sides" spawn (`game/core.py:66`), 180 s clock, 4v4, the simplified
  Imugi core.
- **Options.** v1's conditions first (the yardstick route); v2's conditions directly.
- **Recommendation.** None. YS:585 ranks v1's mixed spawn first among v1's ingredients for the
  size curve.
- **Blocks.** The first run.
- **References.** README "Decisions still with the author" and "When the laptop is back"
  step 7; departure audit Game row "Spawn".
- **Note.** The review (finding 6) lists the "mixed" spawn as used only by tests, a deletion
  candidate, while the README's stage-1 option needs it. Decide this item before deleting it.

#### 4.2 Team sizes in training

- **In the code now.** 4 to 4, equal teams (`learn/config.py:85-87`).
- **Others.** v1 4v4 only. v2's release run: 5v5 at 0.5, 4v5 and 3v5 at 0.25 each, the learner
  always the smaller side. The plan: sizes drawn per environment, unequal included
  (core-brief section 4; MA:371).
- **Options.** Keep 4v4; v2's release mix; a wider draw.
- **Recommendation.** None. Unequal teams have not run with the real game.
- **Blocks.** The first run; 7.9 (the kill level's support was measured to 8v8).
- **References.** departure audit Loop row "Team sizes"; recipe row "Opponent mix"; review 13;
  learn/NOTES "What was run".

#### 4.3 Lineup

- **In the code now.** v2 game: a fixed cycle by rank, Yongma then Jujak, the same for both
  teams (`game/core.py:68`), so every 4v4 game is the same 2 + 2 lineup. Core game: Vizia only.
- **Others.** v2 drew Yongma or Jujak per slot per episode; the game audit recommended
  per-episode mirrored draws.
- **Options.** Keep the cycle; mirrored draws per episode; one type (core).
- **Recommendation.** The game audit's (mirrored draws), for multi-type games.
- **Blocks.** Training with more than one ship type.
- **References.** game/NOTES Q6; departure audit Game row "Lineup"; review 13.

#### 4.4 Spawn state

- **In the code now.** Full shield; store at the type's spawn value (v2 game) or half the
  store (core).
- **Others.** v1 drew health and power uniformly in 75% to 100%. v2's spread was not
  determined.
- **Options.** Keep; draw a spread.
- **Recommendation.** None.
- **References.** departure audit Game row "Spawn resources".

#### 4.5 Which mode comes after elimination, and when

- **What it is.** Only elimination without shrink exists. Shrink, deathmatch (respawn),
  Frontline (zones) are not built.
- **Decided.** Modes "whatever is fastest and cleanest"; modes differ by a few numbers per
  environment (respawn rule, wave counter, shrink rate, zones); deathmatch respawn may be
  random, turned away from the edge; a mode is fixed per environment for the whole run.
- **Open.** Order and timing. Respawn also settles two waiting points: v2 valued ships alive
  or waiting to respawn, lean values living own ships (learn/NOTES reward choice 10); and the
  fog rule for a ship that will respawn.
- **Blocks.** The scripted Frontline layer; zone tokens in the model.
- **References.** core-decisions "Game"; departure audit "v2 complexity lean correctly
  dropped".

### Done or decided, for information

- A mode is a constant of its environment; modes in a fixed share of environments (game/NOTES
  fix 13).
- The "sides" spawn's distribution is checked against v2 (`test_parity_v2.py`).

### Stale notes

- README first-cut table: "The 'sides' spawn untested". It is tested now.

---

## 5. Scripted opponent (Imugi core)

### Decisions needed

#### 5.1 Power actions under the core physics

- **What it is.** The opponent's power rule is Imugi's: boost when slow, or far with store;
  reverse when very close. Under the core physics power 1 is dive (store into speed) and power
  2 is climb (speed into store). So in the core game it dives when slow or far and climbs when
  close.
- **In the code now.** Gate names `boost_slow`, `boost_far`, `reverse`
  (`scripted/opponent.py:28-30`); the rule at `scripted/opponent.py:426-431`. With the
  real-scale card the "slow" gate (2.61 corner speeds) is open nearly always.
- **Options.** A power rule for core (for example climb to bleed speed in a turn, dive to
  extend); keep it.
- **Recommendation.** None in the sources.
- **Blocks.** The opponent and the BC teacher in the core game; backlog E2.
- **References.** review 2; scripted/NOTES "Imugi behaviours kept".

#### 5.2 Target memory for the yardstick and the teacher

- **What it is.** You decided target hysteresis: keep the current target unless another is
  closer than 0.8 of its distance. It lives in an optional memory argument.
- **In the code now.** `target_keep_factor = 0.8` (`scripted/opponent.py:65`), but collect,
  eval and the sandbox call `act` without memory (`learn/collect.py:211`), so the hysteresis
  is off everywhere.
- **Options.** Wire the memory in everywhere (the decision); keep the yardstick memoryless.
- **Recommendation.** scripted/NOTES: whether the yardstick uses memory "should be fixed before
  any measurement".
- **Blocks.** Any measurement against the yardstick (5.6, 8.4).
- **References.** review 2; core-decisions "Scripted opponent" (target); scripted/NOTES
  "Choices the audit left open".

#### 5.3 Persona population and spread

- **What it is.** Personas are per-ship variations of the opponent (gate centres, temperature,
  stand-off, preference for weak targets, lead gain). Spread 0 is the one fixed yardstick.
- **In the code now.** `persona_spread = 0.0` in training and as the BC teacher
  (`learn/config.py:95`); the population's widths are guesses (`scripted/opponent.py:104-109`);
  temperature is drawn per ship (`scripted/opponent.py:148`), where you decided per
  environment; at spread 0 the personas are still carried as an 8-key dictionary of constants.
- **Options.** Spread 0 only (delete the population); a population for the BC teacher and
  the league.
- **Recommendation.** The game audit (GA:1404) recommended a population for BC.
- **Blocks.** 7.1 (the teacher).
- **References.** review 2; scripted/NOTES "Persona population"; departure audit Scripted row
  "Persona and memory in training"; core-decisions "Temperature per environment".

#### 5.4 The scripted operation tripwire

- **What it is.** The budget for one scripted call is 170 operations. The line-of-fire check
  you asked for costs 21.
- **In the code now.** One call for both teams: 188 operations (one team 190).
- **Options.** Sign off 188; ask for savings. Savings named in the notes: precompute the gate
  floor and ln(centre) in the persona (about 6), fold the stand-off into `gate_centre` at
  persona time (about 5); skip the weakest-target term when its weight is 0 (review 2).
- **Recommendation.** None.
- **References.** Q11 (with its update); scripted/NOTES "Operation count"; worklog 03:18.

#### 5.5 The perfectly symmetric head-on pass

- **What it is.** Which way to turn when both ships meet exactly head-on.
- **In the code now.** The side is the sign of the angle error, with a 0.02 floor
  (scripted/NOTES "Imugi behaviours dropped").
- **Options.** A fixed rule (for example "turn clockwise"); a random draw.
- **Recommendation.** None.
- **References.** core-decisions "Scripted opponent" (open point).

#### 5.6 Measure the exchange rate against v2's Imugi

- **What it is.** How strong lean's opponent is against v2's Imugi, so lean's results can be
  compared with v2's (v2's champion figure 1.49 is against Imugi).
- **In the code now.** Not measured.
- **Route** (scripted/NOTES): write this opponent as a v2 controller (about 150 lines), then
  512 games against Imugi, the same against v2's chaser as a floor, and v2's champion against
  it when a GPU is free.
- **Options.** Do it now; after 5.1 and 5.2; skip.
- **Blocks.** Comparing lean's results with v2's.
- **References.** README "When the laptop is back" step 6; scripted/NOTES "Measuring the
  exchange rate".

#### 5.7 Retreat and extend

- **What it is.** Whether the opponent may leave a committed fight.
- **Decided.** "Only by a carefully designed rule, or not at all"; the game should make a badly
  timed escape fatal (worth a test). Which of the two is open.
- **Blocks.** Backlog E2 (manoeuvres).
- **References.** core-decisions "Frontline layer".

### Done or decided, for information

- Built: per-type characteristics from the game (gun range, corner speed, launch speed; worklog
  03:17); no shot with a team-mate in the line of fire (03:18); reads its team's view (E3);
  one call for both teams (fix 6); personas drawn once per segment (fix 7).
- **Decided, not built:** the action hold (a redraw every 1 to 6 ticks at spread above 0,
  `scripted/opponent.py:109`, `:149-152`) is a random switch, against "randomness only from
  sampling"; temperature per environment; spacing from allies in every mode; the rewritten
  Frontline layer; fly straight only when already aimed, and basic fighter manoeuvres (both wait
  for 1.1; backlog E1, E2; `docs/rewrite/bfm-doctrine.md` in progress).
- Decided dropped: dead reckoning, type doctrines, switch-rate commitments, keyed draws, mode
  inference, the random-play share.
- Idea for later: a target closeness that also counts the turn needed to face each other.
- **Known weaknesses (fixes, no decision):**
  - The arena radius is fixed at 4,200 m in `ScriptedConfig` (`scripted/opponent.py:81`); the
    game never provides it, so another arena size steers wrongly (review 4).
  - The lead ignores the medium's refractive index (review 9; departure audit Scripted row
    "Fields").
  - The weakest-target term always runs (review 2).
  - Dead fallbacks and a renaming shim in `ScriptedConfig` (review 4).
- Not measured: what any single behaviour is worth in play (imugi-behaviours "Not measured").
- Checked by reading only: rows with no living enemy take index 0 from the `argmin`; they are
  masked by `has_target` (scripted/NOTES "Check first on the GPU").

### Stale notes

- scripted/NOTES "Characteristics": Yongma and Jujak flown with Vizia's gun range, corner speed
  and launch speed. They now get their own (worklog 03:17).
- scripted/NOTES operation counts (166, 176) and imugi-behaviours "Lean's opponent as it is"
  (172, two calls per tick, reads the true state): now 188 for both teams in one call, reading
  the team's view.
- scripted/NOTES "Persona population": "the line-of-fire check (F) is not written". It is.
- core-decisions "Proposals, not decided": the Imugi keep, drop or simplify table "ready for
  the author". Your decisions from it are already in core-decisions "Scripted opponent".

---

## 6. Model

### Decisions needed

#### 6.1 Rotary layout: two axes per band, or "x, y and attitude"

- **What it is.** Rotary position turns each attention head's dimensions by angles from the
  ships' positions and headings, so attention reads relative position. Your decision says
  "x, y and attitude in each head"; the builder took v2's layout (x and y, two axes per band,
  each band turned 35 degrees from the coarser one) rather than lean's earlier three hexagonal
  axes. Heading is in both layouts.
- **In the code now.** 5 bands, 64 m to 8,603 m, plus heading harmonics 1 and 2
  (`model/policy.py:71-75`); 12 pairs, 37.5% of each head's dimensions.
- **Consequence.** One band alone aliases a separation d with d − 8,603 m, so it is
  unambiguous only to half its period (4.3 km, the radius); further pairs are told apart by
  the finer bands jointly. If that is not enough, add the 29.3 km band (14 pairs, still fits).
- **Options.** Keep v2's two axes; three axes; add the 29.3 km band.
- **Recommendation.** None.
- **Blocks.** Nothing now; revisit with larger maps (decided).
- **References.** model/NOTES choice 2; core-decisions "Fewer rotary dimensions".

#### 6.2 Default position code

- **What it is.** Three swappable position codes: "fourier" (hexagonal Fourier features),
  "grid_cells" (Blair et al. 2007), "v2_codes" (v2's categorical code). You would like grid
  cells as the default.
- **In the code now.** `position_code = "fourier"` (`model/policy.py:66`). The grid-cell
  details (exponential gain, 2 × 2 phases, spacings equal to the Fourier periods) are the
  builder's picks; v2_codes uses a new v2 run's 236,196 m period, most of whose coarse levels
  carry only the offset in a 4.2 km arena.
- **Options.** Keep Fourier; make grid cells the default; first look for papers that used grid
  cells in RL (proposal in core-decisions).
- **Recommendation.** None.
- **References.** core-decisions "Swappable modules" and "Proposals, not decided"; model/NOTES
  choices 11 and 12.

#### 6.3 The next-state head: form, scales and weight

- **What it is.** An auxiliary head predicts seven scaled one-tick changes of the ship's state
  (a regression), at loss weight 0.1.
- **In the code now.** Scales `model/policy.py:78-86` (energy over 400);
  `next_state_coefficient = 0.1` (`learn/config.py:45`).
- **Known problem.** With the v2 game the energy column's mean square was about 1,470 and the
  loss read 209.8, outweighing every other term at 0.1 (learn/NOTES "Found on the way"). The
  scale has not changed since; not re-measured.
- **Others.** v2 used a symlog HL-Gauss delta code; v2's release run turned next-state off;
  v2's default with the belief on was 0.0033.
- **Options.** Set the scales from measured changes; drop the head (as v2's release run);
  keep.
- **Recommendation.** learn/NOTES: set the scales from measured changes before trusting it.
- **Blocks.** The first run (it can dominate the loss).
- **References.** model/NOTES choice 8; README first-cut table ("next-state scales are rough");
  recipe row "Auxiliary losses"; departure audit Observation row "Next-state target scales";
  review 13.

#### 6.4 Builder choices that follow v2 (confirm in bulk)

Each follows v2 or a recorded source, under the rule "anything not listed follows v2". Listed
so you can object to any one (model/NOTES "Choices the author should check"):

1. Game token rotary "centre" (identity), v2's, not lean's "zero" (`model/policy.py:77`).
2. The attention gate acts after the values are rotated back.
3. Depth scale 2/6 on the temporal output and MLP down projections; the spatial sublayers keep
   PyTorch's default initialisation.
4. The temporal MLP initialised the recurrentgemma way.
5. sqrt(1 − a²) kept at a reset (v2), floor 1e-6.
6. Heads as v2 (hidden 256, norm, GELU, orthogonal); the next-state head's last layer starts at
   orthogonal 0.01, not v2's zero.
7. No enemy-action head; the outcome is per ship, no outcome head on the game token.
8. Norm epsilon: torch's default (v2's).
9. The model holds no value supports; the learner owns them.
10. Fields' rotary heading 0 (v2's memory tokens).
11. Sizes as v2: width 128, 2 blocks, 2 heads, 2 spatial sublayers per block, map memory 64.

- **Recommendation.** None.
- **References.** model/NOTES choices 1, 3 to 10, 13, 14; review 13.

### Done or decided, for information

- Built (backlog C1 to C10): doubling scan; Griffin block with the paper's initialisation;
  per-kind encoder; fields as key/value memory; element-wise gated attention, no null key;
  rotary bands to the map with values rotated; 4× MLP; 64 linear HL-Gauss bins per level;
  swappable position code; fog inputs. The hit head and five value levels (model/NOTES "The
  hit head"). 2,048,180 parameters.
- Operations per `step`: 509 compute, under the 650 tripwire (v1's compute count; the "803" in
  model/NOTES adds views) (worklog 03:13).
- Decided: no belief and nothing decoded and fed back; a hidden enemy is a last-seen token
  plus the time since seen; memory is the Griffin block.
- Not built: zone tokens, the absent-zone token, nearest-k attention.
- Known gap: no numerical parity test against v2's modules (candidates: v2's
  `CausalConvolution.sequence` and `RGLRU` against `Temporal` with copied weights).
- Decided for later: revisit the rotary with larger maps.

### Stale notes

- model/NOTES "Critic bins": "Four levels, as the learner has today". There are five.
- model/NOTES "The hit head": "`lean/CONTRACT.md` still lists four [outputs]". CONTRACT now
  lists five.
- model/NOTES "What was not done": no compile of `sequence` or a backward pass; "the ops
  tripwire (650) is exceeded". The update's loss is compiled (forward and backward, D5); the
  tripwire is met on its own counting rule.
- README "Model" first-cut row: 616,677 parameters, temporal block "the recurrent unit only".
- departure audit Model rows (encoder, map memory, null key, temporal block, MLP width,
  heads, parameter count) and top-10 item 9: all ported since.

---

## 7. Training recipe

### Decisions needed

#### 7.1 Behaviour cloning: a stage before RL, or a term inside PPO

- **What it is.** Behaviour cloning (BC) teaches the policy to copy the scripted opponent.
- **In the code now.** A term inside PPO: cross-entropy to the scripted agent's **sampled**
  action (`learn/update.py:111`), weight 2.0 × max(0, 1 − w / 0.45)
  (`learn/config.py:46-47`, `:103-105`), no latch. w is the league's tally against the
  scripted member, draws as half, 0 until the first scripted game, a window of about 100
  games.
- **v1.** The term inside PPO, but to the scripted agent's **probabilities** (soft labels),
  latched off after 3 updates at 0.45.
- **v2.** A separate stage of 100M environment steps cloning Imugi's full distribution
  (coefficient 0.025, no entropy, no KL stop), then RL from the clone's averaged (EMA)
  weights with no BC term and no entropy bonus. Its BC lineup: half self-play, half against
  Imugi.
- **The plan.** A soft-label BC stage, then a decaying distillation term towards the clone
  (core-brief section 4; MA:367-369).
- **Options.** (a) v2's stage; (b) the plan's stage plus distillation; (c) keep the term with
  soft labels (`scripted.marginals` already gives them) and a latch; (d) as is.
- **Recommendation.** None as a decision; the plan prefers (b).
- **Blocks.** The first real run; 5.3 (teacher personas); 8.1 item 1 (gate input).
- **References.** learn/NOTES "Open for the author" and league choice 1; recipe row "BC" and
  main difference 1; departure audit top-10 item 10 and Learner row "BC"; review 13; backlog
  D2.

#### 7.2 Weight averaging

- **What it is.** Keeping an average of the weights over training, which is often stronger
  than the live weights.
- **In the code now.** None.
- **v2.** An EMA (decay 0.95 per update) and a uniform average from the end of warm-up, both
  league members and both rated; BC hands its EMA to RL.
- **v1.** A uniform running mean once BC ends; v1's averaged weights held 0.08 to 0.11 more
  break-even ratio at every size (YS:598).
- **The plan.** "At most one average, and only if measured" (MA:370).
- **Options.** v2's two averages; one EMA; none.
- **Recommendation.** None as a decision; v1's measurement favours averaging.
- **Blocks.** The first run; which weights are rated and shipped; 8.3 (league members).
- **References.** learn/NOTES "Open for the author"; recipe row "Weight averaging" and main
  difference 7; departure audit top-10 item 5; review 13; backlog D2.

#### 7.3 KL early stop

- **What it is.** Stopping an update's epochs early when the policy has moved too far from the
  one that collected the data.
- **In the code now.** None; the approximate KL and the clipped fraction are logged
  (`learn/update.py:138-139`).
- **v2.** 0.02, checked after each epoch but the last. **v1.** 0.1, then 0.02 once the win rate
  passed 0.8.
- **Why it matters now.** At 32 minibatches and 4 epochs an update takes 128 Adam steps, with
  nothing to bound the drift (recipe "Memory note").
- **Options.** v2's 0.02; none (watch the monitors, as the literature decision says).
- **Recommendation.** None as a decision.
- **Blocks.** The first run.
- **References.** learn/NOTES "Open for the author"; recipe row "KL early stop"; literature
  2.3 and 2.4 (monitors); departure audit top-10 item 5; review 13.

#### 7.4 Advantage scaling

- **What it is.** The summed advantage is divided by a scale before the policy loss.
- **In the code now.** Divided by its standard deviation over the whole update, mean not
  subtracted (`learn/config.py:27`, `learn/returns.py:45`). A second option, "percentile",
  is a half-port of v2's rule.
- **v2.** Divided by a slow EMA (decay 0.99) of the 5% to 95% spread of the summed returns,
  floor 1, once per update.
- **Decided.** Normalisation stays over the whole update (core-decisions "Learner").
- **Options.** Keep the standard deviation and delete the half-port; port v2's rule fully.
- **Recommendation.** Literature 1.4: per-update and per-minibatch statistics are practically
  equal at this size; scale-only has a precedent in Acme; Hilton et al. advise watching the
  advantage's standard deviation. No source prefers v2's rule or the standard deviation.
- **Blocks.** The first run; review 6 (the half-port).
- **References.** recipe row "How levels combine" and main difference 4; learn/NOTES "Open for
  the author"; review 6 and 13; departure audit Learner row "Advantages".

#### 7.5 The balance of the loss terms

- **What it is.** The weights of the policy, value, entropy, BC, next-state and hit terms.
- **In the code now** (`learn/config.py:43-48`). Policy 1 (implicit), value 0.05 on the mean
  over levels, entropy 0.003 on the joint entropy (constant), next-state 0.1, hit 0.05, BC
  2.0 (gated).
- **v2.** Policy 27; value 0.007 summed over nine heads; entropy −0.0081 per factor by
  default and 0 in the release run's RL stage (a warm start). **v1.** Entropy 0.005 per factor;
  value 1.0 on a normalised Huber loss. MA:319 recommends re-testing entropy 0.001 to 0.003.
- **Why it is hard.** The coefficients are comparable only together with the advantage and
  value normalisations, and no gradient shares were measured for lean (recipe "Not
  determined").
- **The hit weight.** 0.05 equals the value weight, both cross-entropies in nats per row; v2's
  shot-outcome head had 0.019 against value 0.007 (learn/NOTES reward choice 7).
- **Options.** Keep; port v2's coefficients together with v2's normalisation; measure gradient
  shares first.
- **Recommendation.** None.
- **Blocks.** The first run.
- **References.** recipe rows "Policy term", "Entropy", "Value coefficient" and main difference
  4; learn/NOTES "Departures from v1" and "Open for the author"; review 13.

#### 7.6 The lookahead before the bootstrap

- **What it is.** v2 played 128 extra ticks after each rollout, never trained on, so returns
  were not cut at the segment's end. Lean bootstraps from the critic's value at the segment's
  end.
- **In the code now.** No lookahead (`learn/config.py:38`, one 128-tick segment).
- **Contradiction.** core-decisions "Learner and league" says "No rating lanes and no
  lookahead" (decided 2026-11-22), while learn/NOTES "Open for the author" lists "the 128-tick
  lookahead before the bootstrap" as open. Please confirm whether the decision meant this
  lookahead.
- **References.** recipe row "Bootstrap at the segment's end"; learn/NOTES "Open for the
  author".

#### 7.7 Optimiser details

- **In the code now.** Adam epsilon 1e-5, not fused (`learn/config.py:41`); warm-up 10
  updates, then constant 3e-4 (`:39-40`); gradient clip 1.0.
- **v2.** Epsilon 1e-8, fused on CUDA; warm-up over 10M environment steps; a step with a
  non-finite gradient norm is skipped.
- **Context.** At the decided update size (about 1.05M environment ticks) 10 updates is about
  10.5M steps, close to v2's warm-up (arithmetic).
- **Options.** v2's epsilon and fused Adam; keep.
- **Recommendation.** None.
- **References.** recipe rows "Adam", "Learning rate" and main difference 9; departure audit
  Learner row "Learning-rate schedule"; review 13.

#### 7.8 bf16 as the default precision

- **In the code now.** bf16 autocast for the policy's forward passes on a GPU, losses in
  float32, position phases and recurrence in float32 (`learn/config.py:51`).
- **Measured.** Compiled at 256 environments: 5,160 steps/s against 2,992 in float32; the
  update 1.8 times faster in two thirds of the memory. v2 trained in bf16; earlier lean runs
  were float32.
- **Options.** Confirm; float32.
- **Recommendation.** Q12(a) asks you to confirm; the measurements favour bf16.
- **References.** Q12(a); learn/NOTES "Compiled training on the GPU"; scoreboard "Training on
  the GPU".

#### 7.9 Value supports and range warnings

- **What it is.** Each reward level's critic predicts a distribution over 64 bins; the support
  is the range those bins cover. Targets beyond it are clamped. Warnings fire when too many
  targets are clamped or too few bins are used.
- **In the code now.** Outermost centres ±7 (damage), ±8 (kill_death), ±1, ±1 (the two
  objective levels: placeholders), ±12 (outcome) (`learn/config.py:60-61`). Warnings: clamped
  share above 1%, fewer than 8 bins used, a bin counting from 0.1% of targets
  (`learn/config.py:67-69`), all arbitrary.
- **Measured.** Scripted against scripted, 4v4 to 8v8, both games: damage −4.85 to 5.92,
  kill_death −5.70 to 7.88, outcome ±10. kill_death's 7.88 (one rare multi-kill) sits half a
  bin inside the outermost centre.
- **v2.** 81 symlog bins reaching ±19 reward units.
- **Options.** Accept; other thresholds; revisit the objective levels when a mode pays them.
- **Recommendation.** None.
- **References.** learn/NOTES "Supports, from measured returns" and reward choices 5 and 6;
  review 13.

#### 7.10 How the million ticks per update are split

- **What it is.** You decided about 1 million environment ticks per update (2026-11-23), at
  parity with v2: 2,048 environments × 4 segments × 128 ticks = 1,048,576. The streaming
  build follows v2's split (four 128-tick segments). The open question is only the split:
  the literature review assumed one 512-tick segment per environment.
- **In the code now.** One segment of 128 ticks per update (`learn/config.py:38`) and 256
  environments (`train.py:32`); the streaming build changes both.
- **Options.** Four 128-tick segments (v2's, being built); one 512-tick segment.
- **Why it matters.** The literature review notes that with 128-tick fragments the
  long-horizon levels (discounts 0.999 and 0.9997) are cut at each fragment boundary and
  bootstrapped there. Longer segments mean fewer cuts but more memory per environment.
- **Recommendation.** None from a source. v2's split is what is being built, so a change is
  cheap to make later.
- **Blocks.** Nothing now.
- **References.** core-decisions "Sizes"; recipe row "Update size"; literature (project numbers
  and 1.5).

### Done or decided, for information

- Decided and built: v2's five reward levels, weights, discounts and lambdas
  (`learn/config.py:25-26`, `:72-77`); v2's reward accounting with the damage ledger, kill
  assists and boundary credit; no shaping; no team mixing (rewards additive per ship); the
  outcome as one HL-Gauss level; 64 linear bins, sigma 0.75 bin widths; five levels rather
  than v2's ten offence and defence parts (their sum is the level); the hit head with ghost
  bullet labels; clip 0.2 and 4 epochs (as v2).
- **Decided, not built:**
  - Advantages and value targets once per update (`targets_every_epoch=False`); the code
    still defaults to True (`learn/config.py:36`).
  - 32 minibatches; the code has 8 (`learn/config.py:37`).
  - 2,048 environments; `train.py --envs` defaults to 256 (`train.py:32`).
  - Rollout streaming to the CPU (pinned memory, asynchronous copies, prefetch) and
    micro-batches with gradient accumulation.
  - Compact rollout storage (float32 for absolute position and the PPO numbers, float16 for
    other observation floats, int8 for small integers, int16 ticks for time since seen),
    checked by log-probability difference and KL against full precision.
- Hit head facts: in scripted play about 90% of bullets miss; at 4v4 7% hit an enemy and 3% a
  team-mate; at 8v8 team-mate hits outnumber enemy hits (ghost bullets aimed through a crowd).
  `train.py` plays ghost bullets by default, so the hit head always trains (learn/NOTES reward
  choices 8 and 9).
- Not determined from v2 (recipe "Not determined"): which release file (live, uniform or EMA)
  was shipped; the exact RL budget after the on-box extension (about 200M RL steps by the
  notes); BC's minibatch size in views.

### Stale notes

- `learn/config.py:33-35` comment on `targets_every_epoch`: "Not decided". It is decided
  (False).
- Q12(b) and Q12(c) are decided (core-decisions "Learner"); Q12 still lists them as open.
- learn/NOTES sections "Followed from v1", "Departures from v1", "Choices the plan left open"
  (individual share 0.5, `damage_unit` 1.0, supports ±1.5 to ±4, minibatches 4, facing and
  closing), "Assumptions about the sibling packages" (next-state loss zero, value helpers as
  local copies, a 4-dimensional carry), "First on the GPU", the "Second pass" supports, and
  reward choice 1 (the events bridge is reconstructed; it reads exact `StepOut` fields since
  worklog 03:51). All superseded.
- recipe lean column describes lean at bd24365c (four levels, shaping, team mixing); the
  reward rows have since moved to v2's.
- departure audit top-10 items 3 (shaping fields), 6 (outcome discount 0.998): fixed by v2's
  levels.
- README "Departures from v1's recipe" paragraph: superseded by the items above.

---

## 8. League and evaluation

### Decisions needed

#### 8.1 League choices to confirm (in bulk)

The league is built to your design (half self-play, half league games, the scripted agent a
member, (1 − p)² with a one-eighth uniform floor, AlphaStar's tallies). Seven points the
design did not state (learn/NOTES "Choices the author should check"):

1. **BC gate input:** the tally against the scripted member, draws as half, 0 until a scripted
   game has been played (the sampler's 0.5 prior would switch BC off at the start); a window
   of about 100 games. (See 7.1.)
2. **Simultaneous games** of one member on one tick are counted as every order averaged; exact
   for one game.
3. **A replaced snapshot's running games are abandoned** (re-dealt), not finished against the
   new weights.
4. **Fixed split by index:** environments 0 to E/2 − 1 are always self-play (keeps shapes
   static).
5. **The floor and the weights range over the members present:** before the first snapshot
   every league game is against the scripted agent.
6. **Self-play games update no tally.**
7. **Snapshot opponents sample their actions** at temperature 1.

- **Recommendation.** None.
- **References.** learn/NOTES "The league", "Choices the author should check"; review 13.

#### 8.2 Snapshot cadence and retention

- **In the code now.** A snapshot every 10 updates, 8 slots, the oldest replaced
  (`learn/config.py:91-92`).
- **v2.** Snapshots every 5M steps to 50M, then at doubling intervals, all kept; 4 network
  slots, one redrawn every 2 updates. **v1.** A snapshot at every 200 points of live rating.
- **Context.** At 256 environments the pool spans only the last 2.6M steps (departure audit).
  At the decided update size it would span about 80M (arithmetic: 8 slots × 10 updates × about
  1M ticks).
- **Options.** Keep; v2's schedule; more slots.
- **Recommendation.** None.
- **References.** departure audit League row "Snapshot cadence"; recipe row "Opponent mix";
  review 13.

#### 8.3 Other league members: averages and cheese agents

- **What it is.** v2's league also held the two weight averages and six "cheese" agents
  (scripted exploits). Lean holds the scripted agent and up to 8 snapshots.
- **Options.** Add the averages (with 7.2); add cheese agents; neither.
- **Recommendation.** None.
- **References.** recipe main difference 8; departure audit "v2 complexity lean correctly
  dropped" (cheese agents deferred to after M1).

#### 8.4 In-training evaluation

- **In the code now.** A crossover at network size 4 only (`train.py:50`), every 50 updates
  (`train.py:34`), 64 games per match (`train.py:35`) although `games_for` (v1's formula)
  exists; the network always plays team 0; no position offset is passed to the policy
  (`eval/matches.py:59`).
- **Others.** v1: 5 slots of 512 environments advancing with the rollout. v2: 1,024 rating-lane
  environments (you decided: no rating lanes).
- **Options.** More sizes; `games_for`; draw an offset per match (as v2); keep.
- **Recommendation.** None.
- **Depends on.** 2.3 (the clock rule decides most games, so "ties as non-wins" rarely applies).
- **References.** departure audit Loop rows "In-training evaluation", "Games per cell",
  "Position offset in evaluation"; README first-cut table ("`eval` passes no position offset").

#### 8.5 Batched crossover settings

- **What it is.** The crossover now plays every network size at once, in rounds of batched
  matches, each size walking from the previous evaluation's ratio.
- **In the code now.** `spread = 2` neighbours (`eval/matches.py:159`), a guess; mixed team
  sizes padded to the largest; `maximum_games` uncapped.
- **Measured.** Same answers as the old walk. From the previous ratios about even with the old
  code on the CPU; from parity slower (CPU [4, 8]: 47.6 s to 267.4 s; GPU: 67.6 s to 83.0 s).
  A GPU gain is not shown.
- **Options.** Measure on the GPU from the previous ratios; a smaller spread; group sizes by
  total into separate batches; cap `maximum_games` for memory.
- **Recommendation.** learn/NOTES: measure `start_ratio` from the previous ratios on the GPU.
- **References.** learn/NOTES "Batched crossover (D6)", "Open"; backlog D6; worklog 07:25.

### Done or decided, for information

- Built (backlog D1): the league as designed; snapshots run in one vmapped call, no per-tick
  read-back.
- Decided: no rating lanes; ratings deferred.
- Crossover's start differs from v1's chained start (cannot run in parallel); under the walk's
  assumption it changes which matches are played, not the answer.

### Stale notes

- learn/NOTES "The league", "Opponent actions": per-member grouping with a per-tick
  `bincount` read-back and a loop of up to 8 model calls. Replaced by one vmapped call (same
  file, "Compiled training").
- learn/NOTES "Cost and memory": "`minibatches` stays 4". It is 8 (decided 32).
- learn/NOTES "Followed from v1": half the environments always face the scripted agent. Marked
  wrong in the same file; the league replaced it.
- departure audit League rows (no self-play, fixed scripted half, one snapshot for every league
  environment for a whole update) and top-10 item 1: fixed by the league.

---

## 9. Speed and memory

### Decisions needed

#### 9.1 The core game below the 5,000 steps/s floor

- **Measured.** Compiled, bf16, laptop RTX 4070, league full: v2 game 5,160 (256
  environments) and 5,057 (1,024); core game 4,640 and 4,589. The update dominates (5.1 of
  6.3 s at 256).
- **Levers** (all yours, learn/NOTES "Open"): targets once per update (now decided; 6,457 on
  the v2 game, not measured on the core game); fewer epochs; a smaller update.
- **Options.** Re-measure the core game after targets-once; accept a lower core floor; one of
  the other levers.
- **Recommendation.** None beyond re-measuring with the decided settings.
- **References.** Q12; scoreboard "Training on the GPU"; learn/NOTES "Open".

#### 9.2 Compile start-up

- **Measured.** 199 s cold, 72 s warm to the end of the first update (eager 17 s); repaid after
  about 20 updates cold, 6 warm.
- **In the code now.** `--compile` on by default (`train.py:43`).
- **Options.** Accept; eager for short runs.
- **References.** Q12(d); scoreboard "Start-up".

#### 9.3 The game's operation tripwire

- **What it is.** A budget of operations per game tick, uncompiled: 1,300.
- **Now.** v2 config without fields 1,148 (under); core config 1,522 (over), mostly the flight
  (two kicks of about 400) and the medium.
- **Options.** Accept; a split budget (the game audit's: flight under 1,000, the rest under
  500; the core config's flight is the larger part).
- **Recommendation.** The game audit proposes the split.
- **References.** README "Decisions still with the author"; game/NOTES "Waste fixes done";
  scoreboard "Against the tripwires".

#### 9.4 Play start-up (gate G2)

- **Measured (provisional, loaded machine).** Compiled, every size to 50v50 meets the frame
  budget on the CPU; uncompiled to 20v20. A compile takes 31 to 34 s cold, 12 s warm, against
  "launch to a playable match within 10 s".
- **Options.** Start uncompiled and switch when a background child process has built the
  compiled step (as v2's play shell did); ship the compiled artefact prebuilt.
- **Recommendation.** None; re-measure on a quiet machine first (backlog F4).
- **References.** scoreboard "Play latency"; backlog F4.

#### 9.5 Fields stored every tick in the rollout

- **What it is.** Field rows are constant for an episode but stored every tick: about a sixth
  of observation floats at 4v4 with 10 fields (by shape arithmetic). This matters for the
  decided streaming.
- **Options.** Store once per episode; keep.
- **Recommendation.** None.
- **References.** review 9.

### Done or decided, for information

- Done: compiled training (D5), no graph breaks or recompiles across team sizes, environment
  counts and minibatch sizes (duck sizing off); waste fixes 1 to 9; batched snapshots.
- Policy operations under the tripwire (section 6, information); scripted over (5.4).
- **Known slow spots (no decision; review 9 and learn/NOTES):**
  - Snapshots cost 3.7 ms a tick at 256 environments; empty slots are computed too.
  - Evaluation is eager, re-deals every tick through `reset_done`
    (`eval/matches.py:69`), and reads back a flag every 32 ticks.
  - `observe` runs once per team (two calls a tick).
  - GAE is a Python loop over 128 ticks every epoch (`learn/returns.py:36`).
  - Host reads per epoch (`support_usage`, the normaliser's boolean index).
  - 54 eager `aten.scalar_tensor` operations from `torch.where(..., scalar)`; gone when
    compiled.
- Decided: no bit-exact repeatability (ordinary random generators) and dynamic shapes; sizes
  batch- and GPU-friendly by default (powers of two, fractions such as one eighth); per-tick
  analytics leave the compiled tick (no source records its status).
- Deals are drawn once per segment; a segment with more than E endings would reuse deals
  (counted in `Rollout.repeated_deals`; never seen, shortest episodes 403 ticks).
- Not measured: traced operations across every compiled graph (10,000 tripwire); the core
  game on the GPU after waste fixes 6, 7 and 9.

### Stale notes

- README "First laptop run": "below v2's floor of about 5,000 before any compilation". The v2
  game now meets it compiled.
- scoreboard "Against the tripwires": scripted 172 "at". Now 188 for both teams in one call.
- learn/NOTES "Known weaknesses": "`team_view` calls `observe` twice per tick (three or four
  times with a snapshot)". Now twice per tick in all.

---

## 10. Code cleanup from the review

### Decisions needed

#### 10.1 Where analysis tools live

- **What it is.** `game/tuning.py` (457 lines) is analysis code in the runtime package; it
  patches `card` module globals in place; `per_type_constants` exists only for it; its tests
  check v2's TOML files. The two analysis ship types (p51d, vizia_plain) widen the model's
  type one-hot (ship features 27 to 29; a policy built for 27 does not load).
- **Options.** Move tuning and its types to an analysis folder (for example beside the chart
  scripts in `docs/rewrite/sections/`); keep in `lean/game`.
- **Recommendation.** None.
- **References.** review 6 and 7; p51d-model "Card count".

#### 10.2 Line length

- **What it is.** Lean's files are written at 120 columns against the project's 100; `ruff
  check lean` reports 372 long lines.
- **Options.** Reformat to 100; give `lean/` its own setting.
- **References.** review 14; worklog 03:49.

(Review 1 is item 1.1; review 5 is items 1.1 and the "decided, not built" list in 7; review 6's
percentile normaliser is 7.4; review 13 is spread over sections 1 to 8.)

### Done or decided, for information

Decided test rule (core-decisions "Tests"): a test names the assumption or bug it guards; no
goldens without a reason; comparisons against v2 are transition tests, deleted once a component
is ported; every port change states in advance same, similar (with a tolerance) or different.

Work the review found, no decision needed (estimate: 300 to 400 source lines and about 900
test lines go or shrink):

- **Review 3, test stand-ins.** `learn/tests/standins.py` (191 lines) bends production code
  (package injection in collect and eval, `carry_axis`, an `is_dataclass` guard, generic
  `map_fields` and `seat_view`, `importorskip` on lean's own packages). Fix: test against the
  real packages at tiny sizes.
- **Review 4, duplicates.** Action split and join in collect and contract; the offset draw in
  policy and collect; `scripted_uniforms`; config values held twice (`ModelConfig.actions` and
  `levels`, `Opponent` shares, the League floor and decay); the scripted arena-radius bug
  (section 5).
- **Review 6, dead outputs and test-only code.** `StepOut.damage_dealt`, `friendly_damage`,
  `killer` (two `scatter_reduce`), `fired` unread; rollout counters read only by tests; unused
  parameters and imports in `rewards.py`; `positionless_rotary="zero"`, `left_is_even`,
  `head_to_head`, `crossover(match=)` used only by tests (and the "mixed" spawn, see 4.1);
  `Obs.field_mask` always true; twin dataclasses `ShipCard` and `Constants`; `team_view` and
  `enemy_view` near-duplicates.
- **Review 8.** Importing `lean.learn.compiled` turns duck sizing off process-wide; an
  `importlib` workaround because the function `collect` shadows its module.
- **Review 10.** `core.step` is 247 lines; `collect.py` 597; reward weights as five fields read
  by `getattr`; a few short names.
- **Review 12, tests.** Not earning their place: transition tests, `test_tuning.py`, stand-in
  tests, a test restating a config value, tests of unused memory, hold and spread paths.
  Missing fast tests: compiled collect and update (slow only), crossover with the real game,
  `abandon` with real packages, the scripted opponent at another arena size, scripted power
  under core, `train.main`, the stored position offset reaching the update.
- **Review 1, switches kept side by side** (go with sign-offs): v2/core flight and damage, fog
  off/on, ghosts off/on, fixed/drawn fields, two spawn rules, `advantage_normaliser`,
  `positionless_rotary`, `left_is_even`.

### Stale notes

- None beyond those listed in the other sections.

---

## 11. Docs and notes hygiene

### Decisions needed

#### 11.1 One short current NOTES per package

- **What it is.** The notes pile up history and contradict the code (learn/NOTES is 662
  lines). The review proposes one short, current NOTES per package, with history moved to the
  worklog.
- **Options.** Adopt; keep as is. This document could be the starting point.
- **Recommendation.** The review's (adopt).
- **References.** review 11.

#### 11.2 `CONTRACT.md` and `README.md`

- **What it is.** CONTRACT describes the unpacked `State` and no fog; the README keeps the
  first-cut tables and a "when the laptop is back" list that is done.
- **Options.** Rewrite both to the code; or fold CONTRACT into the code's docstrings.
- **References.** review 11.

#### 11.3 `core-brief.md` and `LAPTOP_PROMPT.md`

- **What it is.** `core-brief.md` (the plan) does not match core-decisions (backlog F1, todo).
  `lean/LAPTOP_PROMPT.md` is a closed one-off prompt.
- **Options.** Rewrite core-brief (F1) or mark it superseded by core-decisions; delete
  LAPTOP_PROMPT.
- **References.** backlog F1; review 11.

#### 11.4 A time box for the skeleton

- **What it is.** Listed in the README under "Decisions still with the author" with no further
  context in the sources. The backlog now orders the work instead.
- **Options.** Set one; drop the line.
- **References.** README "Decisions still with the author".

### Done or decided, for information

- `docs/provenance.md` first version (backlog F2). Scoreboard first numbers (F3; the policy
  and GPU parts done since).
- Gate G1 decided: PyTorch (core-decisions "Framework"); the README's "finish the framework
  benchmark" step is moot.

### Stale notes (by file, what the code does now)

- **README.md:** first-cut tables (36 tests, 616,677 parameters, no fields, no fog, the
  "sides" spawn untested, `reset_done` a third of a step); "Decisions still with the author":
  bullets as tokens is answered (not tokens, Q3); "When the laptop is back" steps 1 to 5 are
  done.
- **CONTRACT.md:** unpacked `State`; `Obs` without fog; `StepOut` without `hit_damage`,
  `boundary_damage`, `shield_restored`, `shot_outcome`.
- **game/NOTES.md:** "Choices where the audit left it open" Q2 (fields default none) and fix 11
  as future work: fields are drawn per environment in `core()`; the boundary ramp's "map scale
  3" (see 2.1); "Outside lean/game": the sandbox "no longer runs": it runs in both configs
  (worklog 04:52, review header).
- **model/NOTES.md:** see section 6 "Stale notes".
- **scripted/NOTES.md:** see section 5 "Stale notes"; "The caller must pass the arena radius"
  is still true (a bug, section 5).
- **learn/NOTES.md:** see sections 7 and 8 "Stale notes".
- **docs/questions.md:** Q12(b, c) decided; Q11's numbers are current.
- **core-decisions.md:** "Proposals, not decided": the Imugi table (already decided).
- **departure-audit.md:** written 2026-11-22; most of its top-10 items are fixed (1, 2, 3, 6,
  7, 9). Still open from it: 4 (batch size: decided, not built), 5 (averaging, KL stop), 8
  (absolute position), 10 (BC), and the "close behind" arena and clock rows.
- **imugi-behaviours.md:** "Lean's opponent as it is" (see section 5).
- **recipe-v2-vs-lean.md:** lean column as of bd24365c (see section 7).

### Contradictions found

- Boundary ramp: game/NOTES ("map scale 3") and the departure audit ("matched") against Q9
  and v2's `rules.py:151` (100 m at map scale 1). Section 2.1.
- The lookahead: core-decisions "no lookahead" against learn/NOTES "open". Section 7.6.
- The "mixed" spawn: a deletion candidate in review 6, needed by the README's stage-1 option
  and ranked first among v1's ingredients by YS:585. Section 4.1.
- Rotary: the decision's wording "x, y and attitude in each head" against the built two axes
  per band. Section 6.1.
- Over-turning: decided "built behind a switch so the author can fly both"; no switch exists.
  Section 1.5.
- Temperature: decided per environment; drawn per ship. Section 5.3.
- `targets_every_epoch`, minibatches, environments: decided, code defaults differ. Section 7.

---

## Coverage: where each source's items went

| Source | Sections |
|---|---|
| lean/README.md | 1.1, 2, 4.1, 6, 8.4, 9.1, 9.3, 11.2, 11.4 |
| lean/CONTRACT.md | 3 (stale), 11.2 |
| lean/game/NOTES.md | 1, 2.1, 2.2, 2.3, 3.1, 4.3, 5, 9.3, 11 |
| lean/model/NOTES.md | 6.1 to 6.4, 6 (stale) |
| lean/scripted/NOTES.md | 5.1 to 5.6, 5 (stale) |
| lean/learn/NOTES.md | 6.3, 7.1 to 7.9, 8.1, 8.5, 9.1, 9 (slow spots), 7 and 8 (stale) |
| lean/sandbox/README.md | 1.1, 1 (stale) |
| docs/questions.md 9 | 2.1 |
| docs/questions.md 10 (a to d) | 1.2, 1.3, 1.4, 1.5 |
| docs/questions.md 11 | 5.4 |
| docs/questions.md 12 (a to d) | 7.8, 7 (decided), 9.1, 9.2 |
| docs/questions.md 1 to 8 (answered) | 3, 4.5, 7, 8 (information) |
| core-decisions.md | every section's "decided" lists; 1.5, 5.3, 5.5, 5.7, 6.1, 6.2, 7.6, 7.10 |
| core-backlog.md | A12 1.2 to 1.5; B2 2.1; B5 2; E1, E2 5; D2 7; D6 8.5; F1 11.3; F3, F4 9.4, 11 |
| lean-review-2026-11-23.md | 1: 1.1; 2: 5.1 to 5.4; 3, 4, 6, 8, 10, 12: 10; 5: 1.1, 7; 6: 7.4, 10; 7: 10.1; 9: 9.5, 9; 11: 11; 13: 1 to 8; 14: 10.2 |
| recipe-v2-vs-lean.md | 7.1 to 7.7, 7.10, 8.2, 8.3 |
| departure-audit.md | 2.2, 2.3, 3.2 to 3.4, 4.1 to 4.4, 5.3, 7, 8.2, 8.4, 11 |
| imugi-behaviours.md | 3.5, 5 |
| core-flight-measurements.md | 1.2, 1.3, 1.5 |
| p51d-model.md | 1.4, 1.5, 1.6, 10.1 |
| literature-ppo-targets-minibatches.md | 7.3, 7.4, 7.10, 7 (decided) |
| scoreboard.md | 7.8, 9.1 to 9.4, 2 (hit test) |
| core-worklog.md (tail) | 1.2, 1.5, 5.4, 8.5, 10.2, 11 |

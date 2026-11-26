# Lean core: the author's decisions (from 2026-11-22)

What `lean` is for, and the deliberate changes to v2. Anything not listed here follows v2's
mathematics; a difference from v2 that is not listed is a departure to fix or to bring to
the author. Proposals are marked and are not decisions.

## The goal

v2's mathematics is right: the game rules, the physics, the model, the PPO update, the
league. v2's code is wrong: about ten times the operations it needs, long compiles, tight
coupling, bloat spread through every file (bit-exact keyed draws, fixed shapes, unused
generality). `lean` re-expresses v2's mathematics in clean code, with the narrow list of
deliberate changes below. Done = every v2 feature the author wants is back. The training
recipe leans towards v2's, with the deliberate changes below.

## Framework

- PyTorch (gate G1, decided 2026-11-22; docs/rewrite/sections/framework-spike/README.md).

## Deliberate changes to v2

Game
- Cheaper damage with the same effects (kinetic term; closeness from the hit test's miss
  distance; head-on term from the cosine of the hit angle). (2026-11-22)
- No bit-exact repeatability: ordinary random generators. Dynamic shapes, not fixed.
- Fields drawn per environment at reset. Field tokens as in v2.
- Fog: a ship is visible to the enemy in the enemy's sight, and on the tick it fires or is
  hit by a bullet; boundary damage does not reveal it. A dead ship that will respawn is
  visible; a ship dead for good is masked out. Every ship is always visible to its own
  team. The scripted opponent reads its team's view, never the state.
- Tell, do not infer: mode, ship type and everything known at reset are inputs, stored.
- Modes: whatever is fastest and cleanest. Observation: the modes differ by a few numbers
  per environment (respawn rule, wave counter, shrink rate, zones). Deathmatch respawn may
  be simply random, turned away from the edge.
- Per-tick analytics leave the compiled tick.

Model
- Swappable modules: no interface framework. Clean, loosely coupled code, so that changing
  one part of the model (position code, temporal block, attention, heads) does not touch
  many places; the choice is made at construction, so the compiled graph holds only the
  chosen module. Position code options: v2's categorical codes, Fourier features
  on a hexagonal layout, and a grid-cell module (the author would like grid cells as the
  default; for now one option among three, 2026-11-22).
- Fewer rotary dimensions (medium or smaller maps for now; larger maps later, with the
  rotary revisited then): drop the bands far larger than the map (removing the lowest
  frequencies beats leaving dimensions unrotated, Barbero et al. 2025). Both heads the same
  layout: x, y and attitude in each (decided 2026-11-22). Values rotated as well as queries
  and keys: the author asked for what is most likely to work; GTA (multi-view vision) found
  the value transform mattered most, QCNet (trajectory prediction) adds relative pose to the
  values; thin evidence, kind (b) at best for this setting. Heading stays in the rotary, with few harmonics.
- Griffin's initialisation (decided 2026-11-22): the decay as the paper's text says
  (a^c uniform in [0.9, 0.999], c = 8); what the paper leaves out from the official
  recurrentgemma code (LeCun initialisation of the gates, the convolution's spread 0.05,
  biases zero). v2's is wrong: its decay gives half-lives of 0.1 to 4.8 ticks, against
  6.6 to 693 in the paper (docs/rewrite/provenance-check.md section 4).
- No belief and nothing decoded and fed back. A hidden enemy is a last-seen token plus the
  time since it was seen. Memory is the Griffin temporal block (v2's block, not a bare
  recurrence).
- No shot-trajectory head. Instead a hit head: would this tick's bullet hit an enemy, an
  ally, or nothing. The last ticks of a segment go unlabelled. Labels from ghost bullets
  (2026-11-22, if it stays simple): every ship puts a bullet in its ring every tick; a flag
  marks it real or ghost; only real bullets cost energy, damage and stop; a ghost records
  its first contact (enemy, ally, none) and is removed. The ring (11 slots, one per tick)
  is already sized for this.
- No null key. Gated attention as the default (Qiu et al. 2025, the Qwen team; the author,
  2026-11-22): a sigmoid gate on each head's attention output before the output
  projection, which v2 already has as an option (src/bnb/models/yemong/spatial.py). Which
  variant: element-wise, the paper's best (decided 2026-11-22). Used without first
  measuring for an attention sink: cheap and low risk. Initialised as released code does
  (Qwen3-Next, Kimi Linear, the official repo): like the query projection, no bias, so it
  starts near 0.5 (decided 2026-11-22; docs/rewrite/rotary-and-gate-check.md).
- Encoder and attention as v2: one first layer per token kind, a shared second; ships and
  the game token are queries; fields and zones are key/value memory in the same
  attention operation. Bullets are not tokens.

Learner and league
- Every reward level the same: HL-Gauss targets, the same number of bins, a scale per
  level; the outcome is one of these levels. Not too hot (the HL-Gauss width checked).
  Decided 2026-11-22 after the source check (docs/rewrite/provenance-check.md): linear
  bins as in the HL-Gauss paper, 64 per level, sigma 0.75 bin widths. Each level's range
  set from its measured returns, and watched during training: a warning when a share of
  targets is clamped at the outer bins, and when targets occupy only a few middle bins.
- No rating lanes and no lookahead. League weighting (1 - p)^2 with a small uniform floor
  (2026-11-22): one eighth of league games uniform over the pool. Win rates per pool
  member kept as AlphaStar's pseudocode does: an unplayed pair counts as 0.5 and every
  game decays a pair's tally by 0.99.
- No shaping level (2026-11-22), as in v2: lean's facing and closing terms go.
- Half self-play (both sides train), half league games (the live side trains); the scripted
  agent is one league member.
- Reward scaling with team size: as now (additive per ship).

Tests
- A test names the assumption or bug it guards (example: holding reverse thrust never takes
  a ship well below its stall speed). No goldens or recorded numbers without a reason.
  Comparisons against v2 are transition tests, deleted once a component is ported. For
  every change during the port, note in advance whether the result should be the same,
  similar (within a stated tolerance) or different, and test that.

Scripted opponent (from docs/rewrite/imugi-behaviours.md, author 2026-11-22)
- Randomness only from sampling the action probabilities, which are the BC soft labels.
  No random switches or changing switch probabilities; hysteresis by concrete rules. Open:
  the tie-break of a perfectly symmetric head-on pass (a fixed rule such as "turn
  clockwise", or a random draw).
- Target: the nearest remembered enemy; keep it unless another is closer than 0.8 of its
  distance. Idea for later: a closeness that also counts the turn needed to face each other.
- Lead (intercept): keep. Power (boost, coast, reverse): keep.
- Fly straight only when already aimed at the target (replaces the second stick call).
- Gun: fire when aimed at the lead point within the target's size and in range; never while
  cooling or short of energy. New: no shot with an ally in the line of fire.
- Boundary steer: keep; the moving boundary handled simply (it shrinks linearly).
- Drop: dead reckoning, type doctrines, switch-rate commitments, keyed draws, mode
  inference, the random-play share.
- Manoeuvres: rewritten as basic fighter manoeuvres, not dropped: recognise one-circle
  against two-circle (and angled merges, scissors), lead-turn before the merge so ships pass
  at their maximum turn rate, commit decisively with rule-based hysteresis (no wiggling,
  no racetracks). No crowd or duel breakaway. Based on real doctrine
  (docs/rewrite/bfm-doctrine.md, in progress).
- Frontline layer: needed for frontline, rewritten cleanly. Spacing from allies applies in
  every mode. Retreat or extend only by a carefully designed rule, or not at all; the game
  should make a badly timed escape from a committed fight fatal (worth a test).
- Mode, ship type and every constant (gun range and so on) and the map are given, never
  inferred. Per-mode behaviour where needed, mostly frontline.
- Temperature per environment, fixed at reset, folded into the gates at no cost.
- Physics question raised: are the speeds of minimum turn radius and maximum turn rate the
  same in our flight model? If so there is no one-circle against two-circle choice
  (docs/rewrite/turn-performance.md, in progress).

Flight physics: agreed direction (author, 2026-11-23), a deliberate change after the v2 port
- Goal: real choices between turn radius, sustained rate and corner speed (today one speed,
  corner, wins nearly everything: docs/rewrite/turn-performance.md), and 2.5D energy
  fighting within a 2D game; collisions stay 2D.
- Weight on the wing: the vertical lift always holds 1 g (today it is paid from the store
  while the store is above 2%, through the `1 - sink_room` factor in `_setting`), so the
  sideways budget is sqrt(budget^2 - g^2). A ship that cannot hold 1 g sinks (store into
  speed).
- Bank implicit and instant (snappy; a roll seen from above looks like nothing): no roll
  state and no roll physics.
- No climb or dive angle: horizontal speed is the speed; altitude is the store.
- The power action (2026-11-23): coast holds altitude; dive (store into speed) and climb
  (speed into store) are equal and opposite trades at a capped rate. At the band's edges
  climb or dive acts as coast. Coast holds altitude only while the wing can carry 1 g;
  otherwise the ship sinks (store into speed), which is also the low-speed safety net.
  No separate reverse thrust.
- The engine always at full effort and only ever drives speed; drag sets cruise speed. The
  engine never feeds the store. Engine power is not tied to the store.
- The store sized to about two vertical loops; fights start at half store and cruise
  speed. Climbing to a full store before a fight is allowed (it costs time). Today the
  store is about 14 loops: 498,000 J/kg, about 12.7 km at the game's 39.24 m/s^2.
- No thinner air with altitude; no energy-for-damage or energy-for-shield.
- Guns (2026-11-23): firing cuts the engine (to zero or a lower fraction) on the ticks it
  fires; bullets no longer cost store energy. To measure: speed lost per second of firing
  at cruise and corner speed, to set the fraction. Known quirk: firing doubles as an air
  brake (the only way to cut thrust); recoil drag is the fix if it proves exploitable.
- Over-turning when slow (2026-11-23): altitude wins. Coast always holds 1 g, so a slow
  ship at full stick turns less; dive holds less than 1 g and frees lift for the turn (the
  descending turn as a deliberate move). Below the speed where the wing cannot carry 1 g
  flying straight, the ship sinks regardless. The other option (the turn wins and the ship
  sinks) built behind a switch so the author can fly both.
- The band's edges (2026-11-23): a smootherstep taper on the exchange rate over 5% of the
  band at the top and at the bottom (today a linear 2%, STORE_FADE_BAND). Smootherstep
  reaches exactly zero at the edge, so the band's limits stay exact.
- The band's edges are a ceiling and a floor (author 2026-11-23): the engine is the same
  throughout; only dive and climb fade out near the edges (a gradual ramp), the width
  whatever flies best (5% for now).
- Required order, slowest to fastest (author 2026-11-23): minimum (sustained) radius, best
  sustained turn rate, corner speed. Anything else means something is wrong. Held as a test
  (tests/test_core_flight.py); the core model meets it for every type (Vizia 124, 160,
  268 m/s). The instantaneous full-stick minimum radius sits near the corner, as the
  constant-lift-coefficient formula gives; a slower one would need a low-speed lift bonus
  (flaps-like), to add only if flying shows it is missing.
- Checked with the turn-performance tool before it becomes the default; the author flies it.

Sizes
- Sizes batch- and GPU-friendly by default (powers of two, fractions such as one eighth)
  unless there is a reason not to (2026-11-22).
- Update size at parity with v2 (2026-11-23): 2,048 environments, 32 minibatches. Fitting
  it on the 8 GB laptop needs rollout streaming to the CPU (pinned memory, asynchronous
  copies, prefetch) and micro-batches with gradient accumulation, built to cost almost
  nothing in speed. Speed comparisons with v2 are like for like (same update size).

Learner
- Advantages and value targets computed once per update from the values stored at collection
  (`targets_every_epoch=False`), the PPO default (author 2026-11-23: follow the literature,
  weighing speed). Literature review: docs/rewrite/literature-ppo-targets-minibatches.md. The
  evidence for recomputing every epoch is weak (Andrychowicz et al. 2021, small MuJoCo runs;
  Pleines et al. 2022, recurrent: "likely not worth the cost"); almost every major library
  computes once per update; the per-epoch pass costs about 25% of loop throughput here.
  Advantage normalisation stays over the whole update.
- 32 minibatches (2026-11-23, the same review): inside common practice, no source measured an
  optimum at this scale. When the update size changes, keep the minibatch size and learning
  rate and change the count (Beukman et al. 2026); fewer, larger minibatches only for hardware
  speed, with Adam's learning rate scaled by the square root of the size ratio (Hilton, Cobbe
  and Schulman 2022). Watch the clipped fraction and the approximate KL every update.

- Compact rollout storage (2026-11-23): absolute position and the PPO numbers (old
  log-probabilities, values, rewards, advantages, returns, position offset) float32; other
  observation floats float16; small integers int8 (signed, -1 sentinels stay); time since
  seen int16 ticks or float. Checked by the log-probability difference and KL against full
  precision.

## Answers from the author review, sections 2 to 6 (2026-11-23)

Item numbers are docs/rewrite/author-review.md's. Section 1 (flight) waits for flying.

Damage, boundary, arena, clock
- 2.1 Boundary ramp fixed at 300 m. Later, with a moving boundary in elimination: the
  boundary may go negative; once it is at -300 m the damage is the maximum everywhere.
- 2.2 Arena 4,200 m for deathmatch and elimination; revisit with moving borders.
- 2.3 Elimination without shrink ends by a wipe or at the clock. The clock is short
  (deathmatch, with respawns, can have a longer one). At the clock: more living ships wins;
  equal living ships, more total shield wins; equal on both, a draw.
- No random cooldown resets (lean has none: cooldowns start at 0).

Fog and observation
- 3.1 All five fog choices confirmed: no reveal at spawn; deaths known only when seen; wrecks
  can be sighted (a visible marker of a death, for players and for attention); only real
  bullets reveal on a hit; time since seen as log(1 + seconds).
- Sight as v2 (checked in the code, `sighted_by_enemy`): within vision range; a field's core
  and its band see each other; a line is blocked when one end is in the core and the other
  outside the field, or when the line crosses the core. The author's example holds: A (left
  band), B (core), C (right band), D (outside): A-B, B-C, C-D see each other; nothing else.
- 3.3 Own team: every ally's pending action is an input (42-way). Actions apply with a
  one-tick delay: the action sampled at tick t is applied at t + 1 (not built: lean applies
  actions on the tick they are sampled). Enemies: no pending action; only what is
  observable (whether it is shooting).
- 3.4 Fixed scales for every ship type, from measured ranges (the flight charts give the
  speeds); linear where the range is narrow (shield), symlog where wide (speed); inputs of
  order 1.
- 3.5 Allies: everything. A visible enemy: everything an ally sees of its own ships
  (position, velocity, shooting, store, shield). A hidden enemy: its last-seen row and the
  time since seen.

Modes, spawns, team sizes, lineup
- 4.1 First runs 4v4, elimination. Spawn rule a setting chosen at launch; default random over
  the arena ("mixed"), with no ship spawning at the edge facing out. A one-side spawn, when
  used, spreads over most of the half disc. Keep the "mixed" spawn.
- 4.2 4v4 only for now; nothing coupled so that unequal sizes (3v5, 3v4) stay easy later.
- 4.3 One ship type for now; build so that more types stay easy; lineup decided then.
- 4.4 Standard spawn (evaluation, human play): full shield, half store. Training: shield and
  store each uniform 0 to 100% per ship. Shield recharge and the core hit stay.
- 4.5 Elimination first; other modes added later.

Scripted opponent
- 5.1 Power rule rethought for the core physics and fighting, kept simple: dive when too
  slow (above all in a turn); climb when too fast or too close; when far from any enemy with
  a low store, climb a little to regain it. Rule details to propose.
- 5.2 Every agent, scripted or network, behaves the same everywhere (collect, evaluation,
  sandbox, play, video). The scripted opponent keeps its target with factor 0.8 everywhere
  (memory wired in everywhere).
- 5.3 Personas: to discuss in depth (the author leans towards a varied opponent and a varied
  BC teacher, as human data would be).
- 5.4 188 operations signed off (the line-of-fire check stays).
- 5.5 The symmetric head-on pass: a random side, drawn once per ship per match and kept.
- 5.6 No exchange-rate measurement against v2's Imugi (different physics).
- 5.7 No retreat or extend for now; in the backlog.
- Known weaknesses are fixed without a decision: the game provides the arena radius (and
  every other number) to the scripted opponent; nothing guesses.

Model
- 6.1 Rotary bands: to discuss. Criteria: the coarsest band unambiguous on the largest map;
  the finest fine enough for close fights but without aliasing from one tick's motion; bands
  close enough in ratio to be read together unambiguously.
- 6.2 Look for papers using grid cells in RL; charts of the grid-cell nodes and coverage.
- 6.3 Every output trained in nats with cross-entropy: actions, values, next state, all.
  So binned next-state targets; literature check on symlog HL-Gauss against symlog two-hot
  for next-state prediction (HL-Gauss was for values).
- 6.4 (1) The game token at the arena centre (the centre the boundary defines). (2 to 6)
  Follow the literature and best practice: gate placement, depth scale, temporal MLP
  initialisation, sqrt(1 - a^2) at a reset, the heads' last-layer initialisation (v2's zero
  came from what v2 used to do). (7) No enemy-action head for now; outcome head per ship or on
  the game token: by the literature. (8) torch's norm epsilon. (9) The learner owns the
  supports. (10) Fields' rotary heading towards the arena centre, not north (rotational
  symmetry). (11) Two blocks, each temporal, spatial, spatial (the code's order).

- Settled by the literature, as the author asked (2026-11-23,
  docs/rewrite/literature-model-choices.md):
  - Next-state head: HL-Gauss cross-entropy per column (sigma 0.75 bin widths, 64 bins);
    linear bins for bounded changes, symlog-uniform bins for heavy-tailed ones (shield,
    store); ranges from measured changes. Targets: x and y as the change minus velocity x
    tick (0 = moved as its velocity said); the other columns as the plain change.
  - Gate: element-wise, after the values are rotated back, before the output projection (as
    built).
  - Initialisation: spatial sublayers take recurrentgemma's rule like the temporal ones
    (depth scale 2/6 over all six residual blocks); sqrt(1 - a^2) replaced by 1 at a reset
    and on the tick after a masked row (released recurrentgemma code); last layers orthogonal
    0.01 for policy, value levels and next state; the hit head 0.01 with biases at the log of
    measured class frequencies.
  - Values per ship for every level, outcome included; a game-token outcome head only as an
    ablation arm.

## Answers from the author review, follow-ups and sections 7 to 8 (2026-11-23)

Follow-ups on sections 2 to 6
- Next-state head: HL-Gauss cross-entropy for every column. A table states, for every column,
  linear or symlog bins and its target form (plain change, or change against an
  extrapolation); a literature check on the target form for position, velocity and shield.
- Initialisation, gate, per-ship values (outcome included): as the literature says (above).
- Rotary bands: ratio between successive bands in 1.65 to 2; the coarsest band above 16.8 km
  (more later for larger maps); successive bands turned by an irrational fraction of 90
  degrees. Open: whether the ratio itself should be irrational (to stop harmonics of one band
  landing on another), and the rotated share of each head (literature varies, and head size
  matters: lean has 2 heads of 64).
- Ship features: drop the raw x/R and y/R (the position code and rotary carry position).
- Training spawns: speed, shield and store each uniform per ship (speed from stall to top);
  one eighth of training games use the standard spawn (full shield, half store, standard
  speed).
- Clock: measure scripted against scripted (4v4 elimination, core game) and set the clock so
  that roughly 80% of games end before it.
  Set 2026-11-25: 1,800 ticks (60 s), the 80th percentile of BFM-CMA2's 4v4 wipe times (54.0 s)
  rounded up to 10 s (docs/rewrite/clock-measurement.md).
- Grid cells: Fourier stays the default; grid cells an experiment arm (no paper compares
  them).
- Experiment arms should favour ideas with some theory but no test in this domain (grid cells,
  weight averaging in RL).

Training recipe (section 7), by the literature unless stated
- 7.1 BC: leaning to a separate BC stage before RL, with distillation (simplest, cleanest);
  a term inside PPO gated by the measured win rate against the scripted agent is the
  alternative. To discuss after the literature check.
- 7.2 Weight averaging wanted (above all for BC). Literature check: who averages weights; does
  a uniform average from the end of warm-up, at constant learning rate, give the benefit of a
  learning-rate decay (shown for LLM pre-training) in RL? If untested in RL, an experiment.
- 7.3 KL early stop if it is standard practice for PPO; otherwise a monitor and a warning.
- 7.4 Advantage scaling: the literature's best.
- 7.5 Loss balancing: the gradient-share tuning loop is to go; literature check on how others
  balance policy, value and auxiliary losses, and on the entropy term (type and amount).
- 7.6 Lookahead before the bootstrap: literature check.
- Rollout shape: literature check on the rollout (truncation) length, and on many shorter
  environments against fewer longer ones (2,048 x 4 segments of 128 against 8,192 x 128
  against 2,048 x 512), with Griffin's cost per length.
- 7.7 Optimiser: standard practice (fused Adam where available, epsilon, gradient clip,
  warm-up length for attention models).
- 7.8 Precision: bf16 wherever standard practice allows, float32 where it requires
  (accumulations, norm statistics, optimiser state, losses).
- 7.9 64 bins; supports calibrated from measured returns, checked by histogram; warnings kept.
- 7.10 About 1M environment ticks per update; how it is split is open (rollout shape above).

League (section 8)
- 8.1 (1) The policy starts as the random agent, assumed to lose every game against the
  scripted agent (the BC gate starts at 0). (3) The live policy's games continue across
  updates. A replaced opponent's games: to decide after looking at the tools (mixture-of-
  experts style routing of rows to league members). (4) Environments 0 to E/2 - 1 self-play;
  the league half to decide with (3). (5) Floor 1/8 of league games, the rest by (1 - p)^2;
  before the first snapshot every league game is against the scripted agent. (7) Snapshot
  opponents sample at temperature 1.
- 8.2 Checkpoints to disk every 10M steps, none deleted. League snapshots on v2's widening
  schedule (every 5M steps to 50M, then wider), roughly one per 100 points of live rating.
- 8.3 The weight averages and the cheese agents join the league (cheese agents checked to be
  reasonable).

- Settled by the literature (2026-11-23, docs/rewrite/literature-recipe-1.md):
  - 7.3 No KL early stop (off by default in SB3, CleanRL, Sample Factory); monitor and warn
    when an epoch's mean approximate KL exceeds 0.02; a stop only if the warning fires
    routinely.
  - 7.4 Advantages divided by their standard deviation over the update, mean not subtracted;
    the half-ported "percentile" option deleted; each level's share of the advantage variance
    logged. Normalising each level before summing: an experiment arm.
  - 7.5 Fixed coefficients on terms that do not depend on reward units (normalised
    advantages; every other term a cross-entropy in nats); no retuning loop; gradient shares
    and cosines kept as diagnostics only; a doubtful term gets an on/off arm.
  - Entropy: a constant 0.003 bonus on the joint 42-way entropy; entropy logged as a fraction
    of ln 42; an adaptive target-entropy coefficient only if entropy collapses.
  - BC trains on the teacher's full distribution (`scripted.marginals`), not its sampled
    action; a BC stage trains the critics too.

- Settled by the literature (2026-11-23, docs/rewrite/literature-recipe-2.md):
  - Optimiser: Adam without weight decay, epsilon 1e-5, fused on CUDA, global gradient clip
    1.0, non-finite steps skipped; warm-up 2 / (1 - beta2) = 2,000 Adam steps (about 16
    updates); constant learning rate after, with weight averaging.
  - Precision: lean's current split (bf16 matrix work; float32 norm statistics, softmax,
    losses, optimiser state, master weights, the scan state).
  - Next-state targets (all HL-Gauss cross-entropy): x, y in the ship's frame (along, across)
    as change minus velocity x tick, linear bins; heading one column as change minus
    nose rate x tick, linear; speed plain change, linear; store plain change, linear; shield
    change minus the scheduled recharge (the damage taken), symlog (needs the one-tick action
    delay). Ranges from measured changes.
  - Rotary: ratio between bands about 2 (irrational optional; the irrational turn between
    bands already separates harmonics in 2D); coarsest band above 16.8 km; about 70 to 75% of
    each 64-wide head rotated (22 to 24 of 32 pairs).
- Bug found by the same review: returns cut at every 128-tick segment (GAE and hit labels per
  segment, bootstrapped at each cut) although an update's four segments per environment are
  consecutive; v2 chained them. To fix: chain across the update's segments, bootstrap only at
  the end.

## Answers from the author review, sections 7 to 10 and follow-ups (2026-11-23)

- Crossover is an evaluation measure only, not run during training (for now).
- League routing: each opponent gets a fixed row capacity (about 1.25 times an even share);
  draws respect the capacities; rows sorted by opponent on the GPU each tick; a replaced
  snapshot keeps playing its running games in a retiring slot until they end (no abandoned
  games).
- Clock measurement once everything else is settled.
- Micro-batches: as large as fits the GPU's memory, all the same size (the size divides the
  minibatch exactly). Streaming always on when micro-batching.
- Position float32 in the rollout store.
- Compile start-up (about 200 s cold) accepted; no recompile at a new team size.
- The operation tripwire is a signal to look for bloat, not a hard limit.
- Play: start uncompiled and switch to the compiled step once a background build is ready.
  The frame budget to be re-measured with the full game, rendering included (rendering
  later, without v2's bloat).
- Section 10 code cleanup as the review lists. Section 11: one short current notes file per
  package (what is, what is open), history in the worklog, closed one-off prompts deleted;
  clean and tidy, so agents' context is not polluted by stale text.
- Engine power, which aircraft to build, and the core flight sign-off wait for the WWII
  research.
- 7.3 KL early stop off (not standard practice). 7.4 as the literature (standard deviation
  over the update). 7.5 fixed coefficients on scale-free terms; gradient shares diagnostic
  only. Entropy 0.003.
- 7.1 BC: a BC stage cloning the teacher's full distribution on the student's own games, all
  heads and critics trained; RL starts from the stage's averaged weights with a teacher term
  that shrinks every update until it switches off (a reverse warm-up).
- 7.2 An EMA and a uniform average kept (both in the league).
- The next-state loss is HL-Gauss cross-entropy on bins; targets computed from the states.
- BC on the full soft distribution.
- Shield next-state target (the damage taken in a tick, after the scheduled recharge) in
  symlog bins (author 2026-11-23): fine near zero for single hits at different angles, and
  the range for several hits in one tick (a crossfire of up to four ships). Range from
  measured per-tick damage.
- Rotary: ratio 2 by default, irrational turns between bands, about 70 to 75% of each head
  rotated.
- No lookahead by default. Update 2,048 x 4 x 128 by default, returns chained over the full
  512 ticks; 512-tick training sequences an arm.

## Flight, after the author flew the core model (2026-11-23)

- The core model feels good. Store size fine; the firing cost fine (if anything weak).
- Coasting holds altitude: altitude changes only on an explicit dive or climb. When the lift
  budget is short (slow, full stick and climb together), the turn has precedence over the
  climb: hold 1 g, then the turn, then the climb gets the remainder.
- Every flight-model term per ship type (no shared drag rise, soft minimums or separation
  drag), so a real aircraft's row can match its fit; the high-level to low-level conversion
  (tuning.py) stays, so rows are set from attributes like corner speed and turn radius.
- A time-scale setting for flight testing (fly at 1x, 1.5x, 2x, 3x); the game stays at 2x.
- Doghouses in game units and converted to 1x real time, to compare the P-51D with
  published charts.
- Learning-rate schedule (author 2026-11-23): warm-up (length from the literature: 2,000 Adam
  steps), then constant. A cooldown is added later, once training curves show what is needed
  (a plateau, a collapse).
- Ship candidates: famous WWII fighters built in the flight model from the most reliable
  data found, compared with published charts; keep the ones that match and feel best, or that
  are best documented. The author found the P-51D feels a bit better than Vizia.
- Engine power to be brought in line with real WWII aircraft, the P-51D in particular (open:
  Vizia is 255 W/kg real thrust power, about 1,810 hp at 4,354 kg; the P-51D 225 W/kg).

## Sign-off and personas (2026-11-23 evening)

- The core flight model is signed off (the author flew Vizia and the P-51D at 2x; the
  scripted opponent reversed a forced two-circle fight into a one-circle). v2's flight and
  damage models go.
- The P-51D (its real point mass, lean.game.tuning) is the frontline ship: the core game's
  lineup.
- Green light for every change agreed in the author review and after (docs/rewrite/
  core-backlog.md "Build batch 2026-11-23").
- P-51D g limit stays 8 (author 2026-11-23; the references put it nearer 6.5 g at the tested
  weight).
- Personas: everything per ship, temperature included (as different humans differ in skill
  and decisiveness); hidden from the student (no persona inputs); drawn from a dense
  continuous distribution over the parameter space, every persona competent (no persona that
  flies out of bounds). The default BC teacher stays the single narrow yardstick (spread 0);
  the persona population is an experiment arm.

## Scripted BFM (2026-11-23 night)

- The author beat the scripted agent in a two-circle fight by banking speed (climb) and
  cashing it later (dive) to shift the circles and gain angles each pass. A teacher that never
  uses the store gives BC no prior for those states, so the scripted agent gets basic fighter
  manoeuvres (E2): circle detection, sustained turns and pursuit choice in two-circle, hard
  turns in one-circle, yo-yos through dive and climb, climbing when not engaged.
- Every scripted parameter is dimensionless: a multiple of one named reference of the ship's
  own performance (speeds by corner speed or stall speed; shooting distances by gun range;
  manoeuvre distances by the sustained turn radius; times by the time to turn 90 degrees at
  the best sustained rate; rates by the best sustained rate; energy as a store fraction;
  angles as they are). So a persona transfers across aircraft and time scales and samples
  around 1. Game rules (arena radius) are inputs, not parameters (author 2026-11-23).
- Built natively, not bolted on: the scripted agent is first reviewed for cohesion and
  coupling, and its parameters must stay tunable and independent, because personas sample
  them.
- The lead turn and the yo-yos are deleted (author 2026-11-25, review S2/S3; on the "Later
  experiments" list): the saturated yo-yo signal is frozen at -1, leaving the chase speed
  (fight speed raised by corner over sustained speed against a turning target).
- Trimmed after the reach check (2026-11-25; agent's call, after the reach check; author to
  confirm): the overshoot rule, the spend term and its three gates, the two-circle cash, the
  break relaxation and the defend range (a defensive ship always breaks) and the speed-floor
  clamp are deleted; turning_floor and boundary_margin are no longer sampled
  (scripted-reach.md "Trim").

## Flight model direction and shared code (2026-11-24 night)

- Real aircraft are a grounded starting point for tuning feel and balance, not a simulation
  target.
- Rows come from the high-level-to-low-level route: the game terms (drag rise, separation
  drag, soft lift and thrust limits) stay on, and the solver sets the low-level coefficients
  so the targets still match, rather than switching terms off.
- Smoothness wherever it is free (soft limits kept moderate; the solver compensates).
- Automatic flaps: maximum lift rises at low speed as if flaps were exactly as far out as
  useful, with a drag cost; no button or slider; per-type parameters. Gives slow ships a real
  radius advantage.
- The engine at the aircraft's maximum combat rating (war emergency power): the game has one
  power setting, so it is set for an all-out fight.
- One source of truth for physics and setup: every tool (live sliders and doghouse, play,
  evaluation, training, any future mode) calls the same game code through shared entry
  points; nothing duplicated without a strong reason.
- A small local web app for the dev tools now (sliders, charts, a plain sandbox), kept simple
  (author 2026-11-24): no styling effort except the charts. An ONNX export spike later, before
  choosing the game client (web or an engine such as Godot).
- Compile time is a budget, not an afterthought (author 2026-11-24: cold starts near 10 min
  are concerning; they must not snowball back to half an hour).

## Evaluation runner and scripted search (2026-11-24 night)

- Order: A1 (physics) -> evaluation runner -> screening of the scripted agent's parameter space
  (Morris about 5 min, then 4,000 Sobol-sequence samples about 25 min with a surrogate;
  extendable) -> a short CMA-ES search -> choose the yardstick. Active subspaces, MAP-Elites,
  ship-space and joint searches are later.
- The scripted agent should not need retuning when the game changes (e.g. the map radius):
  CMA-ES gets a small budget and aims for a robust region, not a sharp peak for one test;
  the screening is there to understand the space.
- The runner keeps results reusable: games gathered for one purpose (screening, Elo ladder,
  win rates, crossover) serve the others where they are valid, stored compactly: per agent
  pair and per match conditions (team sizes, map, spawn rule, clock, game version), wins,
  losses and draws; Sobol sampling continues where it stopped.

- Behavioural variety of scripted configurations is measured as the Jensen-Shannon divergence
  of their action distributions on a fixed set of stored states (author 2026-11-24), beside
  per-game style counters; skill alone does not decide which parameters to freeze.

- **2026-11-25 (branch eval-scale-routing, for the author): evaluation to 32v32, the league's
  routing in evaluation.** The runner routes each seat to its agent with the training league's
  routing (`lean.model.agent.route_rows`, shared with `routed_step`), so one call holds every
  network pairing of a job (networks of one architecture and scripted vectors); writes go in
  place to the network's own seats. Scripted-only games keep their own call, calls split by
  ship bucket (a queue pads to its largest game), and `minimum_envs` 2,048 lets a short job
  fill the GPU. Against the old pairing-by-pairing play on a 7-agent 4v4 ladder: 1.78x at 128
  games per pairing, 1.43x at 512, parity at 2,048; ratings agree within errors. Sizes 4v4 to
  32v32 and unequal sizes compile once (3 graphs with networks, 1 without, no recompile).
  32v32 on the laptop: the network fits E 4,096 (6.2 GiB) and is saturated from 2,048 (about
  15k env ticks/s, 16.5 games/s untrained against the yardstick). Unequal sizes are now read
  in both orientations (counts, Elo, styles). Numbers: lean/eval/NOTES.md.

## How much to measure (author 2026-11-24)

- Every merge: the quick test suite.
- Only when a change touches the hot path (game step, policy, the learner's compiled graphs,
  the evaluation runner's tick): the compile tripwire and one rough speed run (noise
  acknowledged; no waiting for a quiet machine, no repeats).
- A careful quiet-machine measurement only when a decision depends on the number (choosing an
  implementation, a budget, a milestone such as before a dry run).
- Analysis jobs (screening, divergence, reviews, searches) take no speed measurement; agents
  may run in parallel except during the rare careful measurements.

## Names for scripted agents (author 2026-11-24)

- **Old (pre-BFM)**: the scripted agent before step E2; its code no longer exists.
- The **BFM agent** is the only scripted agent in the code; parameter sets are named:
  **BFM-default** (today's defaults in lean/scripted/parameters.py, set by E2's ablations),
  **BFM-design** (the design document's original values; the tests' `BFM` persona),
  **BFM-CMA1** and **BFM-CMA2** (CMA-ES rounds 1 and 2).
- "Yardstick" (and "BC teacher") is a role: whichever parameter set is chosen.
- BFM-CMA2 is the provisional yardstick and BC teacher (author 2026-11-24 ~12:30), pending the
  author's own flying and review of open caveats.

## Charts (author 2026-11-24)

- Matplotlib is the one charting system: the web tool shows static images rendered by the same
  drawing code as the committed charts (a 1-2 s redraw is acceptable; hover and zoom not
  needed).
- Automatic flaps: smooth in speed and in its derivative (C2), scheduled by stick demand (as
  the F-16's), turn rate never decreasing with speed below corner (a test); real aircraft rows
  derived automatically from their targets (no stored copies to regenerate).

## Stick levels (author 2026-11-24)

- Full stick is the most turn the ship can give at that speed: exactly the envelope's top
  line, no blending tax. Smoothness only where it belongs: the true corner (the wing's lift
  against the g limit) stays a soft minimum; a command is not soft-minimised against its own
  limit.
- Every other level is a fraction of that same maximum lateral acceleration at the current
  speed (the same shape as full stick).
- The sustained level's fraction is derived so the sustained level passes exactly through the
  best sustained point, per type, with no stored copy.
- The fine level: a per-type fraction, default half the sustained level (the rule may change).
- Flaps off by default (P-51D `flap_lift` 0); the machinery and its tests stay, switchable per
  ship.
- The separation drag was too small: swept with the solver keeping the targets
  (docs/rewrite/p51d-model.md "Separation drag"); the default (0.5 at exponent 24) is a
  judgement for the author to confirm.

## Ship design (author 2026-11-24, docs/rewrite/ship-design-proposal.md option C)

- Two layers: the documents and best guesses give a reference ship (physical); the design
  sliders start at the reference's values; one `design_row` turns the design inputs into the
  sim's coefficients for both the card and the tool.
- Design inputs: stall speed and g limit (closed form); top speed, best climb (and best-climb
  speed, solving the thrust softness), best sustained rate and the gap (tightest-turn speed
  over best-rate speed, solving the separation drag at exponent 16, P-51D 0.85) by Newton;
  the rest passed straight through. Corner speed, radii, corner rate and the settled
  full-stick turn are readouts. Static thrust stays physical (from the propeller). The
  bullet's reference speed is computed where used, not stored.
- Physical-plausibility warnings (implied e, C_D0, CLmax, propeller efficiency, ...) shown,
  not enforced; the tool lists the implied physical quantities and the solved coefficients.

## Design profiles (author 2026-11-24 evening)

- Design-input sets are named profiles (tables); "documents" stays the card's default.
- "Performance points" profile: g limit, corner speed, top level speed, best climb, best
  sustained speed, and the gap (tightest sustained speed / best sustained speed); best
  sustained rate and tightest speed are readouts. Corner rate dropped (fixed by g limit and
  corner speed). The original six (with best sustained rate, no climb) were ill-conditioned
  (335): speeds and sustained turns depend only on power-to-drag ratios, so a climb input is
  needed to set the level.

## Simpler flight and design (author 2026-11-24 night)

These replace the flap, profile and stick-level points above where they disagree.
- Flaps removed entirely (the P-51D flew with them off): no flap card fields, constants,
  drag term or solve stages; the wing is the clean wing.
- One design profile, "documents": stall speed, g limit, top level speed and best climb
  (the last two solved for the zero-lift drag and power, nothing else solved); span
  efficiency e 0.85, separation drag 0.3 at exponent 8 ("a little, physically realistic")
  and thrust softness 2.83 (what the old solve gave for 165 mph best climb), all closed.
  Best-climb speed, best sustained turn, corner and gap are readouts. Performance points
  and the trial profile are gone.
- Stick levels are fixed thirds of full stick for every type: gradual 1/3, middle 2/3, full 1
  (renamed from fine and sustained; the action space is unchanged). Play shell: A or D
  alone middle, Shift full, left Alt gradual (Ctrl replaced: Ctrl+W closes the tab).
- Tried and removed the same evening: a thrust lapse (thrust = power / speed x
  (speed / 165 mph)^(1 - lapse), 1 a propeller, 0 a jet), b71ef433, reverted in 0ec71cd8.

## Proposals, not decided


- Remove v2's flight and damage models from lean/game once the core model is signed off
  (author 2026-11-23: two flight models in one file go when relevant).

- Grid cells in RL: look for papers that used them (docs/rewrite/provenance-check.md).
- A keep, drop or simplify table of Imugi's behaviours with rough costs (operations, lines,
  time per call, parameters): docs/rewrite/imugi-behaviours.md, ready for the author.

## Answers to author review 2, first pass (author 2026-11-24 night)

- S1: BFM-CMA2 stays provisional; after the stick change and S2, re-check it, and run one or
  two small CMA rounds only if the re-check shows it needs them.
- S2: delete both the lead turn and the yo-yo; both go to the future-ideas list (a lead turn
  belongs with a pre-merge offset by plan, if ever).
- S5: one widened overshoot gate in the S6 pass.
- S4, S6: first check whether the flat mechanisms are reached at all and how much they change
  the action when reached (proposal pending).
- S7: accepted as the game's character (energy matters, as in real fights); no balance work.
  The author wants to stop changing the scripted agent and the ship soon.
- F1: the simplified P-51D is good enough.
- L1: re-measure the ranges when the time comes. L2: as built for the dry runs.
- V1: compile budget confirmed (about 225 s cold is acceptable). V2: keep the strict hash of
  the game files only.
- C1: later. C2: the modern standard, applied automatically by the formatter, run often.
  C3: keep tuning.py in lean/game. C4: rewrite the brief as a short document of goals and
  standards for the author to review.
- Sandbox: the gradual turn on left Alt, not Ctrl (Ctrl+W closes the tab).
- Entropy bonus 0.003 on for the whole run, teacher phase included; no A/B
  (docs/rewrite/literature-entropy-teacher.md: VPT and Wolczyk drop it, kickstarting keeps it,
  none ablates).
- A second design profile, "airframe and engine": zero-lift drag C_D0 and the engine's shaft
  power are inputs (closed form, nothing solved); top speed and climb are readouts. The
  propeller efficiency is held at the card's. Power shows in hp in real imperial units, kW in
  metric.
- The P-51D card: the airframe with a 2,200 hp engine (author 2026-11-24 night)
- Supports with buffer columns (author 2026-11-24 night, built 2026-11-25): every value support
  and next-state column spans its measured range with 64 columns and adds
  `BUFFER_COLUMNS` = 3 of the same width beyond each end (70 bins; the shield's in symlog
  space), in code (`lean.model.value.buffered_centres`), not by hand.

- **2026-11-25 (author): 12-tick bullets.** The P-51D's gun range is 686 m, the full range of a
  12-tick bullet fired at the documented 375 mph (it was 600 m, an 11-tick life), and
  `contract.BULLET_SLOTS` is 12. Gun range stays defined at 375 mph for every design (so the
  design tool round-trips). Quick suite 311 passed; compile check within the baseline (4 graphs,
  5,542 operations, cold 210.6 s).

- **2026-11-25 (author): flat 63 bins, symlog with principled knees.** Every categorical head
  (the five value levels and the six next-state columns) has 63 bins in total: 57 across the
  measured range plus 3 buffer columns at each end, so 0 is a bin centre of a symmetric range.
  No zero class. Bins are uniform in s(x) = sign(x) knee ln(1 + |x| / knee) (one helper,
  `lean.model.value.symlog`, knee 0 = linear); HL-Gauss targets are the Gaussian (sigma 0.75
  bin widths) in that space, and the decode is the expectation over the centres mapped back to
  target units. Knees: the event rule for the value levels (one event x e^-3: damage 0.0286,
  kill_death 0.141; outcome +-10 and the objective placeholders +-1 linear); the flat-core data
  rule for the next-state columns (along 2.4e-3, across 0.0244, heading 7.7e-5, speed 0.156,
  energy 21.8; docs/rewrite/sections/scoreboard/symlog-preview.md). Ranges are the measured
  extremes, symmetric (+-1.89 m, +-1.82 m, +-0.100 rad, +-5.39 m/s, +-976). The shield keeps
  -90 to 0, knee 1, pending its bug check. Parameters 2,172,527 -> 2,152,738. Range
  monitoring thresholds unchanged (counted in bin space). Quick tier lean/model + lean/learn 118
  passed; compile check within the baseline (4 graphs, 5,633 operations, cold 211.9 s). Branch
  bins63.

- **2026-11-25 (author): vision 1,500 m** for every type (was 1,000 m). Two-circle fights reach
  about 1,260 m apart (docs/rewrite/sections/vision/vision.md). Known open item: the scripted
  competence test (every parameter at its box ends) now has one game at level_three -1 reaching
  1.095 arena radii, past the boundary ramp's end (1.071); the same case over 32 other games
  stays within 0.944. Handed to the scripted-ghosts branch, which changes target choice.

- **2026-11-25 (author): training targets.** BC 0.40 against the scripted agent within 100M
  steps; RL to 2000 Elo within 100M more; the scripted agent is whatever it is and is the
  1000 Elo anchor. Train at 4v4; evaluate and keep skill up to 32v32, at least a 99% win
  rate there against the scripted agent.
  Training stays 4v4 only (author: v1 trained only at 4v4 and won 99%+ at 32v32); no
  team-size mixing. The evaluation needs a size ladder up to 32v32 and a 32v32 memory and
  speed check.
## Landed 2026-11-25 (author 2026-11-25, branch land-nov25)

- **45-degree exchange.** The card's dive and zoom trade is g x sin 45 deg (0.707 g, 27.75 game
  m/s^2), not 1 g: `exchange_acceleration = card.GRAVITY * sin(EXCHANGE_ANGLE)` in
  `lean/game/tuning.py` `airframe_row`. The coupled physics stays (the climb gets only the lift
  the turn leaves). Evidence: docs/rewrite/sections/energy-duel/exchange45.md.
- **Decoupled climb dropped, records kept.** `GameConfig.decoupled_exchange` (branch
  energy-duel-decoupled) was tested and is not in the game. Its pages (energy-duel/decoupled.md,
  exchange45.md) and `bench/energy_duel.py --exchange-angle` / `--steady-only` are kept; the
  bench has no decoupled option.
- **Rotary reach at a constant ratio of the arena.** `rotary_reach` = `ROTARY_REACH_ARENA_RADII`
  (4) x the arena radius, set by `ModelConfig.for_arena(game_config.arena_radius)` where the
  model is built from the game (lean/train.py; the model package does not import the game).
  The bare default is the default game's (a test checks it). Revisit if the map gets too large.
- **Tests check rules, not today's data.** The structural-load warning tests move a design's g
  limit a tenth above and below the check's bound (read from `PLAUSIBILITY`) instead of
  asserting the P-51D's own 8 g against 6.6 g; the hit-head label test checks that labelled and
  unlabelled rows both occur and labels cross a game's end, not a labelled share.
- **Scripted ships may be lost to fields.** Field-aware steering only matters as a better prior
  for RL, not now. The per-parameter competence test (every sampled parameter at its box ends)
  plays without fields, its boundary threshold read from that config.
- Checks for this batch: quick tier on lean 318 passed (no failure; the per-parameter
  competence test passes without fields); compile check within the baseline (4 graphs, 5,703
  operations, cold 239.0 s, performance profile).

## Sight ellipse (author 2026-11-25, branch vision-ellipse)

- **Ellipse vision added, configurable, default off.** Sight is an ellipse with the observer at
  one focus along its nose: |d| - e (d . nose) <= p, p the card's vision range (the side reach),
  e the card column `vision_eccentricity` (a design input, 0 <= e < 1). Reach p / (1 - e) ahead,
  p / (1 + e) behind; e = 1 - p / (forward reach). Default e = 0, exactly the old circle, until
  the author has tested it (replay `--vision-eccentricity E`, the sandbox tool's slider).
  Fields' cores and team sharing unchanged. Measurements: docs/rewrite/sections/vision-ellipse/progress.md.
- Checks: quick tier on lean 326 passed; compile check 4 graphs, 5,712 traced operations (core
  5,703: +9), 0 graph breaks, cold 235.7 s (core 234.3 s; performance profile). A first run read
  278.4 s and failed the 260 s budget with the machine busy (load ~6, a video player on the GPU);
  the re-run on a quieter machine passed with the same operation counts.

## Nose cone and collisions (author 2026-11-25, branch nose-cone-collisions)

- **Nose cone, one zone for bullets and collision blame.** No bullet damage when the hit angle
  (the bullet's path against the target's nose) is within 30 deg; full from 45 deg to 180 deg; a
  smootherstep in the cosine between, t = (cos 30 - c) / (cos 30 - cos 45) (no acos). The card's
  `nose_inner_angle` / `nose_outer_angle` (stored as angles, used as cosines) replace
  head_on_fraction_scale, head_on_angle_scale and head_on_sharpness (removed). Kinetic term and
  graze closeness unchanged. `core.aspect_factor(hit_cos, nose_inner_cos, inverse_nose_width)`.
- **Collisions are the game** (no switch): every pair of living ships, any teams, scored once
  per pass on the tick holding the swept closest approach. Blame 0.5 + 0.5 smootherstep on the
  same 30-45 zone at first touch (= 1 - aspect / 2). Damage = max shield x |w| / (2 x cruise) x
  graze(b; summed core radii, summed hit radii) x blame: linear in the closing speed, a full
  shield each head-on at cruise. Zero sum: credited to the other ship in `hit_damage` (rewards,
  kill credit, scoring as a hit; team-mates included); both ships revealed under fog; shield
  rule as for bullets. In the step after the move, with bullet hits and boundary, before the
  shields (`core.collisions`).
- **Scripted miss offset.** `merge_miss_offset` (10/3 hit radii, 40 m) inside
  `merge_miss_range` (1.25 sustained radii, 301 m): closing faster than its own speed on a pass
  nearer than the offset, with any ship (enemy or team-mate), the stick aims the offset beside
  that ship's centre; the gun is unchanged. The gun's worth gate reads the new aspect, so it
  holds fire within 30 deg; its kill case needs a hit worth more than 0 (a fix: it fired nose to
  nose at an empty shield).
- Checks (2026-11-26): quick tier on lean passed; compile check 5,703 operations on core to
  5,918 (act +115, advance +100: the collision test and the miss offset), cold 234.3 to 252.3 s,
  within the baseline (5,721, budget 260 s) and its 10% tolerance; a baseline update to the new
  count is proposed, not made. Sanity (256 scripted games each, before / after): 1v1 draws 93 /
  41, mean length 58.2 / 32.1 s, contacts per head-on pass 0.999 / 0.004; 4v4 contacts per game
  8.49 / 3.07, 4v4 deaths on a collision tick 101 of 1,395; real hits within 30 deg of the nose
  31-85% / 0.7-3.5%. Table: lean/game/NOTES.md "Numbers".
- Resolved 2026-11-26 (author): with both ships turning, the closest approach of the per-tick
  straight moves could fall in two ticks or none (fast passes scored twice 1-4 times and missed
  1-2 times per 256 4v4 games). Now scored at first touch, below.


## Tracking circle (author 2026-11-26, branch tracking-circle)

- **Seen ships are tracked in a circle.** An enemy visible on the tick before, by any means
  (sight, its shot, a hit, a collision), stays visible while some living observer of the other
  team has it within the circle of the observer's forward reach p / (1 - e) with a clear line
  (field cores block it exactly as they block sight); when no observer holds it and nothing
  reveals it, it is hidden and needs the ellipse or a reveal again. No new state: tracked =
  ticks since seen 0 and known (`core.tracked_by_enemy`; the deal's rows are unknown, so a
  never-seen ship is not tracked). Per pair the reach is the circle for a tracked target, else
  the ellipse, square-root free. `GameConfig.fog_tracking` (on; off for ablations), a rule
  switch like `fog`, not a card column (the circle's radius comes from the card's p and e). At
  e = 0 the circle is the ellipse: today's default (e 0) is unchanged, tested.
- Measurements (256 yardstick games per condition, e 0.5, p 1500 and 850, off / on):
  docs/rewrite/sections/tracking-circle/progress.md. Without fields 1v1 contact losses fall to
  0 (from 0.25 and 5.65 per game), 4v4 by 75% and 85%; with fields by 4-41% (most losses there
  are field cores cutting the line). First sightings, draws, length and kills per minute stay
  within paired noise for the yardstick.

## Collisions at first touch (author 2026-11-26, branch collision-first-touch)

- **Once per overlap, on the tick of first touch.** A pair is scored on the tick its gap first
  comes within the two hit radii R: at least R at tick start and the swept straight relative
  move inside R during the tick. Ships that start a tick inside R are never scored; they must
  start a tick at or beyond R again first. State-free (start and end positions), branch-free.
- **b is the pass's predicted closest approach**: the closest distance of that tick's straight
  relative line, extended past the tick end; graze(b; summed core radii, R) as before (cores
  would meet: full; only the hit radii: a partial graze). Blame at first touch unchanged; damage
  formula, zero sum, credit through hit_damage, fog reveal, the 0-shield rule and the step order
  unchanged. Bullets unchanged.
- Checks (2026-11-26): collision tests (a fast pass with both ships turning that the old rule
  scored twice or missed: once; none while overlapping; again only after separating; core touch
  full, graze partial; blame; zero sum); quick tier on lean/game and lean/scripted 141 passed;
  compile check 5,939 operations (core 5,930, +9), cold 222.9 s (core 222.4 s), within the
  baseline. Sanity (256 scripted games each, before / after): closed overlap episodes scored 0
  or 2+ times 0-10 / 0 in every condition; contacts up 12-17% (1v1 248 / 290, 4v4 787 / 883):
  on the same play every overlap episode holds exactly one first touch, and the extra contacts
  are passes cut by a death or the game's end between first touch and the closest-approach
  tick, which the old rule never scored (1v1 68 of 316 episodes, 4v4 103-120 of 853-886).
  Deaths on a collision tick 1v1 24 / 72, 4v4 101 / 115 (fields 112 / 97); mean game length
  1-5% shorter; draws 1v1 41 / 49, 1v1 fields 13 / 30, 4v4 3-4 / 3. Table: lean/game/NOTES.md "Numbers".


## Post-stall flight (author 2026-11-26, branch post-stall)

Supersedes "Stick levels (author 2026-11-24)". Every step's decision and measurement:
docs/rewrite/sections/post-stall/progress.md (steps 1-8); the rules: lean/game/NOTES.md.

- **Three levels, stick indices unchanged**, each all of its own line (`card.LEVEL_NAMES`):
  Sustain (was gradual) the stick holding Ps = 0 coasting at the spawn store, from a per-type
  table over speed (Max where Max gains energy, straight from cruise speed up); Max (was middle)
  the envelope's top line, the wing's maximum lift against the g limit; Snap (was full) Max's
  line at and above corner, below it the angle of attack opening past the stall to the card's
  `deep_stall_angle` (45 deg) over `deep_stall_band` (0.30) of corner speed. The 6 g cap and
  `middle_g_fraction` are gone.
- **Past the stall** the lift falls smoothly and monotonically to the flat plate's (0.75 of the
  maximum) at 45 deg, then C sin 2a; the drag rises smoothly to `post_stall_drag_ratio` (6) x
  the drag at the stall at 45 deg. Gradual lift loss, no asymmetric stall or spin.
  `GameConfig.post_stall` (on) switches the regime off for comparison.
- **Snap never gives up the 1 g:** toward stall speed its deep angle closes to the stall angle.
  The thrust's normal part at the angle flown counts in the 1 g hold, against the g limit. No
  cap at the envelope: near stall speed Snap's thrust turns it a little more than Max.
- **Nose spring per level:** going out the stiffness is 1/3, 2/3, 1 of the full spring for
  Sustain, Max, Snap, the friction fixed (nose_damping 1/sqrt(3)), so the damping ratios are 1,
  0.707, 0.577; going back toward the flight path always the full spring.
- **Firing never takes a ship below stall speed:** at and below it the withheld thrust is paid
  from the store, from 1.1 x stall speed up by the speed, a smootherstep between
  (`GameConfig.fire_store_band` 0.1); an empty store blocks a shot below 1.1 x stall speed.
- **Only straight keeps the shield quiet** (`card.QUIET_STICK_MAX` 0): every turn level
  restarts the recharge wait.
- **The scripted opponent is out of date** with these levels (its level choices, `fire_worth`
  and personas were tuned with the old ones) and is redesigned next; until then its competence
  test is an expected failure (fired share of long-lived ships about 0.957 against the 0.97 bar).
- Open (progress.md step 8): Snap's 2.37 deg/s rise toward the stall near 96-99 m/s, the
  far-side overshoot of a reversal into Sustain or Max, the Snap hold's small store drain.
- Checks (2026-11-26): quick tier on lean/game, lean/tool, lean/scripted tests (one xfail, the
  scripted competence test); ruff clean. Compile check (laptop 4070, performance profile, cold
  cache): `advance` 2,856 traced operations (core 2,300, +24%; old baseline 2,220), total 6,503
  (core 5,947), cold 233.8 s (core 226.9 s), no recompiles or graph breaks. The baseline was
  reset on this branch (author: each change judged on its own).

Hit damage halved (author 2026-11-26)
- `hit_damage` 21 -> 10.5 on the design row (lean/game/tuning.py). The fire rate stays at 30
  rounds/s (it looks and feels right). A typical full hit at 300 m now does about 10.5, so a kill
  (empty the 100-point shield, then one more hit) takes about 11 hits, about 0.4 s of steady
  fire, instead of 6 hits in 0.2 s: long enough to notice being hit before dying. Measured
  before the change (look-ahead agent, 1v1, fog off, no fields, nobody able to die, 32 games x
  60 s): passes every 6.4 s (median); 48% of passes land no hit; one ship landed 11 or more
  hits in 22% of passes; 79% of hits came while one ship sat behind the other within gun range
  with its nose within 30 degrees (5.8% of ship-time, median streak 0.47 s, longest 17.5 s).

## Scripted look-ahead (author 2026-11-26, branch scripted-lookahead)

Every step and measurement: docs/rewrite/sections/scripted-lookahead/progress.md; the agent:
lean/scripted/NOTES.md.

- **The scripted opponent is a look-ahead scoring agent** (replaces the rule-based BFM agent in
  every caller; the old code stays in `lean/scripted/rule_based/` only while the two are
  compared, then goes). Per ship: the 21 flight actions' paths at 0.25 / 0.5 / 1 / 2 s from the
  game's steady flight (`core.steady_flight`, new: `steady_rates` plus the angle) and the nose
  spring per level in closed form; the 4 nearest known enemies' 21 paths each; 21 x 21 pair
  tables (gun, threat, angles, energy, collisions) in one unit, the score second; his reply
  collapsed by w = softmax(S_him / tau_him) and (1 / beta) log sum w e^(beta S_me); own-only
  terms; flight = softmax(score / tau_me) plus stickiness for the pending action; fire by the
  old worth rule. 16 sampled parameters and 12 fixed shapes, each with a meaning
  (lean/scripted/parameters.py).
- **Author additions:** no goal point with no enemy known (an energy preference and a seek toward
  the freshest dropped last-seen position, both personality values); hidden enemies
  dead-reckoned from the last sighting (last speed and nose rate up to 2 s, then straight).
- **Decided by me:** no memory (the stickiness reads the pending action, which Obs carries, so
  the teacher's label is a function of the observation); `fresh_memory` and the callers' memory
  plumbing removed. The BC/RL teacher records the joint 42-way distribution (`TEACHER_WIDTH` =
  42, float16 in the store: 84 bytes per ship-tick instead of 22). The pair stage is a Triton
  kernel (`lean/scripted/kernels.py`) with the PyTorch path as its twin. `energy_weight` 0.25
  from the card's `store_turn`. Personas stay per ship, drawn at each deal.
- **Open, for the author (measured):** (1) cost: 16.6 ms per call at 2,048 envs of 4v4 against
  the policy's 23.0 ms forward (72%, the estimate was about 13%); cold compile of the call
  90.5 s. (2) 1v1 stalemate: self-play draws 74% without fields (rule-based 5.7%); no term
  rewards closing in neutral geometry; parameters alone reach 40% at best. (3) Snap's predicted
  path at low speed and its nose at 2 s (errors up to 19 m and 24 deg). (4) The power choice is
  near-indifferent and flickers. 4v4 is strong: 1,404 wins, 9 draws, 68 losses against the
  rule-based agent.

### Round 3 (author's go 2026-11-27, overnight; unmerged, waits for the author's review)

- **Steady-flight table (decided by me, as planned):** `core.steady_flight` tabulated once per
  ship type with the card (`core.steady_characteristics`; level x power x 17 store nodes x 400
  speeds x 4 outputs), read by `core.steady_lookup`; the store axis is its edge rooms
  (`core.store_room`, factored out of `_setting`). The table rides on a new
  `Constants.per_type` (by type, read with `State.type`; not dealt to slots). Path prediction
  unchanged within 0.3 deg; `act` 7,864 -> 4,451 traced operations; cold compile 333 -> 280.2 s,
  still over the 260 s budget (open: a paths kernel or fewer segments).
- **Hit model as one game function (coupling with the deflector stream):** `core.hit_damage`
  (kinetic x graze x aspect), read by the tick's `_damage` and the agent's trigger and gun
  (`opponent.hit_worth`), so a hit-model change in core reaches the agent without edits.
- **`store_weight` 0.25 (decided by me, decision-safe; question 2):** the store itself in the
  power choice. Store mean 0.07-0.12 -> 0.45-0.51 of full, dive 57-60% -> 23-25%, climb 6-8%
  -> 21-28%; beats the round-2 power choice 68% (1v1) and 57% (4v4). More (0.4, 0.6) wins 1v1
  duels but ties 4v4 and drops the dive out of use. Weakening the speed term (the author's
  suggestion) loses (44% / 47%): not taken.
- **Rule-based agent removed** (code and the tools' comparison modes).
- Measured on the final tree: 5.35 ms per call (25% of the policy forward); draws 2 / 2 / 3 of
  1,002 / 914 / 745 (1v1 fog off, 4v4, 4v4 fields); kills per env-minute 1.86 / 10.12 / 8.74;
  stick TV 0.13-0.14 per tick, by side 0.08-0.09, by steady rate 4-5% of the span.

Bullet deflector and the runtime hit model (PROPOSAL, branch `bullet-deflector`, unmerged; overnight 2026-11-27)
- **Superseded** by the egg (author decision 2026-11-27, entry "The egg deflector" below): the
  closest-approach push, its presets and `hit_fraction` are removed; the runtime hit model and
  the matrix exponential stay.
- **Mechanism** (`core.deflection`, `core.hit_fraction`, `core._deflect`; lean/game/NOTES.md
  "Deflector"): a field around every ship, the same at any shield. A bullet passing a ship at b
  is treated as passing at b + D: contact when b + D < hit radius, damage graze(b + D; core, hit
  radius) x the nose cone weighted by `nose_cone`. A bullet that touches nothing is pushed and
  bent once per pass, on the tick holding its closest approach (0 <= tau < 1), when inside hit
  radius + D: pushed D_eff = min(D, R + D - b) to the side it was passing (smooth sign, eps
  0.1 m), turned away by atan(D_eff / 48 m), speed kept. Ghosts run the same code. A second ship
  later in the same tick sees the unbent path.
- **Presets** (`card.DEFLECTOR_PRESETS`, `GameConfig.deflector`, default "frontal"): "off" (D 0,
  nose cone; skips the code), "symmetric" (D 3 m everywhere, nose cone kept), "frontal" (D 12 m =
  the hit radius in the front 30 deg, fading to 3 m by 45 deg, nose cone dropped: frontal
  bullets glance off). Default frontal: equal to symmetric on every measured play number, one
  mechanism instead of the nose cone's special damage rule, and the glance is visible.
- **Runtime hit model** (`core.HIT_MODEL`, `core.RUNTIME_CONSTANTS`, `core.with_hit_model`):
  hit and core radius, hit damage, spread, the deflector's numbers and the nose cone's are rows
  of the per-slot constants table, never floats baked into the compiled graph; changed between
  calls with no recompile (test under `torch.compile(fullgraph=True)`, error_on_recompile).
  Workbench `python -m lean.tool.hitbench` (about 2 s for three presets); sandbox sliders.
- **The scripted agent's fire-worth** reads `core.hit_fraction` (one call: hit model changes
  reach it without edits).
- **Measured** (rule-based scripted agent both sides, CPU, 4 seeds x 128 games x 60 s; god mode
  1v1 fog off no fields; kills with GameConfig()): damaging hits per minute per game off 50.9 /
  symmetric 38.4 / frontal 37.4; hits per pass 12.2 / 9.2 / 9.0; passes with no hit 0.30 / 0.30
  / 0.31; share of hits in control 0.32 for all; frontal (within 30 deg) contacts 205 / 158 / 0,
  damaging frontal hits 0 for all; kills per minute 1v1 0.80 / 0.61 / 0.62, 4v4 4.68 / 4.01 /
  3.94. Workbench at 300 m: effective hit radius off the nose 8.5 m (off) and 5.5 m (both
  deflector presets). The side D (3 m) is what costs hits; open question 4.
- **Cost** (4070 laptop, E 4096, 4v4, compiled step): off 0.97 ms, frontal 1.21 ms (+0.24 ms,
  +24%) after two cuts (from +66%). Compile check (cold, laptop 4070, performance):
  `advance` traced operations core 2,856 (baseline), branch off 2,998 (+22 the runtime rows, +120
  the elementwise matrix exponential), frontal 3,219 (+221 the deflector): the tripwire FAILS for
  frontal (+13% over the baseline; off is within it); 4 graphs, no recompiles, no graph breaks;
  cold 237.0 s off, 237.4 s frontal (baseline 231.7 s). Baseline not updated.
- **Found on the way** (own commit f583eb0c): the flight's 2x2 matrix exponential as batched
  matrix products was four cuBLAS calls, 43% of the step's CUDA time; written out elementwise
  the step is 1.76 -> 1.07 ms (deflector off).
- Questions for the author: docs/rewrite/questions.md on the branch. Notes and numbers:
  docs/rewrite/sections/bullet-deflector/progress.md.

## Author decisions 2026-11-27 (~05:40, relayed on branch deflector-lookahead)

1. The flight's 2x2 matrix exponential written out entry by entry (f583eb0c) is kept. Decided.
2. Compile budget: provisionally raised, to be revisited. Compile checks over the old limit are
   reported (scripted act; the egg's advance operations and cold compile), not a blocker for
   this branch; the baseline file is not edited.
3. Collision blame stays exactly as it is (the nose cone's angles), also with the nose cone's
   damage rule off: the aggressor stays punished more than the defender.
4. Time-to-kill target: steady fire from behind with perfect range, lead and aim (every shot
   hits, close) about 0.4 s is the floor; typical good tracking 0.6-0.8 s; about 1.0 s the
   upper limit for a well-aimed burst where only dispersion and distance cost hits. The egg
   experiments report which hit damage x spread combinations land in that band (core
   defaults unchanged).
5. Bullet spread: no target yet; decided from the egg experiments' spread grid.
6. Scripted store_weight: undecided (1v1 prefers 0.4-0.6, 4v4 even with fewer kills);
   situation-dependent weights may be tried, lower priority than the egg experiments.

## The egg deflector (PROPOSAL, branch `deflector-lookahead`, unmerged; author decision 2026-11-27 for the mechanism)

- **Mechanism** (`core.egg_entry`, `core.egg_turn`, `core.hump`, `core._deflect`; lean/game/NOTES.md
  "Deflector"): an invisible egg around every ship in its frame (front semi-axis 53.3 m along
  the nose, rear 24 m, half-width 24 m). In the relative frame, a bullet whose straight in-tick
  path first crosses the egg inward is turned away from the centre by 45 deg x hump(b / 24 m),
  hump(x) = 27/4 x (1 - x)^2 (0 at both ends, peak at a third), b the line's distance from the
  centre; the relative speed kept; the swept hit test on the turned remainder. Once per pass
  (only inward crossings; a bullet from inside never turns). Ghosts the same. Same at any
  shield. The old closest-approach push, its presets and `hit_fraction` are removed.
- **Hit radius 12 m, core 10 m** (author). The nose cone's damage rule kept as the hit-model row
  `nose_cone`, 0 by default; collision blame unchanged (author).
- **Presets** "egg" (default) and "off" (`GameConfig.deflector`); every number a runtime row
  (`core.HIT_MODEL`; no recompile, tested), with the derived `hit_widths` column.
- **Hit widths** (the game's function, the sketch's numbers): 24 m with no turn; egg 4.75 m side
  and astern, 2.05 m head-on (30 deg turn: 7.0 / 3.0; 60 deg: 3.5 / 1.6).
- **Scripted agent**: target size = half the hit width at the aspect; the trigger's worth uses
  the hit chance (hit width, spread, `lead_error` 0.25, a reasoned choice). One source of truth
  (the agent reads the column).
- **Cost**: GPU tick (E 4096, 4v4) 0.98 ms off, 1.37 ms egg (+40%, after one cut); compile
  `advance` 3,296 traced operations (+15% over the core baseline), cold 293.4 s (budget 260 s,
  provisionally raised). Baseline not updated.
- **Measured** (look-ahead agent both sides): head-on kills 0.45 -> 0.00 (1v1), hits in
  control 0.05 -> 0.70, kills per minute 1v1 3.25 -> 1.45 and 4v4 11.1 -> 6.6, TTK median
  0.8 -> 7.8 s; spread, hit damage and egg front barely move play; workbench TTK band
  candidates: hit damage about 8.5 with spread up to 5 mrad, or 7 with today's.
  docs/rewrite/sections/bullet-deflector/egg-experiments.md and progress.md.


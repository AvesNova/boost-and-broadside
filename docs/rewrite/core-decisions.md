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

## Proposals, not decided


- Remove v2's flight and damage models from lean/game once the core model is signed off
  (author 2026-11-23: two flight models in one file go when relevant).

- Grid cells in RL: look for papers that used them (docs/rewrite/provenance-check.md).
- A keep, drop or simplify table of Imugi's behaviours with rough costs (operations, lines,
  time per call, parameters): docs/rewrite/imugi-behaviours.md, ready for the author.

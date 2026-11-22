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
  speed. Climbing to a full store before a fight is allowed (it costs time).
- No thinner air with altitude; no energy-for-damage or energy-for-shield.
- Guns (2026-11-23): firing cuts the engine (to zero or a lower fraction) on the ticks it
  fires; bullets no longer cost store energy. To measure: speed lost per second of firing
  at cruise and corner speed, to set the fraction. Known quirk: firing doubles as an air
  brake (the only way to cut thrust); recoil drag is the fix if it proves exploitable.
- Checked with the turn-performance tool before it becomes the default; the author flies it.

Sizes
- Sizes batch- and GPU-friendly by default (powers of two, fractions such as one eighth)
  unless there is a reason not to (2026-11-22).

Provenance
- A provenance table (docs/provenance.md) for every overarching decision (why Griffin, why
  grid cells, why HL-Gauss, and so on): the paper or papers it comes from, or else the
  reasoning or test behind it. Not per line of code.

## Proposals, not decided

- Grid cells in RL: look for papers that used them (docs/rewrite/provenance-check.md).
- A keep, drop or simplify table of Imugi's behaviours with rough costs (operations, lines,
  time per call, parameters): docs/rewrite/imugi-behaviours.md, ready for the author.

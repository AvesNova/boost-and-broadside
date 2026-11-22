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
  on a hexagonal layout, and (later) a grid-cell module.
- Fewer rotary dimensions (medium or smaller maps for now; larger maps later, with the
  rotary revisited then). Heading stays in the rotary, with few harmonics.
- Griffin's initialisation as the paper specifies; v2's is wrong (to be checked against the
  paper, docs/rewrite/provenance-check.md).
- No belief and nothing decoded and fed back. A hidden enemy is a last-seen token plus the
  time since it was seen. Memory is the Griffin temporal block (v2's block, not a bare
  recurrence).
- No shot-trajectory head. Instead a hit head: would this tick's bullet hit an enemy, an
  ally, or nothing. The last ticks of a segment go unlabelled.
- Encoder and attention as v2: one first layer per token kind, a shared second; ships and
  the game token are queries; fields and zones are key/value memory in the same
  attention operation. Bullets are not tokens.

Learner and league
- Every reward level the same: HL-Gauss targets, the same number of bins, a scale per
  level; the outcome is one of these levels. Not too hot (the HL-Gauss width checked).
  Linear against symlog bins to be revisited (2026-11-22): HL-Gauss's paper may use linear
  bins, and DreamerV3's reason for symlog (one configuration across many games) may not
  apply to one game. Check both papers (docs/rewrite/provenance-check.md).
- No rating lanes and no lookahead. League weighting (1 - p)^2 with a small uniform floor
  (2026-11-22); the floor's size open (proposal: 10% uniform, revisit).
- Half self-play (both sides train), half league games (the live side trains); the scripted
  agent is one league member.
- Reward scaling with team size: as now (additive per ship).

Tests
- A test names the assumption or bug it guards (example: holding reverse thrust never takes
  a ship well below its stall speed). No goldens or recorded numbers without a reason.
  Comparisons against v2 are transition tests, deleted once a component is ported. For
  every change during the port, note in advance whether the result should be the same,
  similar (within a stated tolerance) or different, and test that.

Provenance
- A provenance table (docs/provenance.md) for every overarching decision (why Griffin, why
  grid cells, why HL-Gauss, and so on): the paper or papers it comes from, or else the
  reasoning or test behind it. Not per line of code.

## Proposals, not decided

- Shaping: drop it, as v2 did (lean's sixth level: facing the nearest enemy and closing
  speed toward it, v1's terms; not zero-sum, hard to tune, invites reward hacking).
- Hit-head labels from ghost bullets: every ship puts a bullet in its ring every tick; a
  flag marks it real or ghost; only real bullets cost energy, damage and stop; a ghost
  records its first contact (enemy, ally, none) and is removed. The ring (11 slots, one per
  tick) is already sized for this.
- Replace the null key with a published mechanism that removes the need for it and
  stabilises training (the author recalls Kimi; candidates: gated attention, Qiu et al.
  2025, which v2 has as an option; learned per-head sinks as in gpt-oss). Pending the
  source check.
- Grid-cell position code as a third option; look for papers that used grid cells in RL.
- A keep, drop or simplify table of Imugi's behaviours with rough costs (operations, lines,
  time per call, parameters): docs/rewrite/imugi-behaviours.md, in progress.

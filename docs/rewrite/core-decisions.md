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
- Swappable modules: each part of the model (position code, temporal block, attention,
  heads) is a slot with one interface, chosen at construction, so the compiled graph holds
  only the chosen module. Position code options: v2's categorical codes, Fourier features
  on a hexagonal layout, and (later) a grid-cell module.
- Fewer rotary dimensions (medium or smaller maps for now; larger maps later, with the
  rotary revisited then). Heading stays in the rotary, with few harmonics.
- No belief and nothing decoded and fed back. A hidden enemy is a last-seen token plus the
  time since it was seen. Memory is the Griffin temporal block (v2's block, not a bare
  recurrence).
- No shot-trajectory head. Instead a hit head: would this tick's bullet hit an enemy, an
  ally, or nothing. The last ticks of a segment go unlabelled.
- Encoder and attention as v2: one first layer per token kind, a shared second; ships and
  the game token are queries; fields and zones are key/value memory in the same
  attention operation. Bullets are not tokens.

Learner and league
- Every reward level the same: HL-Gauss targets, the same number of bins, symlog, a scale
  per level; the outcome is one of these levels. Not too hot (the HL-Gauss width checked).
- No rating lanes and no lookahead.
- Half self-play (both sides train), half league games (the live side trains); the scripted
  agent is one league member.
- Reward scaling with team size: as now (additive per ship).

Tests
- A test names the assumption or bug it guards (example: holding reverse thrust never takes
  a ship well below its stall speed). No goldens or recorded numbers without a reason.
  Comparisons against v2 are porting scaffolding, removed once a component is ported.

## Proposals, not decided

- Shaping: inside one of v2's five levels or a sixth level.
- Hit-head labels from ghost bullets: every ship puts a bullet in its ring every tick; a
  flag marks it real or ghost; only real bullets cost energy, damage and stop; a ghost
  records its first contact (enemy, ally, none) and is removed. The ring (11 slots, one per
  tick) is already sized for this.
- League weighting (1 - p)^2 with a small uniform floor and no rating lanes; the floor's
  games keep every member's win rate current. The floor's size is open.
- Drop the null key (the game token and fields are always present as keys).
- A keep-or-drop table of Imugi's behaviours for the scripted opponent.

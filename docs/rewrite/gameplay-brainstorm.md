# Gameplay brainstorm: rules, modes and mechanics

Status: **brainstorm, nothing decided.** Holistic design brainstorm with the author,
2026-10-25 to 2026-10-28. "Converging" means both sides liked a direction; it is not a decision.
Items that get agreed move to `decisions.md` (physics items also to `flight-model.md`). Visual
mock-ups are in `visual-mockups/`.

## Goals (from the author)

1. **Fun** for one player and for a team: team play wins matches, but a skilled individual can win
   fights while outnumbered (a high dogfighting skill ceiling, like a Counter-Strike clutch), and a
   new player can do something useful quickly (a low floor, unlike DCS).
2. **A good RL platform**, including one zero-shot agent across team sizes from 1v1 to 50v50.
3. **WW2 dogfighting feel** (IL-2, DCS) in 2D, arcade: ships are always identified once seen,
   resources regenerate (Halo, Overwatch), and fights and matches run on a fast time scale.

## Converging

### Overall shape

- Objectives give the game focus; dogfighting gives the fun. Both must work.
- **Respawns on.** Big-picture strategy over memorisation (AoE2 random maps, not CS2 smoke spots).
- No dumb AI filler units: every ship is flown by a human or by the team's AI.
- **Win condition: destroy the enemy base.** Turrets defend; killing them only makes later attacks
  easier (inspired by LoL turrets and bases, Battlefront II starfighter assault).

### Roles (soft: anyone can do anything, just badly)

- **Fighter:** fast-firing gun with random spread (real-life style; take only favourable shots).
- **Bomber:** low-rate, high-damage, slow, long-range shells that lock a target when fired and home
  with a capped turn rate (cheap: O(bullets), one gather per shell).
- **Scout:** speed, sight and manoeuvrability; weak or difficult weapons; low health.
- Matchups come from physics, not type tables: range ordered fighter < turret < bomber, slow shells
  that hit static targets easily but need big leads on fighters, and some subtractive armour (high on
  turrets). Loose cycle: fighter > bomber > turret > fighter. A 7-type Paley tournament (every pair
  beaten by some third type) is a possible later extension.

### Flight and combat

- Full 2D rigid body; handling based on a P-51D at game scale, run slower than today's ~4x real time
  (2-3x discussed; exact scale left for playtests).
- Kill time: about 3-4 good snap shots or 1-2 good tracking bursts.
- Damage by aspect: a narrow zero-damage cone straight ahead; side and rear equal.
- Shields regenerate; a hit with no shield left kills.

### Map

- One procedural map to start; mild imbalance allowed, no mirroring.
- **Capsule fields** (segment plus radius) for all fields, using the existing union composition;
  the arena is a stadium-shaped capsule with bases at the ends of the long axis.
- **Arena edge:** a high-index field (ships and bullets bend back, can bounce off at shallow angles)
  plus damage that grows with depth.
- Current fog and field rules kept: `perception.py` (team-shared vision, opaque field cores, shot and
  spawn reveal).

### Visuals (see `visual-mockups/README.md`)

- Minimalist flat look. Ship silhouettes from the Komet / Cranked-arrow family.
- Fields drawn as a topographic map of the true refractive index, evenly spaced contours, innermost
  contour on the core edge.

## Open

- How the base and turrets pace a 50v50 match (base health and armour, spawn placement).
- The scout's concrete job beyond sight.
- Exact time scale, kill time, ranges and armour values (needs playtests).
- Field generation rules: core overlaps allowed or not; density as the map grows.
- Linear index ramp (even contours) vs today's smootherstep band in the physics.
- Earlier questions from `flight-model.md` (Open 7-13) still apply: battery story, full-battery
  behaviour, gravity option, scope pass.

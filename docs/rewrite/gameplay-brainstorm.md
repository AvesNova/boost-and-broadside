# Gameplay and visual design brainstorm

Status: **working record of a holistic brainstorm with the author, 2026-10-25 to 2026-10-28.**
Two levels:

- **Chosen (for now)**: the author picked it during the brainstorm. It is the current direction, still
  open to change, and not yet moved to `decisions.md`.
- **Still deliberating**: options on the table, with the current leaning where there is one.

Items that become firm move to `decisions.md` (physics items also to `flight-model.md`). Every visual
mock-up, the interactive colour playground and the scripts that make them are in `visual-mockups/`
(see its README).

## Goals (from the author)

1. **Fun** for one player and for a team: team play wins matches, but a skilled individual can win
   fights while outnumbered (a high dogfighting skill ceiling, like a Counter-Strike clutch), and a
   new player can do something useful quickly (a low floor, unlike DCS).
2. **A good RL platform**, including one zero-shot agent across team sizes from 1v1 to 50v50.
3. **WW2 dogfighting feel** (IL-2, DCS) in 2D, arcade: ships are always identified once seen,
   resources regenerate (Halo, Overwatch), and fights and matches run on a fast time scale.

## Chosen (for now)

### Match and roles

- Objectives give the game focus; dogfighting gives the fun. Both must work.
- **Respawns on.** Scale from 1v1 to 50v50.
- **Win condition: destroy the enemy base.** Turrets defend; destroying them only makes later
  attacks easier (inspired by LoL turrets and bases, Battlefront II starfighter assault).
- **No AI filler units:** every ship is flown by a human or by the team's AI.
- **Three soft roles** (anyone can do anything, just badly): fighter, bomber, scout.
  - Fighter: fast-firing gun with **random spread** (like real life; take only favourable shots).
  - Bomber: low-rate, high-damage, slow, long-range shells that **lock a target when fired and home**
    with a capped turn rate (O(bullets), one gather per shell).
  - Scout: speed, sight and manoeuvrability; weak or difficult weapons; low health.
- **Matchups from physics, not type tables:** range fighter < turret < bomber, plus some subtractive
  armour (high on turrets). Loose cycle: fighter > bomber > turret > fighter.

### Flight and combat

- **Full 2D rigid body**, handling based on a **P-51D** at game scale, run **slower than today's
  ~4x real time** (2-3x discussed; exact scale left for playtests).
- **Kill time:** about 3-4 good snap shots or 1-2 good tracking bursts.
- **Damage by aspect:** a narrow zero-damage cone straight ahead; side and rear equal.
- Shields regenerate; a hit with no shield left kills.
- **No lead indicator.**

### Map, fields and vision

- **One procedural map** to start; mild imbalance allowed, **no mirroring**.
- **Capsule fields** (segment plus radius) using the existing union composition; **overlapping
  fields are fine**. The arena is a stadium-shaped capsule with bases at the ends of the long axis.
- **Arena edge:** a high-index field (ships and bullets bend back and can bounce at shallow angles)
  plus damage that grows with depth.
- **Fog and field rules from `perception.py`:** team-shared vision, opaque field cores, shot and
  spawn reveal; terrain (fields, bases, turrets, edge) always drawn.

### Visuals

- **Flat, minimalist look; near-white sky.**
- **Fields as a topographic map of the true refractive index:** evenly spaced contours (band width
  proportional to level, one line per third of a level), innermost contour exactly on the core edge,
  drawn as the one bold line (it is where sight is blocked). Both field types in a real colour (not
  grey).
- **Fog: darken to 60% brightness, saturation kept** (`fogdark.png`).
- **Every ship outlined** (allies and enemies).
- **Shield and energy:** one ring at the hitbox; shield 240° centred on the nose, energy 120°
  centred on the tail, each shrinking toward its centre (`shields3.png`, row 3).
- **Trail:** fading line plus a faint vortex on the inside wingtip during hard turns.
- **Normal colour vision first;** a separate colour-blind option comes later (a simulator,
  `visual-mockups/colourblind.py`, is ready for it).

## Still deliberating

### Gameplay

- How the base and turrets pace a 50v50 match (base health and armour, spawn placement), turret
  targeting rule (nearest, last attacker, ...), class switching at respawn.
- The scout's concrete job beyond sight (marking, spotting, harassing bombers).
- A 7-type Paley tournament (every pair beaten by some third type) as a later extension of roles.
- Exact time scale, kill time, ranges and armour values (playtests).
- Earlier questions from `flight-model.md` (Open 7-13): battery story, full-battery behaviour,
  gravity option, scope pass.

### Map and fields

- **Band width in the physics:** the code uses a flat 40 m transition for every field; even contours
  need band width proportional to level (20 m per level matches 40 m at the extremes), and a linear
  ramp has kinks where today's smootherstep does not (round only the ends?).
- A faint "zero" contour at the outer edge of each band.
- Field density as the map grows; the arena edge colour (it takes the thick-air colour).

### Visuals

- **Field colours.** Current favourite: **mint (thin) and pink (thick)**. Also liked: lemon / ice,
  purple / ice, ice / peach, and the ice, lemon, mint and peach family generally.
- **Ship colours.** Liked: green / magenta (and green / pink variants), blue / vermilion. With mint
  and pink air, navy / amber and cobalt / gold read best. Ship colours should also work as UI colours.
- **Ship silhouette:** Komet family (Komet, Cranked arrow and their blends in `ships3.png`).
- **"Shields down" indicator:** none of the first set worked; next idea is absence (no shield arc)
  plus shards when the shield breaks.

## Where to look

- `visual-mockups/map-vis.png`: the larger example map with every visibility case.
- `visual-mockups/colour-tool.html`: interactive colour playground on that map (self-contained).
- `visual-mockups/README.md`: every mock-up and the rules the scenes follow.

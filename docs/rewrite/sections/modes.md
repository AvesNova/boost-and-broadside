# Section 3: modes (game rules in `bnb.game`)

Status: decisions settled, 2026-11-01. Spec: `modes-spec.md`.

## Decided (author, 2026-11-01)

### Which modes

- **Now:** team deathmatch (with its arena and win rule) and a port of v1's Frontline.
- **Later:** modes that need capsule fields and a playable area (fortresses, capital
  ships, defense points, ...).
- **Structure:** a mode is a composition of rule components (win rule, zones and capture,
  spawn placement, spawn protection, clock), and each pool plays one mode.
- **Training is always 5v5,** in every mode. Other sizes are for evaluation.

### Team deathmatch

- **Slayer-style win rule:**
  - first to 25 kills (5 per ship) wins;
  - otherwise, at the 5-minute clock, the most kills wins;
  - equal kills is a draw.
- **Spawns, Halo or Call of Duty style:**
  - **The rule:** a random point in the arena with a random heading. Sometimes lucky,
    sometimes not; that is part of the game.
  - **Best-of-K:** each spawning ship draws K candidate points (K = 8) and takes the one
    farthest from the nearest living enemy. This is a single batched, static-shape
    computation, so it fits the captured graph.
  - **At a reset,** every ship first draws one provisional point. It then picks among its
    candidates by distance to the enemies' provisional points. That is one symmetric pass,
    and nothing depends on the team index.
- **Map:** the fields stay point-symmetric.

### Frontline (v1's rules, scaled with the arena)

- **Zones:** five on a ring at 0.615 `R_a`, with zone radius `330 s_a`.
- **Roles,** in order around the ring: team 0's spawn, team 0's defense, neutral, team
  1's defense, team 1's spawn. They rotate one place per capture.
- **Capture:**
  - only the two defense zones can be captured;
  - the meter moves toward the attacker on a strict alive majority inside the zone;
  - the rate is `H(lead) / 10 s`, with `H` the harmonic number, so a lead of one captures
    in 10 s and a lead of two 1.5× faster.
- **The front** moves one step per capture. Three captures ahead wins. At the time limit
  the sign of the front decides, and level is a draw.
- **Respawn:** after 8 s, at the team's *current* spawn zone (read when the wait ends), at
  full shield. Ships cannot be damaged inside their own spawn zone.
- **Symmetry (v2):**
  - the map is a mirror image across the axis through the centre and the neutral zone,
    fields included;
  - which team runs clockwise is drawn at random;
  - the ring's orientation is random.

### Objective events (the interface to rewards)

- **A mode emits two per-ship signed amounts per tick, `objective_progress` and
  `objective`,** each summing to zero per environment, in the mode's own units:
  - Frontline: meter fraction and captures;
  - a later capital-ship mode: damage to it, and its destruction.
- **The mode owns the attribution.** In Frontline (v1's ledgers):
  - meter movement is paid to the attackers present and charged to the defenders absent;
  - a capture is shared half by presence on the completing tick and half by each ship's
    share of the attempt.
- **Rewards apply the weights and assert zero-sum.** There is no per-objective
  bookkeeping when a mode is added.
- **Each environment's result** (±1 or 0 for team 0) comes from the mode on the
  environment's final tick.

### Spawns are mode-owned

The rule, the delay and the placement belong to the mode. The game keeps the spawn
mechanics (spawn state, uniforms) and takes placements from the mode.

### Observation

- **Every piece of mode information** an agent may know goes into the game's observation,
  team-relative like the rest:
  - a `zones` part, empty in team deathmatch, with position, radius, role relative to the
    viewer, meter progress and direction, and how many captures each zone is from ending
    the match;
  - global columns for time remaining, the front relative to the viewer, the win
    threshold, and in team deathmatch both teams' scores relative to the target.
- **Yemong** reads the global columns on its global token and adds a zone token kind.

### Generalising from 5v5 to 50v50 (position features)

Two problems, and the fix for each.

- **Memorised absolute position.** The arena is always centred on the origin, so a model
  could learn "this x is near the boundary" instead of reasoning from the global token and
  the fields.
  - **Fix (Yemong):** each view draws a uniform random offset in `[0, P)` per axis at
    reset (`P`, the position code's period, is 26 km), keeps it for the whole match, and
    adds it to every position it reads. Each team draws its own.
  - Rotary attention depends only on differences, so the offset cancels there exactly. It
    only reaches the categorical position code, which is where memorisation would happen.
  - The global token gets its position code back: it now tells the model where the centre
    is in its own frame.
  - No random rotation: the spawn axis, the fields and Frontline's ring are already drawn
    at random orientations, so the data is isotropic.
- **Coarse levels never exercised.** On a 2600 m map, ships rarely differ at the 8.7 km
  and 26 km position levels or the 4–8 km rotary frequencies, so a model would learn to
  ignore them and then misjudge distance at 50v50.
  - **Fix:** a second team deathmatch pool on the `battle` arena (8200 m, 40 fields: the
    50v50 evaluation map) with 5v5 **clustered spawns**.
  - A few groups start far apart, each mixing both teams (for example a 2v2 and a 3v3).
  - Respawn is near a random living ship, using the same best-of-K rule against enemies.
  - Own ships are always visible, so coarse relative positions show on every tick even
    under fog. Groups shift and merge as the match goes on.
  - Two map sizes, as separate pools in a fixed ratio. This replaces v1's continuous zoom,
    and the arena radius stays per pool, with no game change for it.
  - Frontline stays at squad size.
- **Rejected:**
  - **Separate arenas with their own global tokens:** a dead token in normal training,
    and arenas that never interact teach nothing about distance.
  - **Continuous zoom:** sparse maps where 5 ships rarely meet.

### Later

- **Fast fields as a map-design experiment.** A wider index range would make low-index
  fields highways, since world speed is proper speed / n. But bullets move through fields
  too, so gun reach changes; it would widen the code ranges and shift the ship card's
  balance. It is not the fix for coarse-level training.
- **The match length** (a mode parameter) grows toward about 10 minutes later. The
  outcome discount is revisited when it does.

## Open for the spec

- **The cluster layout:** the number of groups, their spacing, their compositions.
- **The ratio of battle-map to squad pools** (v1 played half its episodes at the
  reference map).

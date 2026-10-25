# Gameplay brainstorm: rules, modes and mechanics

Status: **brainstorm, nothing decided.** Started 2026-10-25 from the author's brief and the
inputs below. Items that get agreed move to `decisions.md`; physics items also to
`flight-model.md`.

Inputs: Goals and Framing in `decisions.md`; `flight-model.md` (especially Open 7-13);
the old rules (`docs/environment.md` on `main`, `docs/frontline-gate1.md` here).

## The brief

Two goals, both required:

1. **Fun** for one player and for a team.
2. **A good RL platform.**

Target feel: WW2 dogfighting as in IL-2 and DCS, in 2D, with arcade choices:

- Ships are always identified once seen (no friend-or-foe guessing).
- Resources replenish (Overwatch / Halo): staying alive never slowly drains ammo or health.
- Fights, preparation and matches run on a Halo / Overwatch time scale, not a simulator's.

## Design pillars (proposed)

A rule earns its place if it serves one of these and passes the cost test (GPU, tuning,
learnability, readability).

1. **Energy is life.** The battery plays the role of altitude: the ship with more total
   energy (speed plus battery) picks the fight. This is the energy-vs-angles choice that
   makes WW2 combat what it is.
2. **Position beats aim.** Where you shoot from matters more than how long you hold the
   trigger: rear hits hurt, head-on hits barely do, and you cannot see behind you.
3. **Dying is cheap, fights are short, the way back is fast.** A few seconds to kill,
   a few seconds to respawn, about ten seconds back to the fight.
4. **Two beats one.** Mutual support (covering a six, crossfire, dragging a pursuer onto a
   wingman) should pay off through the core rules, not through bolt-on bonuses.
5. **Every rule is visible.** Anything that changes the outcome is shown to a human and is
   in the observation for an agent (timers, heat, regeneration delays, zone states).

## Time scale (proposal to tune against)

References: Halo slayer kills take ~1.5-2 s of tracking, respawn ~5 s, matches ~10 min;
Overwatch kills ~1 s, respawn ~10 s, rounds a few minutes.

| Quantity | Proposal | From the current model |
|---|---|---|
| Time to kill under good tracking, full shields | 0.8-1.5 s | 30 rounds/s; sets shield HP vs damage |
| Shield regeneration delay / full refill | 3-4 s / 2 s | old Frontline used 5 s delay |
| Respawn delay | 4-6 s (mode-dependent) | |
| Spawn to first contact | 5-10 s | 1 km at 104 m/s cruise |
| Typical engagement | 10-30 s | sustained 360 at the 1.0 level ~5 s |
| Battery: empty to full while coasting | 15-20 s | 31 s today; probably too slow for this pace |
| Match | 3-6 min, or rounds of 60-90 s | |

At 30 Hz, a 20 s fight is 600 decisions and a 5 min match 9,000: credit assignment within
a fight is easy, across a match it needs an objective signal (see modes).

## Mechanics

### M1. Energy: the battery (Open 7-10)

- **Battery story: yes.** Boost spends the battery for speed, brake (reverse) stores speed
  in the battery, the engine charges it. Show it as a vertical gauge so a player reads it
  as "height". It is the WW2 energy fight without a third dimension.
- **Full battery (Open 8): lean keep the taper** ("engine weakens as the store fills").
  It caps total energy, so a full ship is "high" but not also faster, which prevents a
  snowball where whoever got full first wins every merge. It also gives the store a soft
  ceiling, like a critical altitude. The constant-engine version is simpler to explain;
  if playtests show nobody notices the taper, switch.
- **Gravity (Open 9): lean (B)**, a fixed stall margin. "Slow is deadly" is the WW2 rule
  that most rewards energy management, and (B) keeps it without draining the store.
- **Should bullets cost battery?** Two options:
  - *Battery-paid* (current): elegant, one resource, spraying makes you slow.
  - *Gun heat* (separate bar, Halo plasma / Battlefield style): fire heats, overheating
    locks the guns ~1.5 s, cools fast when idle.
  - **Lean: gun heat.** A player who fires and loses speed without knowing why has a
    readability problem; heat keeps energy decisions and gunnery decisions separate, which
    also helps credit assignment for agents; and WW2 ammunition and energy were separate.
    Spray is still limited, by heat instead of by speed.

### M2. Damage and health

- **Keep shields plus one exposed-core hit** (decided earlier). Halo-like: shields
  regenerate after a delay, the core does not exist as a separate pool.
- **Angle-dependent damage** (the old `head-on damage fraction`, kept and extended):
  head-on hits do a small fraction (0.1-0.3), side hits full, **rear-quarter hits a bonus
  (1.25-1.5x)**. This makes head-on jousting a poor trade (true in WW2 as well) and
  makes getting on a six the goal.
- **Later idea, not v1:** temporary damage states that heal (e.g. a heavy burst makes you
  "smoking": less engine power until the shields refill). Adds a "go home now" decision,
  but more state and tuning.
- **Mid-air collisions:** probably still none, or a soft push. Lethal collisions invite
  ramming strategies in self-play.

### M3. Seeing and being seen

Identification is free, but *spotting* is not. Spotting is where WW2 bounces come from.

- **Rear blind cone:** no vision in a cone behind the ship (e.g. 60-90 degrees) beyond a
  short "you can hear it" radius. "Check six" becomes a real action: you turn to look.
  Cheap (one dot product per pair) and a natural memory task for agents.
- **Spot range** finite (e.g. 1.5-2.5x gun range); **firing reveals you** to everything in
  a wider radius for ~1 s (muzzle flash and tracers). Choosing when to open fire becomes a
  decision.
- **Team-shared spotting:** anything a teammate sees, the whole team sees. Agents share
  observations anyway; giving humans the same picture removes the voice-chat gap between
  agent teams and human teams, and mixed human / agent teams work without communication.
- **Clouds** as pure vision occluders (see M5).
- **Gunsight for humans:** a lead-computing sight is both arcade and authentic (the K-14
  gyro sight on late-war P-51Ds). Agents compute lead themselves; this narrows the human
  vs agent aiming gap.

### M4. Disengaging and regeneration

Regeneration only matters if breaking off is possible, so escape routes are a mechanic:

- **Dive away:** dump the battery into speed; works against a ship with less energy.
- **Break line of sight:** clouds, blind cones.
- **Drag onto friends:** lead a pursuer to a wingman.
- **Home flak:** enemies inside a team's base zone take steady damage (deterministic, not
  random). Protects respawns, gives a retreat destination ("drag them over our flak", a
  real tactic) and ends spawn camping without invulnerability timers.
- **Base zone boosts regeneration** (an airfield): optional, creates a front and a rear.

### M5. Fields and the arena

- Fields cost a lot in the old env (one to four fields dropped env throughput to ~0.13x).
  Re-ask what each buys for fun:
  - **Cloud** (opaque, no physics): highest fun per cost; enables bounces and escapes.
  - **Fast / slow air** (current refractive index): interesting, but hard to read for a
    human. Lean: keep as an optional component, not in the default modes.
  - **Damaging interfaces:** lean drop; damage should come from enemies.
- **Boundary:** keep a soft hazard. Elimination overtime can shrink it to force a finish.
- **Map size follows ship count** (decided). Proposal for 4v4: ~1.5-2 km across (gun range
  ~240 m, min radius ~39 m).

### M6. Spawning

- **Spawn full** (full battery, cruise speed) in your base. A returning ship arrives with an
  energy advantage over campers who spent theirs; with home flak, spawn camping stops
  being worth it.
- **Spawn in pairs** when several ships are waiting (a short queue window), so wingmen
  form naturally.
- **Scramble variant:** spawn empty at low speed (for asymmetric modes, below).

### M7. Ship types (after launch with one)

Rock-paper-scissors in energy-vs-angles space, no hard counters:

- **Turn fighter** (Spitfire / Zero): best sustained turn, light guns, low shields.
- **Energy fighter** (Fw 190 / P-47): best dive and battery, heavy guns, poor turn.
- **Heavy fighter** (Bf 110 / P-38): slow, tough, with a rear turret that fires
  automatically in a rear arc (scripted aim, no extra action head). Later.

### M8. Team rewards inside the rules

- **Assists:** credit for damage on a kill within a window (e.g. 5 s). Shows up in events,
  scores and human feedback.
- Anything beyond that (formation bonuses and the like) is artificial; prefer that M3, M4
  and the damage angles make pairs strong on their own.

## Game modes

Modes are compositions of rule components (decided). Components that cover every mode
below:

| Component | Options |
|---|---|
| Arena | size, boundary rule, shrink in overtime |
| Zones | role (base, objective), static / moving / rotating, capture by presence or by damage, owner |
| Objective entities | HP, path, defensive guns (bombers, balloons, radar) |
| Scoring | kills, captures, time held, objective damage |
| Lives | unlimited with delay, N per round, none |
| Rounds | one match, or best of N with side swap |
| Spawn | full / empty energy, location rule, pairing |
| Vision | fog on / off, blind cone, team-shared, clouds |

### Candidates

1. **Duel (1v1).** Elimination rounds, best of 5, varied start geometries (head-on,
   offset, tail chase, energy advantage). The ranked skill mode for humans and the
   cleanest RL testbed (binary, short, no team noise).
2. **Elimination (team, no respawns).** Rounds of 60-90 s, best of 7; overtime shrinks
   the boundary. High tension (Counter-Strike style), clean outcome per round, strong
   team play since a death is costly.
3. **Furball (team slayer).** Respawn 5 s, first to K kills or 5 min. Casual and
   drop-in, the simplest team mode.
4. **Air superiority (king of the hill).** One zone that moves every 60 s (Halo's moving
   hill). Moving prevents camping, and each move is an energy decision: arrive fast and
   empty or slow and full.
5. **Frontline** (existing): ring of zones, tug of war. Keep; good for large teams.
6. **Escort (asymmetric, swap sides).** A formation of AI bombers crosses the map with
   scripted defensive guns. Escorts protect, interceptors score per bomber destroyed or
   by damage. The Overwatch payload analog, and very WW2 (bomber raids). Bombers reuse
   "zone as an entity with HP"; they are a moving objective.
7. **Scramble (asymmetric).** Attackers arrive full and fast to destroy static targets
   (balloons, radar); defenders spawn empty at their airfield and must charge while the
   raid comes in. Short rounds, swap sides. Could replace capital ship.
8. **Free for all.** N teams of one (N teams are supported from the start).
9. **Training drills** (not competitive): BFM set-ups (defend from a tail chase, convert
   an energy advantage, survive a two-on-one). A tutorial for humans and a fixed,
   interpretable evaluation suite and curriculum for agents.

### RL view of the candidates

| Mode | Outcome signal | Horizon | Team coordination | Extra cost |
|---|---|---|---|---|
| Duel | win / loss per round, short | ~1-2k steps | none | none |
| Elimination | per round | ~2-3k | high | none |
| Furball | kills, dense | ~9k | medium | none |
| Air superiority | hold time, dense | ~7k | high | one zone |
| Frontline | captures | ~9k | high | five zones |
| Escort | bomber damage, dense | ~3-5k | high, asymmetric | AI entities with guns |
| Scramble | target damage | ~3k | medium, asymmetric | static entities |
| Drills | scripted success | short | low | none |

**Suggested v1 set:** Duel, Elimination, Furball, Air superiority (all reuse ships and at
most one zone), plus drills for evaluation. Then Escort and Frontline. Asymmetric modes
need both sides in the observation (role on the global token) and doubled ratings.

## RL platform notes

- **Fairness between humans and agents** as mode options: an agent observation delay
  (e.g. 100-200 ms) and an optional cap on stick reversals per second. Needed for a fun
  human vs agent match and for honest leaderboards.
- **Anti-degenerate checks** to watch in self-play: endless kiting (same-type ships at
  equal energy cannot outrun each other, so objectives must force contact), turtling in
  base (home flak only hurts enemies, so a base-camper also scores nothing), head-on
  jousting (angle damage), spray (heat).
- **Observation additions** from these mechanics: gun heat, shield delay timer, visibility
  flags per pair (blind cone, cloud), revealed-by-firing flag, zone and objective state,
  round number and score, role in asymmetric modes.
- **Events, not rewards** (decided): hit (with angle class), damage, kill, assist, death,
  capture progress, objective damage, round end.

## Questions for the author

1. Ammunition: battery-paid shots, or a separate gun-heat bar?
2. Vision: rear blind cone, firing reveals, team-shared spotting: which of the three?
3. Fields: clouds as pure occluders in the default modes, refractive air optional?
4. Angle damage: add a rear-quarter bonus on top of the head-on reduction?
5. Which modes for v1? (Suggestion above.)
6. Time scale: accept the targets in the table as the tuning goal (notably a faster
   battery refill)?

# lean/scripted

The scripted opponent: the yardstick that the evaluation rates against (1000 Elo), the BC
teacher, and a population of personas around it. Plus a weaker deterministic chaser. A
look-ahead scoring agent (author 2026-11-26; design, steps and measurements:
`docs/rewrite/sections/scripted-lookahead/progress.md`; decisions:
`docs/rewrite/core-decisions.md` "Scripted look-ahead"). It replaced the rule-based BFM agent
(`docs/rewrite/scripted-bfm-design.md`; its code, kept for comparison through round 2, was
removed in round 3).

## Entry points

`from lean import scripted` (API in `lean/CONTRACT.md`): `ScriptedConfig()` (the yardstick;
`replace(name=value)`), `personas`, `draw_uniforms` ((E, N, 2): flight, fire), `explain` (the
`Choice`: the joint 42-way distribution, the flight distribution, the fire probability, the
scores, the scored enemies), `decide` (actions and the `Choice`), `act`, `sample`, `chaser_act`.
No memory: the stickiness reads the state's pending action, which the observation carries, so a
label is a function of the observation. The sandbox flies against it (`World(opponent="scripted"
| "chaser")`).

- `parameters.py`: every parameter is a named row of `PARAMETERS` (default, reference, sampling
  law, half-width at spread 1). The common unit is the **score second**: one second of a full
  gun solution (my nose on his lead point, in range, the hit at full worth). Every term is a rate
  in score seconds per second of the look-ahead; a weight says what one unit of its term is worth
  against my own gun (weight 1 by definition).
- `paths.py`: `predict`, the 7 stick candidates' paths of every ship at its table power at the
  look-ahead points (`LOOK_AHEAD`, a module setting read at call time); `power_outlook`, each
  ship's speed and store at the 3 powers over the power look (`POWER_LOOK`, 1 to 5 s). Both
  read the game's steady-flight table (`core.steady_lookup`, round 3), not the flight
  functions.
- `opponent.py`: `explain` and its stages; `pair_tables` + `collapse` are the plain twin of the
  kernel.
- `kernels.py`: `pair_scores`, the pair stage (and the cover) as one Triton kernel (CUDA;
  `USE_KERNEL`).
- `chaser.py`: the memoryless floor.

## One decision (`explain`, one pass over (E, N) tensors, both teams in one call)

1. Information: the team's view (`core.enemy_view`), reading only what `Obs` carries (positions,
   headings, proper speed, angle of attack, nose rate, store, shield, cooldown, the pending action
   of own ships, the time since seen; ln n and its gradient are functions of the position and the
   fields). A hidden enemy is dead-reckoned from his last sighting by the time since seen (his
   last speed and nose rate for up to 2 s, then straight; `dead_reckon`).
2. Power (round 2, author 2026-11-26): per ship from the own-ship terms only, over the power
   look (`paths.power_outlook`: speed and store at each power holding the pending stick, from
   the game's steady-flight table per segment): minus the speed term, plus energy (at
   `energy_weight` while engaged, `disengaged_energy_weight` while not), plus the store itself
   (`store_weight`, round 3: diving trades the store for speed at almost no change in total
   energy, so without it the store ran nearly empty), discounted by `power_horizon`; its own
   softmax (`power_temperature`) and stickiness (`power_stickiness`) on the pending power. The
   table power: the own ship's pending power; an enemy's (not observable) the power his outlook
   scores best from straight flight with the yardstick's rows.
3. Candidate paths (`paths.predict`, the 7 sticks at the table power): speed and store stepped
   per segment from the game's steady-flight rates at the segment's start (its table: below);
   the angle of attack
   follows the nose spring in closed form per segment (`nose_response`: e'' + 2 sigma e' + k
   omega^2 e = 0 with the tick's stiffness per level going out, the full spring going back, the
   fixed friction; the angle's own rate from `core.current_path_rate`) toward the level's steady
   angle at the segment's start speed (`RETARGET`: Snap's deep angle closes as the speed
   bleeds); the path rate linear in the angle between the segment's start and the steady state;
   chords per segment; the medium's 1 / n and its bend per segment start. Every ship's paths
   once, both views in one pass, materialised once ((P, E, 2 N, fields, C), the kernel's
   layout).
4. The `ENEMIES` (4) nearest known enemies: distance plus roam (seconds since seen x sustained
   speed), dropped past `roam_cutoff` arena radii of roam, weight 1 - roam / cutoff.
5. Pair tables per enemy, 7 x 7 at each point (`pair_tables` / the kernel): my gun on him
   (nose against the drag-law lead point in my medium, tolerance his angular size (half the
   game's hit width at the bullet's aspect, `core.hit_width` on the `hit_widths` column: the
   egg deflector, round 4) plus `aim_blur` x t, a smoothstep range band 0.8 to 1.2 gun
   ranges, the hit's kinetic worth, his nose cone by `core.aspect_factor` weighted by the hit
   model's `nose_cone`, 0 by default), his gun on me (mirrored), angles ((cos ATA_me - cos
   ATA_him) / 2) and energy (speed^2 / 2 + store, in my full stores) both times a closeness 1 /
   (1 + (d / (3 gun ranges))^2), predicted contacts (closest approach on each straight
   segment, exp(-(d / (2 summed hit radii))^2) x closing speed in collision-damage units,
   discounted), and the engagement range (McGrew's form: `aggression` x 1/2 (1 + angles) x
   exp(-|d - `desired_range`| / `range_width`); his mirror in S_him, so neutral geometry
   rewards closing for both). S_him is the mirror with my weights, plus his own-only terms.
6. His reply collapsed per row (`collapse`): w = softmax_b S_him / tau_him; score(a) = log
   sum_b w e^(beta S_me) / beta, the weighted mean at beta -> 0. Summed over the enemies by
   weight.
7. My own-only terms: speed (|ln(v / preferred)|), boundary (in an edge band of
   `boundary_margin` turn radii, more flying outward, growing past the edge), slow medium (ln n
   in half-octaves), cover (behind a field core from an enemy flying straight on, worth his
   angular advantage on me now), team-mates (contacts with the nearest 3 flying their pending
   action), and with no enemy known (author 2026-11-26: no goal point) an energy preference and
   a seek toward the freshest dropped last-seen position. The stickiness bonus for the pending
   stick.
8. Stick = softmax(scores / tau_me); flight = stick x power; fire = the trigger now (aim gate
   at the lead point in target sizes x `fire_tolerance`, the size from the game's hit width at
   the predicted aspect; range gate; the expected hit (damage x hit chance) worth `fire_worth`
   ticks of shield recharge or his whole shield; never through a team-mate; the gun ready).
   The hit chance (round 4, 2026-11-27: with the egg a centred shot always does full damage,
   so the worth must weigh the chance of landing within the narrow width) = erf(half width /
   (sqrt 2 sigma)), sigma = flight time x sqrt(spread^2 + (`lead_error` x 0.5 x his sustained
   turn acceleration x flight time)^2); `lead_error` 0.25 is a reasoned choice, not tuned.
   `store_crowd` (1, on; author 2026-11-27 with `store_weight` 0.5, from egg-experiments.md; 0 is off): scales the store weight by (1 + k) / (1 + k x the
   scored enemies within 2 gun ranges). The hit's worth,
   here and in the pair tables' gun, is the game's hit model through one call
   (`hit_worth` -> `core.hit_damage`: kinetic at the distance, the target's nose cone, dead
   centre), so a change of the hit model in the game reaches the agent without edits (the
   Triton kernel has its own copy of the gun's worth; the kernel-against-twin test flags a
   difference). Joint = stick (7) x power (3) x fire (2).

## Numbers

See `docs/rewrite/sections/scripted-lookahead/progress.md` (speed, usage, entropy, kills,
draws, duels, merge turn, the grouped stick change).

## The steady-flight table (round 3)

`core.steady_flight` tabulated once per ship type when the ship card is built
(`core.steady_characteristics`): turn rate, d speed / dt, d store / dt and the angle of attack
for each stick level, power, 17 store nodes (the store matters only in its edge bands: 9
sink-room nodes, 9 top-room nodes, `core.store_room`) and 400 speeds (half Sustain's step,
from a quarter of stall speed). `core.steady_lookup` reads it bilinearly by the ship's type
(`State.type`); the table rides on `Constants.per_type` (by type, not dealt to slots). Its
error against the direct functions and the path check: progress.md round 3 step 1.

## Tests

`tests/test_scripted.py`: well-formed actions and distributions, the joint in the contract's
order, the yardstick and full tensor rows alike, randomness only from sampling, one call for
both teams, the paths against the game's tick, the nose spring's closed form, the collapse's
limits, roam and cutoff, dead reckoning, no goal point, turning toward an enemy, the gun (tail
and nose cone, team-mate in the line), collision avoidance, the edge, stickiness, the kernel
against its twin (CUDA), the power from the own-ship terms, games against the chaser and the persona population (slow).

## Open

- Compile: the cold compile check still fails after round 3's steady-flight table (280.2 s
  against 260 s on a quiet laptop; `act` 4,451 traced operations, round 2 7,864). What is left
  is the unrolled path loop (medium, nose spring, lookup per segment); a paths kernel or fewer
  segments would cut it (docs/rewrite/questions.md on this branch).
- The stick distribution moves about 0.15 total variation per tick; grouped by side 0.07-0.09,
  weighted by the steady turn rate 4-5% of the rate span (round 3 step 2). Not tuned.
- `store_weight` 0.5 with `store_crowd` 1 (author 2026-11-27; was a fixed 0.25, round 3 step 3): the
  situational weight won 1v1 (0.60 against 0.25) and is even in 4v4.
- Defaults otherwise chosen by reasoning (the energy weight from the card's `store_turn`).

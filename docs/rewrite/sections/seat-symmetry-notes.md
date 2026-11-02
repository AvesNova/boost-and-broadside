# Seat symmetry: notes

A code review (2026-11-02) measured team 1 ahead in self-play with identical controllers:
the scripted agent against itself in `deathmatch` scored 0.536 and 0.540 for team 1 (seeds
5 and 6, 8,192 games each), about 5 standard errors; Frontline about 0.52; the chaser
0.514 and 0.50. This page finds the cause and records the fix.

## Progress

- [x] Mirror test written (scratch, then `tests/sections/test_seat_symmetry.py`).
- [x] Cause found: hit ties at the clamped `τ` went to the lowest sim index.
- [x] Fix: ties go to the smaller miss distance (`combat.contact_order`); kill credit too.
- [x] Mirror test passes after the fix, fails before it.
- [x] Goldens (`tests/reference/`) pass unchanged: no golden blocks the fix.
- [ ] Self-play per-seat scores before and after: queued on the shared GPU, not yet run
  (`/tmp/bnb-scratch-seats/batch.py grid1.txt`, results to `selfplay_results.txt`).
- [ ] Full suite (GPU). The CPU runs of the goldens, `test_game_section.py`,
  `test_learner_shots.py`, `test_ship_card_game.py` and `test_seat_symmetry.py` pass.

## The mirror test

A state's mirror swaps the teams' sim indices (ship `t·S + s` ↔ `(1−t)·S + s`, bullets with
their owners, `ticks_since_seen` by viewing team, the scores, Frontline's ledgers) and maps
every position and direction by the map's symmetry:

- **Team deathmatch** (and sandbox): the maps and spawns are point-symmetric, so the map is
  a rotation by π: positions, headings, velocities, medium gradients and bullet directions
  are negated. The spawn axis stays (team 0's centre `−0.8 R_a â` maps to team 1's).
- **Frontline**: the ring of five zones is not point-symmetric; it is mirror-symmetric
  about the axis through the neutral zone. The map is the reflection about that axis
  (each environment's `ring_rotation`): zone `k` ↔ zone `−k mod 5`, roles `r ↔ 4 − r`,
  `front ↔ −front`. A reflection reverses chirality, so the stick's left and right swap
  (1↔2, 3↔4, 5↔6) and the angle of attack, its sine and the nose rate change sign.

Both games step with mirrored inputs: the actions seat-swapped (and reflected), the spread
normals seat-swapped and mapped as vectors, the tick uniforms mapped so the mirror draws
the mirrored map and spawns (an angle uniform `u` becomes `u + ½` under the rotation and
`2ρ/2π − u` under the reflection; per-ship draws are seat-swapped; the two formation
rotations swap and gain ½). The result is compared with the mirror of the original's
result: every state leaf, every event (damage by shooter and target, deaths, kill credit,
spawns, the episode end and result, the objective events) and the observation (view 0 of
the original against view 1 of the mirror, zones and globals included).

**Exactness.** Under the rotation the comparison is bitwise except the shield: the damage a
ship takes is summed over shooters in sim order, which the seat swap reorders (1 ulp, 2 of
640 ships in one melee tick). Under the reflection every value rounds (`cos 2ρ`, `sin 2ρ`):
positions agree to 2.4e-4 m, directions to 3e-6. The test takes floats to 1e-4 relative and
1e-3 absolute, flags, counts and indices exactly.

**Randomness.** All of the game's randomness is mirror-mappable: the spread is two
standard normals per ship (isotropic, so the mapped draw has the same distribution), and
every tick uniform is either per ship, per team or per environment. One draw is by sim
order: the grouped deathmatch respawn anchor, `⌊u · count⌋`-th living ship in sim order.
It is not mappable draw for draw by a fixed map, but it picks a uniformly random living
ship whatever the order, so it is fair in distribution. The mirror test does not cover
grouped spawns for this reason.

## Coverage

`tests/sections/test_seat_symmetry.py`:

- `test_a_melee_tick_and_its_mirror_agree` (both modes, fast): ships packed into a 30 m disc
  with low shields and bullets among them, so one bullet often touches an ally and an
  enemy at the same `τ`; four batches of 64 environments, one random-action tick each.
  Before the fix: damage, deaths, kill credit, scores (deathmatch) and Frontline's ledgers
  and meters differ.
- `test_chaser_games_and_their_mirror_agree_every_tick` (both modes, slow, about 100 s
  each on CPU): the chaser against itself, fogged, 24 environments, 900 warm-up ticks, then
  300 compared ticks in which every environment's episode ends (new maps, new rings,
  reset spawns), with respawns, captures and deaths.
- `test_a_tau_tie_goes_to_the_closer_pass_whatever_the_index` (`test_game_section.py`):
  the rule itself, both slot orders.

In longer scratch runs (`/tmp/bnb-scratch-seats/mirror*.py`, 64 environments) before the
fix: chaser self-play in team deathmatch differed in 58 of 192,000 environment-ticks over
3,000 ticks (first at tick 153), 15 of them in who died and so in the score; in Frontline, 87 of
160,000 environment-ticks differed in damage over 2,500 ticks, 10 in who died, 14 in kill
credit. Random against random (spread out, rarely in contact) showed no difference. After
the fix the same Frontline run shows none.

## The cause

The swept hit test (game spec 7.1) takes `τ`, the fraction of the tick at the closest
approach, clamped to [0, 1]. A bullet that ends its move inside a ship's hit circle while
still closing on it has `τ = 1` exactly; one that starts inside a circle and moves away
has `τ = 0`. When a bullet ends (or starts) inside two circles at once, both contacts tie
exactly, and the tie went to the lowest ship index.

Ships are stored team-major (team 0 is indices 0…S−1) and friendly fire is on. So in a
tie between an ally and an enemy:

- a team-0 bullet took the lower index, its **ally**: friendly fire;
- a team-1 bullet took the lower index, the **enemy**.

Every death scores for the other team in team deathmatch, so team 0 lost ships to its own
fire that team 1 would have put into the enemy. Hit radii (8 and 11 m) are about one
bullet move (≈ 15 m per tick), so the clamp is common, and the chaser and the scripted
agent both pile onto the nearest enemy, where an ally and an enemy within a circle of the
same bullet are frequent. Kill credit (spec 7.2) had the same order by bullet index, which
credited team 0's shooters in ties; no mode or reward reads it yet, but it would have
biased any kill reward.

Nothing else in the tick broke the mirror: fire, flight, bullets, boundary damage,
shields, respawn and reset spawns, Frontline's captures and protection, visibility and the
observation all mirror exactly (or to the reflection's rounding).

## The fix

`combat.contact_order(tau, miss_squared)`: contacts order by `τ`, then by the miss distance
(the closer pass first), as one int64 key (both are +0.0 or more, so their float bits order
like the floats: `τ`'s 30 bits above the miss's 31). An exact tie in both, a measure-zero
event for distinct ships, still goes to the lower index. Kill credit uses the same order,
then the lower bullet index. The rule is geometric and team-blind, needs no extra
randomness, and leaves the uniform layout unchanged. The learner's shot re-simulation
(`learn/shots.py`) calls the same `first_contact`, so its labels follow.

Game spec 7.1 and 7.2 are amended (marked v2). The goldens in `tests/reference/` pass
unchanged: none of their cases has a tie.

## Before and after

Self-play per-seat score (win 1, draw ½) in each environment's first episode, 8,192
environments per row, both seats the same controller. Standard error ≈ 0.0055.

(pending: the GPU grid)

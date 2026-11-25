# Scripted ghosts, head-on fire, and the head-on stalemate (2026-11-25)

Branch `scripted-ghosts` (worktree `v2-ghosts`), from core with vision 1,500 m merged
(8f7d1ea2, 88541c3f). Laptop, power profile `performance`. Raw outputs:
`data/scripted-ghosts/` (skill stores and json, hits json, loop json, compile log). Scripts
beside this page: `skill_check.py`, `hits.py`, `loops.py`; `before/` is the agent of core
8f7d1ea2, frozen for the head to head.

## 1. Ghosts cost distance and are dropped

`lean/scripted/opponent.py` `engage`. Under fog, a remembered enemy's score uses an effective
distance: the distance to where it was last seen plus its roam, the seconds since seen
(`ticks_since_seen` from the full state x the game's `tick_length`) times its
**sustained speed** (`state.constants["sustained_speed"]`, 168 m/s for the P-51D). Why that
speed: it is the speed of the turning fight, where the yardstick spends its time (mean fight
speed 0.72 corner speeds = 184 m/s, close to it), and the roam should be how far an enemy
typically gets, not the most it could (cruise, 372 m/s, would make every half-second-old
sighting cost 190 m).

**Drop:** a remembered enemy is not a candidate once its roam exceeds `ghost_cutoff` x the
arena radius (new row, 1.0, fixed; author 2026-11-25: on the order of the arena radius, read
from `state.constants["arena_radius"]`). Then it could be anywhere in the arena and its last
spot says nothing. At 4,200 m that is **25.0 s** unseen (17.9 s at 3,000 m, 35.8 s at
6,000 m). The cost does most of the work before that: an enemy seen 3 s ago at 600 m counts
as 1,103 m, so a visible enemy at 900 m wins. A target hidden in a turning fight for the
median 2 s (vision.md) costs 335 m. With the 0.8 keep factor that is a switch only to an enemy
seen now within about 610 m of a 430 m target. There is no swept-spot rule.

The score is now linear in the effective distance (the squared form could not add the roam).
The target order is unchanged when there is no roam. The game now tells `tick_length` as a
constant (`lean/game/core.py` `_build_ship_card`).

**No candidate:** as before, the disengaged tactic flies to the arena centre at merge speed
and does not fire. That is sensible and was kept. The candidate set is the same for every
ship of a team: the same view, and a drop that depends only on time. So "fly to the nearest
ally that has a target" never applies: when one ship has no target, no team-mate has one
either. Searchers of both teams meet at the centre, so a lost fight forms again there.

## 2. No shot in a pure head-on

Gun condition (`explain`, the gun stage): fire only when the hit is worth it. The hit is
`hit_damage` x the game's aspect factor at the predicted aspect, using
`lean.game.core.aspect_factor`, which was pulled out of `_damage` unchanged so the game and
the agent share it. The prediction takes the bullet's launch direction (own velocity plus
muzzle speed along the nose) against the target's nose turned at its nose rate over the
flight time. It must be at least `fire_worth` ticks of the shooter's own shield recharge
(new row, 1.0: shield_rate x tick_length = 0.8 points, 3.8% of a 21-point hit, about 19
degrees off the nose). The principle: in this game a shot costs no store, but it makes the
tick not quiet, so it gives up at least one tick of the shooter's recharge, and a hit worth
less than that loses the shield race even when it lands. The other way to fire is the kill
case: the hit is at least the target's (last seen) shield, so it empties the shield. At 0
shield any damage kills (core.py `_shields`).

## Results

Panel skill (the recheck protocol of scripted-reach.md "Trim": panel of 8, 18 conditions,
64 games a cell, fresh stores; each side's panel on its own code, so the comparison is not
paired):

| | before (core 8f7d1ea2) | after | difference |
|---|---|---|---|
| panel skill | 0.6136 +- 0.0047 | 0.6258 +- 0.0047 | +0.0122 +- 0.0066 |
| head to head, after vs before (1,152 games, seats alternating) | | 0.546 +- 0.015 | |

(The pre-merge core, vision 1,000 m: 0.6163 +- 0.0047, `skill-before-vision1000.json`. The
night's 0.631 was measured on the game before the 12-tick bullets and the 1,800-tick clock.)

Hits (`hits.py`, 256 envs x 1,800 ticks, CPU, training spawn and team sizes, yardstick both
teams):

| | fields before | fields after | no fields before | no fields after |
|---|---|---|---|---|
| hits within 25 deg of the nose | 85.8% | 25.6% | 87.8% | 27.9% |
| share of nominal damage dealt | 7.0% | 48.0% | 5.8% | 45.5% |
| shots | 304,004 | 141,653 | 332,379 | 113,486 |
| hits | 154,188 | 29,948 | 220,920 | 37,647 |
| total damage | 180,572 | 182,704 | 208,986 | 209,421 |
| kills / env-minute | 9.00 | 9.14 | 10.46 | 10.48 |

The skipped shots were almost worthless: half the shots and a fifth of the hits deal the
same total damage.

Compile tripwire (`scoreboard/compile_check.py`, laptop GPU, performance): "within the baseline", 5,703 traced operations (act 1,704, advance 2,173, minibatch_loss 1,080, slot_actions 746), 0 graph breaks, 238 s cold. Scripted call (`scoreboard/game_ops.py`, both teams): 343 -> 386 operations.

Boundary test (`test_every_parameter_alone_at_both_ends_of_its_box_is_competent`): fails
the same way on this branch (1.0953 arena radii, level_three -1). Traced (scratch
`boundary.py`): a defensive ship with an empty store (1%) at 0.94 R heads outward. The
boundary steer is fully on (centre weight 1) and the stick is full right toward the centre
(P 0.92). It flies through a low-index field at the edge (ln n -0.19 to -0.32) twice, and
each time the field turns its nose left against full right stick (nose rate +0.4 to +1.07
rad/s). Steady full stick at that speed in open air turns at 0.69 rad/s without oscillation
(scratch `stickturn.py`), so the stick and the steer behave. The cause is the field's
refraction. Target choice is not involved: the target was seen and 1.4 km away, and the
ghost change does not alter the trajectory. Not fixed here: a real fix would make the steer
aware of fields (e.g. start earlier inside a field, or keep away from fields near the edge),
which is a design question.

## 3. The head-on stalemate (diagnosis, no fix)

`loops.py`: no fields, otherwise the training game (fog, training spawn, 1,800-tick clock),
yardstick both teams, CPU. 1v1: 256 games. 4v4: 128 games. A pass is a mutual pair at a
closing distance minimum, headings within 30 deg of opposite. The loop runs from the pass to
the first tick both noses are within 30 deg of the other ship again. Numbers below are core
(before); the branch's are within a few percent (loops-after-*.json).

| measure | 1v1 | 4v4 |
|---|---|---|
| passes | 1,822 | 913 |
| lateral separation at the pass, p10/p50/p90 | 1.2 / 3.8 / 8.7 m | 1.4 / 4.8 / 17.5 m |
| same, in sustained radii (241 m) | 0.016 median | 0.020 median |
| time from the pass to middle stick or more | 0.033 s (the next tick) | 0.033 s |
| time from the pass to full stick, p50/p90 | 0.13 / 1.87 s | 0.13 / 1.37 s |
| loop length (closed loops) p50 | 4.0 s | 3.7 s |
| turn rate in the loop, sustained rates, p50 | 1.04 | 1.15 |
| full-stick share in the loop, p50 | 0.87 | 0.66 |
| stick cap in the loop, p10/p50/p90 | 0.02 / 0.05 / 0.64 | 0.03 / 0.23 / 0.77 |
| speed in the loop, corner speeds, p50 | 0.70 | 0.80 |
| store in the loop, p50 | 0.014 | 0.091 |
| tactic weights in the loop (mean): disengaged / merge / offensive / turning / defensive | 0.01 / 0.28 / 0.06 / 0.60 / 0.05 | 0.01 / 0.22 / 0.07 / 0.65 / 0.06 |
| two-circle plan share in the loop | 0.53 | 0.60 |
| range / engage range in the loop, mean (max p90) | 0.28 (0.62) | 0.21 (0.60) |
| level_cap gate in the loop, mean | 0.002 | 0.008 |
| range when they face again, p10/p50/p90 | 347 / 406 / 1,045 m | 314 / 414 / 1,018 m |
| head-on passes per pair, p10/p50/p90 | 6 / 7 / 10 (branch: 6 / 8 / 10) | 1 / 1 / 3 |

By pass number in 1v1 (means):

| pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9+ |
|---|---|---|---|---|---|---|---|---|
| store | 0.53 | 0.34 | 0.18 | 0.07 | 0.02 | 0.01 | 0.01 | 0.01 |
| speed / corner | 0.86 | 0.79 | 0.75 | 0.72 | 0.68 | 0.66 | 0.64 | 0.61 |
| turn rate / sustained | 1.30 | 1.22 | 1.16 | 1.07 | 1.03 | 0.96 | 0.93 | 0.92 |
| two-circle share | 0.70 | 0.31 | 0.69 | 0.38 | 0.69 | 0.35 | 0.64 | 0.50 |
| loop ended by a death | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.09 | 0.26 | 0.51 |

What this shows (line numbers: `lean/scripted/opponent.py` on core 8f7d1ea2; on this branch the same lines are about 20 further down):
- **No turning room.** The pass is nose to nose: 4 m median lateral separation against a
  241 m turn radius. Each ship's stick aims at the gun's lead point right up to the pass
  (`lean/scripted/opponent.py:280` aim point, `:377` wanted direction = the aim point; the
  merge tactic has no offset: NOTES.md "Not built: the pre-merge offset").
- **The turn after the pass starts at once.** Middle stick on the next tick, full stick in
  0.13 s. The suspected flip to disengaged does not happen: the range stays within 0.62
  engage ranges, the engaged gate stays on, and the level_cap gate is about 0 (speed far
  below the middle stick speed). So `:295`/`:307` (engaged) and the disengaged cap are not
  the cause.
- **The loop is wide because the ships are slow and out of energy, not because the stick is
  lazy.** By the fourth pass the store is gone (0.07, then 0.01). The speed falls to 0.6-0.7
  corner speeds, and the turn rate falls to the sustained rate or below (0.92 by pass 9),
  even at full stick 60-90% of the loop. The store is spent holding v* = the fight speed,
  1.29 sustained speeds (217 m/s, `:289`, `:344`), against the drag of a full-stick turn
  (steady full stick settles near 150 m/s): the dive is allowed in the turning tactic
  (`:449`) and runs until the store is empty.
- **In two-circle flow the stick is capped to the middle level.** `cap = turning x
  two_circle` (`:349`, `:358`, used at `:420`; two_circle is the flow of the two turn senses,
  so two-circle geometry folds full into middle: "two-circle holds the middle level at the
  fight speed"). In the loop the cap is 0.64 at p90 (0.97 at p90 in the first 3 s). In
  passes where both turn the same way it is the "period of not really turning". In 1v1 the
  median loop is not capped.
- **The plan alternates every pass** (two-circle share 0.70, 0.31, 0.69, 0.38, ...). Every
  pass gains less than the switch gain, so `:327` flips it. Identical ships with zero offset
  make every pass neutral, so the fight goes back to head-on after each 4-second loop at
  about 400 m. That is under the 697 m gun range, so it comes as another pass, not a
  disengage. The stalemate ends only when the energy is gone and the shots of a late pass
  (or a third ship in 4v4) decide it: 1v1 deaths come mostly from pass 7 on.

**Proposed fix (not built):**
1. A pre-merge offset: in the merge tactic, beyond about two gun ranges, aim the stick at a
   point offset sideways by about one sustained turn radius, toward the side the plan will
   turn. Then return to the lead point inside the gun range. This gives the turning room
   (Shaw's "offset at the merge") that the 4 m passes lack. It differs from the deleted lead
   turn: the lead turn began the turn before the pass, timed on the opponent's geometry. The
   offset flies straight and only moves the aim point. The turn still starts after the pass,
   and the stick keeps its existing commit rule.
2. Keep energy for the turn after the pass: stop the dive in the turning tactic once speed is
   at the sustained speed, instead of chasing 1.29 sustained speeds at full stick (`:289`
   with `:449`), so the store lasts past the second pass. Climb back to it in the merge
   approach.
3. After a neutral pass, the first turn at full stick whatever the flow (no two-circle cap
   for that turn, `:358`), so the turn is the best rate the ship has.

All three need a skill check, and 1 is a new mechanism, so it is the author's call.

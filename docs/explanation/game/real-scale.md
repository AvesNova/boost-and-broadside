---
covers: [src/bnb/game/real_scale.py, src/bnb/game/ships/real-scale/, src/bnb/game/handling/similarity.py]
---

# The real-scale variant

The standard game is built for play: its ships turn fast, on a small map, under a gravity
of 27 m/s^2. The real-scale variant keeps the game and changes its units so that one
metre is a metre and one second a second, its ships derived from a P-51D that flies as
the research has it ([The real planes](real-planes.md)). It is
the ship set `real-scale` (`src/bnb/game/ships/real-scale/`) and the presets
real_scale_deathmatch, real_scale_frontline and real_scale_elimination, with the arena and
Clover maps (`bnb.game.modes.rules.PRESETS`). Everything stays per unit mass:
energies are J/kg and powers W/kg.

`bnb.game.real_scale` holds the set's constants (the scales, the gun flight time).

## From the standard fighter to a P-51D

Today's fighter maps onto a P-51D with lengths x 5.82 and times x 4.00: its stall speed,
31.6 m/s, becomes 46 m/s (x 1.455), and its gravity, 27 m/s^2, becomes 9.81 m/s^2
(x 0.3633, length over time squared). The time factor is the one that turns one gravity
into the other, sqrt(5.82 x 27 / 9.81) = 4.0023.

Every flight constant scales by its units (`bnb.game.handling.similarity`): the
exchanges, as accelerations, by 0.3633; the nose frequency floor, a rate, by 1/4.00; the
induced drag factor, a length at the default bleed exponent, by 5.82; the dimensionless
ones (stall angle, engine knee, fade band, damping) not at all. This gives the solver a
starting point in the right proportions; each type's card then fits its own handles. The
bullet drag per metre scales by 1/5.82.

## The map and the clock

The map is three times the standard one in metres: the presets' arena radius is 7800 m
with the map scale's reference radius kept at 2600 m, so the map scale is 3 and the
fields, the zones and the spawn disc scale by themselves. The distances that do not follow
the map scale are set three times in the set's rules: the field band width (120 m) and
team deathmatch's spawn distances (group radius 900 m, respawn distances 750 to 1800 m,
spawn margin 150 m).

The random maps' fields are the author's (game v14): band width 160 m, and radii from
120 m to 1200 m at map scale 1 (360 m to 3600 m on the 7800 m map), drawn lognormal with
median 400 m and log spread 0.6, their centres within 0.85 arena radii, half of them high
and half strong. Boundary damage is at most 40 per s, over a ramp of 100 m at map scale 1
(300 m on the 7800 m map). Team deathmatch plays a 4200 m arena (map scale 1.6), Frontline
and elimination the 7800 m one.

The rules' times follow the map, not the ship. A P-51 cruises about 1.57 times as fast as
today's fighter in metres per second, over a map three times as wide, so crossing between
fields takes about twice as long. The episode doubles to 600 s, and the capture time (20 s)
and the respawn waits (10 s in team deathmatch, 16 s in Frontline) double with it.

## The rules

| Rule | Value | Why |
|---|---|---|
| gravity | 9.81 m/s^2 | real |
| tick | 30 Hz | unchanged |
| bullet drag | 1.375e-4 /m | today's 8e-4 /m over 5.82 |
| vision range | 3072 m | today's 1024 m times the map scale |
| spawn speed | 72.6 m/s | 1.3 times the researched types' fastest stall speed (the Hammerhead's 55.9 m/s) |
| field band width | 120 m | three times today's |
| episode | 600 s | twice today's |
| capture time | 20 s | twice today's |

The gun flight time, 0.6 s (18 ticks), is the time in which the P-51's bullet covers about
600 m (587 m at its researched speeds); every type's gun range is its own bullet's reach in
that time at its cruise and muzzle speed, so every type's bullet life is 0.6 s
(`bnb.game.real_scale.gun_flight_time`). A type's life follows its gun range, so a gun
range tuned later sets its own life. The spawn speed is set above every stall speed so that no ship
starts in a stall: the build took 1.3 times the fastest of the researched types, and the
rules keep it (a player's type that stalls faster starts in a stall).

## The ships

The set holds the ships the game uses: Yongma (the energy fighter) and Jujak (the turn
fighter), the release's two archetypes, first (the game's default lineup alternates the
first two types, so every real-scale preset without a lineup of its own flies them), then
vizia, the P-51D-derived hull both derive from ([Two archetypes](ship-design/archetypes.md)).
The arena's lineup and the play shell's default are Yongma and Jujak too. Player-made types
share the pool: a type made with the tuning panel's **Add a copy as this name** (or written
by hand) goes after them, in the set's `order.toml`, and has only to be a valid ship file
(every handle and card field, a fire rate the tick can hold). The type count is the width
of Yemong's ship-type input: a model trained on the set with another count does not load
into it.

The set first held 27 researched and invented types (the P-51D first), built from
researched quantities by the set's builder; they left the game on 2026-11-13. Their
research, their conversion to handles and their last cards are kept as reference data:
[The real planes](real-planes.md). The rules on this page are the ones those cards were
solved under, and the spawn speed is still 1.3 times their fastest stall speed, above every
kept type's.

## The scripted agent's real-scale sets

The scripted agent plays the real-scale presets with sets of its own,
real_scale_deathmatch and real_scale_frontline in
`bnb.controllers.scripted_defaults.MODE_DEFAULTS` (and real_scale_elimination, a copy of
the deathmatch set without a search), chosen whenever the card's gravity is the
real-scale one ([The scripted agent](../scripted-agent.md)); the standard sets are
unchanged. They are a light retune: each preset's parameter search
(`bench/scripted_search.py --preset <preset> --ship-type 0 --from-defaults`) started from
the standard mode's set and played the P-51D lineup (every ship a P-51D, a type of the set
then), 32 candidates of
64 games a generation, for 12 generations in deathmatch and 13 in Frontline, about 17
minutes of GPU time with compiling. Eight finalists of each (the top four of the last
generation, two of each of the two before) then played 512 games each at T = 1, among
themselves and against the standard set; all eight were within noise of each other and
each beat the standard set in 95-100% of its games. Deathmatch keeps the best rated;
Frontline the one within noise of the best whose self-play drew least and captured second
most.

The check, `bench/scripted_compare.py`, 512 fresh games per row on the P-51D lineup, fog
on, seats drawn per game, seeds above the search's (kills are combat deaths, a front move
is one capture's step of the front):

| Preset | Match | Score of the first | Kills a game | Kills a minute | Shots a game | Hit share | Front moves a minute | Ended by the clock |
|---|---|---|---|---|---|---|---|---|
| deathmatch | real-scale set against the standard set | 0.994 +- 0.003 | 32.6 | 4.8 | 2,271 | 0.22 | | 11% |
| deathmatch | standard set, self-play | | 43.2 | 5.3 | 922 | 0.76 | | 23% |
| deathmatch | real-scale set, self-play | | 46.4 | 7.0 | 3,284 | 0.21 | | 11% |
| Frontline | real-scale set against the standard set | 0.969 +- 0.006 | 24.0 | 4.2 | 1,378 | 0.27 | 0.43 | 12% |
| Frontline | standard set, self-play | | 38.0 | 3.8 | 824 | 0.74 | 0.44 | 97% |
| Frontline | real-scale set, self-play | | 68.1 | 6.8 | 3,424 | 0.30 | 0.20 | 99% |

(A hit is a shot that damaged an enemy. The real-scale self-play drew 9% of its deathmatch
games and 69% of its Frontline games, the standard set 3% and 38%.)

What moved most: both sets fire at targets well beyond a gun range (the shot logistic's
centre from 0.53 to 1.71 gun ranges in deathmatch, 0.73 to 1.90 in Frontline) and accept
a coarser aim (the aim tolerance about doubled in deathmatch, 0.20 to 0.35 target sizes in Frontline), and fight from further
(the combat radius 3.0 to 4.7 gun ranges, 4.0 to 4.8). At 0.96 s of charge a shot, spraying
is cheap: the real-scale set fires three to four times as often for a quarter of the hit
share, and kills faster. In Frontline it captures less: in self-play the front moves
0.20 times a minute against the standard set's 0.44, and two thirds of its games end
drawn on the clock. Every Frontline finalist did the same (0.14 to 0.21 a minute in
self-play), so it is the search's direction, not one candidate's.

The [balancing](real-scale-balance.md) has since replaced both sets with champions
searched on mixed lineups of every type (its round 1). The pass did not cover the other
types, the timing parameters' bounds, the decision floors or the chaser; the backlog holds a proper retune (`docs/rewrite/backlog.md`).

With these sets the scripted agent has flown every type against every other: [The ship-type
tournament](real-scale-tournament.md).

## What is not scaled

The model and the learner (Yemong's position and velocity codes, the delta and correction
bounds, the discounts) are tuned to the standard game; the real-scale variant is for play,
by people and the scripted agent. The scripted agent states its distances in gun ranges,
corner radii and the like, and plays the real-scale presets with its own lightly retuned
sets (above); the chaser keeps the standard distances in the card's lengths. The solver's
windows and error floors follow each card's own scales ([The ship
card](ship-card.md#measuring)); a 125 kJ store climbs for about an hour of game time, so
a real-scale card takes minutes to measure or solve on a CPU, where a standard one takes
seconds. The tuning panel shows its sliders at once from the ship files and measures only
the shown type and the types in the match, 7 to 20 s a real-scale type.

The playback speed and the tuning panel's [time scale](../in-progress/play-shell.md#live-tuning)
are the way to try the variant faster than real time.

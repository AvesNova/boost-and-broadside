---
covers: [src/bnb/game/real_scale.py, src/bnb/game/ships/real-scale/, src/bnb/game/handling/similarity.py]
---

# The real-scale variant

The standard game is built for play: its ships turn fast, on a small map, under a gravity
of 27 m/s^2. The real-scale variant keeps the game and changes its units so that one
metre is a metre and one second a second, with a fighter that flies like a P-51D. It is
the ship set `real-scale` (`src/bnb/game/ships/real-scale/`) and the presets
real_scale_deathmatch and real_scale_frontline (`bnb.game.modes.rules.PRESETS`). Everything stays per unit mass:
energies are J/kg and powers W/kg.

`bnb.game.real_scale` holds the arithmetic and writes the set
(`uv run --no-sync python -m bnb.game.real_scale`).

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

The rules' times follow the map, not the ship. A P-51 cruises about 1.57 times as fast as
today's fighter in metres per second, over a map three times as wide, so crossing between
fields takes about twice as long. The episode doubles to 600 s, and the capture time (20 s)
and the respawn waits (10 s in team deathmatch, 16 s in Frontline) double with it.

## The rules

| Rule | Value | Why |
|---|---|---|
| gravity | 9.81 m/s^2 | real |
| tick | 30 Hz | unchanged |
| bullet life | 0.6 s (18 ticks) | the P-51's gun range of about 600 m |
| bullet drag | 1.375e-4 /m | today's 8e-4 /m over 5.82 |
| vision range | 3072 m | today's 1024 m times the map scale |
| spawn speed | 66.2 m/s | 1.3 times the set's fastest stall speed (the heavy's 51.0 m/s) |
| field band width | 120 m | three times today's |
| episode | 600 s | twice today's |
| capture time | 20 s | twice today's |

Bullet life is one rule for every type, so another type's gun range follows its own
muzzle and cruise speed. The spawn speed is set above every stall speed so that no ship
starts in a stall; the build takes 1.3 times the fastest in the set, so it follows the set
as types are added.

## The ships

The fighter's handles are the P-51D's where they are known: stall speed 46 m/s, g limit 8,
cruise 163 m/s, a store of 125,300 J/kg (the service ceiling of 12,770 m times g), a
sustained turn of 18 deg/s, a stall angle of 16 deg, a half-lift drag of 0.25 (a parabolic
polar), a muzzle speed of 880 m/s (the .50 M2) and a hit radius of 5.64 m (the
half-span). The climb time is the one whose initial store charge, coasting at cruise from
an empty store, is g x 16.3 m/s (the P-51's 3,200 ft/min): 3,918 s to fill the store, as
the charge falls once the store passes the engine knee. The other flight
handles (low-speed acceleration, nose reversal, dive and zoom, zero-lift drag) are today's
scaled by their units, placeholders. The combat handles other than the muzzle speed and
the hit radius (damage, fire rate, shields) keep the game's values; the bullet spread
keeps today's dispersion angle (spread over muzzle speed); the spawn energy keeps today's
fraction of the store.

The heavy is a placeholder: today's heavy, scaled as today's fighter is to the P-51 (each
handle times the P-51's over today's fighter's, or by its units where the P-51 gives
none), with its climb solved for today's heavy's initial charge scaled the same way.

Each ship file's `[notes]` says where each value comes from and which are placeholders.
To add an aircraft, give `bnb.game.real_scale` an `Aircraft` with the handles that are
known and the standard type it is like (and a climb rate, if known), and run the module:
it rewrites the set's rules (the spawn speed follows the fastest stall speed) and solves
every card.

| Handle | Fighter (P-51D) | Heavy (placeholder) |
|---|---|---|
| stall speed | 46.0 m/s | 51.0 m/s |
| g limit | 8 | 5.93 |
| cruise speed | 163 m/s | 166.6 m/s |
| store (ceiling) | 125,300 J/kg | 175,420 J/kg |
| climb time | 3,918 s (initial charge 159.5 W/kg) | 5,369 s (162.9 W/kg) |
| sustained turn | 18.0 deg/s | 18.7 deg/s |
| stall angle | 16 deg | 16 deg |
| muzzle speed | 880 m/s | 792 m/s |
| gun range | 600 m | 554 m |
| hit radius | 5.64 m | 7.76 m |
| vision range | 3072 m | 3072 m |

The solve converged for both types: every flown handle within 3e-6 of its target
(`tests/sections/test_real_scale.py` measures each card again and holds it to the
solver's 0.5%).

## How it plays

Guns draw on the store, and the shots a full store pays for (33) are a placeholder kept
from the standard game. A shot then costs 3,759 J/kg, about 23 s of the P-51's initial
charge, where a standard shot costs about 1 s of its; and a 5.64 m half-span at 600 m is a
quarter of the angle a standard fighter presents at its gun range. Headless matches of
the scripted agent against itself (seed 1, a whole episode each) show the effect:

| Preset | Kills | Shots | Mean speed |
|---|---|---|---|
| frontline (300 s) | 46 | 1,408 | 67 m/s |
| real_scale_frontline (600 s) | 1 | 170 | 58 m/s |
| deathmatch (300 s) | 45 | 1,280 | 66 m/s |
| real_scale_deathmatch (600 s) | 3 | 195 | 86 m/s |

The fighting is rare until the combat handles (shots per store, fire rate, damage,
dispersion) get real values.

## What is not scaled

The model and the learner (Yemong's position and velocity codes, the delta and correction
bounds, the discounts) are tuned to the standard game; the real-scale variant is for play,
by people and the scripted agent. The scripted agent classifies the map in gun ranges and
places its chaser's distances in corner radii and gun ranges, so it plays the real-scale
maps with its deathmatch and Frontline sets, but its timing parameters (memory fade-in,
switch rates) are tuned to the standard game's tempo and are not re-tuned. The solver's
windows and error floors follow each card's own scales ([The ship
card](ship-card.md#measuring)); a 125 kJ store climbs for about an hour of game time, so
a real-scale card takes minutes to measure or solve on a CPU, where a standard one takes
seconds.

The playback speed and the tuning panel's [time scale](../in-progress/play-shell.md#live-tuning)
are the way to try the variant faster than real time.

---
covers: [bench/pacing.py, src/bnb/game/real_scale.py]
---

# Pacing the real-scale fight

The game has no altitude, only the energy store, and the store is where the real-scale set
departs most from real life by design. A P-51D's store (its 12,680 m ceiling, 124 kJ/kg)
charges at its sea-level climb, 149 W/kg: from empty to the 40% a fresh ship spawns with
takes 334 s of coasting, more than half a 600 s episode, and the whole store about 70
minutes. In a flight simulator most of the time is climbing; this is an arcade game, and the
share of time spent fighting should be much higher, with a fighter that has spent its energy
able to reset into the next fight quickly, and the fight itself as little changed as
possible.

The standard game is the reference: its fighter fills its store in 160 s of its time, 640 s
in real-scale seconds (its time factor is 4.00), against the P-51D's 4,163 s: the real-scale
set charges 6.5 times slower than the standard game it was scaled from, over an episode
twice as long.

## How the charge feeds the flight model

The climb handle is the store's initial charge coasting at cruise, g times the climb rate,
and the card meets it with the coast exchange a_0 (the store gains m u a_0 while coasting at
speed u, until the engine knee). The solver pays for it with engine power: below the knee
the engine gives the power that holds the cruise at a full store plus m a_0 u_c. That power
is there in a fight too: in a full-stick turn the exchange falls to zero (it is scaled by
the share of the lift budget not used for the turn) and the whole engine pushes the ship.
So a faster climb is a stronger engine, and a card solved for the same handles at a faster
climb pays for it elsewhere: more induced drag (to hold the sustained turn), a weaker boost
exchange (to hold the dive acceleration) and a stronger reverse exchange (to hold the zoom).

The P-51D solved at the same handles with the climb multiplied (`bench/pacing.py`'s cards;
the turn rates are the scripted agent's settled turns at the three sticks):

| Climb | Engine power below the knee (W/kg) | Induced drag scale | Boost exchange scale | Reverse exchange scale | Turn rates, sticks 1-3 (deg/s) | Corner speed (m/s) | Bleed rate, middle stick (W/kg) | Climb time (s) |
|---|---|---|---|---|---|---|---|---|
| x 1 | 407 | 1.13 | 1.02 | 0.98 | 10.1, 12.8, 15.7 | 126.4 | -1,805 | 4,163 |
| x 2 | 567 | 1.40 | 0.93 | 1.07 | 10.2, 12.9, 15.7 | 126.4 | -2,150 | 2,081 |
| x 4 | 886 | 1.95 | 0.74 | 1.26 | 10.2, 12.9, 15.7 | 126.4 | -2,875 | 1,041 |
| x 8 | 1,542 | 3.48 | 0.38 | 1.63 | 10.0, 12.9, 15.7 | 126.4 | -5,090 | 520 |

Every handle is held (the turn rates to 0.1 deg/s, the corner speed exactly), but the
middle stick's bleed rate, which the handles do not hold, grows with the climb: a
partial-stick turn spends energy 1.6 times as fast at four times the climb and 2.8 times at
eight. The store's size changes nothing on the card but the store (and the top speed of a
boost, which converts it). The engine knee changes nothing below half the store: it only
speeds the climb above it (a later knee keeps the engine's full power higher up).

## The measures

`bench/pacing.py` plays every ship one type, both teams the scripted agent (its real-scale
deathmatch set, T = 1), 512 games of 600 s, and classifies each living ship-tick:

- **engaged**: an enemy is alive within two of the ship's gun ranges;
- **recovering**: not engaged, with less than the type's spawn energy (40% of its store);
- **ready**: not engaged, with at least that.

Per variant: the shares of living time in each, the store fraction during engagements and at
their start (and the share of engagements started below the spawn energy), the time from
leaving a fight to being ready again, and the fight: the speed over cruise and the
flight-path turn rate while engaged, kills a minute, and the engaged ship-seconds per kill.

## The sweep

The P-51D lineup, 512 games a variant (`real-scale-pacing/pacing-dm.jsonl`):

| Climb | Store | Knee | Engaged | Recovering | Ready | Store at a fight's start | Fights started below the spawn energy | Speed over cruise, engaged | Turn while engaged (deg/s) | Kills a minute | Engaged ship-seconds a kill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x 1 | x 0.5 | 0.5 | 0.20 | 0.78 | 0.02 | 0.32 | 0.94 | 0.56 | 8.5 | 7.03 | 14.9 |
| x 1 | x 1 | 0.5 | 0.18 | 0.82 | 0.00 | 0.34 | 0.99 | 0.61 | 8.6 | 7.62 | 12.1 |
| x 1 | x 1 | 0.7 | 0.15 | 0.82 | 0.02 | 0.35 | 0.94 | 0.61 | 8.3 | 7.40 | 11.0 |
| x 1 | x 1 | 0.9 | 0.15 | 0.82 | 0.02 | 0.35 | 0.94 | 0.61 | 8.3 | 7.40 | 11.0 |
| x 2 | x 0.5 | 0.5 | 0.17 | 0.60 | 0.23 | 0.39 | 0.49 | 0.58 | 8.2 | 7.27 | 12.6 |
| x 2 | x 1 | 0.5 | 0.15 | 0.61 | 0.24 | 0.39 | 0.49 | 0.61 | 8.2 | 7.44 | 10.7 |
| x 2 | x 1 | 0.7 | 0.15 | 0.61 | 0.24 | 0.39 | 0.49 | 0.61 | 8.2 | 7.44 | 10.7 |
| x 2 | x 1 | 0.9 | 0.15 | 0.61 | 0.24 | 0.39 | 0.49 | 0.61 | 8.2 | 7.44 | 10.7 |
| x 2 | x 1.5 | 0.5 | 0.14 | 0.62 | 0.23 | 0.39 | 0.49 | 0.62 | 7.9 | 7.52 | 10.0 |
| x 4 | x 0.5 | 0.5 | 0.15 | 0.33 | 0.53 | 0.52 | 0.15 | 0.61 | 7.7 | 7.52 | 10.2 |
| x 4 | x 1 | 0.5 | 0.14 | 0.33 | 0.53 | 0.46 | 0.15 | 0.62 | 7.8 | 7.55 | 9.9 |
| x 4 | x 1 | 0.7 | 0.14 | 0.33 | 0.53 | 0.46 | 0.15 | 0.61 | 7.8 | 7.54 | 10.0 |
| x 4 | x 1 | 0.9 | 0.14 | 0.33 | 0.53 | 0.46 | 0.15 | 0.61 | 7.8 | 7.54 | 10.0 |
| x 4 | x 1.5 | 0.5 | 0.15 | 0.33 | 0.52 | 0.44 | 0.15 | 0.61 | 7.8 | 7.50 | 10.4 |
| x 8 | x 0.5 | 0.5 | 0.14 | 0.17 | 0.69 | 0.69 | 0.05 | 0.58 | 6.7 | 7.43 | 9.7 |
| x 8 | x 1 | 0.5 | 0.14 | 0.17 | 0.69 | 0.59 | 0.05 | 0.59 | 7.2 | 7.48 | 9.9 |
| x 8 | x 1 | 0.7 | 0.14 | 0.17 | 0.69 | 0.61 | 0.05 | 0.60 | 7.2 | 7.50 | 9.8 |
| x 8 | x 1 | 0.9 | 0.14 | 0.17 | 0.69 | 0.61 | 0.05 | 0.60 | 7.3 | 7.49 | 9.7 |
| x 8 | x 1.5 | 0.5 | 0.14 | 0.17 | 0.68 | 0.54 | 0.05 | 0.59 | 7.3 | 7.45 | 10.2 |

What the sweep says:

- **At today's climb the store is spent and stays spent.** 82% of the living time is
  spent below the spawn energy and out of a fight, and 99% of the fights begin there: the
  scripted agent spends the store (boosting) as soon as it has it and fights on what is left.
  The store is not a resource the fight uses; it is a head start each life gets once.
- **The climb is the lever.** The share recovering falls from 0.82 to 0.61, 0.33 and 0.17 at
  two, four and eight times the climb, the fights begun below the spawn energy from 99% to
  49%, 15% and 5%, and the store at a fight's start rises from 0.34 to 0.59.
- **The share engaged does not move** (0.14 to 0.18 everywhere): how often ships meet is
  the map's and the agent's, not the energy's. The fight's pace barely moves either: kills
  7.4 to 7.6 a minute, 10 to 12 engaged ship-seconds a kill, speed 0.6 of cruise while
  engaged. What does move is the turn while engaged: 8.6 deg/s at today's climb, 7.8 at four
  times (-9%), 7.2 at eight (-16%), as the stronger engine's extra induced drag bleeds a
  partial-stick turn faster (the cards above).
- **The store's size and the engine knee do little.** Half or one and a half times the
  store changes the shares by at most 0.03 (the store is a fraction in each); the knee at 0.7
  or 0.9 instead of 0.5 changes nothing at four or eight times the climb and little at one,
  because the ships seldom hold more than half their store: the knee shapes only the top
  half, which the agent does not use.

## The choice

PACING_CHOICE

"""The game: its rules, state, physics, modes and perception, used through `bnb.game.api`.

VERSION rises whenever a change alters what the game computes (a rule, a constant, the
order of a sum that changes its rounding), so policy files and ratings say which game they
were made in. A refactor that computes the same numbers keeps it. FINGERPRINT is a hash
of a short seeded match in each game preset at this version
(`tests/sections/test_game_version.py`), so a change to the numbers fails that test until
the version is raised.

Version 1 named the game from 2026-11-03 to 2026-11-10, though 36 changes to what it
computes went in under it (the observation's accelerations, whole-Hz fire rates and
instant bullet life, the time scale, the real-scale card): ratings keyed on version 1 mix
those games. Version 2 is the game at 2026-11-10. Version 3 (2026-11-11): the shield's
recharge pauses while the pilot gives an input other than a quiet one
(`physics.combat.quiet`), and the arena preset's radius is 1950 m (was 1300 m). On
2026-11-12 each type's bullet life became a card field derived from its gun range, equal to
the old shared life on every ship set, so the game's numbers stayed version 3; the
fingerprint changed only because a play-shell match allocates 60 bullet slots (was 45),
and at 45 it is the old one. Also on 2026-11-12 the elimination mode came in (its presets
real_scale_elimination and clover, fixed map layouts): a new game, rated under its own mode
and map, while every existing game computes the same numbers (the old presets' fingerprint
without the observation's new mode_elimination column, always 0 in them, is the old one),
so the version stayed 3; the fingerprint now covers the two new presets as well. Version 4
(2026-11-12): a random spawn heading leaves room to turn away from the edge
(`modes.spawns._heading`, `Rules.spawn_turn_distance`): within that distance of it a
heading's outward cosine is at most the distance over it, so team deathmatch's and
elimination's spawns change in every preset with random headings. Version 5 (2026-11-12):
an input other than a quiet one restarts the shield's recharge wait at the delay, as a hit
does (it only paused it in versions 3 and 4), so a shield recharges after the delay's
length of unbroken quiet flight (`physics.combat.apply_damage`). The real-scale set gained
two types (Yongma and Jujak) the same day; a new type changes no existing game's numbers.
Version 6 (2026-11-12): bullet damage. A hit's damage is the card's (a centre hit's, by a
bullet at its reference speed) times the explosive share plus the rest times the bullet's
kinetic energy, (proper speed / reference speed)^2, times the closeness of the pass,
(1 - (b / R)^2)^2 for the relative path's closest approach b to the target's centre
(`physics.combat.kinetic`, `closeness`). Each type's bullet drag is its own handle (the
rules' is the floor; the real-scale and design-seed sets' is 5e-4 1/m, the .50 cal's), its
bullet life the whole ticks that fly its gun range under it (no drag trim). Every set's
cards were re-solved: hit damage x 15/8 (the closeness's mean is 8/15), the reference
distance 300 m in the real-scale sets (`Rules.damage_reference_distance`). Version 7
(2026-11-12): elimination spawns by team side (`modes.spawns.side_placement`): each team
in a disc of `ModeRules.side_area_fraction` arena radii around its centre,
`Rules.spawn_offset` arena radii out on the spawn axis, nose toward the enemy area within
`side_heading_jitter`, keeping the edge's turn room; it was team deathmatch's uniform
draw. Team deathmatch, Frontline and fixed layouts (Clover) compute the same numbers.
Version 8 (2026-11-13): Frontline's capture lock and respawn waves, and shorter respawn
waits. After a capture the team that lost the zone gains no progress on it for
`ModeRules.lock_seconds_per_root_km` (19) x sqrt(d / 1 km) s, d the adjacent zones'
spacing (about 45 s at 5v5 on the real-scale map). A dead Frontline ship respawns at its
team's first respawn wave after its minimum wait, each team's waves every
`ModeRules.wave_seconds` (20 s at real scale, 10 s in the standard game) from a random
phase per team (two more mode uniforms per match, before the spawn draws). The respawn
waits: team deathmatch 8 s at real scale (was 10 s), Frontline's minimum 8 s (its fixed
wait was 16 s), halved in the standard game. The observation gains the zones' lock_own
and lock_enemy and the global next_wave_own.
Version 9 (2026-11-13): real-scale flight. Two flight constants, on in the real-scale set's
rules (off elsewhere, where every game computes the same numbers): the direct charge (the
engine's store charge goes into the store as a power; the speed sees only the null power)
and the drag rise (the zero-lift drag times 1 + (u / 197 m/s)^12, which bounds a boost
near 1.35 x cruise). Yongma, Jujak and Vizia were re-solved under them with higher
sustained turn rates (their full-stick turn settles near 70 m/s, not 55); the set's other
types keep their old cards until re-solved.
Version 10 (2026-11-13): per-type stick levels. The stick keeps its three levels each way
(fine, sustained, max: the same action indices), but each type's lift fractions are card
fields (`fine_stick`, `sustained_stick`; max stays 1). A solve sets them from the rules'
shares (`FlightConstants.sustained_rate_share`, `fine_stick_share`): the sustained level
is the lift fraction whose settled coasting turn reaches that share of the full stick's
rate, the fine level that share of it. Without shares (every set but real-scale, whose
shares are 0.8 and 0.3) the cards keep the classic 0.4 and 0.75. Yongma, Jujak and Vizia
were re-solved with vizia's sustained turn 18.7 -> 21.3 deg/s (same archetype factors):
their sustained level settles at 88-91 m/s. The fingerprint's presets fly none of the
three, so it is version 9's.
Version 11 (2026-11-13, rough, iterating): real-scale flight v11, three switches in the
real-scale rules (off elsewhere): flow separation (the induced drag times
1 + s (C / C_max)^m, s = 1, m = 8), the propeller's thrust (softmin exponent 2:
P / sqrt(u^2 + (P / T_0)^2)) and the exchange as vertical lift demand (the lift holds
gravity plus the exchange: a climb asks for more, a dive about none; the turn keeps
priority, the shortfall sinks, and an empty store holds gravity first). The settled level
turn's rate now peaks below full stick; a solve sets the sustained stick to that peak and
the fine to 0.3 of it. Yongma, Jujak and Vizia re-solved.
Version 12 (2026-11-13, rough, iterating): lethality. A card field `core_radius`: full
damage inside it, a smoothstep down to 0 at the hit radius (0, every set's default, keeps
the classic (1 - (b / R)^2)^2 falloff). Yongma, Jujak and Vizia: core 5 m, hit radius 12 m,
a centre hit 21 x the archetype delta (about 6 core hits kill, 0.3-0.5 s of fire), no
damage dead ahead and half at 45 degrees (sharpness 3). The shield keeps its old timing (6 s
of quiet flight, then 12 per s; the wait restarts on a boost, a shot, a hit or a stick past
the fine level). The fingerprint's presets fly none of the three,
so it is version 11's.
Version 13 (2026-11-13): the real-scale set at x2 native speed. Every card and handle of
the set and its rules.toml went through the exact similarity at length 1, time 1/2
(`bnb.game.handling.similarity`; speeds x2, accelerations and gravity x4, rates and fire
rates x2, times /2, energies x4, powers x8, lengths and map sizes kept), with Yongma, Jujak
and Vizia re-solved from their scaled handles (docs/rewrite/sections/lethality/native_x2.py
and resolve_native.py). The mode clocks (match, respawn, capture, waves, the lock) keep
their numbers as native seconds. The real-scale scripted agent's time parameters scaled
the same way (scripted_x2.py). The plane-equivalent values are native over
`real_scale.EQUIVALENT_FACTOR` (2), for display only.
Version 14 (2026-11-13): sight through the band. Each field has a core (inside radius - b/2),
a band (out to radius + b/2) and outside. A line with exactly one end in the core is blocked
only when the other end is outside the band, so the core and the band see each other; a line
with neither end in the core is blocked when it passes within the core radius, as before.
Before, the band counted as outside. No ship's visibility in the fingerprint's 48 ticks
changes, so the fingerprint is version 13's.

Version 15 (2026-11-13): the boundary and the real-scale maps. Boundary damage, in every
mode, is `Rules.boundary_max_damage` per s times the smootherstep of the depth beyond the
boundary over `boundary_ramp_width` (m at map scale 1, times the map scale): 30 per s and
100 m in the standard game, 40 per s at real scale (was 5 + 0.05 depth, and 10 + 0.1 depth
at real scale). Elimination's boundary shrinks: it holds at the arena radius for
`ModeRules.shrink_hold_seconds` (18 s, 36 s at real scale: a fifth of the clock), then
falls to 0 at the clock's end (`shrink_law`: linear in radius, or the smootherstep), so
staying away no longer runs the clock out; the tie-break by health left decides what
survives it. The observation gains the globals boundary_fraction (the boundary's radius
over the arena radius) and boundary_time_to_zero; every length keeps the arena radius as
its scale. The real-scale random maps (the set's rules.toml, the author's): field band
160 m, radii 120-1200 m at map scale 1 drawn lognormal (median 400 m, log spread 0.6).
Real-scale team deathmatch plays a 4200 m arena (was 7800 m) with a 10 s respawn wait
(was 8 s).

Version 16 (2026-11-13): elimination's boundary closes before the clock, and small
deathmatch arenas have a floor. The shrinking boundary no longer stops at radius 0 at the
clock's end: it passes 0 and reaches minus a ramp width, where even a ship at the centre
takes the boundary's most, `ModeRules.shrink_end_lead_seconds` (8 s) before the clock.
From the zero crossing a ship at the centre is hurt every tick, so its shield does not
recharge; 8 s at the most outlasts the largest shield of either set (160 at 30 or 40 per
s), so every ship dies before the clock, and the tie-break by health left only decides
ships that die on the same tick (a draw when both teams' last ships do). The observation
shows the boundary's fraction clamped at 0, and boundary_time_to_zero counts to the zero
crossing. A game's arena radius at another fleet size (`api.arena_radius_for`) follows the
density rule down to its mode's minimum: team deathmatch 1000 m at map scale 1, 3000 m at
real scale (1v1 and 2v2 real-scale deathmatch were 1878 m and 2656 m by the rule alone);
elimination and frontline have none. No preset changes radius.

Version 17 (2026-11-13): team deathmatch's outcome is its kill share, and Frontline's
zones have floors. `Events.win_loss` (B, 2) is team 0's weights of a win and of a loss
where a match ended (a draw the rest): in team deathmatch (share, 1 - share), share =
team 0's kills over both teams' (0.5 with none), as its rating scores a game; in the other
modes the result one-hot. The outcome reward is win - loss (2 share - 1 in deathmatch) and
the outcome head's final target (win, draw, loss) its weights; `result` still says who led.
Frontline's zone ring keeps its zones' radius at least `ModeRules.zone_radius_floor`
(231 m) and adjacent zones at least `zone_spacing_floor` (1315.8 m) apart, both m at map
scale 1 times the ship set's map scale (3 at real scale), 0.7 x the 5v5 map's: no preset
changes, a map under 70% of it is held there. The spawn disc follows the match's zone
radius. Map-size sampling: a pool with `arena_scale` (`bnb.contracts.ArenaScale`; the
training pools by default) draws each environment's arena scale at every reset, ln scale
~ Normal(0, 0.35) truncated to [0.5, 2] by the inverse CDF of one more tick uniform (only
when some pool samples, so a game without sampling draws as before); the episode's arena
radius is the pool's times the scale, at least the mode's minimum (`api.arena_radius_for`),
and every map length follows its map scale. `Match.arena_scale` (B,) holds it (1 without
sampling); that new leaf is the fingerprint's only change.
"""

VERSION = 17
FINGERPRINT = "64c9cbf08b4ca333"

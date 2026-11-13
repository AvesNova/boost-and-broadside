"""The real-scale ship set (`src/bnb/game/ships/real-scale/`): 1 m is a metre and 1 s a
second. Its types are Yongma and Jujak, the release's two archetypes, and vizia, the hull
both derive from (a P-51D-derived card); `docs/explanation/game/real-scale.md` gives the
reasons. This module holds the set's constants that other code reads.

The set was first built from late-war fighters and a few invented designs, each solved
from researched quantities; those 27 types left the game on 2026-11-13. Their research
and their solved cards are reference data now (`bnb.reference.real_planes`,
docs/explanation/game/real-planes.md), and the set's rules (its `rules.toml`) are the ones
they were solved under: today's game in real units, lengths x LENGTH (5.82) and times x
TIME (4.00, the time that turns today's gravity of 27 m/s^2 into 9.81), on a map MAP_SCALE
(3) times today's, its rule times CLOCK_SCALE (2) times today's. The set plays at
EQUIVALENT_FACTOR (2) times the planes' speed (game v13).

A gun's range is its bullet's reach at cruise plus muzzle speed in the gun flight time
(`gun_flight_time`, the whole ticks in which the P-51D's bullet travels about GUN_RANGE)
under RANGE_DRAG; `bnb.game.ship_design` designs types with them.
"""

import math

from bnb.reference.real_planes import AIRCRAFT

from .rules import FlightConstants, Rules

SHIP_SET = "real-scale"
STANDARD_GRAVITY = FlightConstants().gravity  # 27 m/s^2
GRAVITY = 9.81  # m/s^2
# The set plays at this many times the planes' speed (game v13, native): its files hold
# the x2 values; a plane-equivalent display divides by it through the similarity (length 1,
# time EQUIVALENT_FACTOR). It drives no simulation.
EQUIVALENT_FACTOR = 2.0
LENGTH = 5.82  # a standard metre in real metres
TIME = math.sqrt(LENGTH * STANDARD_GRAVITY / GRAVITY)  # a standard second in real ones: 4.00
MAP_SCALE = 3.0  # the map against the standard one (arena radius 7800 m against 2600 m)
CLOCK_SCALE = 2.0  # rule times: a crossing takes 3x the distance at about 1.5x the speed
GUN_RANGE = 600.0  # m, the P-51D's, which sets the gun flight time (whole ticks)
# 1/m: the drag the gun ranges are measured with, today's per standard metre over LENGTH
# (the set's bullet drag before game v6): each type's range is its bullet's reach in the gun
# flight time under it, so the ranges kept their values when each gun got its own drag.
RANGE_DRAG = Rules(1, 1, ()).bullet_drag / LENGTH
# The climb time over the store's fill time at its initial charge: the charge falls once
# the store passes the engine knee, and every solved type landed near 4.99.
CLIMB_FILL = 4.99


def gun_flight_time(tick_length: float = Rules.tick_length) -> float:
    """The whole ticks (s) in which the P-51D's bullet (its researched cruise and muzzle
    speed) travels about GUN_RANGE (ln(1 + k v t) / k = R) under RANGE_DRAG: a type's gun
    range is its own bullet's reach in this time under it."""
    reference = next(one for one in AIRCRAFT if one.key == "p51d").data
    launch = reference["cruise_speed"].value + reference["muzzle_speed"].value
    life = math.expm1(RANGE_DRAG * GUN_RANGE) / (RANGE_DRAG * launch)
    return round(life / tick_length) * tick_length

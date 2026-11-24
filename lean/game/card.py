"""The ship card and the shared rule constants.

Rows of every card tuple are indexed by ship type in the order of TYPE_NAMES. Units: metres,
seconds, radians, joules, at the game's time (twice real time: lean.game.tuning); lift and drag
coefficients are per metre (mass 1).

CARD is derived, not stored: on first use every row is made from its targets by
lean.game.tuning (`load_card`), cached in memory and on disk by a hash of the targets and of
the flight model's source, so a physics change re-solves the rows by itself:

- "p51d": the P-51D (tuning.REAL_AIRCRAFT): its point mass fitted to the flight test (top speed
  and best climb), then the game terms on (tuning.P51D_TERMS: drag rise, separation drag, soft
  minima, automatic flaps) with its zero-lift drag, thrust power and induced drag solved so it
  flies its point mass's top speed, best climb and best sustained turn rate (tuning.game_row,
  docs/rewrite/p51d-model.md); real lift, g limit 8, static thrust, dive and climb at 1 g;
  guns, shield, nose and stick levels from tuning.GAME_FIELDS.
"""

import math

TYPE_NAMES = ("p51d",)

# The flight terms: the drag rise above critical_speed, the separation drag, the soft minima of
# lift against the g limit and of thrust against static thrust, the nose damping, and the
# automatic flaps (core._wing): the bonus flap_lift x the clean maximum lift, deployed by the
# demanded lift coefficient, blown back by flap_blowback_ratio x the clean corner speed; the
# extra lift's drag flap_drag_scale x INDUCED_DRAG_FACTOR x (Delta lift coefficient)^2 with the
# flaps fully used (core._rates).
FLIGHT_TERMS = (
    "critical_speed",
    "drag_rise_exponent",
    "separation_drag",
    "separation_exponent",
    "lift_softmin_exponent",
    "thrust_softmin_exponent",
    "nose_damping",
    "flap_lift",
    "flap_blowback_ratio",
    "flap_drag_scale",
)
# The card's columns, in order. engine_power is thrust power per kg; exchange_acceleration the
# dive and climb trade between speed and store.
CARD_FIELDS = (
    "max_lift_coefficient",
    "max_acceleration",
    "zero_lift_drag",
    "engine_power",
    "static_thrust",
    "exchange_acceleration",
    "corner_nose_frequency",
    "max_shield",
    "hit_radius",
    "core_radius",
    "hit_damage",
    "muzzle_speed",
    "fire_rate",
    "bullet_life",
    "explosive_share",
    "reference_bullet_speed",
    "induced_drag_scale",
    "stall_angle_scale",
    "bullet_spread_scale",
    "bullet_drag_scale",
    "shield_recharge_rate_scale",
    "shield_recharge_delay_scale",
    "head_on_fraction_scale",
    "head_on_angle_scale",
    "head_on_sharpness",
    "fine_stick",
    "sustained_stick",
    "vision_range_scale",
    *FLIGHT_TERMS,
)


def __getattr__(name: str):
    """CARD on first use: {field: one value per type}, derived by lean.game.tuning."""
    if name == "CARD":
        from . import tuning

        return tuning.load_card()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Flight constants shared by every type (world constants and numerical floors).
GRAVITY = 39.24
STALL_ANGLE = 0.2617993877991494
INDUCED_DRAG_FACTOR = 66.348
NOSE_FREQUENCY_FLOOR = 0.4997135491028137
SPEED_FLOOR = 1.454166427889188e-06
# The automatic flaps (core._wing), shared: they deploy as the demanded lift coefficient goes
# from 1 - FLAP_DEMAND_BAND of the clean maximum to all of it, and blow back over the last
# FLAP_BLOWBACK_BAND of the blow-back speed.
FLAP_DEMAND_BAND = 0.1
FLAP_DEPLOY_OFFSET = (1.0 - FLAP_DEMAND_BAND) / FLAP_DEMAND_BAND
FLAP_BLOWBACK_BAND = 0.4

# Guns, shields, boundary, spawns.
BULLET_DRAG = 0.0005
BULLET_SPREAD = 12.0
HEAD_ON_FRACTION = 0.3
HEAD_ON_ANGLE = 0.737832313317803
SHIELD_RECHARGE_DELAY = 6.0
SHIELD_RECHARGE_RATE = 12.0
BOUNDARY_MAX_DAMAGE = 40.0
SPAWN_SPEED = 145.288  # m/s over the map: the standard spawn
SPAWN_AREA_FRACTION = 0.85  # the spawn disc's radius over the arena radius (both rules)
SPAWN_EDGE_BAND = 1000.0  # m inside the boundary: no spawn heading there points outward
SIDE_GAP_FRACTION = 0.05  # "sides": each team's half disc this far (over the arena radius) from the dividing line
SIDE_HEADING_JITTER = 0.2617993877991494  # "sides": noses within 15 degrees of the axis
VISION_RANGE = 3072.0  # m; times the type's vision_range_scale: 1,000 m for every type
LN_SQRT2 = 0.5 * math.log(2.0)
LN_2 = math.log(2.0)

# Stick index to signed level; power index 0 coast, 1 dive, 2 climb.
STICK_LEVELS = (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
QUIET_STICK_MAX = 2  # stick indices 0..2 (straight and the fine turns) keep the shield quiet

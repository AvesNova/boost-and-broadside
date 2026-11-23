"""The ship card and the shared rule constants.

Rows of every card tuple are indexed by ship type in the order of TYPE_NAMES. Units: metres,
seconds, radians, joules, at the game's time (twice real time: lean.game.tuning); lift and drag
coefficients are per metre (mass 1). Rows are made by lean.game.tuning (a test holds each row
equal to the tuning that makes it):

- "p51d": the P-51D (tuning.P51D) with the game terms on (tuning.P51D_TERMS: drag rise,
  separation drag, soft minima, automatic flaps), its zero-lift drag, thrust power and induced
  drag solved so it flies its point mass's top speed, best climb and best sustained turn rate
  (tuning.game_row, docs/rewrite/p51d-model.md); real lift, g limit 8, static thrust, dive and
  climb at 1 g; guns, shield, nose and stick levels from tuning.GAME_FIELDS.
"""

import math

TYPE_NAMES = ("p51d",)

CARD = {
    "max_lift_coefficient": (0.004797063474135985,),
    "max_acceleration": (313.92,),
    "zero_lift_drag": (4.46290716669821e-05,),
    "engine_power": (1775.303807491058,),  # thrust power per kg
    "static_thrust": (23.890784444835262,),
    "exchange_acceleration": (39.24,),  # dive and climb trade between speed and store
    "corner_nose_frequency": (11.23144817352295,),
    "max_shield": (100.0,),
    "hit_radius": (12.0,),
    "core_radius": (5.0,),
    "hit_damage": (21.0,),
    "muzzle_speed": (1712.0,),
    "fire_rate": (30.0,),
    "bullet_life": (0.36666667461395264,),
    "explosive_share": (0.0,),
    "reference_bullet_speed": (1753.606201171875,),
    "induced_drag_scale": (0.3051412439216051,),
    "stall_angle_scale": (1.0333333333333334,),
    "bullet_spread_scale": (0.25704020261764526,),
    "bullet_drag_scale": (1.0,),
    "shield_recharge_rate_scale": (2.0,),
    "shield_recharge_delay_scale": (0.5,),
    "head_on_fraction_scale": (0.0,),
    "head_on_angle_scale": (1.0644670724868774,),
    "head_on_sharpness": (3.0,),
    "fine_stick": (0.21010109782218933,),
    "sustained_stick": (0.7003369927406311,),
    "vision_range_scale": (0.3255208432674408,),
    # The flight terms (FLIGHT_TERMS): the drag rise above critical_speed, the separation drag,
    # the soft minima of lift against the g limit and of thrust against static thrust, the nose
    # damping, and the automatic flaps (core._wing: the maximum lift times 1 + flap_lift below
    # flap_full_speed_ratio x the clean stall speed, clean above flap_clear_speed_ratio x it;
    # the extra lift's drag flap_drag_scale x INDUCED_DRAG_FACTOR x (Delta lift coefficient)^2).
    "critical_speed": (524.062,),
    "drag_rise_exponent": (20.0,),
    "separation_drag": (0.1,),
    "separation_exponent": (8.0,),
    "lift_softmin_exponent": (16.0,),
    "thrust_softmin_exponent": (8.0,),
    "nose_damping": (0.7,),
    "flap_lift": (0.15,),
    "flap_full_speed_ratio": (1.3,),
    "flap_clear_speed_ratio": (1.6,),
    "flap_drag_scale": (1.0054180489951219,),
}
FLIGHT_TERMS = (
    "critical_speed",
    "drag_rise_exponent",
    "separation_drag",
    "separation_exponent",
    "lift_softmin_exponent",
    "thrust_softmin_exponent",
    "nose_damping",
    "flap_lift",
    "flap_full_speed_ratio",
    "flap_clear_speed_ratio",
    "flap_drag_scale",
)

# Flight constants shared by every type (world constants and numerical floors).
GRAVITY = 39.24
STALL_ANGLE = 0.2617993877991494
INDUCED_DRAG_FACTOR = 66.348
NOSE_FREQUENCY_FLOOR = 0.4997135491028137
SPEED_FLOOR = 1.454166427889188e-06

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

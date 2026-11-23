"""The ship card and the shared rule constants.

Rows of every card tuple are indexed by ship type in the order of TYPE_NAMES. Units: metres,
seconds, radians, joules, at the game's time (twice real time: lean.game.tuning); lift and drag
coefficients are per metre (mass 1). Rows are made by lean.game.tuning (a test holds each row
equal to the tuning that makes it):

- "p51d": the P-51D's airframe converted directly (tuning.P51D, tuning.airframe_row): real
  lift, drag, engine power and propeller, g limit 8, dive and climb at 1 g, the game flight
  terms off (its point mass, the October fit, docs/rewrite/p51d-model.md); guns, shield, nose
  and stick levels from tuning.GAME_FIELDS.
"""

import math

TYPE_NAMES = ("p51d",)

CARD = {
    "max_lift_coefficient": (0.004530452337926361,),
    "max_acceleration": (313.92,),
    "zero_lift_drag": (4.500249322340185e-05,),
    "engine_power": (1801.134167607861,),  # thrust power per kg
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
    "induced_drag_scale": (0.3405755313049645,),
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
    # the soft minima of lift against the g limit and of thrust against static thrust, and the
    # nose damping. The P-51D flies with the game terms off (tuning.POINT_MASS_TERMS).
    "critical_speed": (math.inf,),
    "drag_rise_exponent": (12.0,),
    "separation_drag": (0.0,),
    "separation_exponent": (8.0,),
    "lift_softmin_exponent": (400.0,),
    "thrust_softmin_exponent": (400.0,),
    "nose_damping": (0.7,),
}
FLIGHT_TERMS = (
    "critical_speed",
    "drag_rise_exponent",
    "separation_drag",
    "separation_exponent",
    "lift_softmin_exponent",
    "thrust_softmin_exponent",
    "nose_damping",
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
SPAWN_SPEED = 145.288
SPAWN_OFFSET = 0.8
SPAWN_TURN_DISTANCE = 436.5
SIDE_AREA_FRACTION = 0.15
SIDE_HEADING_JITTER = 0.2617993877991494
MIXED_AREA_FRACTION = 0.85
VISION_RANGE = 3072.0  # m; times the type's vision_range_scale: 1,000 m for every type
LN_SQRT2 = 0.5 * math.log(2.0)
LN_2 = math.log(2.0)

# Stick index to signed level; power index 0 coast, 1 dive, 2 climb.
STICK_LEVELS = (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
QUIET_STICK_MAX = 2  # stick indices 0..2 (straight and the fine turns) keep the shield quiet

"""The real-scale ship card and the shared rule constants, as v2 holds them (game version 20).

Rows of every card tuple are indexed by ship type in the order of TYPE_NAMES. Units: metres,
seconds, radians, joules; lift and drag coefficients are per metre (mass 1).

The first three types are v2's. Two are lean's own, made by lean.game.tuning for the core flight
model (docs/rewrite/p51d-model.md; a test holds each row equal to the tuning that makes it):
- "p51d": the P-51D's airframe converted directly at the game's 2x time (tuning.P51D,
  tuning.airframe_row): real lift, drag, engine power and propeller, g limit 8, the game flight
  terms off (its point mass, the October fit); guns, shield,
  nose and stick levels are Vizia's.
- "vizia_plain": Vizia with v2's climb modifications reverted (tuning.unmodified_climb): no
  engine surplus charging the store, dive and climb at 1 g.
"""

import math

TYPE_NAMES = ("yongma", "jujak", "vizia", "p51d", "vizia_plain")

CARD = {
    "max_lift_coefficient": (0.00424512755125761, 0.005678284447640181, 0.0049096886068582535, 0.004530452337926361, 0.0049096886068582535),
    "max_acceleration": (353.1600036621094, 353.1600036621094, 353.1600036621094, 313.92, 353.1600036621094),
    "zero_lift_drag": (4.2485891754040495e-05, 5.8069828810403123e-05, 4.967039785697125e-05, 4.500249322340185e-05, 4.967039785697125e-05),
    "engine_power": (9439.37109375, 11269.7060546875, 10292.57421875, 1801.134167607861, 2038.4674072265625),
    "engine_power_full_store": (2016.1539306640625, 2091.707275390625, 2038.4674072265625, 1801.134167607861, 2038.4674072265625),
    "static_thrust": (49.05462646484375, 33.45197296142578, 35.69484329223633, 23.890784444835262, 35.69484329223633),
    "max_energy": (497563.1875, 497563.1875, 497563.1875, 497563.2, 497563.1875),
    "corner_nose_frequency": (10.925199508666992, 11.559896469116211, 11.23144817352295, 11.23144817352295, 11.23144817352295),
    "max_shield": (141.94908142089844, 79.67646026611328, 100.0, 100.0, 100.0),
    "hit_radius": (12.0, 12.0, 12.0, 12.0, 12.0),
    "core_radius": (5.0, 5.0, 5.0, 5.0, 5.0),
    "hit_damage": (22.98240089416504, 19.187700271606445, 21.0, 21.0, 21.0),
    "muzzle_speed": (1712.0, 1712.0, 1712.0, 1712.0, 1712.0),
    "bullet_energy_cost": (575.0579223632812, 575.0579223632812, 575.0579223632812, 575.0579223632812, 575.0579223632812),
    "fire_rate": (30.0, 30.0, 30.0, 30.0, 30.0),
    "bullet_life": (0.36666667461395264, 0.36666667461395264, 0.36666667461395264, 0.36666667461395264, 0.36666667461395264),
    "spawn_energy": (199025.28125, 199025.28125, 199025.28125, 199025.28125, 199025.28125),
    "explosive_share": (0.5, 0.0, 0.0, 0.0, 0.0),
    "reference_bullet_speed": (1761.7950439453125, 1745.64990234375, 1753.606201171875, 1753.606201171875, 1753.606201171875),
    "induced_drag_scale": (0.5831313729286194, 0.26635676622390747, 0.39490920305252075, 0.3405755313049645, 0.39490920305252075),
    "stall_angle_scale": (1.0333333015441895, 1.0333333015441895, 1.0333333015441895, 1.0333333333333334, 1.0333333015441895),
    "dive_exchange_scale": (4.682722568511963, 4.143795013427734, 4.406386375427246, 4.0, 4.0),
    "climb_exchange_scale": (3.422452211380005, 3.4853007793426514, 3.4585301876068115, 4.0, 4.0),
    "bullet_spread_scale": (0.25704020261764526, 0.25704020261764526, 0.25704020261764526, 0.25704020261764526, 0.25704020261764526),
    "bullet_drag_scale": (1.0, 1.0, 1.0, 1.0, 1.0),
    "shield_recharge_rate_scale": (2.0, 2.0, 2.0, 2.0, 2.0),
    "shield_recharge_delay_scale": (0.5, 0.5, 0.5, 0.5, 0.5),
    "head_on_fraction_scale": (0.0, 0.0, 0.0, 0.0, 0.0),
    "head_on_angle_scale": (1.0644670724868774, 1.0644670724868774, 1.0644670724868774, 1.0644670724868774, 1.0644670724868774),
    "head_on_sharpness": (3.0, 3.0, 3.0, 3.0, 3.0),
    "engine_knee_scale": (1.0, 1.0, 1.0, 1.0, 1.0),
    "fine_stick": (0.1964104026556015, 0.2223183661699295, 0.21010109782218933, 0.21010109782218933, 0.21010109782218933),
    "sustained_stick": (0.654701292514801, 0.7410612106323242, 0.7003369927406311, 0.7003369927406311, 0.7003369927406311),
    "vision_range_scale": (0.3255208432674408, 0.3255208432674408, 0.3255208432674408, 0.3255208432674408, 0.3255208432674408),
    # v2's shared flight terms (rules.toml [flight]), per type in lean (FLIGHT_TERMS). The P-51D
    # flies with the game terms off (tuning.POINT_MASS_TERMS): no drag rise, no separation drag,
    # near-hard minima.
    "critical_speed": (394.0, 394.0, 394.0, math.inf, 394.0),
    "drag_rise_exponent": (12.0, 12.0, 12.0, 12.0, 12.0),
    "separation_drag": (1.0, 1.0, 1.0, 0.0, 1.0),
    "separation_exponent": (8.0, 8.0, 8.0, 8.0, 8.0),
    "lift_softmin_exponent": (16.0, 16.0, 16.0, 400.0, 16.0),
    "thrust_softmin_exponent": (2.0, 2.0, 2.0, 400.0, 2.0),
    "nose_damping": (0.7, 0.7, 0.7, 0.7, 0.7),
}
# The card's columns that are v2's shared flight terms, per type here (v2's names).
FLIGHT_TERMS = (
    "critical_speed",
    "drag_rise_exponent",
    "separation_drag",
    "separation_exponent",
    "lift_softmin_exponent",
    "thrust_softmin_exponent",
    "nose_damping",
)

# Flight constants shared by every type (world constants and numerical floors; the flight terms
# are per type: FLIGHT_TERMS).
GRAVITY = 39.24
STALL_ANGLE = 0.2617993877991494
INDUCED_DRAG_FACTOR = 66.348
ENGINE_KNEE = 0.5
STORE_FADE_BAND = 0.02
CLIMB_EXCHANGE = 9.810000000000002
DIVE_EXCHANGE = 9.810000000000002
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

# Stick index to signed level; power index 0 coast, 1 boost, 2 reverse.
STICK_LEVELS = (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
QUIET_STICK_MAX = 2  # stick indices 0..2 (straight and the fine turns) keep the shield quiet

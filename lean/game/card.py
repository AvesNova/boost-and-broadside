"""The ship card and the shared rule constants.

Rows of every card tuple are indexed by ship type in the order of TYPE_NAMES. Units: metres,
seconds, radians, joules, at the game's time (twice real time: lean.game.tuning); lift and drag
coefficients are per metre (mass 1).

CARD is derived, not stored: on first use every row is made from its targets by
lean.game.tuning (`load_card`), cached in memory and on disk by a hash of the targets and of
the flight model's source, so a physics change re-solves the rows by itself:

- "p51d": the P-51D (tuning.REAL_AIRCRAFT): its reference (the point mass fitted to the flight
  test, tuning.reference_of), then its design inputs (tuning.DESIGN_INPUTS: closed forms, e
  and the separation drag and thrust softness among them, and the top speed and best climb
  solved for the zero-lift drag and thrust power; tuning.design_row,
  docs/rewrite/p51d-model.md "Design inputs").

The bullet's damage reference speed is not stored either (core._derived_characteristics).
The stick levels (Sustain, Max, Snap) are shared rules (core.stick_table and its siblings; their
shapes below).
"""

import math

TYPE_NAMES = ("p51d",)

# The flight terms: the drag rise above critical_speed, the separation drag, the soft minima of
# lift against the g limit and of thrust against static thrust, the nose damping (the angle of
# attack's damping ratio toward the commanded angle at Snap's stiffness) and the post-stall regime
# (core._setting, _lift_coefficient, _rates): Snap's deepest angle of attack below corner, the
# band below corner speed (a fraction of it) over which Snap's angle opens from the stall angle
# to it, and the drag at POST_STALL_DRAG_ANGLE over the drag at the stall.
FLIGHT_TERMS = (
    "critical_speed",
    "drag_rise_exponent",
    "separation_drag",
    "separation_exponent",
    "lift_softmin_exponent",
    "thrust_softmin_exponent",
    "nose_damping",
    "deep_stall_angle",
    "deep_stall_band",
    "post_stall_drag_ratio",
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
    "induced_drag_scale",
    "stall_angle_scale",
    "bullet_spread_scale",
    "bullet_drag_scale",
    "shield_recharge_rate_scale",
    "shield_recharge_delay_scale",
    "nose_inner_angle",
    "nose_outer_angle",
    "blame_inner_angle",
    "blame_outer_angle",
    "vision_ahead",
    "vision_side",
    "vision_behind",
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
# Past the stall (docs/rewrite/sections/post-stall/literature.md): the lift falls smoothly and
# monotonically to the flat plate's, POST_STALL_LIFT_FRACTION of the wing's maximum at 45 deg (a
# flat plate's 1.2 over the P-51D's C_Lmax 1.6), and is 0 at 90 deg; the held vertical comes out
# of the lift over POST_STALL_SHARE_WIDTH (radians) past the stall (core._path_lift); the drag
# reaches the card's post_stall_drag_ratio x the drag at the stall at POST_STALL_DRAG_ANGLE.
POST_STALL_LIFT_FRACTION = 0.75
POST_STALL_SHARE_WIDTH = 0.2
POST_STALL_DRAG_ANGLE = math.radians(45.0)
# The lift's turn floors smoothly where the lift barely carries the held vertical
# (core._path_lift): the softness, a fraction of the held vertical.
PATH_LIFT_SOFTNESS = 0.25

# The bullet deflector (author 2026-11-28, core.circle_entry and core.circle_turn;
# lean/game/NOTES.md "Deflector"): an invisible circle of radius deflector_radius around every
# ship. A bullet whose straight in-tick path first crosses it inward is turned away from the
# ship's centre by deflector_turn x hump(b / deflector_radius) (b the line's distance from the
# centre; core.hump, 0 at both ends, peak at a third), its speed kept. The same for every type
# and any shield. The presets (GameConfig.deflector) set the hit model's deflector inputs
# (core.HIT_MODEL); "off" also skips the deflector's code in the tick. Metres and radians.
DEFLECTOR_RADIUS = 24.0
DEFLECTOR_PRESETS = {
    "circle": dict(deflector_radius=DEFLECTOR_RADIUS, deflector_turn=math.radians(45.0)),
    "off": dict(deflector_radius=DEFLECTOR_RADIUS, deflector_turn=0.0),
}
# The nose cone's armour (core.aspect_factor; author 2026-11-25, on again 2026-11-28): its weight
# in the hit model, 1 on (no bullet damage inside the card's nose_inner_angle, full beyond
# nose_outer_angle), 0 off (a setting for ablations).
NOSE_CONE = 1.0

# Guns, shields, boundary, spawns.
BULLET_DRAG = 0.0005
BULLET_SPREAD = 12.0
DAMAGE_REFERENCE_DISTANCE = (
    300.0  # m: a bullet fired at cruise has the damage reference speed here (v2 rules.toml)
)
SHIELD_RECHARGE_DELAY = 6.0
SHIELD_RECHARGE_RATE = 12.0
BOUNDARY_MAX_DAMAGE = 40.0
SPAWN_SPEED = 145.288  # m/s over the map: the standard spawn
SPAWN_AREA_FRACTION = 0.85  # the spawn disc's radius over the arena radius (both rules)
SPAWN_EDGE_BAND = 1000.0  # m inside the boundary: no spawn heading there points outward
SIDE_GAP_FRACTION = (
    0.05  # "sides": each team's half disc this far (over the arena radius) from the dividing line
)
SIDE_HEADING_JITTER = 0.2617993877991494  # "sides": noses within 15 degrees of the axis
LN_SQRT2 = 0.5 * math.log(2.0)
LN_2 = math.log(2.0)

# Stick index to signed level; power index 0 coast, 1 dive, 2 climb.
STICK_LEVELS = (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
QUIET_STICK_MAX = 0  # stick index 0 (straight) keeps the shield quiet; every turn level resets it
# The stick levels 1, 2, 3 (author 2026-11-26; core._setting), each all of its own line:
# Sustain, the sustained turn (Ps = 0 coasting; the envelope where even it gains energy; a table
# over speed, core._sustain_table); Max, the envelope's top line (the wing's maximum lift against
# the g limit, one soft minimum); Snap, the same above corner and below it the card's
# deep_stall_angle, opening over deep_stall_band of corner speed, closing back to the stall
# angle between the speed where that angle's lift holds 1 g and stall speed
# (core._holding_angle).
LEVEL_NAMES = ("Sustain", "Max", "Snap")
# The nose spring's stiffness going out, by |level| (straight, Sustain, Max, Snap), a factor of
# the full spring; going back it is always 1 (core._kick).
NOSE_STIFFNESS = (1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

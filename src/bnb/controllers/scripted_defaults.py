"""The scripted agent's default parameters per game mode (`MODE_DEFAULTS`), each the choice
of that mode's own parameter search, and each standard search's final population
(`scripted_populations.json`, the 32 sets of each standard mode's last generation, best
first: kept for later use, read by nothing here but `populations`). The section notes
record the standard searches; the real-scale variant's page
(docs/explanation/game/real-scale.md) its light retune."""

import json
import pathlib

from .scripted_parameters import (
    NAMES,
    STANDARD_MODES,
    Logistic,
    ModeParameters,
    Ramp,
    ScriptedParameters,
    Switch,
    from_row,
)

POPULATIONS = pathlib.Path(__file__).with_name("scripted_populations.json")

# Each standard mode's finalist that scored best at T = 1 against round one's agent in its
# own preset (384 games; the section notes), to three significant figures.
STANDARD_DEFAULTS = {
    "deathmatch": ScriptedParameters(
        turn_in_stall_angle=Logistic(0.0304, 0.267, True),
        level_two_in_stall_angle=Logistic(0.231, 0.185, True),
        level_three_in_stall_angle=Logistic(0.441, 0.172, True),
        boost_slow_in_corner_speed=Logistic(0.553, 0.0347, False),
        boost_far_in_store_gun_ranges=Logistic(8.98, 0.152, True),
        reverse_in_gun_range=Logistic(0.296, 0.172, False),
        aim_in_target_size=Logistic(0.308, 0.221, False),
        shoot_in_gun_range=Logistic(0.526, 0.129, False),
        lead_cap_in_gun_range=1.08,
        boundary_margin_in_turn_radius=4.09,
        boundary_steer_ramp_in_margin=Ramp(0.0368, 0.445, 0.0, 1.0),
        centre_hold_in_arena_radius=0.46,
        combat_radius_in_gun_range=2.96,
        support_radius_in_zone_radius=0.919,
        defensive_floor=0.19,
        defensive_margin=0.591,
        aggression=2.06,
        defense_urgency=1.31,
        separation_radius_in_gun_range=0.525,
        separation_weight=0.991,
        recovery_shield_fraction=0.146,
        recovery_threat=1.24,
        memory_fade_in_seconds=Logistic(1.06, 0.145, False),
        target_switch_in_log_distance=Switch(15.4, 0.224, 0.0536),
        side_commit_in_stall_angle=Logistic(4.14, 0.159, True),
        side_switch_in_radians=Switch(7.4, 0.655, 0.155),
        goal_switch_in_strength=Switch(4.07, 0.268, 0.0667),
        recovery_switch_in_log_ratio=Switch(5.73, 0.161, 0.0521),
    ),
    "deathmatch_wide": ScriptedParameters(
        turn_in_stall_angle=Logistic(0.0411, 0.168, True),
        level_two_in_stall_angle=Logistic(0.115, 0.167, True),
        level_three_in_stall_angle=Logistic(0.522, 0.181, True),
        boost_slow_in_corner_speed=Logistic(0.464, 0.0403, False),
        boost_far_in_store_gun_ranges=Logistic(12.7, 0.159, True),
        reverse_in_gun_range=Logistic(0.335, 0.217, False),
        aim_in_target_size=Logistic(0.385, 0.227, False),
        shoot_in_gun_range=Logistic(0.467, 0.127, False),
        lead_cap_in_gun_range=0.649,
        boundary_margin_in_turn_radius=2.5,
        boundary_steer_ramp_in_margin=Ramp(0.031, 0.338, 0.0, 1.0),
        centre_hold_in_arena_radius=0.261,
        combat_radius_in_gun_range=4.52,
        support_radius_in_zone_radius=2.09,
        defensive_floor=0.397,
        defensive_margin=0.471,
        aggression=2.0,
        defense_urgency=2.6,
        separation_radius_in_gun_range=1.46,
        separation_weight=0.606,
        recovery_shield_fraction=0.0919,
        recovery_threat=1.82,
        memory_fade_in_seconds=Logistic(0.881, 0.204, False),
        target_switch_in_log_distance=Switch(21.5, 0.154, 0.0702),
        side_commit_in_stall_angle=Logistic(5.44, 0.197, True),
        side_switch_in_radians=Switch(6.22, 0.663, 0.136),
        goal_switch_in_strength=Switch(6.21, 0.349, 0.0516),
        recovery_switch_in_log_ratio=Switch(4.93, 0.385, 0.0852),
    ),
    "frontline": ScriptedParameters(
        turn_in_stall_angle=Logistic(0.0446, 0.329, True),
        level_two_in_stall_angle=Logistic(0.17, 0.188, True),
        level_three_in_stall_angle=Logistic(0.184, 0.351, True),
        boost_slow_in_corner_speed=Logistic(0.338, 0.0471, False),
        boost_far_in_store_gun_ranges=Logistic(16.1, 0.165, True),
        reverse_in_gun_range=Logistic(0.32, 0.161, False),
        aim_in_target_size=Logistic(0.196, 0.205, False),
        shoot_in_gun_range=Logistic(0.732, 0.132, False),
        lead_cap_in_gun_range=0.82,
        boundary_margin_in_turn_radius=2.16,
        boundary_steer_ramp_in_margin=Ramp(0.018, 0.204, 0.0, 1.0),
        centre_hold_in_arena_radius=0.825,
        combat_radius_in_gun_range=3.95,
        support_radius_in_zone_radius=1.73,
        defensive_floor=0.417,
        defensive_margin=0.345,
        aggression=3.2,
        defense_urgency=4.27,
        separation_radius_in_gun_range=1.12,
        separation_weight=0.543,
        recovery_shield_fraction=0.141,
        recovery_threat=2.48,
        memory_fade_in_seconds=Logistic(1.35, 0.194, False),
        target_switch_in_log_distance=Switch(11.5, 0.287, 0.0979),
        side_commit_in_stall_angle=Logistic(25.7, 0.185, True),
        side_switch_in_radians=Switch(7.02, 0.332, 0.312),
        goal_switch_in_strength=Switch(8.7, 0.486, 0.111),
        recovery_switch_in_log_ratio=Switch(3.33, 0.171, 0.0837),
    ),
}
# Each real-scale preset's set, in `scripted_real_scale.json`: each its mode's champion of the
# balancing rounds (docs/explanation/game/real-scale-balance.md), with the source that chose it.
REAL_SCALE = pathlib.Path(__file__).with_name("scripted_real_scale.json")


def from_named(fields: dict) -> ScriptedParameters:
    """A set from its fields by name, as the real-scale sets' file holds them (a logistic's
    centre and scale, a ramp's four ends, a switch's rate, centre and scale; a logistic's
    direction is the default's)."""
    row = []
    for name in NAMES:
        value = fields[name]
        row.extend(value if isinstance(value, list) else [value])
    return from_row(row)


def read_real_scale(path: pathlib.Path = REAL_SCALE) -> dict[str, dict]:
    """The real-scale sets' file: per mode, its "parameters" (`named`) and "source"."""
    return json.loads(path.read_text())


REAL_SCALE_DEFAULTS = {
    mode: from_named(entry["parameters"]) for mode, entry in read_real_scale().items()
}
MODE_DEFAULTS = ModeParameters(STANDARD_DEFAULTS | REAL_SCALE_DEFAULTS)


def populations() -> dict[str, list[ScriptedParameters]]:
    """Each standard mode's final search population, best first."""
    stored = json.loads(POPULATIONS.read_text())
    return {mode: [from_row(row) for row in stored["populations"][mode]] for mode in STANDARD_MODES}

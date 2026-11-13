# ruff: noqa: E501 (generated data: long source notes)
"""The real planes the real-scale set was researched from (reference data, not ship
types: the game dropped them on 2026-11-13 and kept only Yongma, Jujak and vizia, the
P-51D-derived hull they share; docs/explanation/game/real-planes.md is the table).

`AIRCRAFT`: for each plane, the researched quantities in SI (degrees where marked), each
with its confidence and source. Generated from the research of 2026-11-10 (18 late-war
fighters and 8 invented designs; the raw research is in
docs/explanation/game/real-scale-research/), edited by hand afterwards. The real-scale
builder turned them into ship types (handles, then a solved card); `solved_types` reads
each type's file as it stood at game v16 (the set's x2 native units), from
`real-plane-types/` beside this module, its `[notes]` recording where each value came
from. Not every later rule reached them (game v12's core radius did not).

Confidence: "good" (a primary test document), "medium" (a primary figure via a summary, or a
standard figure from memory), "low" (secondary, forum or simulator), "estimated" (computed
or reasoned; the source says how), "derived" (from other quantities here), "invented" (a
designed type: the P-51D's value times the design's multiplier).
"""

import dataclasses
import pathlib
import tomllib

# The types as last solved (one TOML file each, the ship-file format of bnb.game.ships).
SOLVED = pathlib.Path(__file__).with_name("real-plane-types")


@dataclasses.dataclass(frozen=True)
class Datum:
    """A researched quantity: its value, confidence and source."""

    value: float
    confidence: str
    source: str


@dataclasses.dataclass(frozen=True)
class Aircraft:
    """A type of the set: its file's key, its name, the standard type it is like (for the
    handles nothing here gives), its role, its armament and its researched quantities
    (`QUANTITIES`; one it lacks is a placeholder)."""

    key: str
    name: str
    like: str
    role: str
    armament: str
    data: dict[str, Datum]


# The quantities and their units.
QUANTITIES = {
    "stall_speed": "m/s, clean, power off",
    "g_limit": "g",
    "cruise_speed": "m/s, the sea-level maximum speed at military power",
    "climb_rate": "m/s, sea level, military power",
    "service_ceiling": "m",
    "low_speed_acceleration": "m/s^2, level, near the stall speed",
    "sustained_turn_rate": "deg/s",
    "roll_rate": "deg/s, peak",
    "stall_angle": "deg",
    "drag_coefficient": "C_D0",
    "wing_loading": "kg/m^2",
    "half_span": "m",
    "muzzle_speed": "m/s, the main gun's (the most weight of fire)",
    "dispersion": "mil, the volley's 75% cone diameter",
    "weight_of_fire": "kg/s",
    "fire_rate": "Hz, the main gun's whole-Hz volley rate dividing 30",
    "survivability": "1 fragile - 5 very tough",
}

AIRCRAFT = (
    Aircraft(
        "p51d",
        "P-51D",
        "fighter",
        "long-range escort fighter",
        "6x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(
                44.7,
                "medium",
                "Loftin, NASA SP-468; 100 mph. Loftin: stalling speed 100 mph (at ~10,100 lb gross). At 9,600 lb this scales to ~97.5 mph; 100 kept as the published figure.",
            ),
            g_limit=Datum(8.0, "medium", "recalled standard figure, not re-checked"),
            cruise_speed=Datum(
                162.7,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (364 mph)",
            ),
            climb_rate=Datum(
                15.2,
                "estimated",
                "sea-level climb, military power: NAA calc. NA-46-130 gives 3,410 ft/min at WEP (67 in); military (61 in, ~230 hp less) scaled by excess power t... (3000 ft/min)",
            ),
            service_ceiling=Datum(
                12680.0, "good", "wwiiaircraftperformance.org test report (41600 ft)"
            ),
            low_speed_acceleration=Datum(
                2.728,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.1 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                15.7,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 23.0 s)",
            ),
            roll_rate=Datum(95.0, "medium", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(
                15.5,
                "estimated",
                "NACA 45-100 laminar section, clean CLmax ~1.4; typical fuselage-reference stall angle 15-17 deg",
            ),
            drag_coefficient=Datum(0.0163, "good", "Loftin, NASA SP-468"),
            wing_loading=Datum(
                201.0, "medium", "combat mass 4354 kg (medium) over wing area 21.66 m^2"
            ),
            half_span=Datum(5.645, "good", "half of the span 11.29 m (37.04 ft)"),
            muzzle_speed=Datum(856.0, "medium", "the Browning AN/M2 .50 cal: AAF Manual 200-1"),
            dispersion=Datum(
                6.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.68, "medium", "6x Browning AN/M2 .50 cal (wing): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Browning AN/M2 .50 cal: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                2.0,
                "medium",
                "1 fragile - 5 very tough: Liquid-cooled Merlin with a ventral radiator and long coolant lines: one hit can end the flight. Pilot armour and self-s",
            ),
        ),
    ),
    Aircraft(
        "p47d",
        "P-47D",
        "heavy",
        "fighter / fighter-bomber",
        "8x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(
                49.2,
                "estimated",
                "from W/S 45 lb/sq ft and CLmax ~1.45 (NACA TN 2899 measured F-47D-30 stalls; not opened); 110 mph.",
            ),
            g_limit=Datum(8.0, "medium", "recalled standard figure, not re-checked"),
            cruise_speed=Datum(
                152.0,
                "medium",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (340 mph)",
            ),
            climb_rate=Datum(
                14.1,
                "estimated",
                "sea-level climb, military power: 2,330 ft/min at 12,000 ft at 56 in (Wright Field) plus paddle-blade propeller gain; WEP at SL ~3,100 ft/min (2780 ft/min)",
            ),
            service_ceiling=Datum(
                12500.0, "medium", "recalled standard figure, not re-checked (41000 ft)"
            ),
            low_speed_acceleration=Datum(
                2.218,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (2.52 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                14.4,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 25.0 s)",
            ),
            roll_rate=Datum(85.0, "medium", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(16.0, "estimated", "Seversky S-3 section, typical 15-17 deg"),
            drag_coefficient=Datum(0.0211, "medium", "recalled standard figure, not re-checked"),
            wing_loading=Datum(
                219.7, "medium", "combat mass 6123 kg (medium) over wing area 27.87 m^2"
            ),
            half_span=Datum(6.21, "good", "half of the span 12.42 m (40.75 ft)"),
            muzzle_speed=Datum(856.0, "medium", "the Browning AN/M2 .50 cal: AAF Manual 200-1"),
            dispersion=Datum(
                6.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                4.91, "medium", "8x Browning AN/M2 .50 cal (wing): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Browning AN/M2 .50 cal: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                5.0,
                "medium",
                "1 fragile - 5 very tough: Air-cooled 18-cylinder radial that kept running with cylinders shot away, very large and strong airframe, cockpit armour",
            ),
        ),
    ),
    Aircraft(
        "p38l",
        "P-38L",
        "heavy",
        "twin-engine long-range fighter",
        "1x AN/M2 20 mm (nose); 4x Browning AN/M2 .50 cal (nose)",
        dict(
            stall_speed=Datum(
                52.8,
                "estimated",
                "Loftin gives 105 mph, which implies CLmax 1.84 at combat weight and so is almost certainly with Fowler flaps d...; 118 mph.",
            ),
            g_limit=Datum(
                7.5,
                "estimated",
                "AAF pursuit class design (~8 g at design gross), reduced for combat weight above design gross",
            ),
            cruise_speed=Datum(
                148.0,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (331 mph)",
            ),
            climb_rate=Datum(
                18.1,
                "medium",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (3570 ft/min)",
            ),
            service_ceiling=Datum(13410.0, "medium", "Loftin, NASA SP-468 (44000 ft)"),
            low_speed_acceleration=Datum(
                2.587,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (2.94 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                16.7,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 21.5 s)",
            ),
            roll_rate=Datum(
                90.0, "low", "peak roll rate: recalled standard figure, not re-checked"
            ),
            stall_angle=Datum(16.0, "estimated", "NACA 23016/23009 wing"),
            drag_coefficient=Datum(0.0268, "medium", "recalled standard figure, not re-checked"),
            wing_loading=Datum(
                253.4, "medium", "combat mass 7711 kg (medium) over wing area 30.43 m^2"
            ),
            half_span=Datum(7.925, "good", "half of the span 15.85 m (52 ft)"),
            muzzle_speed=Datum(
                856.0, "medium", "the Browning AN/M2 .50 cal, 2.45 of 3.86 kg/s: AAF Manual 200-1"
            ),
            dispersion=Datum(
                3.6,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.86,
                "medium",
                "1x AN/M2 20 mm (nose); 4x Browning AN/M2 .50 cal (nose): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Browning AN/M2 .50 cal, 2.45 of 3.86 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                4.0,
                "medium",
                "1 fragile - 5 very tough: Two engines: could return on one. Pilot armour, self-sealing tanks. Against it: liquid-cooled engines and turbochargers ",
            ),
        ),
    ),
    Aircraft(
        "spitfire-ix",
        "Spitfire IX",
        "fighter",
        "air-superiority fighter",
        "2x Hispano Mk II 20 mm (wing); 2x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(
                41.1,
                "estimated",
                "W/S 30.2 lb/sq ft with CLmax ~1.35 gives 93 mph; Pilot's Notes quote ~ 90-96 mph IAS flaps and undercarriage u...; 92 mph.",
            ),
            g_limit=Datum(
                8.0,
                "estimated",
                "no published limit found; pilots pulled 8-10 g with wrinkled skins; RAF fighters designed to factor ~ 8 limit",
            ),
            cruise_speed=Datum(
                150.2,
                "good",
                "sea-level maximum speed, military power: spitfireperformance.com trials (336 mph)",
            ),
            climb_rate=Datum(
                23.5,
                "good",
                "sea-level climb, military power: spitfireperformance.com trials (4620 ft/min)",
            ),
            service_ceiling=Datum(12470.0, "good", "spitfireperformance.com trials (40900 ft)"),
            low_speed_acceleration=Datum(
                4.488,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (5.1 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                19.5,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 18.5 s)",
            ),
            roll_rate=Datum(90.0, "low", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(
                16.0, "estimated", "NACA 2213-2209.4 section, washout gives a gentle stall"
            ),
            drag_coefficient=Datum(
                0.0194,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                147.3, "medium", "combat mass 3311 kg (medium) over wing area 22.48 m^2"
            ),
            half_span=Datum(5.613, "good", "half of the span 11.23 m (36.83 ft)"),
            muzzle_speed=Datum(
                880.0, "medium", "the Hispano Mk II 20 mm, 2.60 of 3.83 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                4.6,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.83,
                "medium",
                "2x Hispano Mk II 20 mm (wing); 2x Browning AN/M2 .50 cal (wing): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Hispano Mk II 20 mm, 2.60 of 3.83 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                2.0,
                "medium",
                "1 fragile - 5 very tough: Light structure, liquid-cooled Merlin with wing radiators, main fuel tank ahead of the pilot. Pilot armour and armoured ",
            ),
        ),
    ),
    Aircraft(
        "spitfire-xiv",
        "Spitfire XIV",
        "fighter",
        "air-superiority fighter / interceptor",
        "2x Hispano Mk II 20 mm (wing); 2x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(43.4, "estimated", "W/S 34.7 lb/sq ft and CLmax ~1.35; 97 mph."),
            g_limit=Datum(8.0, "estimated", "as Mk IX"),
            cruise_speed=Datum(
                161.8,
                "medium",
                "sea-level maximum speed, military power: spitfireperformance.com trials (362 mph)",
            ),
            climb_rate=Datum(
                25.6,
                "good",
                "sea-level climb, military power: spitfireperformance.com trials (5040 ft/min)",
            ),
            service_ceiling=Datum(
                13110.0, "medium", "recalled standard figure, not re-checked (43000 ft)"
            ),
            low_speed_acceleration=Datum(
                4.382,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (4.98 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                18.57,
                "estimated",
                "the research's point-mass model (21.1 deg/s) x 0.88, its correction for propeller efficiency (no literature value)",
            ),
            roll_rate=Datum(90.0, "estimated", "peak roll rate: as Mk IX"),
            stall_angle=Datum(16.0, "estimated", "as Mk IX"),
            drag_coefficient=Datum(
                0.0186,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                169.5, "good", "combat mass 3810 kg (good) over wing area 22.48 m^2"
            ),
            half_span=Datum(5.613, "good", "half of the span 11.23 m (36.83 ft)"),
            muzzle_speed=Datum(
                880.0, "medium", "the Hispano Mk II 20 mm, 2.60 of 3.83 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                4.6,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.83,
                "medium",
                "2x Hispano Mk II 20 mm (wing); 2x Browning AN/M2 .50 cal (wing): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Hispano Mk II 20 mm, 2.60 of 3.83 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                2.0, "medium", "1 fragile - 5 very tough: As Mk IX; slightly heavier."
            ),
        ),
    ),
    Aircraft(
        "tempest-v",
        "Tempest V",
        "fighter",
        "low/medium-altitude fighter",
        "4x Hispano Mk V 20 mm (wing)",
        dict(
            stall_speed=Datum(45.2, "estimated", "W/S 37.7 lb/sq ft with CLmax ~1.45; 101 mph."),
            g_limit=Datum(8.0, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                168.1,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (376 mph)",
            ),
            climb_rate=Datum(
                22.3,
                "good",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (4380 ft/min)",
            ),
            service_ceiling=Datum(
                10360.0, "good", "wwiiaircraftperformance.org test report (34000 ft)"
            ),
            low_speed_acceleration=Datum(
                3.546,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (4.03 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                15.7,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 23.0 s)",
            ),
            roll_rate=Datum(
                100.0,
                "estimated",
                "peak roll rate: spring-tab ailerons; AFDU/CFE trials rate it better than the Spitfire XIV at speed, worse than the Fw 190",
            ),
            stall_angle=Datum(16.0, "estimated", "thin semi-elliptical wing (14.5% root, 10% tip)"),
            drag_coefficient=Datum(
                0.0151,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                184.3, "good", "combat mass 5171 kg (good) over wing area 28.06 m^2"
            ),
            half_span=Datum(6.248, "good", "half of the span 12.50 m (41 ft)"),
            muzzle_speed=Datum(840.0, "medium", "the Hispano Mk V 20 mm: Wikipedia"),
            dispersion=Datum(
                4.0,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                6.5, "medium", "4x Hispano Mk V 20 mm (wing): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Hispano Mk V 20 mm: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                4.0,
                "medium",
                "1 fragile - 5 very tough: Big, strongly built; pilot armour. Chin radiator and liquid cooling are vulnerable, but the airframe absorbs a lot.",
            ),
        ),
    ),
    Aircraft(
        "mosquito",
        "Mosquito",
        "heavy",
        "twin-engine fighter-bomber",
        "4x Hispano Mk II 20 mm (nose); 4x .303 Browning Mk II (nose)",
        dict(
            stall_speed=Datum(49.2, "estimated", "W/S 43 lb/sq ft and CLmax ~1.4; 110 mph."),
            g_limit=Datum(6.0, "estimated", "fighter-bomber/light-bomber strength class"),
            cruise_speed=Datum(
                148.4,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (332 mph)",
            ),
            climb_rate=Datum(
                10.2,
                "estimated",
                "sea-level climb, military power: A&AEE HJ.679 at 21,020 lb gives 1,740 ft/min at 10,400 ft in M.S. gear; lighter combat weight and SL density (2000 ft/min)",
            ),
            service_ceiling=Datum(
                9083.0, "good", "wwiiaircraftperformance.org test report (29800 ft)"
            ),
            low_speed_acceleration=Datum(
                2.816,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.2 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                14.96,
                "estimated",
                "the research's point-mass model (17.0 deg/s) x 0.88, its correction for propeller efficiency (no literature value)",
            ),
            roll_rate=Datum(
                60.0, "estimated", "peak roll rate: large twin with heavy roll inertia"
            ),
            stall_angle=Datum(
                15.0, "estimated", "no published value found: the research's estimate"
            ),
            drag_coefficient=Datum(
                0.0215,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                209.7, "estimated", "combat mass 8845 kg (estimated) over wing area 42.18 m^2"
            ),
            half_span=Datum(8.256, "good", "half of the span 16.51 m (54.17 ft)"),
            muzzle_speed=Datum(
                880.0, "medium", "the Hispano Mk II 20 mm, 5.20 of 6.07 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                3.1,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                6.07,
                "medium",
                "4x Hispano Mk II 20 mm (nose); 4x .303 Browning Mk II (nose): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Hispano Mk II 20 mm, 5.20 of 6.07 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                3.0,
                "medium",
                "1 fragile - 5 very tough: Two engines; wooden structure absorbs hits but burns and was hard to repair; little armour beyond the crew bulkhead; coo",
            ),
        ),
    ),
    Aircraft(
        "fw190-a8",
        "Fw 190A",
        "fighter",
        "fighter / bomber-destroyer",
        "2x Rheinmetall-Borsig MG 131 13 mm (cowl); 2x Mauser MG 151/20 (wing_root); 2x Mauser MG 151/20 (wing)",
        dict(
            stall_speed=Datum(
                52.8,
                "estimated",
                "W/S 225 kg/m^2 and CLmax ~1.4 gives 184 km/h; RAE noted a sharp wing drop at the stall; 190 km/h.",
            ),
            g_limit=Datum(
                7.0,
                "estimated",
                "German strength group H5 safe load factor ~ 7 at design weight; not found tonight",
            ),
            cruise_speed=Datum(
                154.7,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (557 km/h)",
            ),
            climb_rate=Datum(
                14.4,
                "good",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (GL/C-E2 Nr 7578/44 (13.4 m/s in the 13.11.43 chart))",
            ),
            service_ceiling=Datum(10300.0, "low", "recalled standard figure, not re-checked"),
            low_speed_acceleration=Datum(
                2.807,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.19 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                15.7,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 23.0 s)",
            ),
            roll_rate=Datum(150.0, "medium", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(15.0, "estimated", "NACA 23015.3-23009 section"),
            drag_coefficient=Datum(
                0.0225,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                229.5, "medium", "combat mass 4200 kg (medium) over wing area 18.3 m^2"
            ),
            half_span=Datum(5.255, "good", "half of the span 10.51 m"),
            muzzle_speed=Datum(805.0, "good", "the Mauser MG 151/20, 4.05 of 4.95 kg/s: Wikipedia"),
            dispersion=Datum(
                4.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                4.95,
                "medium",
                "2x Rheinmetall-Borsig MG 131 13 mm (cowl); 2x Mauser MG 151/20 (wing_root); 2x Mauser MG 151/20 (wing): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Mauser MG 151/20, 4.05 of 4.95 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                4.0,
                "medium",
                "1 fragile - 5 very tough: Air-cooled radial in front of the pilot, armoured oil cooler ring, self-sealing tanks; A-8 added heavier cockpit armour ",
            ),
        ),
    ),
    Aircraft(
        "fw190-d9",
        "Fw 190D",
        "fighter",
        "air-superiority fighter",
        "2x Rheinmetall-Borsig MG 131 13 mm (cowl); 2x Mauser MG 151/20 (wing_root)",
        dict(
            stall_speed=Datum(52.8, "estimated", "same wing, similar weight to the A-8; 190 km/h."),
            g_limit=Datum(7.0, "estimated", "as A-8"),
            cruise_speed=Datum(
                151.4,
                "good",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (545 km/h)",
            ),
            climb_rate=Datum(
                17.0,
                "good",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (Rechlin, 3,250 rpm, 4,300 kg)",
            ),
            service_ceiling=Datum(
                10500.0, "good", "wwiiaircraftperformance.org test report (210002 Rpt 1)"
            ),
            low_speed_acceleration=Datum(
                2.834,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.22 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                16.0,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 22.5 s)",
            ),
            roll_rate=Datum(
                130.0, "low", "peak roll rate: wwiiaircraftperformance.org test report"
            ),
            stall_angle=Datum(15.0, "estimated", "as A-8"),
            drag_coefficient=Datum(
                0.0247,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                233.3, "medium", "combat mass 4270 kg (medium) over wing area 18.3 m^2"
            ),
            half_span=Datum(5.255, "good", "half of the span 10.51 m"),
            muzzle_speed=Datum(805.0, "good", "the Mauser MG 151/20, 1.84 of 2.74 kg/s: Wikipedia"),
            dispersion=Datum(
                4.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                2.74,
                "medium",
                "2x Rheinmetall-Borsig MG 131 13 mm (cowl); 2x Mauser MG 151/20 (wing_root): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Mauser MG 151/20, 1.84 of 2.74 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                3.0,
                "medium",
                "1 fragile - 5 very tough: Liquid-cooled but with an armoured annular radiator; armour as A-8.",
            ),
        ),
    ),
    Aircraft(
        "bf109-k4",
        "Bf 109K",
        "fighter",
        "interceptor / air-superiority fighter",
        "1x Rheinmetall-Borsig MK 108 30 mm (engine); 2x Rheinmetall-Borsig MG 131 13 mm (cowl)",
        dict(
            stall_speed=Datum(
                47.2, "estimated", "W/S 206 kg/m^2 with automatic slats, CLmax ~1.5; 170 km/h."
            ),
            g_limit=Datum(7.0, "estimated", "as Fw 190"),
            cruise_speed=Datum(
                165.3,
                "medium",
                "sea-level maximum speed, military power: forum post quoting kurfurst.org (595 km/h)",
            ),
            climb_rate=Datum(
                22.5,
                "medium",
                "sea-level climb, military power: forum post quoting kurfurst.org (1.80 ata; 25 m/s at 1.98 ata; IL-2 museum page gives 20.5 m/s)",
            ),
            service_ceiling=Datum(12500.0, "medium", "recalled standard figure, not re-checked"),
            low_speed_acceleration=Datum(
                4.743,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (5.39 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                17.6,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 20.5 s)",
            ),
            roll_rate=Datum(
                100.0,
                "estimated",
                "peak roll rate: light at low speed, ailerons stiffen markedly above ~450 km/h IAS",
            ),
            stall_angle=Datum(17.0, "estimated", "slats delay the stall"),
            drag_coefficient=Datum(
                0.0236,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                205.6, "medium", "combat mass 3300 kg (medium) over wing area 16.05 m^2"
            ),
            half_span=Datum(4.96, "good", "half of the span 9.92 m"),
            muzzle_speed=Datum(
                520.0, "medium", "the Rheinmetall-Borsig MK 108 30 mm, 3.41 of 4.31 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                2.4,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                4.31,
                "medium",
                "1x Rheinmetall-Borsig MK 108 30 mm (engine); 2x Rheinmetall-Borsig MG 131 13 mm (cowl): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Rheinmetall-Borsig MK 108 30 mm, 3.41 of 4.31 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                2.0,
                "medium",
                "1 fragile - 5 very tough: Small and light, liquid-cooled with wing radiators; pilot armour and armoured glass; few hits needed.",
            ),
        ),
    ),
    Aircraft(
        "a6m5",
        "Zero",
        "fighter",
        "carrier fighter",
        "2x Type 99 Mk 2 20 mm (wing); 2x Type 97 7.7 mm (cowl)",
        dict(
            stall_speed=Datum(37.5, "estimated", "W/S 127 kg/m^2 with CLmax ~1.45; 135 km/h."),
            g_limit=Datum(7.0, "estimated", "light structure, IJN fighter strength requirement"),
            cruise_speed=Datum(
                130.6,
                "estimated",
                "sea-level maximum speed, military power: 565 km/h at 6,000 m less the usual ~95 km/h altitude gain for a single-stage-two-speed engine (470 km/h)",
            ),
            climb_rate=Datum(
                15.7,
                "low",
                "sea-level climb, military power: Wikipedia (Wikipedia gives 15.7 m/s for the A6M2; the A6M5 is similar (6,000 m in ~7 min))",
            ),
            service_ceiling=Datum(11740.0, "medium", "recalled standard figure, not re-checked"),
            low_speed_acceleration=Datum(
                4.374,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (4.97 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                22.5,
                "low",
                "literature: simulator/community turn-time compilations (360 deg in 16.0 s)",
            ),
            roll_rate=Datum(80.0, "low", "peak roll rate: wwiiaircraftperformance.org test report"),
            stall_angle=Datum(16.0, "estimated", "docile stall reported in US tests"),
            drag_coefficient=Datum(
                0.0219,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                126.8, "medium", "combat mass 2700 kg (medium) over wing area 21.3 m^2"
            ),
            half_span=Datum(5.5, "good", "half of the span 11.00 m"),
            muzzle_speed=Datum(
                750.0, "medium", "the Type 99 Mk 2 20 mm, 2.09 of 2.37 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                2.7,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                2.37,
                "medium",
                "2x Type 99 Mk 2 20 mm (wing); 2x Type 97 7.7 mm (cowl): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Type 99 Mk 2 20 mm, 2.09 of 2.37 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                1.0,
                "medium",
                "1 fragile - 5 very tough: No pilot armour and no self-sealing tanks on the A6M5 base model; light structure. Burns easily.",
            ),
        ),
    ),
    Aircraft(
        "me262",
        "Me 262",
        "heavy",
        "jet interceptor",
        "2x Rheinmetall-Borsig MK 108 30 mm (nose); 2x Rheinmetall-Borsig MK 108 30 mm (nose)",
        dict(
            stall_speed=Datum(
                55.6, "estimated", "W/S 295 kg/m^2 with automatic slats, CLmax ~1.45; 200 km/h."
            ),
            g_limit=Datum(7.0, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                227.8,
                "medium",
                "sea-level maximum speed, military power: recalled standard figure, not re-checked (820 km/h)",
            ),
            climb_rate=Datum(
                20.0,
                "medium",
                "sea-level climb, military power: Wikipedia (At 7,130 kg max weight)",
            ),
            service_ceiling=Datum(11450.0, "good", "Wikipedia"),
            low_speed_acceleration=Datum(
                1.76,
                "estimated",
                "the research's point-mass model at 1.1 x stall (no propeller: no correction)",
            ),
            sustained_turn_rate=Datum(
                14.1,
                "estimated",
                "the research's point-mass model (14.1 deg/s; no propeller, no correction)",
            ),
            roll_rate=Datum(
                60.0, "estimated", "peak roll rate: long span with engines out on the wing"
            ),
            stall_angle=Datum(17.0, "estimated", "slats"),
            drag_coefficient=Datum(
                0.0191,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                294.9, "medium", "combat mass 6400 kg (medium) over wing area 21.7 m^2"
            ),
            half_span=Datum(6.3, "good", "half of the span 12.60 m"),
            muzzle_speed=Datum(520.0, "medium", "the Rheinmetall-Borsig MK 108 30 mm: Wikipedia"),
            dispersion=Datum(
                1.8,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                13.64,
                "medium",
                "2x Rheinmetall-Borsig MK 108 30 mm (nose); 2x Rheinmetall-Borsig MK 108 30 mm (nose): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Rheinmetall-Borsig MK 108 30 mm: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                3.0,
                "medium",
                "1 fragile - 5 very tough: Unarmoured jets that flamed out or caught fire; pilot armour and self-sealing tanks; very vulnerable at take-off/landing",
            ),
        ),
    ),
    Aircraft(
        "ki84",
        "Ki-84",
        "fighter",
        "fighter",
        "2x Ho-5 20 mm (wing); 2x Ho-103 12.7 mm (cowl)",
        dict(
            stall_speed=Datum(44.4, "estimated", "W/S 171 kg/m^2 with CLmax ~1.4; 160 km/h."),
            g_limit=Datum(7.5, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                155.6,
                "estimated",
                "sea-level maximum speed, military power: 624 km/h at 6,120 m (Japanese data) less altitude gain (560 km/h)",
            ),
            climb_rate=Datum(
                18.5,
                "estimated",
                "sea-level climb, military power: 5,000 m in 5 min 54 s (Japanese data)",
            ),
            service_ceiling=Datum(10500.0, "medium", "Wikipedia"),
            low_speed_acceleration=Datum(
                4.488,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (5.1 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                18.74,
                "estimated",
                "the research's point-mass model (21.3 deg/s) x 0.88, its correction for propeller efficiency (no literature value)",
            ),
            roll_rate=Datum(
                90.0,
                "estimated",
                "peak roll rate: no published value found: the research's estimate",
            ),
            stall_angle=Datum(
                16.0, "estimated", "no published value found: the research's estimate"
            ),
            drag_coefficient=Datum(
                0.021,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                171.4, "medium", "combat mass 3600 kg (medium) over wing area 21.0 m^2"
            ),
            half_span=Datum(5.62, "good", "half of the span 11.24 m"),
            muzzle_speed=Datum(750.0, "medium", "the Ho-5 20 mm, 2.13 of 3.04 kg/s: Wikipedia"),
            dispersion=Datum(
                4.4,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.04,
                "medium",
                "2x Ho-5 20 mm (wing); 2x Ho-103 12.7 mm (cowl): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Ho-5 20 mm, 2.13 of 3.04 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                3.0,
                "medium",
                "1 fragile - 5 very tough: Pilot armour and self-sealing tanks, air-cooled radial; but poor build quality late in the war (landing-gear failures, e",
            ),
        ),
    ),
    Aircraft(
        "yak3",
        "Yak-3",
        "fighter",
        "low-altitude air-superiority fighter",
        "1x ShVAK 20 mm (engine); 2x Berezin UBS 12.7 mm (cowl)",
        dict(
            stall_speed=Datum(45.8, "estimated", "W/S 181 kg/m^2 with CLmax ~1.4; 165 km/h."),
            g_limit=Datum(8.0, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                157.5,
                "medium",
                "sea-level maximum speed, military power: airpages.ru (NII VVS-derived) (567 km/h)",
            ),
            climb_rate=Datum(
                21.0,
                "estimated",
                "sea-level climb, military power: 5,000 m in 4.5 min (airpages/Wikipedia)",
            ),
            service_ceiling=Datum(10400.0, "good", "Wikipedia"),
            low_speed_acceleration=Datum(
                4.083,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (4.64 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                19.5, "medium", "literature: airpages.ru (NII VVS-derived) (360 deg in 18.5 s)"
            ),
            roll_rate=Datum(100.0, "estimated", "peak roll rate: small span, light"),
            stall_angle=Datum(
                16.0, "estimated", "no published value found: the research's estimate"
            ),
            drag_coefficient=Datum(
                0.0204,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                181.1, "good", "combat mass 2690 kg (good) over wing area 14.85 m^2"
            ),
            half_span=Datum(4.6, "good", "half of the span 9.20 m"),
            muzzle_speed=Datum(
                850.0, "medium", "the Berezin UBS 12.7 mm, 1.28 of 2.48 kg/s: Wikipedia"
            ),
            dispersion=Datum(
                2.8,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                2.48,
                "medium",
                "1x ShVAK 20 mm (engine); 2x Berezin UBS 12.7 mm (cowl): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Berezin UBS 12.7 mm, 1.28 of 2.48 kg/s: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                2.0,
                "medium",
                "1 fragile - 5 very tough: Very light mixed wood/metal structure, liquid-cooled; some pilot armour, protected tanks.",
            ),
        ),
    ),
    Aircraft(
        "f4u-1d",
        "Corsair",
        "fighter",
        "carrier fighter / fighter-bomber",
        "6x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(
                45.4,
                "good",
                "wwiiaircraftperformance.org test report; 101.5 mph. Navy test F4U-1 No. 02158, normal fighter 11,194 lb, clean, power off.",
            ),
            g_limit=Datum(7.5, "medium", "recalled standard figure, not re-checked"),
            cruise_speed=Datum(
                155.6,
                "medium",
                "sea-level maximum speed, military power: wwiiaircraftperformance.org test report (348 mph)",
            ),
            climb_rate=Datum(
                16.5,
                "good",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (3250 ft/min)",
            ),
            service_ceiling=Datum(
                11800.0, "good", "wwiiaircraftperformance.org test report (38700 ft)"
            ),
            low_speed_acceleration=Datum(
                2.913,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.31 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                14.7,
                "estimated",
                "the research's point-mass model (16.7 deg/s) x 0.88, its correction for propeller efficiency (no literature value)",
            ),
            roll_rate=Datum(85.0, "medium", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(
                16.0, "estimated", "estimated; note the left-wing-drop stall (stall strip fitted)"
            ),
            drag_coefficient=Datum(
                0.0165,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                186.6, "medium", "combat mass 5443 kg (medium) over wing area 29.17 m^2"
            ),
            half_span=Datum(6.236, "good", "half of the span 12.47 m (40.92 ft)"),
            muzzle_speed=Datum(856.0, "medium", "the Browning AN/M2 .50 cal: AAF Manual 200-1"),
            dispersion=Datum(
                6.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.68, "medium", "6x Browning AN/M2 .50 cal (wing): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Browning AN/M2 .50 cal: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                4.0,
                "medium",
                "1 fragile - 5 very tough: Air-cooled radial, pilot armour, self-sealing tanks, strong carrier airframe.",
            ),
        ),
    ),
    Aircraft(
        "la7",
        "La-7",
        "fighter",
        "low/medium-altitude fighter",
        "2x ShVAK 20 mm (cowl)",
        dict(
            stall_speed=Datum(45.8, "estimated", "W/S 186 kg/m^2 with CLmax ~1.4; 165 km/h."),
            g_limit=Datum(8.0, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                160.8,
                "medium",
                "sea-level maximum speed, military power: airpages.ru (NII VVS-derived) (579 km/h)",
            ),
            climb_rate=Datum(
                18.0,
                "estimated",
                "sea-level climb, military power: 5,000 m in 5.25 min (4.6 min forced)",
            ),
            service_ceiling=Datum(10750.0, "medium", "recalled standard figure, not re-checked"),
            low_speed_acceleration=Datum(
                4.242,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (4.82 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                18.9, "medium", "literature: airpages.ru (NII VVS-derived) (360 deg in 19.0 s)"
            ),
            roll_rate=Datum(
                110.0,
                "estimated",
                "peak roll rate: metal-spar wing, light ailerons; often compared with the Fw 190",
            ),
            stall_angle=Datum(
                16.0, "estimated", "no published value found: the research's estimate"
            ),
            drag_coefficient=Datum(
                0.0205,
                "estimated",
                "back-solved so the point-mass model (propeller efficiency 0.8, Oswald 0.8) reaches the measured sea-level mili...",
            ),
            wing_loading=Datum(
                185.6, "medium", "combat mass 3265 kg (medium) over wing area 17.59 m^2"
            ),
            half_span=Datum(4.9, "good", "half of the span 9.80 m"),
            muzzle_speed=Datum(770.0, "medium", "the ShVAK 20 mm: Wikipedia"),
            dispersion=Datum(
                3.5,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                1.92, "medium", "2x ShVAK 20 mm (cowl): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the ShVAK 20 mm: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                3.0,
                "medium",
                "1 fragile - 5 very tough: Air-cooled radial and pilot armour; wooden wing structure (metal spars) less resilient than all-metal designs.",
            ),
        ),
    ),
    Aircraft(
        "me163",
        "Me 163",
        "fighter",
        "rocket point-defence interceptor",
        "2x Rheinmetall-Borsig MK 108 30 mm (wing_root)",
        dict(
            stall_speed=Datum(44.4, "estimated", "tailless, W/S ~180 kg/m^2 at burnout; 160 km/h."),
            g_limit=Datum(7.0, "estimated", "no published value found: the research's estimate"),
            cruise_speed=Datum(
                230.6,
                "estimated",
                "sea-level maximum speed, military power: Mach-limited airframe, ~960 km/h at altitude (830 km/h)",
            ),
            climb_rate=Datum(
                81.3, "medium", "sea-level climb, military power: Wikipedia (16000 ft/min)"
            ),
            service_ceiling=Datum(12100.0, "medium", "recalled standard figure, not re-checked"),
            low_speed_acceleration=Datum(
                3.01,
                "estimated",
                "the research's point-mass model at 1.1 x stall (no propeller: no correction)",
            ),
            sustained_turn_rate=Datum(
                19.3,
                "estimated",
                "the research's point-mass model (19.3 deg/s; no propeller, no correction)",
            ),
            roll_rate=Datum(120.0, "estimated", "peak roll rate: elevons, very light"),
            stall_angle=Datum(18.0, "estimated", "fixed leading-edge slots give a benign stall"),
            drag_coefficient=Datum(0.016, "estimated", "very clean tailless glider-like airframe"),
            wing_loading=Datum(178.6, "low", "combat mass 3500 kg (low) over wing area 19.6 m^2"),
            half_span=Datum(4.65, "good", "half of the span 9.30 m"),
            muzzle_speed=Datum(520.0, "medium", "the Rheinmetall-Borsig MK 108 30 mm: Wikipedia"),
            dispersion=Datum(
                2.5,
                "estimated",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                6.82,
                "medium",
                "2x Rheinmetall-Borsig MK 108 30 mm (wing_root): count x rate x projectile mass",
            ),
            fire_rate=Datum(
                10.0,
                "derived",
                "the Rheinmetall-Borsig MK 108 30 mm: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                1.0,
                "medium",
                "1 fragile - 5 very tough: Hypergolic T-Stoff/C-Stoff propellants: leaks, hard landings or hits often meant explosion; landed on a skid. Some pilot",
            ),
        ),
    ),
    Aircraft(
        "f6f-5",
        "Hellcat",
        "fighter",
        "carrier fighter",
        "6x Browning AN/M2 .50 cal (wing)",
        dict(
            stall_speed=Datum(
                43.4,
                "estimated",
                "Loftin's 84 mph (F6F-3) implies CLmax >1.8 and is a landing-configuration figure; clean power-off taken at CLm...; 97 mph.",
            ),
            g_limit=Datum(7.5, "medium", "recalled standard figure, not re-checked"),
            cruise_speed=Datum(
                142.2,
                "low",
                "sea-level maximum speed, military power: recalled standard figure, not re-checked (318 mph)",
            ),
            climb_rate=Datum(
                14.8,
                "good",
                "sea-level climb, military power: wwiiaircraftperformance.org test report (2910 ft/min)",
            ),
            service_ceiling=Datum(
                11370.0, "good", "wwiiaircraftperformance.org test report (37300 ft)"
            ),
            low_speed_acceleration=Datum(
                2.816,
                "estimated",
                "the research's point-mass model at 1.1 x stall, military power (3.2 m/s^2) x 0.88, its correction for propeller efficiency",
            ),
            sustained_turn_rate=Datum(
                15.4,
                "estimated",
                "the research's point-mass model (17.5 deg/s) x 0.88, its correction for propeller efficiency (no literature value)",
            ),
            roll_rate=Datum(80.0, "medium", "peak roll rate: NACA TR-868 via secondary summaries"),
            stall_angle=Datum(16.0, "estimated", "benign stall"),
            drag_coefficient=Datum(0.0211, "good", "Loftin, NASA SP-468"),
            wing_loading=Datum(
                175.4, "medium", "combat mass 5443 kg (medium) over wing area 31.03 m^2"
            ),
            half_span=Datum(6.527, "good", "half of the span 13.05 m (42.83 ft)"),
            muzzle_speed=Datum(856.0, "medium", "the Browning AN/M2 .50 cal: AAF Manual 200-1"),
            dispersion=Datum(
                6.0,
                "low",
                "the volley's 75% cone (mil), weight-of-fire-weighted over the gun groups, installed",
            ),
            weight_of_fire=Datum(
                3.68, "medium", "6x Browning AN/M2 .50 cal (wing): count x rate x projectile mass"
            ),
            fire_rate=Datum(
                15.0,
                "derived",
                "the Browning AN/M2 .50 cal: the whole-Hz volley rate dividing 30 nearest its rate per gun",
            ),
            survivability=Datum(
                5.0,
                "medium",
                "1 fragile - 5 very tough: Air-cooled radial, ~200 lb of armour, self-sealing tanks, rugged Grumman structure.",
            ),
        ),
    ),
    Aircraft(
        "hammerhead",
        "Hammerhead",
        "fighter",
        "Energy interceptor: a huge engine on a small wing.",
        "2x MK 108-class 30 mm @ 5/s (nose)",
        dict(
            stall_speed=Datum(
                55.88,
                "invented",
                "the P-51D's x 1.25: Energy interceptor: a huge engine on a small wing.",
            ),
            g_limit=Datum(7.0, "invented", "Energy interceptor: a huge engine on a small wing."),
            cruise_speed=Datum(
                179.0,
                "invented",
                "the P-51D's x 1.1: Energy interceptor: a huge engine on a small wing.",
            ),
            climb_rate=Datum(
                28.88,
                "invented",
                "the P-51D's x 1.9: Energy interceptor: a huge engine on a small wing.",
            ),
            service_ceiling=Datum(
                17750.0,
                "invented",
                "the P-51D's x 1.4: Energy interceptor: a huge engine on a small wing.",
            ),
            low_speed_acceleration=Datum(
                4.092,
                "invented",
                "the P-51D's x 1.5: Energy interceptor: a huge engine on a small wing.",
            ),
            sustained_turn_rate=Datum(
                11.77,
                "invented",
                "the P-51D's x 0.75: Energy interceptor: a huge engine on a small wing.",
            ),
            roll_rate=Datum(
                80.75,
                "invented",
                "the P-51D's x 0.85: Energy interceptor: a huge engine on a small wing.",
            ),
            drag_coefficient=Datum(
                0.01548,
                "invented",
                "the P-51D's x 0.95: Energy interceptor: a huge engine on a small wing.",
            ),
            wing_loading=Datum(
                311.6,
                "invented",
                "the P-51D's x 1.55: Energy interceptor: a huge engine on a small wing.",
            ),
            muzzle_speed=Datum(520.0, "invented", "the MK 108-class 30 mm"),
            dispersion=Datum(
                1.74,
                "invented",
                "the P-51D's x 0.29: Energy interceptor: a huge engine on a small wing.",
            ),
            weight_of_fire=Datum(
                3.312, "invented", "2x MK 108-class 30 mm @ 5/s (nose): the P-51D's x 0.9"
            ),
            fire_rate=Datum(5.0, "invented", "the MK 108-class 30 mm"),
            survivability=Datum(3.0, "invented", "Armoured engine bay, nothing else."),
        ),
    ),
    Aircraft(
        "wren",
        "Wren",
        "fighter",
        "Nimble brawler: very low wing loading, light guns.",
        "2x Browning .50-class 12.7 mm @ 15/s (cowl)",
        dict(
            stall_speed=Datum(
                32.18,
                "invented",
                "the P-51D's x 0.72: Nimble brawler: very low wing loading, light guns.",
            ),
            g_limit=Datum(10.0, "invented", "Nimble brawler: very low wing loading, light guns."),
            cruise_speed=Datum(
                139.9,
                "invented",
                "the P-51D's x 0.86: Nimble brawler: very low wing loading, light guns.",
            ),
            climb_rate=Datum(
                17.48,
                "invented",
                "the P-51D's x 1.15: Nimble brawler: very low wing loading, light guns.",
            ),
            service_ceiling=Datum(
                10780.0,
                "invented",
                "the P-51D's x 0.85: Nimble brawler: very low wing loading, light guns.",
            ),
            low_speed_acceleration=Datum(
                3.546,
                "invented",
                "the P-51D's x 1.3: Nimble brawler: very low wing loading, light guns.",
            ),
            sustained_turn_rate=Datum(
                22.76,
                "invented",
                "the P-51D's x 1.45: Nimble brawler: very low wing loading, light guns.",
            ),
            roll_rate=Datum(
                133.0,
                "invented",
                "the P-51D's x 1.4: Nimble brawler: very low wing loading, light guns.",
            ),
            drag_coefficient=Datum(
                0.01874,
                "invented",
                "the P-51D's x 1.15: Nimble brawler: very low wing loading, light guns.",
            ),
            wing_loading=Datum(
                110.6,
                "invented",
                "the P-51D's x 0.55: Nimble brawler: very low wing loading, light guns.",
            ),
            muzzle_speed=Datum(856.0, "invented", "the Browning .50-class 12.7 mm"),
            dispersion=Datum(
                4.02,
                "invented",
                "the P-51D's x 0.67: Nimble brawler: very low wing loading, light guns.",
            ),
            weight_of_fire=Datum(
                1.362, "invented", "2x Browning .50-class 12.7 mm @ 15/s (cowl): the P-51D's x 0.37"
            ),
            fire_rate=Datum(15.0, "invented", "the Browning .50-class 12.7 mm"),
            survivability=Datum(1.0, "invented", "No armour; Zero-like."),
        ),
    ),
    Aircraft(
        "anvil",
        "Anvil",
        "heavy",
        "Heavy gunship: one big slow cannon, armoured bathtub.",
        "1x Vickers S-class 40 mm @ 2/s (nose); 2x Browning .50-class 12.7 mm (ranging) @ 6/s (nose)",
        dict(
            stall_speed=Datum(
                51.41,
                "invented",
                "the P-51D's x 1.15: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            g_limit=Datum(6.0, "invented", "Heavy gunship: one big slow cannon, armoured bathtub."),
            cruise_speed=Datum(
                146.4,
                "invented",
                "the P-51D's x 0.9: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            climb_rate=Datum(
                11.4,
                "invented",
                "the P-51D's x 0.75: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            service_ceiling=Datum(
                10140.0,
                "invented",
                "the P-51D's x 0.8: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            low_speed_acceleration=Datum(
                2.046,
                "invented",
                "the P-51D's x 0.75: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            sustained_turn_rate=Datum(
                12.56,
                "invented",
                "the P-51D's x 0.8: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            roll_rate=Datum(
                66.5,
                "invented",
                "the P-51D's x 0.7: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            drag_coefficient=Datum(
                0.02119,
                "invented",
                "the P-51D's x 1.3: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            wing_loading=Datum(
                261.3,
                "invented",
                "the P-51D's x 1.3: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            muzzle_speed=Datum(615.0, "invented", "the Vickers S-class 40 mm, 2.16 of 2.71 kg/s"),
            dispersion=Datum(
                2.4,
                "invented",
                "the P-51D's x 0.4: Heavy gunship: one big slow cannon, armoured bathtub.",
            ),
            weight_of_fire=Datum(
                2.723,
                "invented",
                "1x Vickers S-class 40 mm @ 2/s (nose); 2x Browning .50-class 12.7 mm (ranging) @ 6/s (nose): the P-51D's x 0.74",
            ),
            fire_rate=Datum(2.0, "invented", "the Vickers S-class 40 mm, 2.16 of 2.71 kg/s"),
            survivability=Datum(
                5.0, "invented", "Armoured engine, cockpit and tanks (Il-2 style)."
            ),
        ),
    ),
    Aircraft(
        "albatross",
        "Albatross",
        "fighter",
        "Glider-like energy fighter: very low drag, long wing, weak engine.",
        "4x Browning .50-class 12.7 mm @ 10/s (wing)",
        dict(
            stall_speed=Datum(
                35.76,
                "invented",
                "the P-51D's x 0.8: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            g_limit=Datum(
                7.0,
                "invented",
                "Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            cruise_speed=Datum(
                149.7,
                "invented",
                "the P-51D's x 0.92: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            climb_rate=Datum(
                12.16,
                "invented",
                "the P-51D's x 0.8: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            service_ceiling=Datum(
                16480.0,
                "invented",
                "the P-51D's x 1.3: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            low_speed_acceleration=Datum(
                1.637,
                "invented",
                "the P-51D's x 0.6: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            sustained_turn_rate=Datum(
                15.7,
                "invented",
                "the P-51D's x 1.0: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            roll_rate=Datum(
                71.25,
                "invented",
                "the P-51D's x 0.75: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            drag_coefficient=Datum(
                0.01141,
                "invented",
                "the P-51D's x 0.7: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            wing_loading=Datum(
                150.8,
                "invented",
                "the P-51D's x 0.75: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            muzzle_speed=Datum(856.0, "invented", "the Browning .50-class 12.7 mm"),
            dispersion=Datum(
                4.5,
                "invented",
                "the P-51D's x 0.75: Glider-like energy fighter: very low drag, long wing, weak engine.",
            ),
            weight_of_fire=Datum(
                1.84, "invented", "4x Browning .50-class 12.7 mm @ 10/s (wing): the P-51D's x 0.5"
            ),
            fire_rate=Datum(10.0, "invented", "the Browning .50-class 12.7 mm"),
            survivability=Datum(2.0, "invented", "Light structure."),
        ),
    ),
    Aircraft(
        "firecracker",
        "Firecracker",
        "fighter",
        "Rocket sprinter: enormous thrust, a tiny store (propellant).",
        "2x MK 108-class 30 mm @ 5/s (wing_root)",
        dict(
            stall_speed=Datum(
                49.17,
                "invented",
                "the P-51D's x 1.1: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            g_limit=Datum(
                8.0, "invented", "Rocket sprinter: enormous thrust, a tiny store (propellant)."
            ),
            cruise_speed=Datum(
                235.9,
                "invented",
                "the P-51D's x 1.45: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            climb_rate=Datum(
                60.8,
                "invented",
                "the P-51D's x 4.0: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            service_ceiling=Datum(
                3170.0,
                "invented",
                "the P-51D's x 0.25: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            low_speed_acceleration=Datum(
                8.184,
                "invented",
                "the P-51D's x 3.0: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            sustained_turn_rate=Datum(
                15.7,
                "invented",
                "the P-51D's x 1.0: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            roll_rate=Datum(
                114.0,
                "invented",
                "the P-51D's x 1.2: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            drag_coefficient=Datum(
                0.01467,
                "invented",
                "the P-51D's x 0.9: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            wing_loading=Datum(
                241.2,
                "invented",
                "the P-51D's x 1.2: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            muzzle_speed=Datum(520.0, "invented", "the MK 108-class 30 mm"),
            dispersion=Datum(
                2.52,
                "invented",
                "the P-51D's x 0.42: Rocket sprinter: enormous thrust, a tiny store (propellant).",
            ),
            weight_of_fire=Datum(
                3.312, "invented", "2x MK 108-class 30 mm @ 5/s (wing_root): the P-51D's x 0.9"
            ),
            fire_rate=Datum(5.0, "invented", "the MK 108-class 30 mm"),
            survivability=Datum(1.0, "invented", "Explosive propellants."),
        ),
    ),
    Aircraft(
        "zerstorer",
        "Zerstörer",
        "heavy",
        "Twin-engine destroyer with a concentrated nose battery.",
        "4x Hispano-class 20 mm @ 10/s (nose); 2x Browning .50-class 12.7 mm @ 15/s (nose)",
        dict(
            stall_speed=Datum(
                53.64,
                "invented",
                "the P-51D's x 1.2: Twin-engine destroyer with a concentrated nose battery.",
            ),
            g_limit=Datum(
                6.5, "invented", "Twin-engine destroyer with a concentrated nose battery."
            ),
            cruise_speed=Datum(
                157.8,
                "invented",
                "the P-51D's x 0.97: Twin-engine destroyer with a concentrated nose battery.",
            ),
            climb_rate=Datum(
                14.44,
                "invented",
                "the P-51D's x 0.95: Twin-engine destroyer with a concentrated nose battery.",
            ),
            service_ceiling=Datum(
                12050.0,
                "invented",
                "the P-51D's x 0.95: Twin-engine destroyer with a concentrated nose battery.",
            ),
            low_speed_acceleration=Datum(
                2.592,
                "invented",
                "the P-51D's x 0.95: Twin-engine destroyer with a concentrated nose battery.",
            ),
            sustained_turn_rate=Datum(
                11.77,
                "invented",
                "the P-51D's x 0.75: Twin-engine destroyer with a concentrated nose battery.",
            ),
            roll_rate=Datum(
                57.0,
                "invented",
                "the P-51D's x 0.6: Twin-engine destroyer with a concentrated nose battery.",
            ),
            drag_coefficient=Datum(
                0.02445,
                "invented",
                "the P-51D's x 1.5: Twin-engine destroyer with a concentrated nose battery.",
            ),
            wing_loading=Datum(
                231.1,
                "invented",
                "the P-51D's x 1.15: Twin-engine destroyer with a concentrated nose battery.",
            ),
            muzzle_speed=Datum(860.0, "invented", "the Hispano-class 20 mm, 5.2 of 6.58 kg/s"),
            dispersion=Datum(
                3.18,
                "invented",
                "the P-51D's x 0.53: Twin-engine destroyer with a concentrated nose battery.",
            ),
            weight_of_fire=Datum(
                6.587,
                "invented",
                "4x Hispano-class 20 mm @ 10/s (nose); 2x Browning .50-class 12.7 mm @ 15/s (nose): the P-51D's x 1.79",
            ),
            fire_rate=Datum(10.0, "invented", "the Hispano-class 20 mm, 5.2 of 6.58 kg/s"),
            survivability=Datum(4.0, "invented", "Two engines, crew armour."),
        ),
    ),
    Aircraft(
        "shrike",
        "Shrike",
        "fighter",
        "Roll-and-dive slasher (Fw 190 vibe).",
        "4x MG 151/20-class 20 mm @ 10/s (wing_root); 2x MG 131-class 13 mm @ 10/s (cowl)",
        dict(
            stall_speed=Datum(
                50.06, "invented", "the P-51D's x 1.12: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            g_limit=Datum(9.0, "invented", "Roll-and-dive slasher (Fw 190 vibe)."),
            cruise_speed=Datum(
                167.6, "invented", "the P-51D's x 1.03: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            climb_rate=Datum(
                15.2, "invented", "the P-51D's x 1.0: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            service_ceiling=Datum(
                11410.0, "invented", "the P-51D's x 0.9: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            low_speed_acceleration=Datum(
                2.728, "invented", "the P-51D's x 1.0: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            sustained_turn_rate=Datum(
                13.82, "invented", "the P-51D's x 0.88: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            roll_rate=Datum(
                156.8, "invented", "the P-51D's x 1.65: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            drag_coefficient=Datum(
                0.01793, "invented", "the P-51D's x 1.1: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            wing_loading=Datum(
                231.1, "invented", "the P-51D's x 1.15: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            muzzle_speed=Datum(805.0, "invented", "the MG 151/20-class 20 mm, 3.68 of 4.40 kg/s"),
            dispersion=Datum(
                3.72, "invented", "the P-51D's x 0.62: Roll-and-dive slasher (Fw 190 vibe)."
            ),
            weight_of_fire=Datum(
                4.416,
                "invented",
                "4x MG 151/20-class 20 mm @ 10/s (wing_root); 2x MG 131-class 13 mm @ 10/s (cowl): the P-51D's x 1.2",
            ),
            fire_rate=Datum(10.0, "invented", "the MG 151/20-class 20 mm, 3.68 of 4.40 kg/s"),
            survivability=Datum(4.0, "invented", "Radial engine in front, armour."),
        ),
    ),
    Aircraft(
        "lantern",
        "Lantern",
        "fighter",
        "Fast, fragile skirmisher.",
        "2x Browning .50-class 12.7 mm @ 15/s (cowl)",
        dict(
            stall_speed=Datum(44.7, "invented", "the P-51D's x 1.0: Fast, fragile skirmisher."),
            g_limit=Datum(8.0, "invented", "Fast, fragile skirmisher."),
            cruise_speed=Datum(187.1, "invented", "the P-51D's x 1.15: Fast, fragile skirmisher."),
            climb_rate=Datum(16.72, "invented", "the P-51D's x 1.1: Fast, fragile skirmisher."),
            service_ceiling=Datum(
                13950.0, "invented", "the P-51D's x 1.1: Fast, fragile skirmisher."
            ),
            low_speed_acceleration=Datum(
                2.864, "invented", "the P-51D's x 1.05: Fast, fragile skirmisher."
            ),
            sustained_turn_rate=Datum(
                14.44, "invented", "the P-51D's x 0.92: Fast, fragile skirmisher."
            ),
            roll_rate=Datum(95.0, "invented", "the P-51D's x 1.0: Fast, fragile skirmisher."),
            drag_coefficient=Datum(
                0.01385, "invented", "the P-51D's x 0.85: Fast, fragile skirmisher."
            ),
            wing_loading=Datum(201.0, "invented", "the P-51D's x 1.0: Fast, fragile skirmisher."),
            muzzle_speed=Datum(856.0, "invented", "the Browning .50-class 12.7 mm"),
            dispersion=Datum(4.02, "invented", "the P-51D's x 0.67: Fast, fragile skirmisher."),
            weight_of_fire=Datum(
                1.362, "invented", "2x Browning .50-class 12.7 mm @ 15/s (cowl): the P-51D's x 0.37"
            ),
            fire_rate=Datum(15.0, "invented", "the Browning .50-class 12.7 mm"),
            survivability=Datum(1.0, "invented", "No armour, unprotected tanks."),
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class SolvedType:
    """A type as the real-scale set last held it: its key and name, its handles (as it
    flew), its solved card, and each handle's note (where the value came from)."""

    key: str
    name: str
    handles: dict[str, float]
    card: dict[str, float]
    notes: dict[str, str]


def solved_types() -> dict[str, SolvedType]:
    """Every researched or invented type as the set last held it (game v16, x2 native
    units), by key, in `AIRCRAFT`'s order."""
    out = {}
    for one in AIRCRAFT:
        data = tomllib.loads((SOLVED / f"{one.key}.toml").read_text())
        out[one.key] = SolvedType(
            one.key, data["name"], data["handles"], data["card"], data.get("notes", {})
        )
    return out

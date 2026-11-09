"""The real-scale ship set (`src/bnb/game/ships/real-scale/`): 1 m is a metre and 1 s a
second, and the fighter flies like a P-51D. `docs/explanation/game/real-scale.md` gives the
reasons; this module holds the arithmetic, and writes the set:

    uv run --no-sync python -m bnb.game.real_scale

**The rules.** Today's game maps onto a P-51D with lengths x LENGTH (5.82) and times x TIME
(4.00, the time that turns today's gravity of 27 m/s^2 into 9.81). The flight constants are
today's in those units (`similarity.scale_flight`); the bullet drag per metre is today's
over LENGTH; the bullet life is the whole ticks that give the P-51 a gun range of about
GUN_RANGE; vision is today's times MAP_SCALE; the spawn speed is SPAWN_SPEED_MARGIN times
the set's fastest stall speed. The map is MAP_SCALE (3) times today's (the presets' arena
radius of 7800 m scales the fields, zones and spawn disc by itself); the field band and the
team-deathmatch spawn distances, which do not follow the map scale, are scaled here. Rule
times follow the map, not the ship (CLOCK_SCALE, 2): crossing between fields takes about
twice as long, so the capture time doubles (the presets double the episode and respawn).

**The ships.** Each `Aircraft` gives the handles that are known for it (`handles`, SI units,
angles in radians) and the standard type it is like (`like`). Every handle it does not
give is a placeholder: its `like` type's standard handle times the factor that turns the
standard fighter into the P-51 (`placeholder_factors`: the P-51's given handle over the
standard fighter's, or for a flight handle the similarity factor of its units where the
P-51 gives none; the combat handles keep the game's values). A few
follow from others instead: the gun range from the bullet life, cruise and muzzle speed
(drag scale 1); the spawn energy at the `like` type's fraction of the store; the bullet
spread at the `like` type's dispersion angle (spread over muzzle speed). `climb_rate`
(m/s, as a climbing aircraft's rate) sets the climb time: the one whose initial store
charge, coasting at cruise, is gravity times it. Each ship file records in `[notes]` where
each value comes from. To add an aircraft, add an `Aircraft` to AIRCRAFT and run the module.
"""

import dataclasses
import math

import torch

from bnb.contracts import GameConfig

from . import ships
from .handling import FLIGHT, Handles
from .handling import solve as solve_handles
from .handling.manoeuvres import airframe_of, empty_map
from .handling.manoeuvres import ships as flying_ships
from .handling.similarity import HANDLE_UNITS, factor, scale_flight
from .physics.flight import flight_tick
from .rules import FlightConstants, Rules, rules_of
from .ship_types import FIELDS, ShipTypes

SHIP_SET = "real-scale"
STANDARD_GRAVITY = FlightConstants().gravity  # 27 m/s^2
GRAVITY = 9.81  # m/s^2
LENGTH = 5.82  # a standard metre in real metres
TIME = math.sqrt(LENGTH * STANDARD_GRAVITY / GRAVITY)  # a standard second in real ones: 4.00
MAP_SCALE = 3.0  # the map against the standard one (arena radius 7800 m against 2600 m)
CLOCK_SCALE = 2.0  # rule times: a crossing takes 3x the distance at about 1.5x the speed
GUN_RANGE = 600.0  # m, the P-51's, which sets the bullet life (whole ticks)
SPAWN_SPEED_MARGIN = 1.3  # the spawn speed over the set's fastest stall speed
CLIMB_TOLERANCE = 0.01  # the initial climb charge is met to this fraction
CLIMB_ROUNDS = 6
DEGREE = math.pi / 180.0


@dataclasses.dataclass(frozen=True)
class Aircraft:
    """One type of the set: its file's key, its name, the standard type it is like, its known
    handles (SI, radians), `climb_rate` (m/s: the climb time is solved for it; None keeps the
    placeholder climb time), and a note per given handle (its source)."""

    key: str
    name: str
    like: str
    handles: dict[str, float]
    climb_rate: float | None = None
    notes: dict[str, str] = dataclasses.field(default_factory=dict)


P51D = Aircraft(
    "fighter",
    "Fighter",
    "fighter",
    dict(
        stall_speed=46.0,
        g_limit=8.0,
        cruise_speed=163.0,
        ceiling=125_300.0,
        sustained_turn_rate=18.0 * DEGREE,
        stall_angle=16.0 * DEGREE,
        half_lift_drag=0.25,
        muzzle_speed=880.0,
        hit_radius=5.64,
        vision_range=MAP_SCALE * 1024.0,
    ),
    climb_rate=16.3,
    notes=dict(
        stall_speed="P-51D, clean, power off (100-106 mph)",
        g_limit="P-51D",
        cruise_speed="P-51D maximum level speed at sea level, military power (364 mph)",
        ceiling="P-51D service ceiling 12,770 m (41,900 ft) times g, rounded: the store, J/kg",
        climb_time="solved for an initial charge of g x 16.3 m/s (P-51D, 3,200 ft/min)",
        sustained_turn_rate="P-51D, about 18 deg/s (simulator-based)",
        stall_angle="P-51D, about 16 deg",
        half_lift_drag="a parabolic drag polar",
        muzzle_speed="the .50 M2's 880 m/s",
        hit_radius="P-51D half-span",
        vision_range="the standard 1024 m times the map scale (3)",
    ),
)
HEAVY_PLACEHOLDER = Aircraft("heavy", "Heavy", "heavy", {}, notes={})
AIRCRAFT = (P51D, HEAVY_PLACEHOLDER)
REFERENCE = P51D  # the aircraft whose given handles define the placeholder factors


def placeholder_factors(standard_fighter: dict[str, float]) -> dict[str, float]:
    """What a standard handle is multiplied by for a placeholder: the P-51's given handle
    over the standard fighter's; else, for a flight handle, the similarity factor of its
    units; else (a combat handle: damage, fire rate, shields) 1, the game's value kept."""
    return {
        name: REFERENCE.handles[name] / standard_fighter[name]
        if name in REFERENCE.handles
        else factor(units, LENGTH, TIME)
        if name in FLIGHT
        else 1.0
        for name, units in HANDLE_UNITS.items()
    }


def real_scale_flight() -> FlightConstants:
    """Today's flight constants in real units, gravity GRAVITY exactly."""
    scaled = scale_flight(FlightConstants(), LENGTH, TIME)
    return dataclasses.replace(scaled, gravity=GRAVITY)


def rule_values(stall_speeds: list[float], tick_length: float = Rules.tick_length) -> dict:
    """The set's rules.toml values (`ships.rules` keys)."""
    standard = Rules(1, 1, ())
    standard_mode = rules_of(GameConfig(1, 1, 1.0)).pools[0].mode
    bullet_drag = standard.bullet_drag / LENGTH
    launch = REFERENCE.handles["cruise_speed"] + REFERENCE.handles["muzzle_speed"]
    # The life whose range is GUN_RANGE: ln(1 + k v t) / k = R.
    life = math.expm1(bullet_drag * GUN_RANGE) / (bullet_drag * launch)
    values = {
        "rules.bullet_life": round(life / tick_length) * tick_length,
        "rules.bullet_drag": bullet_drag,
        "rules.vision_range": MAP_SCALE * standard.vision_range,
        "rules.spawn_speed": SPAWN_SPEED_MARGIN * max(stall_speeds),
        "rules.field_band_width": MAP_SCALE * standard.field_band_width,
        "mode.group_radius": MAP_SCALE * standard_mode.group_radius,
        "mode.anchor_distance_min": MAP_SCALE * standard_mode.anchor_distance_min,
        "mode.anchor_distance_max": MAP_SCALE * standard_mode.anchor_distance_max,
        "mode.spawn_margin": MAP_SCALE * standard_mode.spawn_margin,
        "mode.capture_seconds": CLOCK_SCALE * standard_mode.capture_seconds,
    }
    flight = real_scale_flight()
    values |= {
        f"flight.{field.name}": getattr(flight, field.name)
        for field in dataclasses.fields(FlightConstants)
    }
    return values


def standard_climb_charges() -> dict[str, float]:
    """The standard types' initial climb charge (W/kg), by key, under the standard rules."""
    found = ships.load()
    rules = rules_of(GameConfig(1, 1, 1.0))
    types = ShipTypes(**{name: torch.tensor([s.card[name] for s in found]) for name in FIELDS})
    cruise = torch.tensor([s.handles["cruise_speed"] for s in found])
    charge = initial_climb_charge(rules, types, cruise)
    return {ship.key: float(value) for ship, value in zip(found, charge)}


def climb_rate_of(aircraft: Aircraft) -> tuple[float, str]:
    """The climb rate (m/s) an aircraft's climb time is solved for, and its note: given, or
    its `like` type's standard initial charge scaled as the standard fighter's is to the
    P-51's."""
    if aircraft.climb_rate is not None:
        return aircraft.climb_rate, aircraft.notes.get(
            "climb_time", f"solved for an initial charge of g x {aircraft.climb_rate:g} m/s"
        )
    charges = standard_climb_charges()
    rate = REFERENCE.climb_rate * charges[aircraft.like] / charges[REFERENCE.like]
    return rate, (
        f"placeholder: solved for an initial charge of g x {rate:.3g} m/s, the standard "
        f"{aircraft.like}'s scaled as the standard fighter's is to the P-51's"
    )


def target_handles(aircraft: Aircraft, rules: Rules) -> tuple[dict[str, float], dict[str, str]]:
    """An aircraft's handles under the set's rules (the climb time a placeholder until
    `solve_set` meets its climb rate) and a note for each."""
    standard = {ship.key: ship.handles for ship in ships.load()}
    factors = placeholder_factors(standard[REFERENCE.like])
    like = standard[aircraft.like]
    values, notes = {}, {}
    for name in HANDLE_UNITS:
        if name in aircraft.handles:
            values[name] = aircraft.handles[name]
            notes[name] = aircraft.notes.get(name, "given")
        else:
            values[name] = like[name] * factors[name]
            notes[name] = (
                f"placeholder: the standard {aircraft.like}'s, scaled as the standard "
                f"{REFERENCE.like} is to the P-51"
                if factors[name] != 1.0
                else f"placeholder: the standard {aircraft.like}'s"
            )
    if "bullet_spread" not in aircraft.handles:
        values["bullet_spread"] = (
            like["bullet_spread"] * values["muzzle_speed"] / like["muzzle_speed"]
        )
        notes["bullet_spread"] = f"placeholder: the standard {aircraft.like}'s dispersion angle"
    if "spawn_energy" not in aircraft.handles:
        values["spawn_energy"] = like["spawn_energy"] / like["ceiling"] * values["ceiling"]
        notes["spawn_energy"] = f"placeholder: the standard {aircraft.like}'s fraction of the store"
    drag = rules.bullet_drag
    launch = values["cruise_speed"] + values["muzzle_speed"]
    life = rules.bullet_moves * rules.tick_length
    values["gun_range"] = math.log1p(drag * launch * life) / drag
    notes["gun_range"] = "follows from the bullet life, cruise and muzzle speed (drag scale 1)"
    notes["climb_time"] = climb_rate_of(aircraft)[1]
    return values, notes


def initial_climb_charge(rules: Rules, types: ShipTypes, cruise_speed: torch.Tensor):
    """(num_types,) the store's charge (W/kg) over the first tick of the climb: coasting
    straight at cruise from an empty store."""
    types = ShipTypes(**{name: getattr(types, name).double()[None] for name in FIELDS})
    speed = cruise_speed.double()[None]
    kinds = torch.arange(speed.shape[-1], dtype=torch.int32)[None]
    zeros = torch.zeros_like(speed)
    start = flying_ships(speed, zeros, zeros, zeros, kinds)
    end = flight_tick(
        rules.flight,
        airframe_of(types, kinds),
        rules.tick_length,
        start,
        zeros,
        zeros,
        empty_map(speed),
    )
    return (end.energy / rules.tick_length)[0]


def solve_set(aircraft=AIRCRAFT, log=print) -> tuple[list, object]:
    """Write the set's rules.toml, then solve its ships (each climb time until its initial
    charge meets its climb rate): the ships (with notes) and the last solution."""
    stall_speeds = [
        target_handles(one, rules_of(GameConfig(1, 1, 1.0)))[0]["stall_speed"] for one in aircraft
    ]
    ships.write_rules(rule_values(stall_speeds), SHIP_SET)
    rules = rules_of(GameConfig(1, 1, 7800.0, ship_set=SHIP_SET))
    found = [target_handles(one, rules) for one in aircraft]
    rates = [climb_rate_of(one)[0] for one in aircraft]
    table = {name: [values[name] for values, _ in found] for name in HANDLE_UNITS}
    start = None
    for attempt in range(CLIMB_ROUNDS):
        targets = Handles(
            **{name: torch.tensor(value, dtype=torch.float64) for name, value in table.items()}
        )
        solution = solve_handles(targets, rules, start=start, strict=False)
        start = solution.types
        charge = initial_climb_charge(rules, solution.types, targets.cruise_speed)
        done = True
        for index, one in enumerate(aircraft):
            wanted = rules.flight.gravity * rates[index]
            ratio = float(charge[index]) / wanted
            log(
                f"round {attempt}: {one.name} climb time {table['climb_time'][index]:.1f} s, "
                f"initial charge {float(charge[index]):.1f} W/kg (wanted {wanted:.1f})"
            )
            if abs(ratio - 1.0) > CLIMB_TOLERANCE:
                table["climb_time"][index] *= ratio
                done = False
        if done:
            break
    built = [
        ships.Ship(
            one.key,
            one.name,
            {name: float(table[name][index]) for name in HANDLE_UNITS},
            {name: float(getattr(solution.types, name)[index]) for name in FIELDS},
            notes,
        )
        for index, (one, (_, notes)) in enumerate(zip(aircraft, found))
    ]
    return built, solution


def main() -> None:
    """Build the set: rules.toml, then the ship files."""
    built, solution = solve_set()
    ships.write(built, SHIP_SET)
    print(f"wrote {len(built)} ships to {ships.directory_of(SHIP_SET)}")
    print(f"worst flown error by type: {solution.error.abs().amax(-1).tolist()}")
    print(f"converged: {solution.converged.tolist()}")


if __name__ == "__main__":
    main()

"""The real-scale ship set (`src/bnb/game/ships/real-scale/`): 1 m is a metre and 1 s a
second, and the types are late-war fighters (the first the P-51D) and a few invented
designs. `docs/explanation/game/real-scale.md` gives the reasons; this module holds the
arithmetic, and writes the set:

    uv run --no-sync python -m bnb.game.real_scale [--resume] [--only KEY,...]

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

**The ships.** Each `Aircraft` (`bnb.game.real_scale_aircraft`) gives researched quantities
with their confidence and source; `given_handles` converts them to handles: the ceiling's
store is g times the service ceiling, the zero-lift drag rho C_D0 / (2 W/S), the hit radius
the half-span, the bullet spread the volley's dispersion angle times the main gun's muzzle
speed, the nose reversal the P-51's (a placeholder) times the P-51's roll rate over the
type's, the shield the survivability rating (SHIELD_PER_RATING). The combat economy is a
proposal (COMBAT, `combat_economy`): a shot costs the same seconds of a type's own climb
charge as a standard shot does of the standard fighter's, at full rate; damage per second
follows the weight of fire, the P-51 keeping the standard fighter's 180 per second.

Every handle a type does not give is a placeholder: its `like` type's standard handle times
the factor that turns the standard fighter into the P-51 (`placeholder_factors`: the P-51's
given handle over the standard fighter's, or for a flight handle the similarity factor of
its units where the P-51 gives none; the other combat handles keep the game's values). The
gun range follows from the bullet life, cruise and muzzle speed (drag scale 1); the spawn
energy is the `like` type's fraction of the store. The climb rate (m/s) sets the climb
time: the one whose initial store charge, coasting at cruise, is gravity times it.

Each type is solved in its own process (`solve_set`), and its ship file written as it
finishes, so a stopped build loses at most the cards in flight (`--resume` keeps every
written card whose targets are unchanged, refreshing its notes). Each ship file records in
`[notes]` where each value comes from and how sure it is, and any handle its card misses.
"""

import argparse
import concurrent.futures
import dataclasses
import datetime
import math
import os
import time

import torch

from bnb.contracts import GameConfig

from . import ships
from .handling import FLIGHT, FLOWN, Handles, Infeasible
from .handling import solve as solve_handles
from .handling.manoeuvres import airframe_of, empty_map
from .handling.manoeuvres import ships as flying_ships
from .handling.similarity import HANDLE_UNITS, factor, scale_flight
from .physics.flight import flight_tick
from .real_scale_aircraft import AIRCRAFT as RESEARCHED
from .real_scale_aircraft import Aircraft
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
AIR_DENSITY = 1.225  # kg/m^3, sea level
# A round's 2-D normal spread: the 75% circle's radius over the per-axis sigma, sqrt(2 ln 4).
SPREAD_RADIUS_75 = math.sqrt(2.0 * math.log(4.0))
SHIELD_PER_RATING = 0.2  # the shield against the P-51's per survivability rating step
WORKERS = max(1, (os.cpu_count() or 2) - 2)
ACCEPTED = 5e-3  # the solver's acceptance: every flown handle within 0.5%


@dataclasses.dataclass(frozen=True)
class Combat:
    """The proposed combat economy (`docs/explanation/game/real-scale.md`): `fire_power`
    is continuous fire's store cost over the type's initial climb charge (None: the
    standard fighter's, about 14.5: a shot about 1 s of charge at 15 Hz), and
    `damage_per_second` the P-51's (None: the standard fighter's, 12 x 15)."""

    fire_power: float | None = None
    damage_per_second: float | None = None


COMBAT = Combat()

HEAVY = Aircraft("heavy", "Heavy", "heavy", "placeholder", "", {})
AIRCRAFT = (RESEARCHED[0], HEAVY, *RESEARCHED[1:])
REFERENCE = RESEARCHED[0]  # the P-51D: its given handles define the placeholder factors


def standard_handles() -> dict[str, dict[str, float]]:
    """The standard set's handles, by key."""
    return {ship.key: ship.handles for ship in ships.load()}


def standard_climb_charges() -> dict[str, float]:
    """The standard types' initial climb charge (W/kg), by key, under the standard rules."""
    found = ships.load()
    rules = rules_of(GameConfig(1, 1, 1.0))
    types = ShipTypes(**{name: torch.tensor([s.card[name] for s in found]) for name in FIELDS})
    cruise = torch.tensor([s.handles["cruise_speed"] for s in found])
    charge = initial_climb_charge(rules, types, cruise)
    return {ship.key: float(value) for ship, value in zip(found, charge)}


def combat_constants(combat: Combat = COMBAT) -> tuple[float, float]:
    """(fire power over climb charge, the P-51's damage per second), the standard fighter's
    where the economy leaves them out."""
    fighter = standard_handles()["fighter"]
    power = combat.fire_power
    if power is None:
        cost = fighter["ceiling"] / fighter["shots_per_store"]
        power = cost * fighter["fire_rate"] / standard_climb_charges()["fighter"]
    damage = combat.damage_per_second
    if damage is None:
        damage = fighter["hit_damage"] * fighter["fire_rate"]
    return power, damage


def kind_of(confidence: str) -> str:
    """A datum's kind in the notes: researched, estimated or invented."""
    if confidence in ("good", "medium", "low"):
        return f"researched ({confidence})"
    return confidence


def note_of(aircraft: Aircraft, quantity: str, how: str = "") -> str:
    """A handle's note from the datum it comes from (`how`: the conversion)."""
    datum = aircraft.data[quantity]
    converted = f"; {how}" if how else ""
    return f"{kind_of(datum.confidence)}: {quantity} {datum.value:g}{converted}. {datum.source}"


def reference_reversal() -> float:
    """The P-51's nose reversal time (s): a placeholder, the standard fighter's in real
    units."""
    units = HANDLE_UNITS["nose_reversal_time"]
    return standard_handles()["fighter"]["nose_reversal_time"] * factor(units, LENGTH, TIME)


def given_handles(
    aircraft: Aircraft, combat: tuple[float, float] | None = None
) -> tuple[dict[str, float], dict[str, str]]:
    """The handles an aircraft's researched quantities give, with a note for each (the climb
    rate as "climb_rate", m/s). combat: `combat_constants()` (computed when None)."""
    data = {name: datum.value for name, datum in aircraft.data.items()}
    values, notes = {}, {}

    def give(name, value, quantity, how=""):
        values[name] = value
        notes[name] = note_of(aircraft, quantity, how)

    for name in ("stall_speed", "g_limit", "cruise_speed", "low_speed_acceleration"):
        if name in data:
            give(name, data[name], name)
    if "climb_rate" in data:
        give("climb_rate", data["climb_rate"], "climb_rate")
    if "service_ceiling" in data:
        give("ceiling", GRAVITY * data["service_ceiling"], "service_ceiling", "the store g h, J/kg")
    for name in ("sustained_turn_rate", "stall_angle"):
        if name in data:
            give(name, data[name] * DEGREE, name, "degrees")
    if "roll_rate" in data and aircraft is not REFERENCE:
        reference_roll = REFERENCE.data["roll_rate"].value
        give(
            "nose_reversal_time",
            reference_reversal() * reference_roll / data["roll_rate"],
            "roll_rate",
            f"the P-51D's nose reversal (a placeholder) x its roll rate {reference_roll:g} "
            "over this one",
        )
    if "drag_coefficient" in data and "wing_loading" in data:
        give(
            "zero_lift_drag",
            0.5 * AIR_DENSITY * data["drag_coefficient"] / data["wing_loading"],
            "drag_coefficient",
            f"rho C_D0 / (2 W/S) with W/S {data['wing_loading']:.4g} kg/m^2 "
            f"({kind_of(aircraft.data['wing_loading'].confidence)})",
        )
    if "half_span" in data:
        give("hit_radius", data["half_span"], "half_span")
    if "muzzle_speed" in data:
        give("muzzle_speed", data["muzzle_speed"], "muzzle_speed")
        if "dispersion" in data:
            give(
                "bullet_spread",
                spread_of(data["dispersion"], data["muzzle_speed"]),
                "dispersion",
                f"sigma = D75 / (2 x {SPREAD_RADIUS_75:.3f}) x the muzzle speed; gun spacing "
                "ignored (nose guns, no convergence)",
            )
    if "vision_range" not in data and aircraft is REFERENCE:
        values["vision_range"] = MAP_SCALE * 1024.0
        notes["vision_range"] = "derived: the standard 1024 m times the map scale (3)"
    if "survivability" in data:
        rating = data["survivability"]
        reference_rating = REFERENCE.data["survivability"].value
        give(
            "max_shield",
            100.0 * (1.0 + SHIELD_PER_RATING * (rating - reference_rating)),
            "survivability",
            f"proposal: the standard 100 x (1 + {SHIELD_PER_RATING:g} per rating step "
            f"from the P-51D's {reference_rating:g})",
        )
    if {"weight_of_fire", "fire_rate", "climb_rate", "service_ceiling"} <= set(data):
        power, damage = combat_constants() if combat is None else combat
        economy = combat_economy(
            data["weight_of_fire"],
            data["fire_rate"],
            data["climb_rate"],
            values["ceiling"],
            REFERENCE.data["weight_of_fire"].value,
            power,
            damage,
        )
        values |= economy
        notes["fire_rate"] = note_of(aircraft, "fire_rate", "proposal: the main gun's volley rate")
        notes["hit_damage"] = note_of(
            aircraft,
            "weight_of_fire",
            f"proposal: {damage:g} per second (the P-51D's) x the weight of fire over the "
            f"P-51D's {REFERENCE.data['weight_of_fire'].value:g} kg/s, over the fire rate",
        )
        notes["shots_per_store"] = (
            f"proposal: continuous fire costs {power:.3g} x the initial climb charge "
            f"(g x {data['climb_rate']:g} m/s), the standard fighter's ratio: a shot costs "
            f"{economy['shot_cost']:.4g} J/kg, {economy['shot_seconds']:.3g} s of charge"
        )
        del values["shot_cost"], values["shot_seconds"]
    return values, notes


def spread_of(dispersion: float, muzzle_speed: float) -> float:
    """The bullet spread (m/s per axis) of a volley's 75% cone diameter (mil) at this muzzle
    speed."""
    return dispersion / 1000.0 / (2.0 * SPREAD_RADIUS_75) * muzzle_speed


def combat_economy(
    weight_of_fire: float,
    fire_rate: float,
    climb_rate: float,
    ceiling: float,
    reference_weight: float,
    fire_power: float,
    damage_per_second: float,
) -> dict[str, float]:
    """The proposed combat handles (and the shot's cost, J/kg, and its seconds of charge):
    damage per second follows the weight of fire, and continuous fire costs `fire_power`
    times the initial climb charge g x climb_rate."""
    charge = GRAVITY * climb_rate
    cost = fire_power * charge / fire_rate
    return dict(
        fire_rate=fire_rate,
        hit_damage=damage_per_second * weight_of_fire / reference_weight / fire_rate,
        shots_per_store=ceiling / cost,
        shot_cost=cost,
        shot_seconds=cost / charge,
    )


def placeholder_factors(
    standard_fighter: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    """What a standard handle is multiplied by for a placeholder: the P-51's given handle
    over the standard fighter's; else, for a flight handle, the similarity factor of its
    units; else (a combat handle: shields, recharge, head-on) 1, the game's value kept."""
    return {
        name: reference[name] / standard_fighter[name]
        if name in reference
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
    reference = REFERENCE.data
    launch = reference["cruise_speed"].value + reference["muzzle_speed"].value
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


def climb_rate_of(aircraft: Aircraft, given: dict[str, float]) -> tuple[float, str]:
    """The climb rate (m/s) an aircraft's climb time is solved for, and its note: given, or
    its `like` type's standard initial charge scaled as the standard fighter's is to the
    P-51's."""
    if "climb_rate" in given:
        rate = given["climb_rate"]
        return rate, (
            f"solved for an initial charge of g x {rate:g} m/s; " + note_of(aircraft, "climb_rate")
        )
    charges = standard_climb_charges()
    rate = REFERENCE.data["climb_rate"].value * charges[aircraft.like] / charges["fighter"]
    return rate, (
        f"placeholder: solved for an initial charge of g x {rate:.3g} m/s, the standard "
        f"{aircraft.like}'s scaled as the standard fighter's is to the P-51's"
    )


def target_handles(
    aircraft: Aircraft, rules: Rules, combat: tuple[float, float] | None = None
) -> tuple[dict[str, float], dict[str, str], float]:
    """An aircraft's handles under the set's rules (the climb time a placeholder until
    `solve_set` meets its climb rate), a note for each, and its climb rate (m/s)."""
    combat = combat_constants() if combat is None else combat
    standard = standard_handles()
    reference, _ = given_handles(REFERENCE, combat)
    factors = placeholder_factors(standard["fighter"], reference)
    like = standard[aircraft.like]
    given, given_notes = given_handles(aircraft, combat)
    values, notes = {}, {}
    for name in HANDLE_UNITS:
        if name in given:
            values[name] = given[name]
            notes[name] = given_notes[name]
        else:
            values[name] = like[name] * factors[name]
            notes[name] = (
                f"placeholder: the standard {aircraft.like}'s, scaled as the standard "
                "fighter is to the P-51D"
                if factors[name] != 1.0
                else f"placeholder: the standard {aircraft.like}'s"
            )
    if "bullet_spread" not in given:
        fighter = standard["fighter"]
        angle = reference["bullet_spread"] / reference["muzzle_speed"]
        angle *= like["bullet_spread"] / like["muzzle_speed"]
        angle /= fighter["bullet_spread"] / fighter["muzzle_speed"]
        values["bullet_spread"] = angle * values["muzzle_speed"]
        notes["bullet_spread"] = (
            f"placeholder: the standard {aircraft.like}'s dispersion angle, scaled as the "
            "standard fighter's is to the P-51D's"
        )
    if "spawn_energy" not in given:
        values["spawn_energy"] = like["spawn_energy"] / like["ceiling"] * values["ceiling"]
        notes["spawn_energy"] = f"placeholder: the standard {aircraft.like}'s fraction of the store"
    drag = rules.bullet_drag
    launch = values["cruise_speed"] + values["muzzle_speed"]
    life = rules.bullet_moves * rules.tick_length
    values["gun_range"] = math.log1p(drag * launch * life) / drag
    notes["gun_range"] = "follows from the bullet life, cruise and muzzle speed (drag scale 1)"
    rate, notes["climb_time"] = climb_rate_of(aircraft, given)
    return values, notes, rate


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


@dataclasses.dataclass
class Solved:
    """One type's solve: its ship (handles, card, notes), its flown handles' worst error,
    whether it converged, and the reasons it did not."""

    ship: ships.Ship
    worst: float
    converged: bool
    failures: list[str]


def solve_type(
    key: str,
    name: str,
    targets: dict[str, float],
    notes: dict[str, str],
    rate: float,
    rules: Rules,
    log=print,
) -> Solved:
    """One type's card (the climb time iterated until its initial charge meets g x rate),
    with a note for each handle it misses. A type no card can meet raises Infeasible."""
    table = dict(targets)
    start = None
    solution = None
    for attempt in range(CLIMB_ROUNDS):
        handles = Handles(
            **{n: torch.tensor([value], dtype=torch.float64) for n, value in table.items()}
        )
        solution = solve_handles(handles, rules, start=start, strict=False)
        start = solution.types
        charge = float(initial_climb_charge(rules, solution.types, handles.cruise_speed)[0])
        wanted = rules.flight.gravity * rate
        log(
            f"{key} round {attempt}: climb time {table['climb_time']:.1f} s, initial charge "
            f"{charge:.1f} W/kg (wanted {wanted:.1f}), worst flown error "
            f"{float(solution.error.abs().amax()):.2e}"
        )
        if abs(charge / wanted - 1.0) <= CLIMB_TOLERANCE:
            break
        table["climb_time"] *= charge / wanted
    notes = dict(notes)
    error = solution.error[0]
    for index, handle in enumerate(FLOWN):
        if float(error[index]) > ACCEPTED:
            notes[handle] += f" MISSED: the card flies it {float(error[index]):.1%} off"
    return Solved(
        ships.Ship(
            key,
            name,
            {n: float(value) for n, value in table.items()},
            {n: float(getattr(solution.types, n)[0]) for n in FIELDS},
            notes,
        ),
        float(error.abs().amax()),
        bool(solution.converged[0]),
        list(solution.failures),
    )


def _solve_in_process(arguments) -> Solved:
    """`solve_type` in a worker process, one thread, logging to stdout."""
    torch.set_num_threads(1)
    return solve_type(*arguments, log=lambda line: print(line, flush=True))


def clock(seconds_from_now: float) -> str:
    """A clock time this many seconds from now."""
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds_from_now)).strftime(
        "%H:%M"
    )


def solve_set(
    aircraft=AIRCRAFT, workers: int = WORKERS, resume: bool = False, only=None, log=print
) -> list[Solved]:
    """Write the set's rules.toml, then solve each type in its own process, writing its ship
    file as it finishes, and finally the set's order: the solves, in the set's order.
    resume: keep a written card whose targets (all but the climb time) match, its notes
    and name refreshed;
    only: solve just these keys (the others must be written already)."""
    combat = combat_constants()
    standard_rules = rules_of(GameConfig(1, 1, 1.0))
    stall_speeds = [
        target_handles(one, standard_rules, combat)[0]["stall_speed"] for one in aircraft
    ]
    ships.write_rules(rule_values(stall_speeds), SHIP_SET)
    rules = rules_of(GameConfig(1, 1, 7800.0, ship_set=SHIP_SET))
    written = {}
    directory = ships.directory_of(SHIP_SET)
    for one in aircraft:
        if (directory / f"{one.key}.toml").is_file():
            written[one.key] = ships.read(directory / f"{one.key}.toml", one.key)
    jobs, results = [], {}
    for one in aircraft:
        values, notes, rate = target_handles(one, rules, combat)
        old = written.get(one.key)
        same = old is not None and all(
            math.isclose(old.handles.get(n, math.nan), v, rel_tol=1e-9)
            for n, v in values.items()
            if n != "climb_time"
        )
        if (resume and same) or (only is not None and one.key not in only):
            if old is None:
                raise FileNotFoundError(f"{one.key} is not written yet: solve it too")
            if same:  # the notes and name may have changed: keep a MISSED mark
                missed = {
                    n: v[v.index(" MISSED") :] for n, v in old.notes.items() if " MISSED" in v
                }
                fresh = {n: note + missed.get(n, "") for n, note in notes.items()}
                old = dataclasses.replace(old, name=one.name, notes=fresh)
                ships.write_ship(old, SHIP_SET)
            results[one.key] = Solved(old, math.nan, True, [])
            continue
        if old is not None and same:
            values["climb_time"] = old.handles["climb_time"]  # a close start
        jobs.append((one.key, one.name, values, notes, rate, rules))
    log(f"solving {len(jobs)} types in {min(workers, max(len(jobs), 1))} processes")
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_solve_in_process, job): job[0] for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key = futures[future]
            try:
                solved = future.result()
            except Infeasible as error:
                log(f"{key}: INFEASIBLE: {error}")
                raise
            ships.write_ship(solved.ship, SHIP_SET)
            results[key] = solved
            elapsed = time.time() - started
            waves = math.ceil(len(jobs) / workers)
            per_wave = elapsed / math.ceil(done / workers)
            left = max(waves * per_wave - elapsed, 0.0)
            log(
                f"done {done}/{len(jobs)} {key}: worst flown error {solved.worst:.2e}, "
                f"converged {solved.converged} ({elapsed / 60:.1f} min); "
                f"ETA {clock(left)} (worst {clock(1.5 * left + 60)})"
            )
    ordered = [results[one.key] for one in aircraft]
    ships.write([solved.ship for solved in ordered], SHIP_SET)
    return ordered


def main() -> None:
    """Build the set: rules.toml, then the ship files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", action="store_true", help="keep unchanged written cards")
    parser.add_argument("--only", help="comma-separated keys to solve (the rest kept)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    only = set(arguments.only.split(",")) if arguments.only else None
    solved = solve_set(
        workers=arguments.workers,
        resume=arguments.resume,
        only=only,
        log=lambda line: print(line, flush=True),
    )
    print(f"wrote {len(solved)} ships to {ships.directory_of(SHIP_SET)}")
    for one in solved:
        print(f"{one.ship.key}: worst flown error {one.worst:.2e}, converged {one.converged}")


if __name__ == "__main__":
    main()

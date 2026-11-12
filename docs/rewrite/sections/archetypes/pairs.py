"""Archetype pairs as `sizes.py` candidates: ship A in the card table's first slot (the
Yongma row), ship B in the second (the Jujak row), each a hull and a set of factors.

A hull replaces the slot's flight fields with absolute values: "Y" and "J" are the ship
files' (the frozen cards), "M" the geometric mean of the two in flight, shield, damage and size (a neutral ship,
the same in both slots), "D2" the
balance-sizes D2 card's (Yongma's flight, Jujak's with induced drag x1.6). Factors then
multiply the hulled row (a field -> factor, or {"set": value}).

    python docs/rewrite/sections/archetypes/pairs.py round1   # writes round1/candidates.json
"""

import json
import math
import pathlib
import sys
import tomllib

HERE = pathlib.Path(__file__).parent
SHIPS = HERE.parents[3] / "src/bnb/game/ships/real-scale"
FLIGHT = (
    "max_lift_coefficient",
    "max_acceleration",
    "zero_lift_drag",
    "engine_power",
    "engine_power_full_store",
    "static_thrust",
    "max_energy",
    "corner_nose_frequency",
    "induced_drag_scale",
    "stall_angle_scale",
    "coast_exchange_scale",
    "dive_exchange_scale",
    "climb_exchange_scale",
    "bleed_exponent",
    "engine_knee_scale",
)
NEUTRAL = ("max_shield", "hit_damage", "reference_bullet_speed", "hit_radius")
CARD = {k: tomllib.loads((SHIPS / f"{k}.toml").read_text())["card"] for k in ("yongma", "jujak")}
Y, J = CARD["yongma"], CARD["jujak"]
DEG = math.radians(42.27)  # rules.head_on_angle
# The balance-sizes D2 card (factors on the frozen files).
D2_A = {"max_shield": 0.5, "bullet_life": 0.75, "head_on_fraction_scale": 0.7, "hit_damage": 1.1}
D2_B = {"induced_drag_scale": 1.6, "head_on_fraction_scale": 0.5}


def hull(name: str) -> dict[str, float]:
    if name == "Y":
        return {f: Y[f] for f in FLIGHT}
    if name == "J":
        return {f: J[f] for f in FLIGHT}
    if name == "M":  # the flight and the guns and armour: one neutral ship in both slots
        out = {f: math.sqrt(Y[f] * J[f]) for f in FLIGHT + NEUTRAL}
        out["explosive_share"] = 0.5 * (Y["explosive_share"] + J["explosive_share"])
        return out
    if name == "D2J":
        return {f: J[f] * (1.6 if f == "induced_drag_scale" else 1.0) for f in FLIGHT}
    raise ValueError(name)


def ship(slot: str, hull_name: str | None, factors: dict) -> dict:
    """The slot's card dict: the hull's flight fields set, then the factors."""
    base = Y if slot == "A" else J
    out: dict = {}
    if hull_name:
        out = {f: {"set": v} for f, v in hull(hull_name).items() if v != base[f]}
    for field, factor in factors.items():
        if isinstance(factor, dict):
            out[field] = factor
        elif field in out:
            out[field] = {"set": out[field]["set"] * factor}
        else:
            out[field] = factor
    return out


def cone(f0: float, degrees: float, p: float) -> dict:
    """The armour cone: f0 the dead-on fraction, its half angle and sharpness."""
    return {
        "head_on_fraction_scale": {"set": f0 / 0.3},
        "head_on_angle_scale": {"set": math.radians(degrees) / DEG},
        "head_on_sharpness": {"set": p},
    }


def arm(metres: float) -> dict:
    return {"arming_distance": {"set": metres}} if metres else {}


def pair(name: str, a: tuple, b: tuple) -> dict:
    """a, b: (hull or None, factors)."""
    return {"name": name, "energy": ship("A", *a), "turn": ship("B", *b)}


def merge(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in out and not isinstance(v, dict) and not isinstance(out[k], dict):
                out[k] = out[k] * v
            elif k in out and not isinstance(v, dict):  # a factor on a set value
                out[k] = {"set": out[k]["set"] * v}
            else:
                out[k] = v
    return out


def round1() -> list[dict]:
    armour = merge(cone(0.085, 12, 8), {"hit_damage": 1.5})
    slasher = {
        "hit_damage": 2.0,
        "fire_rate": {"set": 7.5},
        "bullet_energy_cost": 2.0,
        "zero_lift_drag": 0.8,
        "max_lift_coefficient": 0.85,
    }
    dogfighter = {
        "hit_damage": 0.5,
        "fire_rate": {"set": 30.0},
        "bullet_energy_cost": 0.5,
        "max_lift_coefficient": 1.1,
        "zero_lift_drag": 1.1,
    }
    gunner = {"bullet_life": 1.6, "hit_damage": 0.6}
    brawler = {"bullet_life": 0.5, "hit_damage": 1.6}
    zoomer = {"dive_exchange_scale": 2.5, "climb_exchange_scale": 1.5, "max_energy": 1.3}
    level = {"engine_power_full_store": 1.1, "dive_exchange_scale": 0.5}
    hit_and_run = {
        "shield_recharge_rate_scale": 4.0,
        "shield_recharge_delay_scale": 0.4,
        "max_shield": 0.7,
    }
    tough = {"max_shield": 1.3, "shield_recharge_rate_scale": 0.25}
    rate = {"max_lift_coefficient": 1.3, "induced_drag_scale": 1.6}
    sustained = {"max_lift_coefficient": 0.85, "induced_drag_scale": 0.75}
    m = ("M", {})
    return [
        pair("D2", (None, D2_A), (None, D2_B)),
        *[
            pair(f"D2, arming {x}", (None, merge(D2_A, arm(x))), (None, merge(D2_B, arm(x))))
            for x in (50, 150, 300)
        ],
        *[
            pair(
                f"D2 + armour 0.085 12deg p8 dmg x1.5, arming {x}",
                (None, merge(D2_A, armour, arm(x))),
                (None, merge(D2_B, armour, arm(x))),
            )
            for x in (150, 300)
        ],
        pair("frozen, arming 300", (None, arm(300)), (None, arm(300))),
        pair("same ship M", m, m),
        pair("same ship M, arming 150", ("M", arm(150)), ("M", arm(150))),
        pair("slasher v dogfighter (M)", ("M", slasher), ("M", dogfighter)),
        pair(
            "slasher v dogfighter (M), arming 150",
            ("M", merge(slasher, arm(150))),
            ("M", merge(dogfighter, arm(150))),
        ),
        pair("gunner v brawler (M)", ("M", gunner), ("M", brawler)),
        pair("gunner arming 300 v brawler (M)", ("M", merge(gunner, arm(300))), ("M", brawler)),
        pair(
            "gunner arming 300 v brawler (M), J armour",
            ("M", merge(gunner, arm(300))),
            ("M", merge(brawler, cone(0.05, 12, 8))),
        ),
        pair("zoomer v level (D2 hulls)", (None, merge(D2_A, zoomer)), (None, merge(D2_B, level))),
        pair("hit-and-run v tough (M)", ("M", hit_and_run), ("M", tough)),
        pair(
            "scout v brawler (M)",
            ("M", {"vision_range_scale": 1.6, "max_shield": 0.8}),
            ("M", {"vision_range_scale": 0.7, "hit_damage": 1.25}),
        ),
        pair(
            "armoured nose v agile (M)",
            ("M", merge(cone(0.03, 12, 8), {"induced_drag_scale": 1.3})),
            ("M", {"max_lift_coefficient": 1.1}),
        ),
        pair("rate v sustained (M)", ("M", rate), ("M", sustained)),
        pair(
            "rate v sustained (M), arming 150",
            ("M", merge(rate, arm(150))),
            ("M", merge(sustained, arm(150))),
        ),
    ]


def knobs(name: str, a: dict, b: dict, a_knobs: dict, b_knobs: dict) -> dict:
    """A pair with extra factors on each side (a_knobs, b_knobs), named after them."""
    tags = [f"A {k} x{v:g}" for k, v in a_knobs.items()] + [
        f"B {k} x{v:g}" for k, v in b_knobs.items()
    ]
    return {
        "name": name + (", " + ", ".join(tags) if tags else ""),
        "energy": merge(a, a_knobs),
        "turn": merge(b, b_knobs),
    }


def round2() -> list[dict]:
    """The mirror fix: arming and damage (and the cone) on D2 and on one neutral ship, 1v1."""
    out = [pair("D2", (None, D2_A), (None, D2_B)), pair("same ship M", ("M", {}), ("M", {}))]
    for x in (100, 150, 200):
        for d in (1.25, 1.5, 2.0):
            extra = merge(arm(x), {"hit_damage": d})
            out.append(
                pair(f"D2, arming {x}, dmg x{d:g}", (None, merge(D2_A, extra)),
                     (None, merge(D2_B, extra)))
            )
    for x, d in ((100, 1.5), (150, 1.5), (150, 2.0)):
        extra = merge(arm(x), {"hit_damage": d})
        out.append(pair(f"same ship M, arming {x}, dmg x{d:g}", ("M", extra), ("M", extra)))
    for f0, x, d in ((0.085, 100, 2.0), (0.15, 150, 2.0), (0.15, 100, 1.5)):
        extra = merge(cone(f0, 12, 8), arm(x), {"hit_damage": d})
        out.append(
            pair(f"D2, cone {f0} 12deg p8, arming {x}, dmg x{d:g}", (None, merge(D2_A, extra)),
                 (None, merge(D2_B, extra)))
        )
    return out


ROUNDS = {"round1": round1, "round2": round2}

if __name__ == "__main__":
    name = sys.argv[1]
    out = HERE / name / "candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = ROUNDS[name]()
    out.write_text(json.dumps(candidates, indent=1))
    print(f"{out}: {len(candidates)} candidates")

"""Archetype pairs as `sizes.py` candidates: ship A in the card table's first slot (the
Yongma row), ship B in the second (the Jujak row), each a hull and a set of factors.

A hull replaces the slot's flight fields with absolute values: "Y" and "J" are the ship
files' (the frozen cards), "M" the geometric mean of the two in flight, shield, damage and
size (a neutral ship, the same in both slots), "D2J" the Jujak's with induced drag x1.6.
Factors then
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
                pair(
                    f"D2, arming {x}, dmg x{d:g}",
                    (None, merge(D2_A, extra)),
                    (None, merge(D2_B, extra)),
                )
            )
    for x, d in ((100, 1.5), (150, 1.5), (150, 2.0)):
        extra = merge(arm(x), {"hit_damage": d})
        out.append(pair(f"same ship M, arming {x}, dmg x{d:g}", ("M", extra), ("M", extra)))
    for f0, x, d in ((0.085, 100, 2.0), (0.15, 150, 2.0), (0.15, 100, 1.5)):
        extra = merge(cone(f0, 12, 8), arm(x), {"hit_damage": d})
        out.append(
            pair(
                f"D2, cone {f0} 12deg p8, arming {x}, dmg x{d:g}",
                (None, merge(D2_A, extra)),
                (None, merge(D2_B, extra)),
            )
        )
    return out


SLASHER = {
    "hit_damage": 2.0,
    "fire_rate": {"set": 7.5},
    "bullet_energy_cost": 2.0,
    "zero_lift_drag": 0.8,
    "max_lift_coefficient": 0.85,
}
DOGFIGHTER = {
    "hit_damage": 0.5,
    "fire_rate": {"set": 30.0},
    "bullet_energy_cost": 0.5,
    "max_lift_coefficient": 1.1,
    "zero_lift_drag": 1.1,
}


def round3() -> list[dict]:
    """Balance grids, every size: slasher v dogfighter (vision and shield on the slasher, the
    dogfighter's bleed), gunner v brawler (the damage split), scout v brawler, a milder rate v
    sustained pair, and a sniper v shotgun pair (spread, fire rate, range)."""
    out = []
    for av, ash, bi in (
        (1.4, 0.8, 1),
        (1.4, 0.65, 1),
        (1.8, 0.65, 1),
        (1, 0.65, 1.3),
        (1.4, 0.75, 1.3),
        (1.8, 0.8, 1.3),
    ):
        out.append(
            pair(
                f"slasher v dogfighter (M), S vision x{av:g} shield x{ash:g}, D induced x{bi:g}",
                ("M", merge(SLASHER, {"vision_range_scale": av, "max_shield": ash})),
                ("M", merge(DOGFIGHTER, {"induced_drag_scale": bi})),
            )
        )
    for gl, gd, bl, bd in (
        (1.6, 0.8, 0.5, 1.3),
        (1.6, 0.9, 0.5, 1.2),
        (1.6, 1.0, 0.5, 1.3),
        (1.3, 0.85, 0.6, 1.3),
    ):
        out.append(
            pair(
                f"gunner v brawler (M), G life x{gl:g} dmg x{gd:g}, B life x{bl:g} dmg x{bd:g}",
                ("M", {"bullet_life": gl, "hit_damage": gd}),
                ("M", {"bullet_life": bl, "hit_damage": bd}),
            )
        )
    for sv, ssh, bv, bd, bsh in (
        (1.3, 1.0, 1.0, 1.25, 1.0),
        (1.3, 0.9, 0.8, 1.15, 1.0),
        (1.6, 1.0, 1.0, 1.25, 1.1),
    ):
        out.append(
            pair(
                f"scout v brawler (M), S vision x{sv:g} shield x{ssh:g}, "
                f"B vision x{bv:g} dmg x{bd:g} shield x{bsh:g}",
                ("M", {"vision_range_scale": sv, "max_shield": ssh}),
                ("M", {"vision_range_scale": bv, "hit_damage": bd, "max_shield": bsh}),
            )
        )
    for cl, ind, rd, scl, sind in (
        (1.2, 1.3, 1.0, 0.9, 0.8),
        (1.2, 1.3, 1.25, 0.9, 0.8),
        (1.3, 1.4, 1.25, 0.85, 0.75),
    ):
        out.append(
            pair(
                f"rate CL x{cl:g} ind x{ind:g} dmg x{rd:g} v sustained "
                f"CL x{scl:g} ind x{sind:g} (M)",
                ("M", {"max_lift_coefficient": cl, "induced_drag_scale": ind, "hit_damage": rd}),
                ("M", {"max_lift_coefficient": scl, "induced_drag_scale": sind}),
            )
        )
    sniper = {
        "bullet_spread_scale": 0.3,
        "fire_rate": {"set": 7.5},
        "hit_damage": 2.0,
        "bullet_energy_cost": 2.0,
        "bullet_life": 1.4,
    }
    shotgun = {
        "bullet_spread_scale": 3.0,
        "fire_rate": {"set": 30.0},
        "hit_damage": 0.6,
        "bullet_energy_cost": 0.5,
        "bullet_life": 0.6,
    }
    out.append(pair("sniper v shotgun (M)", ("M", sniper), ("M", shotgun)))
    out.append(
        pair(
            "sniper v shotgun (M), shotgun dmg x0.75",
            ("M", sniper),
            ("M", merge(shotgun, {"hit_damage": 0.75 / 0.6})),
        )
    )
    return out


ROUNDS = {"round1": round1, "round2": round2, "round3": round3}


def mirror_study() -> list[dict]:
    """One ship a candidate (slot A, played as the A mirror at 1v1): what shapes a mirror.
    The neutral ship M with single changes, and the D2 Yongma with its parts removed."""
    singles = {
        "vision x0.7": {"vision_range_scale": 0.7},
        "vision x1.3": {"vision_range_scale": 1.3},
        "drag x0.8 (faster)": {"zero_lift_drag": 0.8},
        "CL x1.15": {"max_lift_coefficient": 1.15},
        "induced x0.8": {"induced_drag_scale": 0.8},
        "induced x1.25": {"induced_drag_scale": 1.25},
        "explosive 0.5": {"explosive_share": {"set": 0.5}},
        "bullet life x0.7": {"bullet_life": 0.7},
        "bullet life x1.4": {"bullet_life": 1.4},
        "damage x1.5": {"hit_damage": 1.5},
        "shield x0.6": {"max_shield": 0.6},
        "fire 30 Hz dmg x0.5": {"fire_rate": {"set": 30.0}, "hit_damage": 0.5},
        "fire 7.5 Hz dmg x2": {"fire_rate": {"set": 7.5}, "hit_damage": 2.0},
        "spread x3": {"bullet_spread_scale": 3.0},
        "head-on fraction x0.5": {"head_on_fraction_scale": 0.5},
        "head-on angle x2": {"head_on_angle_scale": 2.0},
        "hit radius x1.5": {"hit_radius": 1.5},
        "engine x1.3": {"engine_power": 1.3, "engine_power_full_store": 1.3},
        "bleed exponent 1.5": {"bleed_exponent": {"set": 1.5}},
        "stall angle x1.2": {"stall_angle_scale": 1.2},
    }
    out = [pair("M", ("M", {}), ("M", {}))]
    out += [pair(f"M, {k}", ("M", v), ("M", v)) for k, v in singles.items()]
    d2y = {
        "D2 Yongma": {},
        "D2 Yongma, explosive 0": {"explosive_share": {"set": 0.0}},
        "D2 Yongma, bullet life x1.33": {"bullet_life": 1 / 0.75},
        "D2 Yongma, head-on fraction x1.43": {"head_on_fraction_scale": 1 / 0.7},
        "D2 Yongma, shield x2": {"max_shield": 2.0},
        "frozen Yongma": None,
    }
    for k, v in d2y.items():
        a = (None, {}) if v is None else (None, merge(D2_A, v))
        out.append(pair(k, a, a))
    return out


ROUNDS["mirror"] = mirror_study


def mirror_study_2() -> list[dict]:
    """Vision between x0.7 and 1, alone and with damage, on M and on D2's two ships."""
    out = []
    for v in (0.75, 0.8, 0.85, 0.9):
        for d in (1.0, 1.3):
            f = {"vision_range_scale": v, "hit_damage": d}
            out.append(pair(f"M, vision x{v:g}, dmg x{d:g}", ("M", f), ("M", f)))
    for v in (0.7, 0.8, 0.9):
        vision = {"vision_range_scale": v}
        out.append(
            pair(f"D2, vision x{v:g}", (None, merge(D2_A, vision)), (None, merge(D2_B, vision)))
        )
    return out


ROUNDS["mirror2"] = mirror_study_2

GUNNER = {"bullet_life": 1.6}
BRAWLER = {"bullet_life": 0.5}
SNIPER = {
    "bullet_spread_scale": 0.3,
    "fire_rate": {"set": 7.5},
    "hit_damage": 2.0,
    "bullet_energy_cost": 2.0,
    "bullet_life": 1.4,
}
SHOTGUN = {
    "bullet_spread_scale": 3.0,
    "fire_rate": {"set": 30.0},
    "hit_damage": 0.6,
    "bullet_energy_cost": 0.5,
    "bullet_life": 0.6,
}


def round4() -> list[dict]:
    """Two levers a pair: the sustained turn (induced drag, the duel lever) and damage,
    shield or sight (the crowd levers)."""
    out = []
    for gd, bd, gi, bi, gv in (
        (0.9, 1.2, 1, 1.25, 1),
        (0.9, 1.2, 1, 1.45, 1),
        (0.85, 1.2, 1, 1.45, 1),
        (0.9, 1.2, 0.85, 1, 1),
        (0.9, 1.2, 1, 1, 1.25),
        (0.85, 1.25, 1, 1.25, 1.2),
    ):
        out.append(
            pair(
                f"gunner v brawler (M), G dmg x{gd:g} induced x{gi:g} vision x{gv:g}, "
                f"B dmg x{bd:g} induced x{bi:g}",
                (
                    "M",
                    merge(
                        GUNNER,
                        {"hit_damage": gd, "induced_drag_scale": gi, "vision_range_scale": gv},
                    ),
                ),
                ("M", merge(BRAWLER, {"hit_damage": bd, "induced_drag_scale": bi})),
            )
        )
    for sd, bi, si in ((1, 1.25, 1), (1, 1.45, 1), (0.9, 1.45, 1), (1, 1, 0.85)):
        out.append(
            pair(
                f"sniper v shotgun (M), sniper dmg x{sd:g} induced x{si:g}, "
                f"shotgun induced x{bi:g}",
                ("M", merge(SNIPER, {"hit_damage": sd, "induced_drag_scale": si})),
                ("M", merge(SHOTGUN, {"induced_drag_scale": bi})),
            )
        )
    for sv, ssh, bv, bd in (
        (1.3, 1.1, 0.8, 1.15),
        (1.3, 1.0, 0.8, 1.0),
        (1.2, 1.0, 0.85, 1.1),
        (1.15, 1.0, 1.0, 1.15),
    ):
        out.append(
            pair(
                f"scout v brawler (M), S vision x{sv:g} shield x{ssh:g}, "
                f"B vision x{bv:g} dmg x{bd:g}",
                ("M", {"vision_range_scale": sv, "max_shield": ssh}),
                ("M", {"vision_range_scale": bv, "hit_damage": bd}),
            )
        )
    for si, ssh in ((0.8, 0.8), (0.8, 0.65), (0.7, 0.65), (0.7, 0.5)):
        out.append(
            pair(
                f"slasher v dogfighter (M), S induced x{si:g} shield x{ssh:g}",
                ("M", merge(SLASHER, {"induced_drag_scale": si, "max_shield": ssh})),
                ("M", DOGFIGHTER),
            )
        )
    return out


ROUNDS["round4"] = round4


def mirror_study_3() -> list[dict]:
    """Short sight (flank kills) with damage for the pace, and the cone for the joust, on M."""
    out = []
    for v, d, f0 in (
        (0.8, 1.6, None),
        (0.8, 2.0, None),
        (0.85, 1.6, None),
        (0.8, 1.6, 0.15),
        (0.8, 2.0, 0.15),
        (0.8, 2.0, 0.1),
        (0.85, 2.0, 0.1),
        (0.9, 2.0, 0.1),
        (1.0, 2.0, 0.1),
        (1.0, 2.0, 0.15),
    ):
        f = merge({"vision_range_scale": v, "hit_damage": d}, cone(f0, 12, 8) if f0 else {})
        name = f"M, vision x{v:g}, dmg x{d:g}" + (f", cone {f0:g} 12deg p8" if f0 else "")
        out.append(pair(name, ("M", f), ("M", f)))
    return out


ROUNDS["mirror3"] = mirror_study_3


def kit(vision: float, f0: float, damage: float) -> dict:
    """The mirror kit: sight, the 12-degree p8 armour cone at f0, and damage."""
    return merge({"vision_range_scale": vision, "hit_damage": damage}, cone(f0, 12, 8))


def mirror_study_4() -> list[dict]:
    """The kit pushed toward a 300 s mirror with little joust, on M and on D2."""
    out = []
    for v, f0, d in (
        (1.0, 0.1, 2.5),
        (0.9, 0.1, 2.5),
        (0.9, 0.12, 2.0),
        (0.9, 0.08, 2.5),
        (1.0, 0.07, 2.5),
        (0.85, 0.1, 2.5),
        (0.9, 0.1, 3.0),
        (0.85, 0.08, 3.0),
    ):
        out.append(
            pair(
                f"M, kit vision x{v:g} f0 {f0:g} dmg x{d:g}",
                ("M", kit(v, f0, d)),
                ("M", kit(v, f0, d)),
            )
        )
    for v, f0, d in ((1.0, 0.1, 2.0), (0.9, 0.1, 2.0), (0.9, 0.1, 2.5)):
        out.append(
            pair(
                f"D2, kit vision x{v:g} f0 {f0:g} dmg x{d:g}",
                (None, merge(D2_A, kit(v, f0, d))),
                (None, merge(D2_B, kit(v, f0, d))),
            )
        )
    return out


ROUNDS["mirror4"] = mirror_study_4


KIT = (0.9, 0.08, 2.5)  # mirror study 4's pick: sight x0.9, cone f0 0.08, damage x2.5


def round5() -> list[dict]:
    """Refinements of round 4's live pairs (scout v brawler, sniper v shotgun, slasher v
    dogfighter, gunner v brawler), each also with the mirror kit on both ships."""
    out = []

    def add(name, a, b, with_kit=False):
        out.append(pair(name, ("M", a), ("M", b)))
        if with_kit:
            out.append(pair(name + ", kit", ("M", merge(a, kit(*KIT))), ("M", merge(b, kit(*KIT)))))

    for sv, bv, bd, k in (
        (1.15, 0.85, 1.1, True),
        (1.1, 0.85, 1.1, False),
        (1.2, 0.85, 1.0, False),
        (1.15, 0.9, 1.1, True),
    ):
        add(
            f"scout v brawler (M), S vision x{sv:g}, B vision x{bv:g} dmg x{bd:g}",
            {"vision_range_scale": sv},
            {"vision_range_scale": bv, "hit_damage": bd},
            k,
        )
    for si, sd in ((0.85, 0.9), (0.9, 1.0)):
        add(
            f"sniper v shotgun (M), sniper induced x{si:g} dmg x{sd:g}",
            merge(SNIPER, {"induced_drag_scale": si, "hit_damage": sd}),
            SHOTGUN,
            True,
        )
    for si, ssh in ((0.8, 0.55), (0.85, 0.65)):
        add(
            f"slasher v dogfighter (M), S induced x{si:g} shield x{ssh:g}",
            merge(SLASHER, {"induced_drag_scale": si, "max_shield": ssh}),
            DOGFIGHTER,
            True,
        )
    for gi in (0.92, 0.95):
        add(
            f"gunner v brawler (M), G dmg x0.9 induced x{gi:g} vision x1.25, B dmg x1.2",
            merge(
                GUNNER, {"hit_damage": 0.9, "induced_drag_scale": gi, "vision_range_scale": 1.25}
            ),
            merge(BRAWLER, {"hit_damage": 1.2}),
        )
    return out


ROUNDS["round5"] = round5


def screen1() -> list[dict]:
    """New pair ideas at 1v1 and 5v5, each with the mirror kit on both ships."""
    ideas = {
        "interceptor v brawler": (
            {
                "zero_lift_drag": 0.8,
                "engine_power_full_store": 1.15,
                "vision_range_scale": 1.2,
                "hit_damage": 0.8,
            },
            {"zero_lift_drag": 1.15, "vision_range_scale": 0.85, "hit_damage": 1.2},
        ),
        "big gun big target v small light": (
            {"hit_radius": 1.4, "hit_damage": 1.4},
            {"hit_radius": 0.75, "hit_damage": 0.85},
        ),
        "glass cannon v steady": (
            {"max_shield": 0.6, "hit_damage": 1.5},
            {"max_shield": 1.3, "hit_damage": 0.8},
        ),
        "armed gunner v close fighter": (
            {"arming_distance": {"set": 150.0}, "hit_damage": 1.4, "bullet_life": 1.3},
            {"bullet_life": 0.8},
        ),
        "armed gunner (dmg x1.2) v close fighter": (
            {"arming_distance": {"set": 150.0}, "hit_damage": 1.2, "bullet_life": 1.3},
            {"bullet_life": 0.8},
        ),
        "wide nose armour v agile": (
            merge(cone(0.3, 30, 4), {"induced_drag_scale": 1.1}),
            {"max_lift_coefficient": 1.1},
        ),
        "hungry stream v efficient bursts": (
            {"fire_rate": {"set": 30.0}, "hit_damage": 0.55, "bullet_energy_cost": 1.5},
            {"fire_rate": {"set": 10.0}, "hit_damage": 1.5, "bullet_energy_cost": 1.0},
        ),
        "fast recharge v big shield": (
            {
                "shield_recharge_rate_scale": 3.0,
                "shield_recharge_delay_scale": 0.5,
                "max_shield": 0.85,
            },
            {"max_shield": 1.15, "shield_recharge_rate_scale": 0.5},
        ),
    }
    out = []
    for name, (a, b) in ideas.items():
        out.append(pair(name + " (M), kit", ("M", merge(a, kit(*KIT))), ("M", merge(b, kit(*KIT)))))
    return out


ROUNDS["screen1"] = screen1


def round6() -> list[dict]:
    """Screen 1's live ideas at every size, with the kit: big gun big target v small light,
    the armed gunner v the close fighter, the interceptor v the brawler."""
    out = []

    def add(name, a, b):
        out.append(pair(name + ", kit", ("M", merge(a, kit(*KIT))), ("M", merge(b, kit(*KIT)))))

    for br, bd, sr, sd in (
        (1.4, 1.4, 0.75, 0.85),
        (1.4, 1.4, 0.75, 0.95),
        (1.4, 1.4, 0.8, 0.9),
        (1.3, 1.3, 0.8, 0.9),
        (1.5, 1.5, 0.75, 0.9),
    ):
        add(
            f"big (radius x{br:g}, dmg x{bd:g}) v small (radius x{sr:g}, dmg x{sd:g}) (M)",
            {"hit_radius": br, "hit_damage": bd},
            {"hit_radius": sr, "hit_damage": sd},
        )
    for gd, gv in ((1.1, 1.15), (1.2, 1.15), (1.0, 1.2)):
        add(
            f"armed gunner (arming 150, life x1.3, dmg x{gd:g}, vision x{gv:g}) v close fighter "
            "(life x0.8) (M)",
            {
                "arming_distance": {"set": 150.0},
                "hit_damage": gd,
                "bullet_life": 1.3,
                "vision_range_scale": gv,
            },
            {"bullet_life": 0.8},
        )
    for iv, idmg in ((1.0, 0.8), (1.0, 0.9)):
        add(
            f"interceptor (vision x{iv:g}, dmg x{idmg:g}) v brawler (M)",
            {
                "zero_lift_drag": 0.8,
                "engine_power_full_store": 1.15,
                "vision_range_scale": iv,
                "hit_damage": idmg,
            },
            {"zero_lift_drag": 1.15, "vision_range_scale": 0.85, "hit_damage": 1.2},
        )
    return out


ROUNDS["round6"] = round6


KIT2 = (1.0, 0.08, 2.5)  # the kit without the short sight


def round7() -> list[dict]:
    """Scout v brawler, sniper v shotgun and slasher v dogfighter with the kit at full sight
    (KIT2), rebalanced."""
    out = []

    def add(name, a, b, k=KIT2, tag="kit2"):
        out.append(pair(f"{name}, {tag}", ("M", merge(a, kit(*k))), ("M", merge(b, kit(*k)))))

    for sv, bv, bd in ((1.15, 0.9, 1.1), (1.1, 0.9, 1.15), (1.05, 0.9, 1.1), (1.1, 0.95, 1.1)):
        add(
            f"scout v brawler (M), S vision x{sv:g}, B vision x{bv:g} dmg x{bd:g}",
            {"vision_range_scale": sv},
            {"vision_range_scale": bv, "hit_damage": bd},
        )
    for si, sd in ((0.9, 1.0), (0.95, 1.0), (0.95, 0.95)):
        add(
            f"sniper v shotgun (M), sniper induced x{si:g} dmg x{sd:g}",
            merge(SNIPER, {"induced_drag_scale": si, "hit_damage": sd}),
            SHOTGUN,
        )
    for si, ssh in ((0.8, 0.45), (0.85, 0.5)):
        add(
            f"slasher v dogfighter (M), S induced x{si:g} shield x{ssh:g}",
            merge(SLASHER, {"induced_drag_scale": si, "max_shield": ssh}),
            DOGFIGHTER,
            KIT,
            "kit",
        )
    return out


ROUNDS["round7"] = round7


if __name__ == "__main__":
    name = sys.argv[1]
    out = HERE / name / "candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = ROUNDS[name]()
    out.write_text(json.dumps(candidates, indent=1))
    print(f"{out}: {len(candidates)} candidates")

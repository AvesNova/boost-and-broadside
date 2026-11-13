"""Yongma and Jujak cards under r9m2 as maneuvers-2 `sizes.py` candidates.

    python docs/rewrite/sections/ships-r9m2/make.py round0

A card is factors on the frozen ship files (a field -> factor, or {"set": value}; the Yongma
slot's on yongma.toml, the Jujak slot's on jujak.toml). Each card is flown by r9m2
(maneuvers-2's `champion2.json`) with the Jujak slot's duel-extension gate at 0.7 ("g0.7")
or 1.0 ("g1.0"); r9m2 itself has 0.4. Writes `<round>/candidates.json`.
"""

import json
import math
import pathlib
import sys
import tomllib

HERE = pathlib.Path(__file__).parent
SHIPS = HERE.parents[3] / "src/bnb/game/ships/real-scale"
CARD = {k: tomllib.loads((SHIPS / f"{k}.toml").read_text())["card"] for k in ("yongma", "jujak")}
Y, J = CARD["yongma"], CARD["jujak"]
# the archetypes' neutral hull M: the geometric mean of the two in flight, shield, damage, size
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
# The balance-sizes D2 card (factors on the frozen files).
D2_Y = {"max_shield": 0.5, "bullet_life": 0.75, "head_on_fraction_scale": 0.7, "hit_damage": 1.1}
D2_J = {"induced_drag_scale": 1.6, "head_on_fraction_scale": 0.5}


def middle(factors: dict | None = None, slot: str = "yongma") -> dict:
    """The neutral ship M on a slot's base row, then factors."""
    base = Y if slot == "yongma" else J
    out = {f: {"set": math.sqrt(Y[f] * J[f])} for f in FLIGHT + NEUTRAL}
    out["explosive_share"] = {"set": 0.5 * (Y["explosive_share"] + J["explosive_share"])}
    out = {f: v for f, v in out.items() if v["set"] != base[f]}
    for field, factor in (factors or {}).items():
        out[field] = (
            ({"set": out[field]["set"] * factor} if field in out else factor)
            if not isinstance(factor, dict)
            else factor
        )
    return out


def doctrine(gate: float) -> dict:
    """r9m2 with the Jujak slot's duel-extension gate at `gate`."""
    champion = json.loads((HERE.parent / "maneuvers-2" / "champion2.json").read_text())
    centre, scale = champion["jujak"]["duel_extend_in_shield_ratio"]
    return dict(
        doctrine=dict(
            energy=champion["yongma"],
            turn=dict(champion["jujak"], duel_extend_in_shield_ratio=[gate, scale]),
        ),
        parameters={str(i): v for i, v in enumerate(champion["parameters"])},
    )


def candidate(name: str, energy: dict, turn: dict, gate: float = 0.7) -> dict:
    return dict(name=f"{name} | g{gate}", energy=energy, turn=turn, **doctrine(gate))


def round0() -> list[dict]:
    return [
        candidate("frozen", {}, {}, 0.7),
        candidate("frozen", {}, {}, 1.0),
        candidate("d2", D2_Y, D2_J, 0.7),
        candidate("d2", D2_Y, D2_J, 1.0),
    ]


def flight_grid() -> list[dict]:
    """Jujak's induced drag and Yongma's flight, for the turn numbers only."""
    out = [candidate("frozen", {}, {})]
    for drag in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
        out.append(candidate(f"J drag x{drag}", {}, {"induced_drag_scale": drag}))
    out.append(candidate("M", middle(), middle(slot="jujak")))
    return out


def round1() -> list[dict]:
    """The first screen: Jujak's induced drag (x1.3: +10% sustained turn over Yongma; x1.4:
    +5%) with Yongma's damage, shield, range and dive exchange."""
    cards = [
        ("Jd1.3", {}, {"induced_drag_scale": 1.3}),
        ("Jd1.4", {}, {"induced_drag_scale": 1.4}),
        ("Jd1.3 Ydmg1.2", {"hit_damage": 1.2}, {"induced_drag_scale": 1.3}),
        ("Jd1.4 Ydmg1.2", {"hit_damage": 1.2}, {"induced_drag_scale": 1.4}),
        ("Jd1.3 Ysh1.3", {"max_shield": 1.3}, {"induced_drag_scale": 1.3}),
        (
            "Jd1.4 Ydmg1.15 Ysh1.15",
            {"hit_damage": 1.15, "max_shield": 1.15},
            {"induced_drag_scale": 1.4},
        ),
        ("Jd1.3 Ydive1.5", {"dive_exchange_scale": 1.5}, {"induced_drag_scale": 1.3}),
        ("Jd1.3 Yrange1.15", {"bullet_life": 1.15}, {"induced_drag_scale": 1.3}),
        ("Jd1.2 Ydmg1.3", {"hit_damage": 1.3}, {"induced_drag_scale": 1.2}),
        ("Ydmg1.3", {"hit_damage": 1.3}, {}),
    ]
    return [candidate(name, e, t, 0.7) for name, e, t in cards]


def cone(scale: float) -> dict:
    """The nose cone's dead-on fraction as a factor on the frozen 0.3 (lower: harder)."""
    return {"head_on_fraction_scale": scale}


def round2() -> list[dict]:
    """The second screen: Yongma's nose cone (a duel lever) on Jujak drag x1.3-1.4, with
    damage, Jujak's cone and Yongma's shield."""
    j14, j135, j13 = ({"induced_drag_scale": d} for d in (1.4, 1.35, 1.3))
    cards = [
        ("Jd1.4 Yf0.6", cone(0.6), j14),
        ("Jd1.4 Yf0.4", cone(0.4), j14),
        ("Jd1.4 Yf0.25", cone(0.25), j14),
        ("Jd1.35 Yf0.4", cone(0.4), j135),
        ("Jd1.4 Yf0.4 Ydmg1.15", cone(0.4) | {"hit_damage": 1.15}, j14),
        ("Jd1.4 Yf0.4 Jf1.3", cone(0.4), j14 | cone(1.3)),
        ("Jd1.3 Yf0.25 Ydmg1.15", cone(0.25) | {"hit_damage": 1.15}, j13),
        ("Jd1.4 Yf0.4 Ysh0.85", cone(0.4) | {"max_shield": 0.85}, j14),
        ("Jd1.4 Yf0.6 Ydmg1.3", cone(0.6) | {"hit_damage": 1.3}, j14),
        ("Yf0.25", cone(0.25), {}),
    ]
    return [candidate(name, e, t, 0.7) for name, e, t in cards]


def round3() -> list[dict]:
    """Yongma's lift (its instantaneous turn only: x1.2 is 0.623 rad/s against Jujak's 0.661)
    as a duel lever, and its dive exchange (x1.5) as a crowd counter, on Jujak drag x1.3-1.4."""
    j14, j13 = ({"induced_drag_scale": d} for d in (1.4, 1.3))
    lift = lambda f: {"max_lift_coefficient": f}  # noqa: E731
    dive = {"dive_exchange_scale": 1.5}
    cards = [
        ("Jd1.3 Yl1.2", lift(1.2), j13),
        ("Jd1.4 Yl1.2", lift(1.2), j14),
        ("Jd1.4 Yl1.1", lift(1.1), j14),
        ("Jd1.4 Yl1.2 Ydmg1.2", lift(1.2) | {"hit_damage": 1.2}, j14),
        ("Jd1.4 Ydive1.5 Ydmg1.3", dive | {"hit_damage": 1.3}, j14),
        ("Jd1.4 Ydive1.5", dive, j14),
        ("Jd1.4 Yl1.2 Ydive1.5", lift(1.2) | dive, j14),
        ("Jd1.3 Yl1.2 Ydmg1.2", lift(1.2) | {"hit_damage": 1.2}, j13),
        ("Jd1.4 Yl1.2 Ydmg1.3", lift(1.2) | {"hit_damage": 1.3}, j14),
        ("Jd1.4 Yl1.2 Ydmg1.3 Ydive1.5", lift(1.2) | dive | {"hit_damage": 1.3}, j14),
    ]
    return [candidate(name, e, t, 0.7) for name, e, t in cards]


def lift_card(
    drag: float = 1.4, lift: float = 1.2, damage: float = 1.3, dive: float = 1.5, **more
) -> tuple[dict, dict]:
    """Round 3's best: Jujak drag, Yongma lift, damage and dive exchange (and more Yongma
    factors)."""
    energy = {"max_lift_coefficient": lift, "hit_damage": damage, "dive_exchange_scale": dive}
    return energy | more, {"induced_drag_scale": drag}


def round4() -> list[dict]:
    """Refine round 3's Jd1.4 Yl1.2 Ydmg1.3 Ydive1.5 (1v1 0.45, 5v5 0.53), all sizes."""
    cards = [
        ("A Jd1.4 Yl1.2 Ydmg1.3 Ydive1.5", lift_card(), 0.7),
        ("A Jd1.4 Yl1.2 Ydmg1.3 Ydive1.5", lift_card(), 1.0),
        ("Jd1.4 Yl1.2 Ydmg1.3 Ydive1.3", lift_card(dive=1.3), 0.7),
        ("Jd1.4 Yl1.2 Ydmg1.3 Ydive1.4", lift_card(dive=1.4), 0.7),
        ("Jd1.4 Yl1.2 Ydmg1.4 Ydive1.5", lift_card(damage=1.4), 0.7),
        ("Jd1.4 Yl1.2 Ydmg1.3 Ysh1.1 Ydive1.5", lift_card(max_shield=1.1), 0.7),
        ("Jd1.4 Yl1.25 Ydmg1.3 Ydive1.5", lift_card(lift=1.25), 0.7),
        ("Jd1.35 Yl1.2 Ydmg1.4 Ydive1.5", lift_card(drag=1.35, damage=1.4), 0.7),
    ]
    return [candidate(name, e, t, gate) for name, (e, t), gate in cards]


ROUNDS = {
    "round4": round4,
    "round0": round0,
    "flight_grid": flight_grid,
    "round1": round1,
    "round2": round2,
    "round3": round3,
}

if __name__ == "__main__":
    tag = sys.argv[1]
    out = ROUNDS[tag]()
    path = HERE / tag / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")

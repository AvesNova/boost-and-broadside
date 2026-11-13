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


def absolute(base: dict, factors: dict) -> dict:
    """A card dict of factors as absolute card values on its base row."""
    out = dict(base)
    for field, factor in factors.items():
        out[field] = factor["set"] if isinstance(factor, dict) else base[field] * factor
    return out


def middle_of(energy: dict, turn: dict, slot: str = "yongma", factors: dict | None = None) -> dict:
    """The neutral ship between a card's Yongma and Jujak (the geometric mean of the two in
    flight, shield, damage and size, as the archetypes' M on the frozen pair), on a slot's
    base row, then factors."""
    y, j = absolute(Y, energy), absolute(J, turn)
    base = Y if slot == "yongma" else J
    fields = set(FLIGHT + NEUTRAL) | set(energy) | set(turn)
    fields -= {"explosive_share"}
    out = {f: {"set": math.sqrt(y[f] * j[f])} for f in fields}
    out["explosive_share"] = {"set": 0.5 * (y["explosive_share"] + j["explosive_share"])}
    out = {f: v for f, v in out.items() if v["set"] != base[f]}
    for field, factor in (factors or {}).items():
        if isinstance(factor, dict):
            out[field] = factor
        elif field in out:
            out[field] = {"set": out[field]["set"] * factor}
        else:
            out[field] = factor
    return out


def three_set(name: str, energy: dict, turn: dict, gate: float, middle: dict | None = None):
    """A three-ship set file for `threeship.py`."""
    m = middle or {}
    return dict(
        name=name,
        yongma=energy,
        jujak=turn,
        middle_energy=middle_of(energy, turn, "yongma", m),
        middle_turn=middle_of(energy, turn, "jujak", m),
        player=dict(doctrine=dict(turn=dict(duel_extend_in_shield_ratio=[gate, 0.1]))),
    )


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


def round5() -> list[dict]:
    """The dive exchange raises Yongma's sustained turn (x1.5: 0.248 rad/s), so card A's Jujak
    is only +2.4%: back to Jujak drag x1.3 (+7%) and x1.35 (+5%), with Jujak's nose cone
    softer (round 2: x1.3 was +0.05 in the duel, +0.03 at 5v5) and Yongma's damage up."""
    cards = []
    for drag, damage, dive, jf in (
        (1.35, 1.3, 1.5, 1.5),
        (1.35, 1.4, 1.5, 1.5),
        (1.35, 1.3, 1.5, 2.0),
        (1.3, 1.4, 1.5, 2.0),
        (1.3, 1.5, 1.5, 1.5),
        (1.35, 1.4, 1.3, 1.5),
        (1.3, 1.5, 1.5, 2.0),
        (1.35, 1.5, 1.5, 2.0),
    ):
        energy, turn = lift_card(drag=drag, damage=damage, dive=dive)
        name = f"Jd{drag} Yl1.2 Ydmg{damage} Ydive{dive} Jf{jf}"
        cards.append(candidate(name, energy, turn | cone(jf), 0.7))
    return cards


def card_b(drag=1.35, damage=1.2, dive=1.3, jf=1.5, **more) -> tuple[dict, dict]:
    """Round 5's shape: Jujak drag, Yongma lift x1.2, damage and dive, Jujak's softer nose."""
    energy, turn = lift_card(drag=drag, damage=damage, dive=dive, **more)
    return energy, turn | cone(jf)


def round6() -> list[dict]:
    """Round 5's Jd1.35 Ydmg1.4 Ydive1.3 Jf1.5 (0.65 / 0.75 / 0.59 at 1v1 / 5v5 / 10v10: flat
    but Yongma's) with less damage, all sizes."""
    cards = [
        dict(damage=1.1),
        dict(damage=1.2),
        dict(damage=1.0),
        dict(damage=1.2, jf=1.3),
        dict(damage=1.2, dive=1.4),
        dict(damage=1.1, dive=1.2),
        dict(damage=1.2, drag=1.3),
        dict(damage=1.3, jf=1.3),
    ]
    out = []
    for c in cards:
        k = dict(drag=1.35, damage=1.2, dive=1.3, jf=1.5) | c
        name = f"Jd{k['drag']} Yl1.2 Ydmg{k['damage']} Ydive{k['dive']} Jf{k['jf']}"
        out.append(candidate(name, *card_b(**c), 0.7))
    return out


def round7() -> list[dict]:
    """Round 6's Jd1.35 Yl1.2 Ydmg1.1 Ydive1.2 Jf1.5 (0.58 / 0.74 / 0.67 / 0.53 at 1v1 / 2v2 /
    5v5 / 10v10): single levers looking for one that takes 2v2 and 5v5 down more than 1v1
    and 10v10."""
    base = dict(damage=1.1, dive=1.2)
    levers = [
        ("", {}, {}),
        (" Yvis0.9", {"vision_range_scale": 0.9}, {}),
        (" Jvis1.1", {}, {"vision_range_scale": 1.1}),
        (" Yrange0.85", {"bullet_life": 0.85}, {}),
        (" Ysh0.8 Ydmg1.2", {"max_shield": 0.8, "hit_damage": 1.2}, {}),
        (" Yrad1.1", {"hit_radius": 1.1}, {}),
        (" Jdmg1.1", {}, {"hit_damage": 1.1}),
        (" Jsh1.15", {}, {"max_shield": 1.15}),
        (" Ypower0.9", {"engine_power": 0.9}, {}),
        (" Ydmg1.0", {"hit_damage": 1.0}, {}),
    ]
    out = []
    for suffix, energy_more, turn_more in levers:
        energy, turn = card_b(**base)
        energy = energy | energy_more
        name = "Jd1.35 Yl1.2 Ydmg1.1 Ydive1.2 Jf1.5" + suffix
        out.append(candidate(name, energy, turn | turn_more, 0.7))
    return out


ROUNDS = {
    "round7": round7,
    "round6": round6,
    "round5": round5,
    "round4": round4,
    "round0": round0,
    "flight_grid": flight_grid,
    "round1": round1,
    "round2": round2,
    "round3": round3,
}

GAME = dict(
    ships_per_team=5,
    num_fields=10,
    arena_radius=7800.0,
    episode_seconds=600.0,
    respawn_seconds=10.0,
    mode="team_deathmatch",
    spawn_groups=[],
    kills_per_ship=5,
    ship_set="real-scale",
    layout=None,
)


def write_map(name: str, energy: dict, turn: dict) -> pathlib.Path:
    """An opt-in play map `yj-<name>`: the card's changed fields as absolute values, the
    release lineup (2 Yongma + 3 Jujak)."""
    ships = {}
    for slot, base, factors in (("yongma", Y, energy), ("jujak", J, turn)):
        values = absolute(base, factors)
        ships[slot] = {f: float(values[f]) for f in factors}
    data = dict(game=GAME, lineup=["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"], ships=ships)
    path = HERE.parents[3] / "src/bnb/game/maps" / f"yj-{name}.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


if __name__ == "__main__":
    tag = sys.argv[1]
    out = ROUNDS[tag]()
    path = HERE / tag / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")

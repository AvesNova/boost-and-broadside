"""Candidates for `sizes.py`: each card x each doctrine, the doctrine's fields on both types
on top of r9m (the maneuvers section's champion: the duel one-circle and the crowd
extension).

    python docs/rewrite/sections/maneuvers-2/make.py round1 [card ...]

`ROUNDS[name]` lists the doctrines (a name, and the fields over r9m; a tuple gives the
Yongma's and the Jujak's). The cards: D2, D2 with the 12-degree p8 cone at f0 0.085 and
damage x1.5, and the archetypes branch's big v small pair (confirm 1's, on hull M).
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

D2_ENERGY = {
    "max_shield": 0.5,
    "bullet_life": 0.75,
    "head_on_fraction_scale": 0.7,
    "hit_damage": 1.1,
}
D2_TURN = {"induced_drag_scale": 1.6, "head_on_fraction_scale": 0.5}
CONE = {"head_on_angle_scale": {"set": 0.28388928317955997}, "head_on_sharpness": {"set": 8}}


def armour(f0: float, damage: float) -> tuple[dict, dict]:
    fraction = {"head_on_fraction_scale": {"set": f0 / 0.3}}
    energy = dict(D2_ENERGY, **fraction, **CONE, hit_damage=1.1 * damage)
    turn = dict(D2_TURN, **fraction, **CONE, hit_damage=damage)
    return energy, turn


BIG_SMALL = json.loads((HERE / "big_small.json").read_text())
CARDS = {
    "D2": (D2_ENERGY, D2_TURN),
    "cone 0.085": armour(0.085, 1.5),
    "big v small": (BIG_SMALL["energy"], BIG_SMALL["turn"]),
    "frozen": ({}, {}),
}

ONE = dict(sense=1.0, sense_one_circle=1.0, sense_in_gun_range=[4.0, 0.2])
DUEL = dict(ONE, sense_mirror_only=1.0, sense_in_crowd=[0.5, 0.1])
STRAIGHT = dict(crowd_sense=1.0, crowd_sense_stick=0.0, crowd_sense_boost=1.0)
R9M = dict(DUEL, **STRAIGHT)


def press_energy(ratio: float) -> dict:
    """Extend in the crowd only below this own-over-target specific energy."""
    return dict(crowd_sense_in_energy_ratio=[ratio, 0.05])


def press_shield(ratio: float) -> dict:
    return dict(crowd_sense_in_shield_ratio=[ratio, 0.1])


def threat(radians: float) -> dict:
    """Count an other enemy only with its track within this angle of the line to the ship."""
    return dict(sense_crowd_in_nose_angle=[radians, 0.15])


def duel_extend(energy=None, shield=None, lift=None) -> dict:
    out = dict(duel_extend=1.0)
    if energy:
        out["duel_extend_in_energy_ratio"] = [energy, 0.05]
    if shield:
        out["duel_extend_in_shield_ratio"] = [shield, 0.1]
    if lift:
        out["duel_extend_in_lift_ratio"] = [lift, 0.03]
    return out


def hold(gun_ranges: float, stall_angles: float = 1.0) -> dict:
    return dict(
        hold_fire=1.0,
        hold_fire_in_gun_range=[gun_ranges, 0.15],
        hold_fire_in_stall_angle=[stall_angles, 0.2],
    )


def de(shield=0.6, scale=0.1, energy=None) -> dict:
    out = dict(duel_extend=1.0, duel_extend_in_shield_ratio=[shield, scale])
    if energy:
        out["duel_extend_in_energy_ratio"] = [energy, 0.05]
    return out


MIRROR = dict(duel_extend_mirror_only=1.0)
ROUNDS = {
    "round1": [
        ("r9m", {}),
        ("press energy 1.3", press_energy(1.3)),
        ("press energy 1.0", press_energy(1.0)),
        ("press shield 1.5", press_shield(1.5)),
        ("press shield 1.0", press_shield(1.0)),
        ("threat 0.8 rad", threat(0.8)),
        ("threat 0.4 rad", threat(0.4)),
        ("allies 1", dict(sense_crowd_ally_weight=1.0)),
        ("duel extend shield 0.6", duel_extend(shield=0.6)),
        ("duel extend energy 0.8", duel_extend(energy=0.8)),
        ("duel extend lift 0.95, any type", dict(duel_extend(lift=0.95), sense_mirror_only=0.0)),
        ("hold 1 gr", hold(1.0)),
        ("hold 0.5 gr", hold(0.5)),
    ],
    "round2": [
        ("r9m", {}),
        ("DE shield 0.4", de(0.4)),
        ("DE shield 0.6", de(0.6)),
        ("DE shield 0.8", de(0.8)),
        ("DE shield 1.0", de(1.0)),
        ("DE shield 0.6 soft", de(0.6, 0.3)),
        ("DE shield 0.6 + energy 0.6", de(0.6, energy=0.6)),
        ("DE shield 0.6, Yongma only", (de(0.6), {})),
        ("DE shield 0.6 + energy 0.8, Yongma only", (de(0.6, energy=0.8), {})),
    ],
    "round3": [
        ("r9m", {}),
        ("DE shield 0.4", de(0.4)),
        ("DE shield 0.5", de(0.5)),
        ("DE Y 0.6 / J 0.4", (de(0.6), de(0.4))),
        ("DE Y 0.8 / J 0.4", (de(0.8), de(0.4))),
        ("DE Y 1.0 / J 0.4", (de(1.0), de(0.4))),
        ("DE Y 0.6 + energy 0.6 / J 0.4", (de(0.6, energy=0.6), de(0.4))),
    ],
    "round4": [
        ("r9m", {}),
        ("DE 0.4", de(0.4)),
        ("DE 0.4 mirror", dict(de(0.4), **MIRROR)),
        ("DE 0.5 mirror", dict(de(0.5), **MIRROR)),
        ("DE Y 0.6 / J 0.4 mirror", (dict(de(0.6), **MIRROR), dict(de(0.4), **MIRROR))),
        ("DE Y 1.0 / J 0.4 mirror", (dict(de(1.0), **MIRROR), dict(de(0.4), **MIRROR))),
    ],
    "confirm": [
        ("r9m", {}),
        ("DE 0.4", de(0.4)),
        ("DE 0.4 mirror", dict(de(0.4), **MIRROR)),
        ("DE 0.5 mirror", dict(de(0.5), **MIRROR)),
        ("DE Y 1.0 / J 0.4", (de(1.0), de(0.4))),
    ],
    "round5": [
        ("r9m", {}),
        ("DE Y 1.0 / J 0.4", (de(1.0), de(0.4))),
        ("DE Y 1.0 / J 0.6", (de(1.0), de(0.6))),
        ("DE Y 0.7 / J 0.4", (de(0.7), de(0.4))),
        ("DE Y 1.0 / J 0.4 mirror", (de(1.0), dict(de(0.4), **MIRROR))),
        ("DE Y 0.7 / J 0.6", (de(0.7), de(0.6))),
    ],
}


def main() -> None:
    name = sys.argv[1]
    cards = sys.argv[2:] or list(CARDS)
    out = []
    for card in cards:
        energy, turn = CARDS[card]
        for label, fields in ROUNDS[name]:
            pair = fields if isinstance(fields, tuple) else (fields, fields)
            out.append(
                dict(
                    name=f"{card} | {label}",
                    energy=energy,
                    turn=turn,
                    doctrine=dict(energy=dict(R9M, **pair[0]), turn=dict(R9M, **pair[1])),
                )
            )
    path = HERE / name / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()

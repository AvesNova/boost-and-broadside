"""Candidates for `sizes.py`: each card x each maneuver set, the maneuver set's doctrine
fields applied to both types on top of r9.

    python docs/rewrite/sections/maneuvers/make.py round1 [card ...]

`ROUNDS[name]` lists the maneuver sets (a name and the doctrine fields); the cards are
`CARDS` (the frozen card, balance-d2, and d2 with the tank-armour cones).
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
    """D2 with the 12-degree p8 cone at a dead-on fraction f0 (of a full hit) on both types,
    the damage times `damage`."""
    fraction = {"head_on_fraction_scale": {"set": f0 / 0.3}}
    energy = dict(D2_ENERGY, **fraction, **CONE, hit_damage=1.1 * damage)
    turn = dict(D2_TURN, **fraction, **CONE, hit_damage=damage)
    return energy, turn


CARDS = {
    "frozen": ({}, {}),
    "D2": (D2_ENERGY, D2_TURN),
    "D2 cone 0.085": armour(0.085, 1.5),  # tank-armour round 3's pick (inside the 300 s band)
    "D2 cone 0.01": armour(0.01, 1.5),  # the near-immune nose: the side-by-side stalemate
}


def lead(seconds, stick=1.0, one_circle=0.0, crossing=0.5):
    return dict(
        lead_turn=1.0,
        lead_turn_in_seconds=[seconds, 0.1],
        lead_turn_in_crossing=[crossing, 0.2],
        lead_turn_stick=stick,
        lead_turn_one_circle=one_circle,
    )


def yoyo(after=20.0, phase=1.5, climb=4.0, dive=4.0, ease=0.4, low=0.0, to=10.0):
    return dict(
        yoyo=1.0,
        yoyo_low=low,
        yoyo_to_corner_speed=to,
        yoyo_after_seconds=after,
        yoyo_phase=phase,
        yoyo_climb_seconds=climb,
        yoyo_dive_seconds=dive,
        yoyo_ease=ease,
    )


ONE = dict(sense=1.0, sense_one_circle=1.0, sense_in_gun_range=[4.0, 0.2])
DUEL = dict(ONE, sense_mirror_only=1.0, sense_in_crowd=[0.5, 0.1])
STRAIGHT = dict(crowd_sense=1.0, crowd_sense_stick=0.0, crowd_sense_boost=1.0)
ROUNDS = {
    "round1": [
        ("r9", {}),
        ("lead 1 s", lead(1.0)),
        ("lead 2 s", lead(2.0)),
        ("lead 3 s", lead(3.0)),
        ("lead 5 s", lead(5.0)),
        ("lead 2 s stick 0.6", lead(2.0, stick=0.6)),
        ("lead 2 s one-circle", lead(2.0, one_circle=1.0)),
        ("lead 4 s one-circle", lead(4.0, one_circle=1.0)),
        ("yoyo 20 s", yoyo()),
        ("yoyo 10 s", yoyo(after=10.0)),
        ("yoyo 20 s climb 8", yoyo(climb=8.0, dive=6.0)),
        ("yoyo 20 s phase 0.5", yoyo(phase=0.5)),
        ("brake", dict(brake=1.0)),
    ],
    "round2": [
        ("r9", {}),
        ("lead 0.5 s", lead(0.5)),
        ("lead 1 s", lead(1.0)),
        ("lead 1 s stick 0.6", lead(1.0, stick=0.6)),
        ("lead 1.5 s", lead(1.5)),
        ("low 10 s ph1.5 4 s", yoyo(10.0, 1.5, 4.0, 2.0, 0.0, 1.0, 1.0)),
        ("low 10 s ph3.1 4 s", yoyo(10.0, 3.1, 4.0, 2.0, 0.0, 1.0, 1.0)),
        ("low 20 s ph1.5 6 s", yoyo(20.0, 1.5, 6.0, 2.0, 0.0, 1.0, 1.0)),
        ("low 10 s ph1.5 3 s ease 0.4 to 0.8", yoyo(10.0, 1.5, 3.0, 2.0, 0.4, 1.0, 0.8)),
        ("low 5 s ph1.5 4 s", yoyo(5.0, 1.5, 4.0, 2.0, 0.0, 1.0, 1.0)),
        ("low 10 s ph0.5 4 s", yoyo(10.0, 0.5, 4.0, 2.0, 0.0, 1.0, 1.0)),
        ("low 10 s ph1.5 8 s to 1.3", yoyo(10.0, 1.5, 8.0, 2.0, 0.0, 1.0, 1.3)),
        ("high 20 s (gated)", yoyo()),
    ],
    "round3": [
        ("r9", {}),
        ("lead 0.25 s", lead(0.25)),
        ("lead 0.5 s", lead(0.5)),
        ("lead 0.75 s", lead(0.75)),
        ("lead 0.5 s cross 1.0", lead(0.5, crossing=1.0)),
        ("lead 0.5 s cross 1.5", lead(0.5, crossing=1.5)),
        ("lead 0.75 s cross 1.0", lead(0.75, crossing=1.0)),
        ("lead 1 s cross 1.0", lead(1.0, crossing=1.0)),
        ("lead 0.5 s stick 0.75", lead(0.5, stick=0.75)),
        ("lead 0.5 s one-circle", lead(0.5, one_circle=1.0)),
        ("lead 0.5 s + brake", dict(lead(0.5), brake=1.0)),
    ],
    "round4": [
        ("r9", {}),
        ("sense two-circle", dict(sense=1.0)),
        ("sense one-circle", dict(sense=1.0, sense_one_circle=1.0)),
        ("sense two-circle 1 gr", dict(sense=1.0, sense_in_gun_range=[1.0, 0.2])),
        (
            "sense one-circle 1 gr",
            dict(sense=1.0, sense_one_circle=1.0, sense_in_gun_range=[1.0, 0.2]),
        ),
        (
            "sense one-circle 4 gr",
            dict(sense=1.0, sense_one_circle=1.0, sense_in_gun_range=[4.0, 0.2]),
        ),
        ("sense one-circle 0.5", dict(sense=0.5, sense_one_circle=1.0)),
        ("lead 0.5 s + sense two-circle", dict(lead(0.5), sense=1.0)),
        ("lead 0.5 s + sense one-circle", dict(lead(0.5), sense=1.0, sense_one_circle=1.0)),
    ],
    "confirm": [
        ("r9", {}),
        ("lead 0.5 s", lead(0.5)),
        ("lead 0.5 s cross 1.0", lead(0.5, crossing=1.0)),
    ],
    "round5": [
        ("r9", {}),
        ("lead 0.5 s", lead(0.5)),
        ("one-circle", ONE),
        ("one-circle mirror", dict(ONE, sense_mirror_only=1.0)),
        ("one-circle lift 0.95", dict(ONE, sense_in_lift_ratio=[0.95, 0.03])),
        ("lead + one-circle mirror", dict(lead(0.5), **ONE, sense_mirror_only=1.0)),
        ("lead + one-circle lift 0.95", dict(lead(0.5), **ONE, sense_in_lift_ratio=[0.95, 0.03])),
        ("J one-circle, Y lead", (lead(0.5), dict(lead(0.5), **ONE))),
    ],
    "round6": [
        ("r9", {}),
        ("one-circle mirror duel", DUEL),
        ("one-circle mirror duel, crowd straight boost", dict(DUEL, **STRAIGHT)),
        ("one-circle mirror duel, crowd straight", dict(DUEL, crowd_sense=1.0)),
        (
            "one-circle mirror duel 5 gr, crowd straight boost",
            dict(DUEL, **STRAIGHT, sense_crowd_in_gun_range=[5.0, 0.1]),
        ),
        (
            "one-circle mirror duel, crowd half one-circle",
            dict(DUEL, crowd_sense=0.5, crowd_sense_stick=1.0),
        ),
        ("one-circle at 0.5 (round 4)", dict(sense=0.5, sense_one_circle=1.0)),
        ("crowd straight boost only", dict(DUEL, sense=0.0, **STRAIGHT)),
        (
            "lead + one-circle mirror duel, crowd straight boost",
            dict(lead(0.5), **DUEL, **STRAIGHT),
        ),
    ],
}


def main() -> None:
    name = sys.argv[1]
    cards = sys.argv[2:] or list(CARDS)
    out = []
    for card in cards:
        energy, turn = CARDS[card]
        for label, fields in ROUNDS[name]:
            out.append(
                dict(
                    name=f"{card} | {label}",
                    energy=energy,
                    turn=turn,
                    doctrine=(
                        dict(energy=fields[0], turn=fields[1])
                        if isinstance(fields, tuple)
                        else dict(energy=fields, turn=fields)
                    ),
                )
            )
    path = HERE / name / "candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


if __name__ == "__main__":
    main()
